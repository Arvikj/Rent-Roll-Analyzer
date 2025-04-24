#!/usr/bin/env python3
"""
Rent-Roll Analyzer Streamlit App
"""

from __future__ import annotations

import hashlib, io, json, logging, tempfile, zipfile, os
from pathlib import Path
from typing import Any
import concurrent.futures
import time

import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ── internal modules ────────────────────────────────────────────────
from src import file_parser
from src.data_processing import clamp_budget, compute_metadata, serialize, validate_and_clean
from src.llm_interface import (
    CALL_TIMEOUT, JSON_DELAY, build_prompt, call_gemini_with_retry,
    extract_json_slice, quick_insights, compare_insights, _call_with_timeout 
)
# ────────────────────────────────────────────────────────────────────

load_dotenv()
MAX_MB = 2
st.set_page_config("Rent-Roll Analyzer", layout="wide")

############################################################################
#  Minimal logging (console only)
############################################################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

############################################################################
#  Sidebar – file-uploader
############################################################################
uploader_key = st.session_state.get("uploader_key", "u0")
blobs = st.sidebar.file_uploader(
    "Upload Rent Roll Files",
    type=["csv", "xls", "xlsx", "pdf", "zip"],
    accept_multiple_files=True,
    key=uploader_key,
    help=f"Upload individual files or a ZIP archive. Max individual file size: {MAX_MB}MB."
)


############################################################################
#  Session-state holders
############################################################################
ss = st.session_state
ss.setdefault("results", {})
ss.setdefault("failed", {})
ss.setdefault("multi_insight", "")
# Add state to track active tab for conditional sidebar controls
ss.setdefault("active_tab_key", "Preview") # Default to first tab

############################################################################
#  Helpers 
############################################################################
sha1 = lambda b: hashlib.sha1(b).hexdigest()

def preview(name: str, data: bytes):
    """Displays a preview of the uploaded file."""
    ext = Path(name).suffix.lower(); bio = io.BytesIO(data)
    try:
        if ext in {".xls",".xlsx"}:
            st.dataframe(pd.read_excel(bio,nrows=15).astype(str))
        elif ext==".csv":
            st.dataframe(pd.read_csv(bio,nrows=15).astype(str))
        elif ext==".pdf":
            with pdfplumber.open(bio) as pdf:
                if pdf.pages:
                    img = pdf.pages[0].to_image(resolution=110).original
                    # Use Pillow to check image mode (optional, good practice)
                    if isinstance(img, Image.Image):
                       st.image(img, caption="First page preview")
                    else: # Handle cases where to_image might fail subtly
                       st.warning("Could not render PDF page preview.")
                else:
                    st.warning("PDF has no pages.")
    except Exception as e:
        st.warning(f"Preview failed for {name}: {e}")

def enum_files(uploaded_blobs):
    """Handles direct uploads and extracts files from ZIP archives."""
    files_to_process = []
    if not uploaded_blobs: return files_to_process # Return empty list if no blobs

    for uf in uploaded_blobs:
        if uf.type == "application/zip":
            try:
                with zipfile.ZipFile(io.BytesIO(uf.getvalue())) as zf:
                    for zinfo in zf.infolist():
                        if zinfo.is_dir() or zinfo.filename.startswith('__'): continue
                        if zinfo.file_size > MAX_MB * 1024 * 1024:
                            logging.warning(f"Skipping large file in ZIP '{uf.name}': {zinfo.filename} ({zinfo.file_size / (1024*1024):.2f}MB)")
                            # Consider adding to failed state based on filename? Hashing is hard here.
                            continue
                        if Path(zinfo.filename).suffix.lower() not in ['.csv', '.xls', '.xlsx', '.pdf']:
                            logging.warning(f"Skipping unsupported file type in ZIP '{uf.name}': {zinfo.filename}")
                            continue
                        file_content = zf.read(zinfo)
                        files_to_process.append((zinfo.filename, file_content))
            except zipfile.BadZipFile:
                logging.error(f"Could not read ZIP file: {uf.name}. It might be corrupted.")
                st.error(f"Failed to read ZIP file: {uf.name}. Skipping.")
            except Exception as e:
                 logging.error(f"Error processing ZIP file {uf.name}: {e}", exc_info=True)
                 st.error(f"Unexpected error processing ZIP: {uf.name}. Skipping.")
        else:
            file_content = uf.getvalue()
            file_hash = sha1(file_content) # Calculate hash early
            if len(file_content) > MAX_MB * 1024 * 1024:
                 logging.warning(f"Skipping large file: {uf.name} ({len(file_content) / (1024*1024):.2f}MB)")
                 ss.failed[file_hash] = f"File size exceeds {MAX_MB}MB limit." # Use hash as key
                 continue
            files_to_process.append((uf.name, file_content))

    return files_to_process

def gemini_extract(fname:str, parsed_data: Any, meta_data: dict) -> dict | None:
    """Builds prompt, calls Gemini, validates and cleans the result."""
 
    logging.info(f"Building prompt for {fname}...")
    prompt = build_prompt(fname, meta_data, serialize(parsed_data))

    logging.info(f"Calculating thinking budget for {fname}...")
    data_rows_for_budget = 0
    if meta_data.get("type") == "table":
        data_rows_for_budget = meta_data.get("data_rows_estimate", 0)
    elif meta_data.get("type") == "excel":
        data_rows_for_budget = sum(v.get("data_rows_estimate", 0) for v in meta_data.get("sheets", {}).values())
    budget = clamp_budget(data_rows_for_budget)
    logging.info(f"Budget set to {budget} for {fname}")

    parsed_json = None
    logging.info(f"Calling Gemini for {fname}...")
    for pass_ix, current_budget in enumerate([budget, 0], start=1):
         logging.info(f"   Attempting Gemini call (Pass {pass_ix}, Budget: {current_budget}) for {fname}")
         try:
            # Note: _call_with_timeout wraps call_gemini_with_retry
            resp = _call_with_timeout(
                lambda: call_gemini_with_retry(prompt, current_budget),
                CALL_TIMEOUT if pass_ix == 1 else CALL_TIMEOUT * 2,
            )
            # Check response validity before accessing text
            if not resp or not hasattr(resp, 'text'):
                 raise ValueError("Invalid or empty response received from Gemini call.")

            raw_text = resp.text or ""
            snippet = extract_json_slice(raw_text)
            try:
                parsed_json = json.loads(snippet)
                logging.info(f"   Successfully parsed JSON response (Pass {pass_ix}) for {fname}")
                break
            except json.JSONDecodeError as json_err:
                logging.warning(f"   JSON parse failed (Pass {pass_ix}) for {fname}: {json_err}")
                if pass_ix == 1: time.sleep(JSON_DELAY)
         except concurrent.futures.TimeoutError:
              logging.warning(f"   Gemini call timed out (Pass {pass_ix}) for {fname}")
              if pass_ix == 1: time.sleep(JSON_DELAY)
         except Exception as e:
              logging.error(f"   Gemini API or processing error (Pass {pass_ix}) for {fname}: {e}", exc_info=True)
              break # Break loop on other errors

    if parsed_json is None:
        logging.error(f"Failed to get valid JSON from Gemini for {fname}.")
        return None

    logging.info(f"Validating and cleaning result for {fname}...")
    try:
        result = validate_and_clean(parsed_json, meta_data)
        return result
    except Exception as e:
        logging.error(f"Error during validation/cleaning for {fname}: {e}", exc_info=True)
        return None

############################################################################
#  Idle Screen Enhancement
############################################################################
if not blobs:
    st.title("Rent-Roll Analyzer")
    st.markdown("Welcome! Upload your rent roll files using the sidebar to get started.")

    st.markdown("""
    **How it works:**
    1.  **Upload:** Use the sidebar to upload Excel, PDF, or ZIP files containing rent rolls.
    2.  **Process:** The app parses the files and uses AI to extract/generate key summary metrics and provide actionable insights.
    3.  **Analyze:** View results, visualize data, and compare summaries across multiple files.
    """)
    st.stop() # Stop execution if no files are uploaded

############################################################################
#  Tabs Setup
############################################################################
st.title("Rent-Roll Analyzer") # Keep title visible when files are loaded
tab_titles = ["File Preview", "Results", "Visualize", "Compare"]
prev_tab, res_tab, vis_tab, cmp_tab = st.tabs(tab_titles)


############################################################################
#  Process uploads once per session (Loop)
############################################################################
files_to_process = enum_files(blobs)
# ── Purge any no-longer‐uploaded files from session_state ──
# compute the set of hashes for all currently uploaded items
current_hashes = { sha1(data) for (_, data) in files_to_process }

# find any old hashes we stored but the user has since removed
to_remove = (set(ss.results.keys()) | set(ss.failed.keys())) - current_hashes
if to_remove:
    for h in to_remove:
        ss.results.pop(h, None)
        ss.failed.pop(h, None)
    ss.multi_insight = ""

total_files = len(files_to_process)
progress_bar = st.progress(0, text="Initializing..." if total_files > 0 else "No files selected.")

processed_count = 0
if total_files > 0:
    for i, (fname, raw_data) in enumerate(files_to_process, 1):
        file_hash = sha1(raw_data)

        # --- Check Session State ---
        if file_hash in ss.results or file_hash in ss.failed:
            logging.info(f"Skipping already processed/failed file: {fname}")
            processed_count += 1 # Increment count even for skipped files for progress bar logic
            continue # Skip to next file iteration

        # --- Process New File ---
        progress_text = f"Processing file {i}/{total_files}: {fname}"
        # Update progress before starting the actual processing for this file
        progress_bar.progress(i / total_files, text=progress_text)

        # Display preview in its tab
        with prev_tab:
            st.subheader(fname)
            preview(fname, raw_data)
            st.markdown("---") # Separator

        # Save to temp file for parsing
        temp_file_path = None # Ensure path is defined outside try
        try:
            # Use a context manager where possible, but need path outside
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(fname).suffix) as tmp:
                tmp.write(raw_data)
                temp_file_path = Path(tmp.name) # Get the path

            logging.info(f"Parsing file: {fname} from path {temp_file_path}")
            parsed_data = file_parser.parse_file(str(temp_file_path))

            if parsed_data is None:
                logging.error(f"Parse failed for: {fname}")
                ss.failed[file_hash] = "File parsing error" # More specific error
            else:
                logging.info(f"Computing metadata for: {fname}")
                meta_data = compute_metadata(parsed_data)

                logging.info(f"Extracting summary using Gemini for: {fname}")
                result_data = gemini_extract(fname, parsed_data, meta_data)

                if result_data:
                    logging.info(f"Successfully extracted summary for: {fname}")
                    ss.results[file_hash] = {"file": fname, **result_data}
                else:
                    logging.error(f"Gemini extraction/validation failed for: {fname}")
                    ss.failed[file_hash] = "AI processing error" # More specific

        except Exception as e:
             logging.error(f"Unexpected error processing {fname}: {e}", exc_info=True)
             ss.failed[file_hash] = "Unexpected processing error"
        finally:
            # --- Temporary File Cleanup ---
            # This deletes the temporary copy on the server disk.
            if temp_file_path and temp_file_path.exists():
                try:
                    os.unlink(temp_file_path)
                    logging.info(f"Deleted temporary file: {temp_file_path}")
                except Exception as del_err:
                    # Log error but don't stop the app
                    logging.error(f"Could not delete temporary file {temp_file_path}: {del_err}")
            # --- End Cleanup ---

        processed_count += 1 # Increment after processing attempts

    progress_bar.empty() # Remove progress bar when done

# Display summary of failures, if any, in the sidebar
if ss.failed:
    st.sidebar.error("Processing Issues Detected")
    # Use an expander in the sidebar
    with st.sidebar.expander(f"Files with Errors ({len(ss.failed)})"):
        # Try to map hash back to filename for display
        hash_to_name = {sha1(data): name for name, data in files_to_process}
        for file_hash, error_msg in ss.failed.items():
            display_name = hash_to_name.get(file_hash, f"File ID: {file_hash[:8]}...")
            st.error(f"`{display_name}`: {error_msg}")


############################################################################
#  Results tab (Displays results from session state)
############################################################################
with res_tab:
    if not ss.results:
        st.warning("No successful extractions yet. Upload files or check error messages in the sidebar.")
    else:
        results_list = list(ss.results.values()) # Get list of result dicts
        file_options = sorted([res_data["file"] for res_data in results_list]) # Sort names
        if not file_options:
             st.warning("No valid results available to display.")
        else:
            selected_file_name = st.selectbox("Select File to View Results", file_options, key="results_selector")

            selected_result_data = next((res_data for res_data in results_list if res_data["file"] == selected_file_name), None)

            if selected_result_data:
                inner_tab_kpi, inner_tab_json = st.tabs(["Metrics & Insights", "JSON Output"])

                with inner_tab_kpi:
                    st.subheader(f"Summary for: `{selected_file_name}`")
                    cols=st.columns(6)
                    def format_metric(value, fmt="{:,.0f}"):
                        return "—" if value in (None, "") else fmt.format(value)

                    kpi_map = [ # Key, Label, Format String 
                        ("total_units", "Total Units"),
                        ("occupancy_rate", "Occupancy", "{:.1%}"),
                        ("average_rent", "Avg Rent", "${:,.0f}"),
                        ("total_actual_rent", "Actual Rent", "${:,.0f}"),
                        ("total_market_rent", "Market Rent", "${:,.0f}"),
                        ("total_square_feet", "Total SqFt")
                    ]
                    for col, (key, label, *fmt_str) in zip(cols, kpi_map):
                        col.metric(label, format_metric(selected_result_data.get(key), *(fmt_str or ["{:,.0f}"])))

                    st.markdown("---")
                    if st.button("Generate Quick Insights", key=f"insight_{selected_file_name}"):
                        with st.spinner("Generating insights..."):
                            insights = quick_insights(selected_result_data)

                        st.markdown("#### Quick Insights:")
                        if insights:
                            # Render as plain text (no Markdown parsing)
                            st.text(insights)
                        else:
                            st.error("Could not generate insights.")


                    st.markdown("---")
                    try:
                        json_string_to_download = json.dumps(selected_result_data, indent=2)
                        st.download_button(
                            label=f"Download JSON Summary", # Simplified label
                            data=json_string_to_download,
                            file_name=f"{Path(selected_file_name).stem}_summary.json",
                            mime="application/json",
                            key=f"download_{selected_file_name}"
                        )
                    except Exception as e:
                        st.error(f"Could not prepare JSON for download: {e}")

                with inner_tab_json:
                    st.subheader(f"Full JSON Output for: `{selected_file_name}`")
                    # Use expanded=True to show the full JSON by default
                    st.json(selected_result_data, expanded=True)
            else:
                st.error(f"Could not retrieve data for the selected file: {selected_file_name}")

############################################################################
#  Visualise tab - Conditional Controls & Fixed Charts
############################################################################
with vis_tab:
    # Set active tab state when this tab is viewed
    ss.active_tab_key = "Visualize"

    st.subheader("Visualize Key Metrics")

    if not ss.results:
        st.info("Charts require successfully processed files. Upload files or check errors.")
    else:
        results_list = list(ss.results.values()) # Get list of result dicts
        vis_file_options = sorted([res_data["file"] for res_data in results_list])

        st.sidebar.markdown("---")
        st.sidebar.subheader("Chart Options")
        vis_selected_file_name = st.sidebar.selectbox(
            "Select File for Charts", vis_file_options, key="vis_file_selector_sidebar" # Unique key
        )
        vis_chart_choice = st.sidebar.radio(
            "Select Chart Type",
             ["Unit Status", "Occupancy Pie", "Charge Codes", "Actual vs Market Rent"],
            key="vis_chart_radio_sidebar" # Unique key
        )
        

        vis_result_data = next((res_data for res_data in results_list if res_data["file"] == vis_selected_file_name), None)

        if vis_result_data:
            st.markdown(f"#### Displaying: `{vis_selected_file_name}`")

 
            if vis_chart_choice == "Unit Status":
                try:
                    status_data = vis_result_data.get("status_breakdown", [])
                    if not status_data: raise ValueError("No status breakdown data.")
                    df = pd.DataFrame(status_data)
                    if "status" not in df.columns or "count" not in df.columns: raise ValueError("Missing 'status' or 'count'.")
                    fig = px.bar(df, x="status", y="count", title="Unit Status Counts", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Unit Status chart: {e}")

            elif vis_chart_choice == "Occupancy Pie":
                try:
                    
                    status_data = vis_result_data.get("status_breakdown", [])
                    if not status_data: raise ValueError("No status breakdown data.")
                    occupied_count=sum(item.get("count",0) for item in status_data if "occupied" in item.get("status","").lower())
                    vacant_count=sum(item.get("count",0) for item in status_data if "vacant" in item.get("status","").lower())
                    total_calc = sum(item.get("count",0) for item in status_data)
                    other_count = total_calc - occupied_count - vacant_count

                    pie_values = [occupied_count, vacant_count, other_count]
                    pie_names = ["Occupied", "Vacant", "Other Status"]
                    filtered_values = [v for v in pie_values if v > 0]
                    filtered_names = [n for v, n in zip(pie_values, pie_names) if v > 0]
                    if not filtered_values: raise ValueError("No data for pie chart.")

                    fig = px.pie(names=filtered_names, values=filtered_values, title="Occupancy Distribution", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Occupancy Pie chart: {e}")

            elif vis_chart_choice == "Charge Codes":
                try:
                    charge_data = vis_result_data.get("charge_codes", [])
                    if not charge_data: raise ValueError("No charge code data.")
                    df = pd.DataFrame(charge_data)
                    if "charge_type" not in df.columns or "total_amount" not in df.columns: raise ValueError("Missing 'charge_type' or 'total_amount'.")
                    df_filtered = df.dropna(subset=["total_amount"])
                    df_filtered = df_filtered[df_filtered["total_amount"] != 0]
                    if df_filtered.empty: raise ValueError("No non-zero charge amounts.")

                    fig = px.bar(df_filtered, x="charge_type", y="total_amount", title="Charge Code Totals ($)", template="plotly_white")
                    fig.update_layout(xaxis_title="Charge Type", yaxis_title="Total Amount ($)")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate Charge Codes chart: {e}")

            elif vis_chart_choice == "Actual vs Market Rent":
                try:
                    # 1. pull values
                    actual = vis_result_data.get("total_actual_rent")
                    market = vis_result_data.get("total_market_rent")
                    if actual is None or market is None:
                        raise ValueError("Missing actual or market rent data.")

                    # 2. compute difference
                    diff = market - actual
                    pct_diff = (diff / market) if market else None

                    # 3. build DataFrame including the delta bar
                    df_rent = pd.DataFrame({
                        'Rent Type': ['Actual Rent', 'Market Rent', 'Difference'],
                        'Amount':   [actual,      market,      diff      ]
                    })

                    # 4. plot
                    fig = px.bar(
                        df_rent,
                        x='Rent Type',
                        y='Amount',
                        title='Actual vs Market Rent (with Difference)',
                        text='Amount',
                        template='plotly_white'
                    )
                    # show full dollar values on bars
                    fig.update_traces(
                        texttemplate='$%{y:,.0f}',
                        textposition='outside',
                        cliponaxis=False
                    )
                    # zoom y-axis tightly around our 3 values
                    y_min = df_rent.Amount.min() * 0.95
                    y_max = df_rent.Amount.max() * 1.05
                    fig.update_layout(
                        yaxis_title='Amount ($)',
                        yaxis_range=[y_min, y_max]
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # 5. call out percentage difference as a metric
                    if pct_diff is not None:
                        st.metric(
                            label="Market vs Actual Rent Gap",
                            value=f"${diff:,.0f}",
                            delta=f"{pct_diff:.1%}"
                        )

                except Exception as e:
                    st.warning(f"Could not generate Actual vs Market Rent chart: {e}")

        else:
            st.error(f"Could not load data for '{vis_selected_file_name}' to generate visualizations.")


############################################################################
#  Compare tab
############################################################################
with cmp_tab:
    # Set active tab state
    ss.active_tab_key = "Compare"

    results_list = list(ss.results.values())
    if len(results_list) < 2:
        st.info("Upload and successfully process at least two files to enable comparison features.")
    else:
        st.subheader("Multi-File Comparison Table")
        try:
            compare_cols = [
                "file", "total_units", "occupancy_rate", "average_rent",
                "total_actual_rent", "total_market_rent", "total_square_feet"
            ]
            data_for_df = []
            for res_dict in results_list:
                row_data = {'file': res_dict.get('file', 'Unknown File')}
                for col in compare_cols[1:]:
                    row_data[col] = res_dict.get(col)
                data_for_df.append(row_data)

            cmp_df = pd.DataFrame(data_for_df).set_index("file")
            st.dataframe(cmp_df.style.format({
                "occupancy_rate": "{:.1%}", "average_rent": "${:,.0f}",
                "total_actual_rent": "${:,.0f}", "total_market_rent": "${:,.0f}",
                "total_square_feet": "{:,.0f}", "total_units": "{:,.0f}",
            }, na_rep="-"))

            st.markdown("---")
            st.subheader("Comparison Insights")
 
            if st.button("Generate Comparison Insights", key="compare_insight_button"):
                with st.spinner("Generating comparison insights... This may take a moment."):
                    ss.multi_insight = compare_insights(ss.results)

            # Always display the latest insights if they exist; otherwise prompt
            if ss.multi_insight:
                st.text(ss.multi_insight)
            else:
                st.caption("Click the button above to generate comparative insights.")



            st.markdown("---")
            # Download all results button
            try:
              
                all_results_jsonl = "\n".join(json.dumps(res, default=str) for res in results_list) 
                st.download_button(
                    label="Download All Results (JSONL)",
                    data=all_results_jsonl,
                    file_name="all_rentroll_summaries.jsonl",
                    mime="application/jsonl",
                    key="download_all"
                )
            except Exception as e:
                st.error(f"Failed to prepare all results for download: {e}")

        except Exception as e:
            st.error(f"Failed to create comparison view: {e}")
            logging.error("Error creating comparison view:", exc_info=True)