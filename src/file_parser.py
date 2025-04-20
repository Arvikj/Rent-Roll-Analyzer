# src/file_parser.py

import pandas as pd
import pdfplumber
import os
from pathlib import Path
import logging

# Configure basic logging to show info messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_excel(file_path: Path) -> dict | None:
    """
    Parses an Excel file (.xlsx, .xls) and extracts data from all sheets.

    Args:
        file_path: Path object pointing to the Excel file.

    Returns:
        A dictionary where keys are sheet names and values are pandas DataFrames
        containing data read as strings, or None if parsing fails completely.
        Returns an empty dictionary if the file has no sheets.
    """
    logging.info(f"Attempting to parse Excel file: {file_path.name}")
    try:
        # Use pd.ExcelFile to handle multiple sheets efficiently
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_data = {}
        if not xls.sheet_names:
            logging.warning(f"Excel file has no sheets: {file_path.name}")
            return {} # Return empty dict if no sheets

        for sheet_name in xls.sheet_names:
            try:
                # Read sheet without assuming header, prevent 'NA' interpretation, keep blank rows
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None, na_filter=False, keep_default_na=False)
                # Convert all data to string initially to avoid pandas type inference issues
                sheet_data[sheet_name] = df.astype(str)
                logging.info(f"Successfully parsed sheet '{sheet_name}' from {file_path.name} ({len(df)} rows)")
            except Exception as e:
                logging.error(f"Could not parse sheet '{sheet_name}' in {file_path.name}: {e}", exc_info=True)
                # Skip problematic sheet but continue with others
                continue
        # Return data only if at least one sheet was parsed successfully
        return sheet_data if sheet_data else None
    except FileNotFoundError:
        logging.error(f"Excel file not found: {file_path}")
        return None
    except Exception as e:
        # Catch other potential errors like file corruption, permission issues
        logging.error(f"Failed to parse Excel file {file_path.name}: {e}", exc_info=True)
        return None

def parse_csv(file_path: Path) -> pd.DataFrame | None:
    """
    Parses a CSV file. (Note: No CSVs visible in screenshots, but kept for completeness)

    Args:
        file_path: Path object pointing to the CSV file.

    Returns:
        A pandas DataFrame containing the CSV data read as strings, or None if parsing fails.
    """
    logging.info(f"Attempting to parse CSV file: {file_path.name}")
    try:
        # Read without header, prevent 'NA' interpretation, skip initial spaces
        df = pd.read_csv(file_path, header=None, na_filter=False, keep_default_na=False, encoding='utf-8', skipinitialspace=True)
         # Convert all data to string initially
        df = df.astype(str)
        logging.info(f"Successfully parsed CSV file: {file_path.name} ({len(df)} rows)")
        return df
    except FileNotFoundError:
        logging.error(f"CSV file not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.warning(f"CSV file is empty: {file_path.name}")
        return pd.DataFrame() # Return empty DataFrame for empty files
    except Exception as e:
        logging.error(f"Failed to parse CSV file {file_path.name}: {e}", exc_info=True)
        return None

def parse_pdf(file_path: Path) -> dict | None:
    """
    Parses a PDF file, extracting text and attempting to extract tables from each page.

    Args:
        file_path: Path object pointing to the PDF file.

    Returns:
        A dictionary containing 'file_name' and 'pages' (a list of dicts,
        each with 'page_number', 'text', and 'tables' (list of DataFrames with string data)),
        or None if parsing fails.
    """
    logging.info(f"Attempting to parse PDF file: {file_path.name}")
    pages_data = []
    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                logging.warning(f"PDF file has no pages: {file_path.name}")
                return {'file_name': file_path.name, 'pages': []}

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                page_content_found = False # Flag to check if page yielded anything
                try:
                    text = page.extract_text() or "" # Ensure text is always a string
                    if text.strip():
                        page_content_found = True

                    # Attempt table extraction (can be slow on complex pages)
                    # Consider adding table_settings if needed for specific layouts
                    tables = page.extract_tables() # Returns list of lists of strings/None

                    page_tables_df = []
                    if tables:
                        page_content_found = True
                        for table_data in tables:
                            if table_data: # Ensure table_data list is not empty
                                try:
                                    # Create DataFrame, handle potential None values gracefully
                                    df = pd.DataFrame(table_data)
                                    # Replace None with empty string BEFORE converting to string
                                    df = df.fillna('')
                                    page_tables_df.append(df.astype(str))
                                except Exception as table_err:
                                     logging.warning(f"Could not convert extracted table on page {page_num} of {file_path.name} to DataFrame: {table_err}", exc_info=False) # Don't need full trace here

                    pages_data.append({
                        'page_number': page_num,
                        'text': text,
                        'tables': page_tables_df # List of DataFrames
                    })
                    if page_content_found:
                        logging.info(f"Successfully processed page {page_num} from {file_path.name}")
                    else:
                         logging.info(f"Processed page {page_num} from {file_path.name} (No text or tables extracted)")

                except Exception as page_err:
                    logging.error(f"Error processing page {page_num} in PDF {file_path.name}: {page_err}", exc_info=True)
                    # Add placeholder for the failed page or skip? Let's add placeholder.
                    pages_data.append({
                        'page_number': page_num,
                        'text': f"Error processing page: {page_err}",
                        'tables': []
                    })
                    continue # Move to next page

            return {
                'file_name': file_path.name,
                'pages': pages_data
            }
    except FileNotFoundError:
        logging.error(f"PDF file not found: {file_path}")
        return None
    except Exception as e:
        # pdfplumber can raise various errors on corrupted/complex PDFs
        logging.error(f"Failed to parse PDF file {file_path.name}: {e}", exc_info=True)
        return None

def parse_file(file_path_str: str) -> dict | pd.DataFrame | None:
    """
    Detects the file type and calls the appropriate parser.

    Args:
        file_path_str: String representing the path to the file.

    Returns:
        The parsed data structure (dict for Excel/PDF, DataFrame for CSV),
        or None if the file type is unsupported or parsing fails. Returns specific structure based on file type.
    """
    try:
        file_path = Path(file_path_str) # Convert string path to Path object

        if not file_path.is_file():
            logging.error(f"File does not exist or is not a file: {file_path_str}")
            return None

        file_extension = file_path.suffix.lower()
        logging.info(f"Processing file: {file_path.name} (Extension: {file_extension})")

        if file_extension in ['.xlsx', '.xls']:
            return parse_excel(file_path)
        elif file_extension == '.csv':
            return parse_csv(file_path)
        elif file_extension == '.pdf':
            return parse_pdf(file_path)
        else:
            logging.warning(f"Unsupported file type: {file_extension} for file {file_path.name}")
            return None
    except Exception as e:
        logging.error(f"An unexpected error occurred in parse_file for {file_path_str}: {e}", exc_info=True)
        return None

# --- Main Execution Block for Testing ---
if __name__ == '__main__':
    logging.info("Starting Phase 1 parser testing...")

    # Define the base path to the assignment data
    base_data_path = Path("sample_data") 

    # Define subdirectories
    summary_dir = base_data_path / "rent_rolls_with_summary"
    no_summary_dir = base_data_path / "rent_rolls_without_summary"

    # List files to process
    files_to_process = []

    # Add files from summary directory
    if summary_dir.is_dir():
        files_to_process.extend(list(summary_dir.glob('*.xlsx')))
        files_to_process.extend(list(summary_dir.glob('*.xls')))
        files_to_process.extend(list(summary_dir.glob('*.pdf')))
    else:
        logging.warning(f"Directory not found: {summary_dir}")

    # Add files from no_summary directory
    if no_summary_dir.is_dir():
        files_to_process.extend(list(no_summary_dir.glob('*.xlsx')))
        files_to_process.extend(list(no_summary_dir.glob('*.xls')))
        files_to_process.extend(list(no_summary_dir.glob('*.pdf')))
    else:
        logging.warning(f"Directory not found: {no_summary_dir}")

    if not files_to_process:
        logging.error("No files found to process in the specified directories.")
    else:
        logging.info(f"Found {len(files_to_process)} files to process.")

        success_count = 0
        failure_count = 0

        for file_path_obj in files_to_process:
            file_path_str = str(file_path_obj) # Convert Path object to string for the function
            logging.info(f"\n--- Processing: {file_path_obj.name} ---")
            parsed_data = parse_file(file_path_str)

            if parsed_data is not None:
                logging.info(f"[SUCCESS] Successfully parsed: {file_path_obj.name}")
                success_count += 1
                # --- Optional: Inspect the output structure for one file ---
                if "274187" in file_path_obj.name: # Example: Inspect one specific file
                   logging.info("Inspecting parsed data structure...")
                   if isinstance(parsed_data, dict) and 'pages' in parsed_data: # PDF
                       logging.info(f"  Type: PDF. Pages: {len(parsed_data['pages'])}")
                       if parsed_data['pages']:
                            logging.info(f"  Page 1 Text Snippet: '{parsed_data['pages'][0]['text'][:100]}...'")
                            logging.info(f"  Page 1 Tables Found: {len(parsed_data['pages'][0]['tables'])}")
                   elif isinstance(parsed_data, dict): # Excel
                       logging.info(f"  Type: Excel. Sheets: {list(parsed_data.keys())}")
                       if parsed_data:
                           first_sheet_name = list(parsed_data.keys())[0]
                           logging.info(f"  First sheet ('{first_sheet_name}') shape: {parsed_data[first_sheet_name].shape}")
                   elif isinstance(parsed_data, pd.DataFrame): # CSV
                       logging.info(f"  Type: CSV. Shape: {parsed_data.shape}")

            else:
                logging.error(f"[FAILURE] Failed to parse: {file_path_obj.name}")
                failure_count += 1

        logging.info("\n--- Parsing Test Summary ---")
        logging.info(f"Total Files Processed: {len(files_to_process)}")
        logging.info(f"Successful Parses: {success_count}")
        logging.info(f"Failed Parses: {failure_count}")
        logging.info("--- End of Phase 1 parser testing ---")