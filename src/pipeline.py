#!/usr/bin/env python3
"""
Single-file workflow pulled out of the monolith.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from pathlib import Path

import file_parser
from llm_interface import (
    CALL_TIMEOUT,
    JSON_DELAY,
    _call_with_timeout,
    build_prompt,
    call_gemini_with_retry,
    extract_json_slice,
)
from data_processing import (
    clamp_budget,
    compute_metadata,
    serialize,
    validate_and_clean,
)

# --------------------------------------------------------------------
# Outputs directory (unchanged)
# --------------------------------------------------------------------
OUTPUT_DIR = Path("Outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    base = Path("./sample_data")
    exts = [".xlsx", ".xls", ".csv", ".pdf"]

    files = sum((list(base.rglob(f"*{e}")) for e in exts), [])

    if not files:
        logging.error("No files found under ./sample_data")
        return

    success, fail = 0, 0

    for fp in files:
        logging.info(f"→ {fp.name}")
        parsed = file_parser.parse_file(str(fp))
        if parsed is None:
            logging.warning("   parse_file failed → skipping")
            fail += 1
            continue

        meta = compute_metadata(parsed)
        data = serialize(parsed)
        prompt = build_prompt(fp.name, meta, data)

        data_rows_for_budget = 0
        if meta.get("type") == "table":
            data_rows_for_budget = meta.get("data_rows_estimate", 0)
        elif meta.get("type") == "excel":
            data_rows_for_budget = sum(
                v["data_rows_estimate"] for v in meta["sheets"].values()
            )

        budget = clamp_budget(data_rows_for_budget)

        parsed_json = None
        for pass_ix, this_budget in enumerate([budget, 0], start=1):
            try:
                resp = _call_with_timeout(
                    lambda: call_gemini_with_retry(prompt, this_budget),
                    CALL_TIMEOUT if pass_ix == 1 else CALL_TIMEOUT * 2,
                )
            except concurrent.futures.TimeoutError:
                logging.warning(
                    f"   Gemini call exceeded {CALL_TIMEOUT}s — cancelling "
                    f"and retrying with thinking_budget = 0"
                )
                if pass_ix == 1:
                    time.sleep(JSON_DELAY)
                    continue
                else:
                    logging.error("   Still timing out with budget-0 — skipping file")
                    fail += 1
                    break
            except Exception as e:
                logging.error(f"   API error after retries: {e}")
                fail += 1
                break

            raw_text = resp.text or ""
            snippet = extract_json_slice(raw_text)
            try:
                parsed_json = json.loads(snippet)
                break
            except json.JSONDecodeError:
                if pass_ix == 1:
                    logging.warning(
                        f"   JSON parse failed; waiting {JSON_DELAY}s then "
                        f"retrying with thinking_budget = 0"
                    )
                    time.sleep(JSON_DELAY)
                    continue
                else:
                    logging.error("   JSON parse failed even with budget-0")
                    logging.debug(raw_text)
                    fail += 1
                    break

        if parsed_json is None:
            continue

        result = validate_and_clean(parsed_json, meta)
        out_f = OUTPUT_DIR / f"{fp.stem}.json"
        with open(out_f, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logging.info(f"   ✔ saved {out_f}")
        success += 1
        time.sleep(5)  # stay well under rate limits
        print("Sleeping....")

    logging.info("\n--- run summary ---")
    logging.info(f"success: {success}  •  failure: {fail}  •  total: {len(files)}")
