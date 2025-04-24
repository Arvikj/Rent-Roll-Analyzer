#!/usr/bin/env python3
"""
Gemini-API helpers split from the original monolith.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
from typing import Any, Dict, Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#  Setup
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env")

client = genai.Client(api_key=API_KEY)

# --------------------------------------------------------------------
#  Config & schema (unchanged)
# --------------------------------------------------------------------
REQUIRED_KEYS: Dict[str, Union[type, Tuple[type, ...]]] = {
    "summary_section_found": bool,
    "total_units": int,
    "total_actual_rent": (int, float),
    "average_rent": (int, float),
    "occupancy_rate": (int, float),
    "total_market_rent": (int, float),
    "total_square_feet": (int, float),
    "status_breakdown": list,
    "charge_codes": list,
}

# –– RETRY / TIMEOUT TUNING ––––––––––––––––––––––––––––––––––––––––––
MAX_RETRIES   = 3
RETRY_DELAY   = 15          # seconds
JSON_RETRIES  = 3
CALL_TIMEOUT  = 120         # seconds
JSON_DELAY    = 10          # seconds

# –– HELPER FUNCTIONS (identical) ––––––––––––––––––––––––––––––––––––
def _is_retryable_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    return any(
        kw in txt
        for kw in (
            "rate limit",
            "quota",
            "tokens per minute",
            "too many requests",
            "resource exhausted",
            "exceeded",
            "429",
        )
    )

_re = re  # alias preserved for the regex helper below

def _extract_retry_delay(exc: Exception) -> int | None:
    m = _re.search(r'"retryDelay"\s*:\s*"(\d+)s"', str(exc))
    return int(m.group(1)) if m else None


def _call_with_timeout(fn, timeout: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        return fut.result(timeout=timeout)


def call_gemini_with_retry(
    prompt: str,
    budget: int,
    model: str = "gemini-2.5-flash-preview-04-17",
) -> types.GenerateContentResponse:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=budget),
                ),
            )
        except Exception as e:
            if attempt == MAX_RETRIES or not _is_retryable_error(e):
                raise
            delay = _extract_retry_delay(e) or RETRY_DELAY
            logging.warning(
                f"Rate-limit hit (attempt {attempt}/{MAX_RETRIES}) – "
                f"sleeping {delay}s then retrying…"
            )
            time.sleep(delay)


# –– PROMPT BUILDER (verbatim) ––––––––––––––––––––––––––––––––––––––
def build_prompt(fname: str, meta: Dict[str, Any], data: Any) -> str:
    rows_hint: int | None = None
    if meta.get("type") == "table":
        rows_hint = meta.get("data_rows_estimate")
    elif meta.get("type") == "excel":
        rows_hint = sum(s["data_rows_estimate"] for s in meta["sheets"].values())

    min_units = max_units = None
    bounds_clause = ""
    if rows_hint:
        min_units = int(rows_hint * 0.80)
        max_units = int(rows_hint * 1.05)
        bounds_clause = (
            f"**total_units must be between {min_units} and {max_units}, **unless** the “Unit ID” column in the parsed data has **> 15%** empty or null cells—in that case, you may override and infer a smaller count, but only in that case.**"
        )

    meta_j = json.dumps(meta, indent=2)
    data_j = json.dumps(data, indent=2)
    keys = json.dumps(list(REQUIRED_KEYS.keys()), indent=2)

    return f"""
You are a multifamily rent-roll analyst.  **Return exactly one JSON object** with
these keys and *no others*:
{keys}

Rules
-----
1  If the document contains an explicit summary table that already lists these
   totals, set `"summary_section_found": true` and copy those numbers verbatim.

2  Otherwise set `"summary_section_found": false` **and**:

   • {bounds_clause or 'total_units must be a realistic value based on the data.'}
   • total_actual_rent  = sum of actual/tenant rent for occupied units
   • average_rent       = total_actual_rent / total_units  (2-dec float)
   • occupancy_rate     = occupied_units / total_units     (0–1 float)
   • total_market_rent  = sum of market / contract rent
   • total_square_feet  = sum of sqft / area columns (0 if none)
   • status_breakdown   = list of {{status, count}} for every status present
   • charge_codes       = list of {{charge_type (snake_case), total_amount}}

Cleaning rules
--------------
• Strip "$", ","; treat "(123)" as –123; blanks/dashes → 0.
• If a field cannot be derived, output `null`.
• **Respond ONLY with the JSON.  No markdown, text or commentary.**

File name : "{fname}"

Metadata:
{meta_j}

Parsed data (full):
{data_j}
""".strip()


# –– RESPONSE-SLICE EXTRACTOR (unchanged) –––––––––––––––––––––––––––
def extract_json_slice(text: str) -> str:
    i = text.find("{")
    j = text.rfind("}")
    return text[i : j + 1] if 0 <= i < j else text
