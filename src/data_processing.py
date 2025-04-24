#!/usr/bin/env python3
"""
Pure data-handling helpers (moved verbatim).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Union, cast

import pandas as pd
from .llm_interface import REQUIRED_KEYS
# --------------------------------------------------------------------
# Helpers (unchanged)
# --------------------------------------------------------------------
def _snake(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def _row_has_numbers(row: pd.Series) -> bool:
    for cell in row.astype(str):
        if re.search(r"[0-9]", cell):
            return True
    return False


# –– METADATA ––
def compute_metadata(parsed: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(parsed, pd.DataFrame):
        total_rows = parsed.shape[0]
        data_start = 0
        for idx, row in parsed.iterrows():
            if _row_has_numbers(row):
                data_start = idx
                break
        data_rows = total_rows - data_start

        non_empty_rows = int(
            parsed.astype(str).apply(lambda r: r.str.strip().any(), axis=1).sum()
        )
        return {
            "type": "table",
            "total_rows": total_rows,
            "data_rows_estimate": data_rows,
            "non_empty_rows": non_empty_rows,
        }

    if isinstance(parsed, dict) and "pages" in parsed:  # PDF
        pages = cast(List[Dict[str, Any]], parsed["pages"])
        return {
            "type": "pdf",
            "pages": len(pages),
            "tables_per_page": [len(p["tables"]) for p in pages],
        }

    if isinstance(parsed, dict):  # Excel with sheets
        sheets_meta = {}
        for name, df in parsed.items():
            total_rows = df.shape[0]
            data_start = 0
            for idx, row in df.iterrows():
                if _row_has_numbers(row):
                    data_start = idx
                    break
            data_rows = total_rows - data_start
            non_empty_rows = int(
                df.astype(str).apply(lambda r: r.str.strip().any(), axis=1).sum()
            )
            sheets_meta[name] = {
                "total_rows": total_rows,
                "data_rows_estimate": data_rows,
                "non_empty_rows": non_empty_rows,
            }
        return {"type": "excel", "sheets": sheets_meta}

    return {"type": "raw"}


# –– SERIALISATION ––
def serialize(parsed: Union[pd.DataFrame, Dict[str, Any]]) -> Any:
    if isinstance(parsed, pd.DataFrame):
        return parsed.to_dict(orient="records")

    if isinstance(parsed, dict) and "pages" in parsed:  # PDF
        return [
            {
                "page": p["page_number"],
                "text": p["text"],
                "tables": [t.to_dict(orient="records") for t in p["tables"]],
            }
            for p in parsed["pages"]
        ]

    if isinstance(parsed, dict):  # Excel dict-of-DFs
        return {s: df.to_dict(orient="records") for s, df in parsed.items()}

    return parsed


# –– THINKING-BUDGET HEURISTIC ––
def clamp_budget(data_rows: int) -> int:
    if data_rows <= 150:
        return 0
    return min(max(data_rows * 2, 512), 2048)


# –– VALIDATION / CLEANUP ––
def _coerce_number(x: Any) -> float | None:
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    x = re.sub(r"[,$]", "", x)
    if re.match(r"^\(.*\)$", x):
        x = "-" + x[1:-1]
    try:
        return float(x)
    except ValueError:
        return None


def validate_and_clean(obj: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}

    max_units = None
    if meta.get("type") == "table":
        max_units = meta.get("data_rows_estimate")
    elif meta.get("type") == "excel":
        max_units = sum(v["data_rows_estimate"] for v in meta["sheets"].values())

    for k, exp_type in REQUIRED_KEYS.items():
        v = obj.get(k)

        if k in {"total_units"}:
            try:
                v = int(v)
            except Exception:
                v = None
            if max_units and v and (v < 1 or v > max_units * 1.2):
                v = max_units

        elif k in {
            "total_actual_rent",
            "average_rent",
            "occupancy_rate",
            "total_market_rent",
            "total_square_feet",
        }:
            v = _coerce_number(v)
            if k == "occupancy_rate" and v is not None and v > 1:
                v = v / 100 if v <= 100 else 1.0

        elif k == "status_breakdown" and isinstance(v, list):
            fixed: List[Dict[str, Any]] = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "")).title()
                try:
                    cnt = int(item.get("count", 0))
                except Exception:
                    cnt = 0
                fixed.append({"status": status, "count": cnt})
            v = fixed

        elif k == "charge_codes" and isinstance(v, list):
            fixed: List[Dict[str, Any]] = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                ctype_raw = str(item.get("charge_type", ""))
                ctype = _snake(ctype_raw)
                amt = _coerce_number(item.get("total_amount"))
                fixed.append({"charge_type": ctype, "total_amount": amt})
            v = fixed

        if v is not None and not isinstance(v, exp_type):
            v = None
        clean[k] = v

    return clean
