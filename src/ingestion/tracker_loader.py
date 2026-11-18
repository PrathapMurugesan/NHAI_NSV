"""
Tracker and survey metadata loader for the DLM Extraction Pipeline.

Migrated from Notebook Cell 3 (Tracker and D2 Threshold Parsing).
Preserves all business logic:
- Sheet lookup by Zone (e.g. 'Zone B')
- Fuzzy UPC normalization (replacing '/' and '_' and lowercase matching)
- Phase and Cycle matching
- Extracting survey dates, PIU, contractor, AE/IE, state, pavement type, PCI
- Extracting defect standards and reporting intervals
- D2 filename override ('As per CA' or 'As per IRC')
"""

import copy
import os
import pandas as pd
from typing import Any, Dict, Optional
from src.common.logger import logger
from src.common.constants import LEVEL_STANDARDS
from src.common.fs_utils import path_exists, list_dir, read_excel_df


def parse_std(val: Any) -> Optional[str]:
    """Parse standard string to either 'IRC', 'CA', or None."""
    s = str(val).strip().upper() if pd.notna(val) else ""
    if "IRC" in s:
        return "IRC"
    if "CA" in s:
        return "CA"
    return None


def parse_interval(val: Any) -> Optional[int]:
    """Parse reporting interval to integer meters."""
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def get_default_tracker_context() -> Dict[str, Any]:
    defs = {
        d: {"standard": "CA", "interval_m": 100}
        for d in ["ravelling", "rutting", "cracking", "roughness", "pothole"]
    }
    return {
        "survey_start_date": None,
        "survey_end_date": None,
        "piu": None,
        "ae_ie": None,
        "contractor": None,
        "state": None,
        "pavement_type": None,
        "pci": None,
        "threshold_label_override": None,
        "defect_standards": defs,
        "raw_defect_standards": copy.deepcopy(defs),
    }


def load_tracker_context(
    tracker_path: str,
    d2_base: str,
    zone: str,
    phase: str,
    cycle: str,
    upc_slash: str,
) -> Dict[str, Any]:
    """
    Load tracker metadata and defect reporting standards for a specific UPC.
    """
    sheet_map = {"A": "Zone A", "B": "Zone B", "C": "Zone C", "D": "Zone D", "E": "Zone E"}
    sheet_name = sheet_map.get(zone.upper(), f"Zone {zone}")
    ctx = get_default_tracker_context()

    if not path_exists(tracker_path):
        logger.warning(f"Tracker path not found: {tracker_path}. Using default context.")
        _apply_d2_override(ctx, d2_base)
        return ctx

    try:
        df = read_excel_df(tracker_path, sheet_name=sheet_name, header=1)
    except Exception as e:
        logger.warning(f"Could not load tracker from {tracker_path} (sheet {sheet_name}): {e}")
        _apply_d2_override(ctx, d2_base)
        return ctx

    upc_norm = upc_slash.replace("/", "_").replace("_", "").lower()
    best_row = None
    for _, row in df.iterrows():
        if len(row) < 13:
            continue
        upc_val = str(row.iloc[12]).replace("/", "_").replace("_", "").lower()
        if upc_norm not in upc_val and upc_val not in upc_norm:
            continue
        phase_val = str(row.iloc[2]).lower().replace("phase", "").strip()
        cycle_val = str(row.iloc[3]).lower().replace("cycle", "").strip()
        if phase_val == str(phase).strip() and cycle_val == str(cycle).strip():
            best_row = row
            break
        if best_row is None:
            best_row = row

    if best_row is None:
        logger.warning(f"UPC {upc_slash} not found in tracker sheet {sheet_name}.")
        _apply_d2_override(ctx, d2_base)
        return ctx

    r = best_row
    survey_start = pd.to_datetime(r.iloc[21], errors="coerce")
    survey_end = pd.to_datetime(r.iloc[22], errors="coerce")

    defect_standards = {
        "ravelling": {"standard": parse_std(r.iloc[46]) or "CA", "interval_m": parse_interval(r.iloc[47]) or 100},
        "rutting":   {"standard": parse_std(r.iloc[48]) or "CA", "interval_m": parse_interval(r.iloc[49]) or 100},
        "cracking":  {"standard": parse_std(r.iloc[50]) or "CA", "interval_m": parse_interval(r.iloc[51]) or 100},
        "roughness": {"standard": parse_std(r.iloc[52]) or "CA", "interval_m": parse_interval(r.iloc[53]) or 100},
    }
    defect_standards["pothole"] = defect_standards["cracking"].copy()
    raw_defect_standards = copy.deepcopy(defect_standards)

    pci_val = None
    if pd.notna(r.iloc[43]):
        pci_str = str(r.iloc[43]).replace(".", "", 1).replace("-", "", 1).strip()
        if pci_str.isdigit():
            pci_val = float(r.iloc[43])

    ctx.update({
        "survey_start_date": survey_start.date() if pd.notnull(survey_start) else None,
        "survey_end_date": survey_end.date() if pd.notnull(survey_end) else None,
        "piu": str(r.iloc[7]).strip() if pd.notna(r.iloc[7]) else None,
        "ae_ie": str(r.iloc[9]).strip() if pd.notna(r.iloc[9]) else None,
        "contractor": str(r.iloc[8]).strip() if pd.notna(r.iloc[8]) else None,
        "state": str(r.iloc[6]).strip() if pd.notna(r.iloc[6]) else None,
        "pavement_type": str(r.iloc[42]).strip() if pd.notna(r.iloc[42]) else None,
        "pci": pci_val,
        "defect_standards": defect_standards,
        "raw_defect_standards": raw_defect_standards,
    })

    _apply_d2_override(ctx, d2_base)
    logger.info(f"Loaded tracker context for UPC {upc_slash}: Survey {ctx['survey_start_date']} to {ctx['survey_end_date']}")
    return ctx


def _apply_d2_override(ctx: Dict[str, Any], d2_base: str) -> None:
    """Check D2 deliverable directory for filename override."""
    if not d2_base or not path_exists(d2_base):
        return
    d2_files = [f.lower() for f in list_dir(d2_base) if f.endswith(".parquet")]
    if any("as per ca" in fl or "with ca" in fl for fl in d2_files):
        ctx["threshold_label_override"] = "As per CA"
        for dn in ctx["defect_standards"]:
            ctx["defect_standards"][dn]["standard"] = "CA"
    elif any("as per irc" in fl or "with irc" in fl for fl in d2_files):
        ctx["threshold_label_override"] = "As per IRC"
        for dn in ctx["defect_standards"]:
            ctx["defect_standards"][dn]["standard"] = "IRC"


def get_threshold_name(tracker_ctx: Dict[str, Any], defect_name: str) -> Optional[str]:
    override = tracker_ctx.get("threshold_label_override")
    if override:
        return override
    std = tracker_ctx.get("defect_standards", {}).get(defect_name, {}).get("standard")
    if std == "IRC":
        return "As per IRC"
    if std == "CA":
        return "As per CA"
    return "As per CA"


def get_reporting_interval(tracker_ctx: Dict[str, Any], defect_name: str) -> int:
    return tracker_ctx.get("defect_standards", {}).get(defect_name, {}).get("interval_m", 100) or 100


def get_level_standard(level: str) -> str:
    prefs = LEVEL_STANDARDS.get(level, ["CA"])
    return f"As per {prefs[0]}"
