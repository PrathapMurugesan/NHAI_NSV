"""
D2 threshold parser for CA and IRC defect limits.

Migrated from Notebook Cell 3 (Tracker and D2 Threshold Parsing).
Preserves all business logic:
- Dynamic parsing from D2 Comparison parquet sheets
- Dual parsing: CA comparison file vs IRC file
- Discovers limits for Roughness (BI/IRI), Rutting, Cracking, Ravelling
- Unit detection for Ravelling ('pct' vs 'sqm')
- Default fallback values if D2 is absent or incomplete
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple
from src.common.logger import logger
from src.common.fs_utils import path_exists, list_dir, read_parquet_df


DEFAULT_THRESHOLDS = {
    "roughness_bi": 2400.0,
    "roughness_iri": 2400.0,
    "rutting_mm": 10.0,
    "cracking_pct": 10.0,
    "ravelling": 10.0,
    "ravelling_unit": "pct",
}


def parse_d2_thresholds(d2_base_path: str) -> Dict[str, Any]:
    """
    Parse concession agreement (CA) and general thresholds from D2 Parquets.
    """
    if not d2_base_path or not path_exists(d2_base_path):
        logger.warning(f"D2 base path not found ({d2_base_path}), using defaults.")
        return dict(DEFAULT_THRESHOLDS)

    thresholds = dict(DEFAULT_THRESHOLDS)
    ca_set, irc_set = set(), set()

    for pf in sorted(list_dir(d2_base_path)):
        if not pf.endswith(".parquet"):
            continue
        if "roughness,rut,crack,rav" in pf.lower():
            continue
        is_irc_file = "irc" in pf.lower()
        file_path = f"{d2_base_path}/{pf}" if d2_base_path.startswith("s3://") else os.path.join(d2_base_path, pf)

        try:
            df = read_parquet_df(file_path)
        except Exception as e:
            logger.warning(f"Failed reading D2 file {pf}: {e}")
            continue

        for col in df.columns:
            cl = col.lower()
            if "limitation" not in cl and "limit" not in cl:
                continue
            if ">" in cl:
                continue
            val = pd.to_numeric(df[col], errors="coerce").dropna()
            if val.empty:
                continue
            limit_val = float(val.iloc[0])
            is_ca = "concession" in cl

            if ("irc" in cl and "roughness" in cl) or ("iri" in cl and "limitation" in cl):
                if is_irc_file:
                    if "roughness_iri" not in irc_set:
                        thresholds["roughness_iri"] = limit_val
                        irc_set.add("roughness_iri")
                else:
                    if "roughness_bi" not in ca_set:
                        thresholds["roughness_bi"] = limit_val
                        ca_set.add("roughness_bi")
            elif "bi" in cl or ("roughness" in cl and "irc" not in cl):
                if is_irc_file:
                    if "roughness_iri" not in irc_set:
                        thresholds["roughness_iri"] = limit_val
                        irc_set.add("roughness_iri")
                else:
                    if is_ca or "roughness_bi" not in ca_set:
                        thresholds["roughness_bi"] = limit_val
                        if is_ca:
                            ca_set.add("roughness_bi")
            elif "rut" in cl:
                if not is_irc_file:
                    thresholds["rutting_mm"] = limit_val
                    ca_set.add("rutting_mm")
                elif "rutting_mm" not in ca_set:
                    thresholds["rutting_mm"] = limit_val
            elif "crack" in cl:
                if not is_irc_file:
                    thresholds["cracking_pct"] = limit_val
                    ca_set.add("cracking_pct")
                elif "cracking_pct" not in ca_set:
                    thresholds["cracking_pct"] = limit_val
            elif "ravelling" in cl or "ravel" in cl:
                if not is_irc_file:
                    thresholds["ravelling"] = limit_val
                    ca_set.add("ravelling")
                elif "ravelling" not in ca_set:
                    thresholds["ravelling"] = limit_val
                if "sq" in cl or "area" in cl:
                    thresholds["ravelling_unit"] = "sqm"
                elif "%" in cl or "percent" in cl:
                    thresholds["ravelling_unit"] = "pct"

    return thresholds


def parse_irc_specific_thresholds(d2_base_path: str, ca_thresholds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse IRC-specific thresholds from D2 IRC Parquet sheet.
    Falls back to CA thresholds if IRC-specific values are not defined.
    """
    irc_crack_pct = None
    irc_rut_threshold = None
    irc_ravelling_threshold = None
    irc_ravelling_unit = None

    if d2_base_path and path_exists(d2_base_path):
        for pf in sorted(list_dir(d2_base_path)):
            if not pf.endswith(".parquet") or "irc" not in pf.lower():
                continue
            file_path = f"{d2_base_path}/{pf}" if d2_base_path.startswith("s3://") else os.path.join(d2_base_path, pf)
            try:
                df_irc = read_parquet_df(file_path)
            except Exception as e:
                logger.warning(f"Could not read D2 IRC parquet {pf}: {e}")
                continue

            for col in df_irc.columns:
                cl = col.lower()
                if "limit" not in cl:
                    continue
                val = pd.to_numeric(df_irc[col], errors="coerce").dropna()
                if val.empty:
                    continue
                lv = float(val.iloc[0])
                if "crack" in cl and irc_crack_pct is None:
                    irc_crack_pct = lv
                elif "rut" in cl and irc_rut_threshold is None:
                    irc_rut_threshold = lv
                elif "ravel" in cl and irc_ravelling_threshold is None:
                    irc_ravelling_threshold = lv
                    if "%" in cl or "pct" in cl:
                        irc_ravelling_unit = "pct"
                    elif "sq" in cl or "area" in cl:
                        irc_ravelling_unit = "sqm"
            break

    return {
        "roughness_iri": ca_thresholds.get("roughness_iri", 2400.0),
        "cracking_pct": irc_crack_pct or ca_thresholds.get("cracking_pct", 10.0),
        "rutting_mm": irc_rut_threshold or ca_thresholds.get("rutting_mm", 10.0),
        "ravelling": irc_ravelling_threshold or ca_thresholds.get("ravelling", 10.0),
        "ravelling_unit": irc_ravelling_unit or ca_thresholds.get("ravelling_unit", "pct"),
    }
