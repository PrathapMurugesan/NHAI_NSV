"""
Deliverable-1 (D1) loader and chainage lookup functions.

Migrated from Notebook Cell 7 (Load Workbook Schemas and Helper Functions).
Preserves all business logic:
- Loading LRPMaster, PavementType, CarriagewayType, ServiceRoad parquets
- Cleaning junk values ('lcms issue', 'uc', 'na', 'n/a', '')
- Preparing LRMS (LRP) lookup table with chainage sorting
- Preparing interval lookups (start_chainage, end_chainage)
- Direction isolation: lane's own direction filtered first so sides are never mixed
- Half-open interval matching [start, end) with boundary fallback to next interval
- Nearest edge fallback when chainage is outside surveyed range
- Narrowest interval preference when multiple intervals match
- Road context construction (NH number, section, direction, pavement type, carriageway)
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from src.common.logger import logger
from src.common.fs_utils import path_exists
from src.common.constants import (
    as_chainage_m,
    as_float,
    as_text,
    first_non_null,
    lane_to_direction,
    normalize_direction,
    normalize_lane,
    pick_col,
    standardize_nh_number,
)
from src.ingestion.backup_loader import load_matching_parquets

_JUNK_VALUES = {"lcms issue", "uc", "na", "n/a", ""}


def clean_d1_lookup(pdf: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Filter out known placeholder and junk values from D1 tables."""
    if pdf.empty or value_col not in pdf.columns:
        return pdf
    mask = pdf[value_col].fillna("").astype(str).str.strip().str.lower().isin(_JUNK_VALUES)
    return pdf[~mask].copy()


def prepare_lrms_lookup(pdf: pd.DataFrame) -> pd.DataFrame:
    """Prepare LRPMaster DataFrame for chainage-based nearest lookup."""
    cols = ["chainage", "nh_number", "section_code", "direction"]
    if pdf.empty:
        return pd.DataFrame(columns=cols)

    pcols = list(pdf.columns)
    chain_c = pick_col(pcols, "Chainage")
    nh_c = pick_col(pcols, "NHNumber", "New NH number", "New NH number NHNumber")
    section_c = pick_col(pcols, "SectionCode", "Section")
    direction_c = pick_col(pcols, "Direction")

    rows = []
    for _, row in pdf.iterrows():
        ch_val = as_chainage_m(row[chain_c]) if chain_c else None
        if ch_val is None:
            continue
        rows.append({
            "chainage": ch_val,
            "nh_number": standardize_nh_number(row[nh_c]) if nh_c else None,
            "section_code": as_text(row[section_c]) if section_c else None,
            "direction": normalize_direction(row[direction_c]) if direction_c else None,
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("chainage").reset_index(drop=True)


def prepare_interval_lookup(
    pdf: pd.DataFrame,
    start_cands: List[str],
    end_cands: List[str],
    field_cands: Dict[str, List[str]],
    float_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Prepare interval-based lookup tables (e.g. PavementType, CarriagewayType)."""
    float_set = set(float_fields or [])
    cols = ["start_chainage", "end_chainage", *field_cands.keys()]
    if pdf.empty:
        return pd.DataFrame(columns=cols)

    pcols = list(pdf.columns)
    start_c = pick_col(pcols, *start_cands)
    end_c = pick_col(pcols, *end_cands)
    resolved = {name: pick_col(pcols, *c) for name, c in field_cands.items()}

    rows = []
    for _, row in pdf.iterrows():
        s = as_chainage_m(row[start_c]) if start_c else None
        e = as_chainage_m(row[end_c]) if end_c else None
        if s is None and e is None:
            continue
        if s is None:
            s = e
        if e is None:
            e = s
        if e < s:
            s, e = e, s

        rec = {"start_chainage": s, "end_chainage": e}
        for fn, cn in resolved.items():
            if cn:
                rec[fn] = as_float(row[cn]) if fn in float_set else as_text(row[cn])
            else:
                rec[fn] = None

        if "nh_number" in rec:
            rec["nh_number"] = standardize_nh_number(rec["nh_number"])
        if "direction" in rec:
            rec["direction"] = normalize_direction(rec["direction"])
        rows.append(rec)

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


_DIR_SPLIT_CACHE: Dict[Tuple[int, int, str], pd.DataFrame] = {}


def restrict_direction(pdf: pd.DataFrame, direction: Optional[str]) -> pd.DataFrame:
    """
    Restrict D1 lookup to rows matching the lane's direction.
    Falls back to all rows only when D1 has no row for that side.
    """
    if pdf.empty or direction is None or "direction" not in pdf.columns:
        return pdf
    dir_str = str(direction).strip().lower()
    key = (id(pdf), len(pdf), dir_str)
    if key in _DIR_SPLIT_CACHE:
        return _DIR_SPLIT_CACHE[key]

    same = pdf[pdf["direction"].fillna("").astype(str).str.strip().str.lower() == dir_str]
    out = same if not same.empty else pdf
    _DIR_SPLIT_CACHE[key] = out
    return out


def lookup_interval_record(
    pdf: pd.DataFrame,
    ch_val: Optional[float],
    lane: Optional[str] = None,
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lookup record in an interval table using half-open interval [start, end).
    """
    if pdf.empty or ch_val is None:
        return {}
    filtered = restrict_direction(pdf, direction)

    matches = filtered[(filtered["start_chainage"] <= ch_val) & (filtered["end_chainage"] > ch_val)]
    if matches.empty:
        matches = filtered[(filtered["start_chainage"] <= ch_val) & (filtered["end_chainage"] >= ch_val)]
    if matches.empty:
        probe = filtered.copy()
        probe["_dist"] = np.minimum((probe["start_chainage"] - ch_val).abs(), (probe["end_chainage"] - ch_val).abs())
        sorted_probe = probe.sort_values("_dist")
        return sorted_probe.iloc[0].drop(labels=["_dist"]).to_dict() if not sorted_probe.empty else {}

    if lane and "lane" in matches.columns:
        lane_norm = normalize_lane(lane)
        lane_match = matches[matches["lane"].fillna("").astype(str).str.strip().str.lower() == str(lane_norm).lower()]
        if not lane_match.empty:
            matches = lane_match

    if len(matches) > 1 and "start_chainage" in matches.columns:
        widths = (matches["end_chainage"] - matches["start_chainage"]).values
        matches = matches.iloc[widths.argsort()]

    return matches.iloc[0].to_dict() if not matches.empty else {}


def lookup_lrms_record(
    lrms_df: pd.DataFrame,
    ch_val: Optional[float],
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """Find the nearest LRP / LRMS record for a given chainage."""
    if lrms_df.empty or ch_val is None:
        return {}
    probe = restrict_direction(lrms_df, direction).copy()
    probe["_dist"] = (probe["chainage"] - ch_val).abs()
    return probe.sort_values("_dist").iloc[0].drop(labels=["_dist"]).to_dict() if not probe.empty else {}


class D1Context:
    """Encapsulates D1 tables and provides lookup helpers."""

    def __init__(self, d1_base: str):
        self.d1_base = d1_base
        self.lrms_lookup = pd.DataFrame()
        self.pavement_lookup = pd.DataFrame()
        self.carriageway_lookup = pd.DataFrame()
        self.service_road_lookup = pd.DataFrame()
        self._load()

    def _load(self):
        if not self.d1_base or not path_exists(self.d1_base):
            logger.warning(f"D1 path not found: {self.d1_base}")
            return

        df_lrms, _ = load_matching_parquets(self.d1_base, ["LRPMaster"])
        df_pav, _ = load_matching_parquets(self.d1_base, ["PavementType"])
        df_cw, _ = load_matching_parquets(self.d1_base, ["CarriagewayType"])
        df_sr, _ = load_matching_parquets(self.d1_base, ["ServiceRoad"])

        df_pav = clean_d1_lookup(df_pav, "PavementType")
        df_cw = clean_d1_lookup(df_cw, "CarriagewayType")
        if not df_sr.empty and "PavementType" in df_sr.columns:
            df_sr = clean_d1_lookup(df_sr, "PavementType")

        self.lrms_lookup = prepare_lrms_lookup(df_lrms)

        self.pavement_lookup = prepare_interval_lookup(
            df_pav,
            ["New NH number StartChainage", "Start_Chainage"],
            ["New NH number EndChainage", "End_Chainage"],
            {
                "nh_number": ["New NH number NHNumber", "NHNumber"],
                "section_code": ["SectionCode"],
                "direction": ["Direction"],
                "pavement_type": ["PavementType", "Surface Type"],
            },
        )

        if not df_sr.empty:
            self.service_road_lookup = prepare_interval_lookup(
                df_sr,
                ["New NH number StartChainage", "Start_Chainage"],
                ["New NH number EndChainage", "End_Chainage"],
                {
                    "nh_number": ["NHNumber", "New NH number NHNumber"],
                    "section_code": ["SectionCode"],
                    "direction": ["Direction"],
                    "pavement_type": ["PavementType", "Surface Type"],
                },
            )

        self.carriageway_lookup = prepare_interval_lookup(
            df_cw,
            ["New NH number StartChainage", "Start_Chainage"],
            ["New NH number EndChainage", "End_Chainage"],
            {
                "nh_number": ["New NH number", "NHNumber"],
                "section_code": ["SectionCode"],
                "direction": ["Direction"],
                "carriageway": ["CarriagewayType"],
            },
        )
        logger.info(f"Loaded D1 tables: LRMS ({len(self.lrms_lookup)} rows), Pavement ({len(self.pavement_lookup)} rows), Carriageway ({len(self.carriageway_lookup)} rows)")

    def get_road_context(
        self,
        start_ch: Optional[float],
        end_ch: Optional[float] = None,
        lane: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Derive road context (NH number, section, direction, pavement type, carriageway).
        Direction follows lane (L* -> Increasing, R* -> Decreasing).
        """
        if not hasattr(self, '_context_cache'):
            self._context_cache = {}

        probe_ch = first_non_null(start_ch, end_ch)
        cache_key = (probe_ch, lane, direction)
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        resolved_dir = first_non_null(lane_to_direction(lane), normalize_direction(direction))

        lrms = lookup_lrms_record(self.lrms_lookup, probe_ch, direction=resolved_dir)
        pav = lookup_interval_record(self.pavement_lookup, probe_ch, direction=resolved_dir)
        cw = lookup_interval_record(self.carriageway_lookup, probe_ch, direction=resolved_dir)

        res = {
            "nh_number": first_non_null(lrms.get("nh_number"), pav.get("nh_number")),
            "section_code": first_non_null(lrms.get("section_code"), pav.get("section_code")),
            "direction": first_non_null(resolved_dir, lrms.get("direction"), pav.get("direction")),
            "pavement_type": first_non_null(pav.get("pavement_type")),
            "carriageway": first_non_null(cw.get("carriageway")),
        }
        self._context_cache[cache_key] = res
        return res


def load_d1_context(d1_base: str, upc: Optional[str] = None) -> D1Context:
    """Helper function to load and initialize D1Context."""
    return D1Context(d1_base)
