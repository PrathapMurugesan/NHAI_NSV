"""
Bronze defect data orchestrator and schema standardizer.

Migrated from Notebook Cell 4 (Load All Defect Data Sources).
Preserves all business logic:
- Condition classification functions using dynamic D2 threshold breakpoints:
  * crack_condition_from_pct (breakpoints at t, 2t, 4t, 6t)
  * roughness_condition_from_metrics (breakpoint at 0.8*t, t)
  * rutting_condition_from_mm (breakpoint at 0.5*t, t)
  * ravelling_condition_from_pct (breakpoints at 0, 0.5*t, t)
- Roughness BI & IRI derivation:
  * Formula: Lane IRI = (LwpIri + RwpIri) / 2
  * Formula: BI = 630 × IRI^1.12
  * Fills remaining null BI values via 630 × IRI^1.12
- Ravelling percentage formula:
  * Ravelling % = Area_sq_m * 100.0 / 35.0 (for 10m segments)
- Rutting renaming: 'Rutting Average' -> 'Rutting'
- Cracking renaming: 'Percent_Area' -> 'Cracking', 'Area_Sum' -> 'Area_sq_m'
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional, List
from src.common.logger import logger
from src.ingestion.backup_loader import load_backup_lane_parquets, discover_lane_folders
from src.ingestion.d4_loader import load_d4_potholes


def crack_condition_from_pct(value: Any, ca_crack_pct: float = 10.0) -> Optional[str]:
    """Classify cracking condition using dynamic breakpoints from CA_CRACK_PCT."""
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    t = ca_crack_pct
    b2, b3, b4 = round(t * 2), round(t * 4), round(t * 6)
    if val < t:
        return f"Very Good (<{t:.0f}%)"
    if val <= b2:
        return f"Good ({t:.0f}-{b2}%)"
    if val <= b3:
        return f"Fair ({b2}-{b3}%)"
    if val <= b4:
        return f"Poor ({b3}-{b4}%)"
    return f"Very Poor (>{b4}%)"


def roughness_condition_from_metrics(
    iri_value: Any,
    bi_value: Any,
    ca_bi_threshold: float = 2400.0,
) -> Optional[str]:
    """Classify roughness condition using dynamic breakpoints from CA_BI_THRESHOLD."""
    iri_v = pd.to_numeric(pd.Series([iri_value]), errors="coerce").iloc[0]
    bi_v = pd.to_numeric(pd.Series([bi_value]), errors="coerce").iloc[0]
    probe = bi_v if pd.notna(bi_v) else (iri_v * 1000 if pd.notna(iri_v) else np.nan)
    if pd.isna(probe):
        return None
    t = ca_bi_threshold
    fair_lower = round(t * 0.8)
    if probe < fair_lower:
        return f"Good (<{fair_lower}mm/km)"
    if probe <= t:
        return f"Fair ({fair_lower}-{t:.0f}mm/km)"
    return f"Poor (>{t:.0f}mm/km)"


def rutting_condition_from_mm(value: Any, ca_rut_threshold: float = 10.0) -> Optional[str]:
    """Classify rutting condition using dynamic breakpoints from CA_RUT_THRESHOLD."""
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    t = ca_rut_threshold
    fair_lower = round(t * 0.5)
    if val < fair_lower:
        return f"Good (<{fair_lower}mm)"
    if val <= t:
        return f"Fair ({fair_lower}-{t:.0f}mm)"
    return f"Poor (>{t:.0f}mm)"


def ravelling_condition_from_pct(value: Any, ravelling_threshold: float = 10.0) -> Optional[str]:
    """Classify ravelling condition using dynamic breakpoints from RAVELLING_THRESHOLD."""
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return None
    t = ravelling_threshold
    half_t = round(t * 0.5)
    if val == 0:
        return "Very Good (0)"
    if val <= half_t:
        return f"Good (1-{half_t}%)"
    if val < t:
        return f"Fair ({half_t}-{t:.0f}%)"
    return f"Poor (>={t:.0f}%)"


def load_all_raw_defects(
    backup_base: str,
    d4_base: str,
    ca_thresholds: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load all defect data sources and apply schema standardizations.
    """
    ca_crack_pct = ca_thresholds.get("cracking_pct", 10.0)
    ca_bi_threshold = ca_thresholds.get("roughness_bi", 2400.0)
    ca_rut_threshold = ca_thresholds.get("rutting_mm", 10.0)
    ravelling_threshold = ca_thresholds.get("ravelling", 10.0)

    # 1. Load Potholes from D4
    df_pothole_raw = load_d4_potholes(d4_base)

    # 2. Load Raw Backup Lane Parquets
    df_cracking_raw, _ = load_backup_lane_parquets(backup_base, "_Cracking.parquet")
    df_roughness_raw, _ = load_backup_lane_parquets(backup_base, "_Roughness_IRI.parquet")
    df_rutting_raw, _ = load_backup_lane_parquets(backup_base, "_Rutting.parquet")
    df_ravelling_raw, _ = load_backup_lane_parquets(backup_base, "_Raveling.parquet")

    # Standardize Cracking
    if not df_cracking_raw.empty:
        df_cracking_raw.rename(columns={
            "LANE": "Lane", "Percent_Area": "Cracking", "cracking_image_name": "Image_Name",
            "cracking_image_url": "Image_URL", "Start_Latitude": "Start_Lat",
            "Start_Longitude": "Start_Lon", "End_Latitude": "End_Lat",
            "End_Longitude": "End_Lon", "survey_date_crack": "SurveyDate", "Area_Sum": "Area_sq_m"
        }, inplace=True)
        if "Lane" not in df_cracking_raw.columns:
            df_cracking_raw["Lane"] = df_cracking_raw["Source_Lane_Folder"]
        crack_pct = pd.to_numeric(df_cracking_raw["Cracking"], errors="coerce").fillna(0)
        df_cracking_raw["Condition"] = crack_pct.apply(lambda v: crack_condition_from_pct(v, ca_crack_pct))
        df_cracking_raw["Type_of_Defect"] = "Cracking"

    # Standardize Roughness (with LWP+RWP Derivation)
    if not df_roughness_raw.empty:
        df_roughness_raw.rename(columns={
            "RoughnessIndex": "BI m/km", "LaneIri": "IRI",
            "Start_Latitude": "Start_Lat", "Start_Longitude": "Start_Lon",
            "End_Latitude": "End_Lat", "End_Longitude": "End_Lon",
            "ConditionOfRoadSurface": "Condition",
        }, inplace=True)
        if "SurveyDate" not in df_roughness_raw.columns:
            df_roughness_raw["SurveyDate"] = pd.NaT

        has_lwp = "LwpIri" in df_roughness_raw.columns
        has_rwp = "RwpIri" in df_roughness_raw.columns
        if has_lwp and has_rwp:
            lwp = pd.to_numeric(df_roughness_raw["LwpIri"], errors="coerce")
            rwp = pd.to_numeric(df_roughness_raw["RwpIri"], errors="coerce")
            both_valid = lwp.notna() & rwp.notna()

            derived_iri = (lwp + rwp) / 2
            derived_bi = 630.0 * (derived_iri ** 1.12)

            df_roughness_raw["IRI_original"] = df_roughness_raw["IRI"]
            df_roughness_raw["BI_original"] = df_roughness_raw["BI m/km"]

            df_roughness_raw.loc[both_valid, "IRI"] = derived_iri[both_valid]
            df_roughness_raw.loc[both_valid, "BI m/km"] = derived_bi[both_valid]
            logger.info(f"Derived Roughness BI from LWP+RWP for {both_valid.sum()} rows.")

        rough_iri = pd.to_numeric(df_roughness_raw.get("IRI"), errors="coerce")
        rough_bi = pd.to_numeric(df_roughness_raw.get("BI m/km"), errors="coerce")
        derived_condition = [
            roughness_condition_from_metrics(i, b, ca_bi_threshold)
            for i, b in zip(rough_iri, rough_bi)
        ]
        df_roughness_raw["Condition"] = df_roughness_raw.get(
            "Condition", pd.Series([None] * len(df_roughness_raw))
        ).fillna(pd.Series(derived_condition))
        df_roughness_raw["Type_of_Defect"] = "Roughness"

        # Fill remaining null BI from IRI: BI = 630 * (IRI)^1.12
        df_roughness_raw["BI m/km"] = pd.to_numeric(df_roughness_raw["BI m/km"], errors="coerce")
        df_roughness_raw["IRI"] = pd.to_numeric(df_roughness_raw["IRI"], errors="coerce")
        null_bi = df_roughness_raw["BI m/km"].isna()
        if null_bi.any():
            filled_bi = 630.0 * (df_roughness_raw.loc[null_bi, "IRI"] ** 1.12)
            df_roughness_raw.loc[null_bi, "BI m/km"] = filled_bi
            logger.info(f"Filled {null_bi.sum()} null BI values via 630*(IRI)^1.12")

    # Standardize Rutting
    if not df_rutting_raw.empty:
        df_rutting_raw.rename(columns={
            "LANE": "Lane", "Rutting Average": "Rutting",
            "Start_Latitude": "Start_Lat", "Start_Longitude": "Start_Lon",
            "End_Latitude": "End_Lat", "End_Longitude": "End_Lon",
            "survey_date_rutting": "SurveyDate",
        }, inplace=True)
        if "Lane" not in df_rutting_raw.columns:
            df_rutting_raw["Lane"] = df_rutting_raw["Source_Lane_Folder"]
        rut_val = pd.to_numeric(df_rutting_raw["Rutting"], errors="coerce")
        df_rutting_raw["Condition"] = rut_val.apply(lambda v: rutting_condition_from_mm(v, ca_rut_threshold))
        df_rutting_raw["Type_of_Defect"] = "Rutting"

    # Standardize Ravelling
    if not df_ravelling_raw.empty:
        df_ravelling_raw.rename(columns={
            "Raveling_Area_m2": "Ravelling_Area_sq_m",
            "Start_Latitude": "Start_Lat", "Start_Longitude": "Start_Lon",
            "End_Latitude": "End_Lat", "End_Longitude": "End_Lon",
            "survey_date_raveling": "SurveyDate",
        }, inplace=True)
        rav_pct = pd.to_numeric(df_ravelling_raw["Ravelling_Area_sq_m"], errors="coerce").fillna(0) * 100.0 / 35.0
        df_ravelling_raw["Ravelling"] = rav_pct
        df_ravelling_raw["Condition"] = rav_pct.apply(lambda v: ravelling_condition_from_pct(v, ravelling_threshold))
        df_ravelling_raw["Type_of_Defect"] = "Ravelling"

    return {
        "pothole": df_pothole_raw,
        "cracking": df_cracking_raw,
        "roughness": df_roughness_raw,
        "rutting": df_rutting_raw,
        "ravelling": df_ravelling_raw,
    }
