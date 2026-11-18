"""
Backup parquet loader for NSV lane-wise defect data.

Migrated from Notebook Cell 4 (Load All Defect Data Sources).
Preserves all business logic:
- Discovering Lane=* folders in BACKUP_BASE
- Loading matching Parquets per lane folder for each defect suffix
- Normalizing lane names (e.g. Lane=R-1 -> R1)
- Populating Source_Lane_Folder and Lane columns
"""

import os
import pandas as pd
from typing import List, Tuple, Union
from src.common.logger import logger
from src.common.constants import normalize_lane
from src.common.fs_utils import path_exists, list_dir, is_dir, read_parquet_df


def discover_lane_folders(base_path: str) -> List[str]:
    """Discover all 'Lane=*' subdirectories in base_path."""
    if not base_path or not path_exists(base_path):
        return []
    is_s3 = base_path.startswith("s3://")
    return sorted(
        d.replace("Lane=", "")
        for d in list_dir(base_path)
        if d.startswith("Lane=") and (is_s3 or os.path.isdir(os.path.join(base_path, d)))
    )


def load_matching_parquets(base_path: str, patterns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Load all Parquet files directly under base_path matching any pattern in patterns."""
    if not base_path or not path_exists(base_path):
        return pd.DataFrame(), []
    files = [
        f for f in sorted(list_dir(base_path))
        if f.endswith(".parquet") and any(p.lower() in f.lower() for p in patterns)
    ]
    if not files:
        return pd.DataFrame(), []

    frames = []
    for f in files:
        file_path = f"{base_path}/{f}" if base_path.startswith("s3://") else os.path.join(base_path, f)
        try:
            frames.append(read_parquet_df(file_path))
        except Exception as e:
            logger.warning(f"Could not read parquet {file_path}: {e}")

    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), files


def load_backup_lane_parquets(
    base_path: str,
    suffixes: Union[str, List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load lane-wise Parquets across all Lane=* directories in backup base.
    """
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    frames, files = [], []
    lanes = discover_lane_folders(base_path)

    for lane in lanes:
        lane_path = f"{base_path}/Lane={lane}" if base_path.startswith("s3://") else os.path.join(base_path, f"Lane={lane}")
        matches = [
            f for f in sorted(list_dir(lane_path))
            if f.endswith(".parquet") and any(sfx.lower() in f.lower() for sfx in suffixes)
        ]
        lane_n = normalize_lane(lane)

        for pf in matches:
            file_path = f"{lane_path}/{pf}" if lane_path.startswith("s3://") else os.path.join(lane_path, pf)
            try:
                part = read_parquet_df(file_path)
            except Exception as e:
                logger.warning(f"Could not read lane parquet {file_path}: {e}")
                continue

            if "Lane" not in part.columns and "LANE" not in part.columns:
                part["Lane"] = lane_n
            for lc in ("Lane", "LANE"):
                if lc in part.columns:
                    part[lc] = part[lc].map(normalize_lane).fillna(lane_n)
            part["Source_Lane_Folder"] = lane_n
            frames.append(part)
            files.append(f"Lane={lane}/{pf}")

    res_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return res_df, files
