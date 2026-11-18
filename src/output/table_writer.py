"""
Table writer module for persisting pipeline output DataFrames.
Ported from Databricks DLM Extraction Pipeline (Cell 15).

Persists tables as Parquet datasets partitioned or indexed by UPC, Phase, and Cycle.
Provides replaceWhere semantics (atomically replaces rows for the current UPC/Phase/Cycle).
"""

from typing import Dict, Any, Optional
import os
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.logger import get_logger

logger = get_logger(__name__)


class TableWriter:
    """
    Manages persistence of Silver/DLM tables to Parquet on local storage or AWS EBS/S3.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not output_dir.startswith("s3://"):
            os.makedirs(self.output_dir, exist_ok=True)

    def get_table_path(self, table_name: str) -> str:
        return os.path.join(self.output_dir, table_name)

    def write_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        upc: str,
        phase: str,
        cycle: int,
    ) -> int:
        """
        Write a table to Parquet with replaceWhere semantics:
        Replaces existing records for (upc, phase, cycle) with df.
        """
        if df is None:
            logger.warning("DataFrame for %s is None; skipping write", table_name)
            return 0

        tbl_path = self.get_table_path(table_name)
        is_s3 = tbl_path.startswith("s3://")
        if not is_s3:
            os.makedirs(tbl_path, exist_ok=True)

        parquet_file = f"{tbl_path}/data.parquet" if is_s3 else os.path.join(tbl_path, "data.parquet")

        clean_df = df.copy()
        # Ensure partition metadata exists
        if 'upc_code' in clean_df.columns:
            clean_df['upc_code'] = str(upc)
        if 'phase' in clean_df.columns:
            clean_df['phase'] = str(phase)
        if 'cycle' in clean_df.columns:
            clean_df['cycle'] = int(cycle)

        # Handle existing data (replaceWhere)
        if is_s3:
            from src.common.fs_utils import path_exists, read_parquet_df, write_parquet_df
            if path_exists(parquet_file):
                try:
                    existing = read_parquet_df(parquet_file)
                    mask = (
                        (existing.get('upc_code') == str(upc)) &
                        (existing.get('phase') == str(phase)) &
                        (existing.get('cycle') == int(cycle))
                    )
                    retained = existing[~mask]
                    combined = pd.concat([retained, clean_df], ignore_index=True)
                except Exception as e:
                    logger.warning("Could not read existing parquet %s: %s; overwriting", parquet_file, e)
                    combined = clean_df
            else:
                combined = clean_df
        elif os.path.exists(parquet_file):
            try:
                existing = pd.read_parquet(parquet_file)
                # Filter out current UPC/Phase/Cycle
                mask = (
                    (existing.get('upc_code') == str(upc)) &
                    (existing.get('phase') == str(phase)) &
                    (existing.get('cycle') == int(cycle))
                )
                retained = existing[~mask]
                combined = pd.concat([retained, clean_df], ignore_index=True)
            except Exception as e:
                logger.warning("Could not read existing parquet %s: %s; overwriting", parquet_file, e)
                combined = clean_df
        else:
            combined = clean_df

        # Serialize list/dict columns to strings or PyArrow compatibility
        out_to_save = combined.copy()
        for col in out_to_save.columns:
            if out_to_save[col].dtype == object:
                out_to_save[col] = out_to_save[col].apply(
                    lambda v: ', '.join(map(str, v)) if isinstance(v, (list, tuple))
                    else (str(v) if isinstance(v, dict) else v)
                )

        if is_s3:
            from src.common.fs_utils import write_parquet_df
            write_parquet_df(out_to_save, parquet_file)
        else:
            temp_file = parquet_file + ".tmp"
            out_to_save.to_parquet(temp_file, index=False, engine='pyarrow')
            os.replace(temp_file, parquet_file)

        written_count = len(clean_df)
        logger.info("Saved %s: %d rows (total in table: %d)", table_name, written_count, len(combined))
        return written_count

    def write_all(
        self,
        tables: Dict[str, pd.DataFrame],
        upc: str,
        phase: str,
        cycle: int,
    ) -> Dict[str, int]:
        """
        Write all output tables to their Parquet destinations.
        """
        counts = {}
        for tbl_name, df in tables.items():
            counts[tbl_name] = self.write_table(tbl_name, df, upc, phase, cycle)
        return counts

    def read_table(
        self,
        table_name: str,
        upc: Optional[str] = None,
        phase: Optional[str] = None,
        cycle: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Read back a table from Parquet, optionally filtered by UPC/Phase/Cycle.
        """
        parquet_file = os.path.join(self.get_table_path(table_name), "data.parquet")
        if not os.path.exists(parquet_file):
            return None

        df = pd.read_parquet(parquet_file)
        if upc is not None and 'upc_code' in df.columns:
            df = df[df['upc_code'] == str(upc)]
        if phase is not None and 'phase' in df.columns:
            df = df[df['phase'] == str(phase)]
        if cycle is not None and 'cycle' in df.columns:
            df = df[df['cycle'] == int(cycle)]

        return df.reset_index(drop=True)
