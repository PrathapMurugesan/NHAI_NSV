"""
CSV exporter module for DLM tables.
Ported from Databricks DLM Extraction Pipeline (Cell 22).

Exports each table as an individual CSV file, converts complex objects
to JSON strings, and produces an _index.csv integrity report.
"""

from typing import Dict, Any, List, Optional
import os
import json
import pandas as pd
import numpy as np

from src.common.constants import ALL_DEFECTS, all_tables
from src.common.logger import get_logger

logger = get_logger(__name__)


def export_to_csv(
    tables: Dict[str, pd.DataFrame],
    export_dir: str,
) -> pd.DataFrame:
    """
    Export each table in tables to individual CSV files in export_dir.
    Returns _index DataFrame with row verification.
    """
    is_s3 = export_dir.startswith("s3://")
    if not is_s3:
        os.makedirs(export_dir, exist_ok=True)

    csv_tables: List[str] = []
    for d in ALL_DEFECTS:
        for t in all_tables(d):
            if t not in csv_tables:
                csv_tables.append(t)
    for extra_t in ['geometry_parent', 'geometry_child', 'master_chainage']:
        if extra_t not in csv_tables:
            csv_tables.append(extra_t)

    csv_index = []

    for tbl in csv_tables:
        if tbl not in tables or tables[tbl] is None:
            csv_index.append({
                'table': tbl,
                'table_rows': 0,
                'csv_rows': 0,
                'status': 'table missing',
            })
            continue

        df = tables[tbl].copy()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda v: json.dumps(v) if isinstance(v, (dict, list, tuple))
                    else (None if pd.isna(v) else v)
                )

        file_path = f"{export_dir}/{tbl}.csv" if is_s3 else os.path.join(export_dir, f"{tbl}.csv")
        if is_s3:
            from src.common.fs_utils import write_file_bytes
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            write_file_bytes(file_path, csv_bytes)
            n_back = len(df)
            status = 'OK'
        else:
            df.to_csv(file_path, index=False)
            n_back = len(pd.read_csv(file_path, usecols=[0])) if os.path.exists(file_path) and os.path.getsize(file_path) > 0 else 0
            status = 'OK' if n_back == len(df) else 'ROW MISMATCH'

        csv_index.append({
            'table': tbl,
            'table_rows': len(df),
            'csv_rows': n_back,
            'status': status,
        })
        logger.debug("CSV exported: %s (%d rows)", file_path, n_back)

    index_df = pd.DataFrame(csv_index)
    index_path = f"{export_dir}/_index.csv" if is_s3 else os.path.join(export_dir, "_index.csv")
    if is_s3:
        from src.common.fs_utils import write_file_bytes
        write_file_bytes(index_path, index_df.to_csv(index=False).encode('utf-8'))
    else:
        index_df.to_csv(index_path, index=False)

    logger.info("Exported %d CSV tables to %s", len(csv_tables), export_dir)
    return index_df
