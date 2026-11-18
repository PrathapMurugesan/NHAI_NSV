"""
Validation report generator.
Ported from Databricks DLM Extraction Pipeline (Cell 18).

Formats and presents validation check results for a single UPC run.
"""

from typing import Dict, Any, Optional
import os
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.common.logger import get_logger

logger = get_logger(__name__)


def generate_validation_summary(val_df: pd.DataFrame) -> str:
    """
    Format validation results into a human-readable console string.
    """
    if val_df is None or val_df.empty:
        return "No validation results available."

    order_map = {'FAIL': 0, 'WARN': 1, 'SKIP': 2, 'PASS': 3}
    sorted_df = val_df.copy()
    sorted_df['_order'] = sorted_df['status'].map(lambda s: order_map.get(s, 99))
    sorted_df = sorted_df.sort_values(['_order', 'section', 'check'])

    display_cols = ['section', 'check', 'scope', 'checked', 'failed', 'status', 'detail']
    existing_cols = [c for c in display_cols if c in sorted_df.columns]

    if HAS_TABULATE:
        table_str = tabulate(
            sorted_df[existing_cols],
            headers='keys',
            tablefmt='psql',
            showindex=False,
        )
    else:
        table_str = sorted_df[existing_cols].to_string(index=False)

    counts = val_df['status'].value_counts().to_dict()
    summary_header = f"\nValidation Results Summary: {counts}\n"
    return summary_header + table_str


def save_validation_report(
    val_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Save validation report DataFrame to CSV or Parquet.
    """
    if val_df is None or val_df.empty:
        return

    is_s3 = output_path.startswith("s3://")
    if not is_s3:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if is_s3:
        from src.common.fs_utils import write_file_bytes, write_parquet_df
        if output_path.endswith('.parquet'):
            write_parquet_df(val_df, output_path)
        else:
            write_file_bytes(output_path, val_df.to_csv(index=False).encode('utf-8'))
    elif output_path.endswith('.csv'):
        val_df.to_csv(output_path, index=False)
    elif output_path.endswith('.parquet'):
        val_df.to_parquet(output_path, index=False)
    logger.info("Saved validation report to %s", output_path)
