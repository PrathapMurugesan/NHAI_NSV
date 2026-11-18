"""
Unified filesystem and S3 I/O utilities for the DLM Extraction Pipeline.

Provides transparent abstraction over both local POSIX paths and S3 URIs (s3://...):
  - path_exists(path)
  - list_dir(path)
  - is_dir(path)
  - read_parquet_df(path)
  - read_excel_df(path, sheet_name, header)
  - walk_files(path, suffix)
"""

import io
import os
from typing import Any, List, Optional, Tuple, Dict
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs

from src.common.logger import get_logger

logger = get_logger(__name__)

# Cache S3 filesystems to avoid repeated connection initializations
_FS_CACHE: Dict[str, pafs.FileSystem] = {}


def _get_fs_and_path(path: str) -> Tuple[pafs.FileSystem, str]:
    """Resolve PyArrow filesystem and relative path from a URI or local path."""
    if path.startswith("s3://"):
        parts = path[5:].split("/", 1)
        bucket = parts[0]
        if bucket not in _FS_CACHE:
            _FS_CACHE[bucket] = pafs.S3FileSystem()
        return _FS_CACHE[bucket], path[5:]
    else:
        return pafs.LocalFileSystem(), os.path.abspath(path)


def path_exists(path: str) -> bool:
    """Check if a file or directory exists (supports local paths and s3:// URIs)."""
    if not path:
        return False
    if path.startswith("s3://"):
        try:
            fs, path_in_fs = _get_fs_and_path(path)
            info = fs.get_file_info(path_in_fs)
            return info.type != pafs.FileType.NotFound
        except Exception as e:
            logger.debug(f"Error checking S3 path existence {path}: {e}")
            return False
    return os.path.exists(path)


def is_dir(path: str) -> bool:
    """Check if path is a directory (supports local paths and s3:// URIs)."""
    if not path:
        return False
    if path.startswith("s3://"):
        try:
            fs, path_in_fs = _get_fs_and_path(path)
            info = fs.get_file_info(path_in_fs)
            return info.type == pafs.FileType.Directory
        except Exception:
            return False
    return os.path.isdir(path)


def list_dir(path: str) -> List[str]:
    """List entries in a directory (supports local paths and s3:// URIs)."""
    if not path:
        return []
    if path.startswith("s3://"):
        try:
            fs, path_in_fs = _get_fs_and_path(path)
            selector = pafs.FileSelector(path_in_fs, allow_not_found=True, recursive=False)
            infos = fs.get_file_info(selector)
            return [info.base_name for info in infos if info.base_name]
        except Exception as e:
            logger.debug(f"Error listing S3 directory {path}: {e}")
            return []
    try:
        return os.listdir(path) if os.path.exists(path) else []
    except Exception:
        return []


def read_parquet_df(path: str) -> pd.DataFrame:
    """Read a Parquet file into pandas DataFrame (supports local paths and s3:// URIs)."""
    if path.startswith("s3://"):
        fs, path_in_fs = _get_fs_and_path(path)
        table = pq.read_table(path_in_fs, filesystem=fs)
        return table.to_pandas()
    return pd.read_parquet(path)


def read_excel_df(path: str, sheet_name: Any = 0, header: Any = 0) -> pd.DataFrame:
    """Read an Excel file into pandas DataFrame (supports local paths and s3:// URIs)."""
    if path.startswith("s3://"):
        fs, path_in_fs = _get_fs_and_path(path)
        with fs.open_input_stream(path_in_fs) as stream:
            content = stream.read()
            return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=header)
    return pd.read_excel(path, sheet_name=sheet_name, header=header)


def walk_files(root: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Recursively find all files under root matching optional extensions.
    Returns full paths (local or s3://).
    """
    if not root:
        return []
    matched = []
    exts = [e.lower() for e in extensions] if extensions else None

    if root.startswith("s3://"):
        try:
            fs, path_in_fs = _get_fs_and_path(root)
            selector = pafs.FileSelector(path_in_fs, allow_not_found=True, recursive=True)
            infos = fs.get_file_info(selector)
            for info in infos:
                if info.type == pafs.FileType.File:
                    full_p = f"s3://{info.path}"
                    if exts is None or any(full_p.lower().endswith(e) for e in exts):
                        matched.append(full_p)
        except Exception as e:
            logger.warning(f"Error walking S3 root {root}: {e}")
        return matched

    if not os.path.exists(root):
        return []

    for r, _, files in os.walk(root):
        for f in files:
            full_p = os.path.join(r, f)
            if exts is None or any(full_p.lower().endswith(e) for e in exts):
                matched.append(full_p)
    return matched
