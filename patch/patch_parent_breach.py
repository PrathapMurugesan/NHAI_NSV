#!/usr/bin/env python3
"""
Patch: apply the "parent breach follows the 10m children" rule to UPCs that are already processed.

Rule (same as src/hierarchy/parent_builder.py and lane_builder.py): a LHS / RHS 1000m parent of roughness and
rutting - and a lane's 1000m record in master_<defect>_1000m_lane (roughness, rutting, cracking, ravelling) - is a
breach when its value is above the limit OR when any one 10m child (of that side / of that lane) in those 1000m is a
breach. CA and IRC are decided separately, each from its own 10m rows. The extraction is NOT run again: the patch
reads the UPC's existing parquets, changes only the parent and lane tables and writes them back.

Per parent row that becomes a breach (a row that is already a breach never changes, so the patch can run twice):
    CA   breach_flag = 1, defect_id = parent_id, road_condition = 'Bad',
         affected_length_ca / affected_length = total_length, display_defect_id = new ID
    IRC  affected_length_irc = total_length, rating_irc = 'Poor'
    measures JSON gets breach_* / breach_from_child_* (and patched_parent_breach)
The same columns change on a lane record, except the display ID: the lane tables carry no display_defect_id.
New display IDs of the parents continue the UPC's counter (D/<ddmmyy>/N/<max + 1>); no existing ID of any table is
changed, so the child rows (and the image upload log) keep their IDs.
A UPC processed before the lane tables existed has no master_<defect>_1000m_lane file: it is reported as skipped
(run that UPC again to get the tables).

Nothing is written unless --apply is given. With --apply every changed parquet is first copied to
<silver-root>/_patch_backup/parent_breach_<time>/Zone=../Phase=../Cycle=../UPC=../dlm_kpi_new/ .

Which UPCs:
    --all                         every UPC found under the silver root (Zone=*/Phase=*/Cycle=*/UPC=*)
    --zone B --phase 2 --cycle 1 --upc N_07024_01001_GJ        one UPC
    --upc-file batches/upc_list1.txt                           lines "UPC ZONE PHASE CYCLE" (# = comment)
    --zone A --all                                             --zone / --phase / --cycle also narrow --all

    python3 patch/patch_parent_breach.py --all                    dry run, report only
    python3 patch/patch_parent_breach.py --all --apply            patch them
Report: patch/logs/parent_breach_<time>.csv (and <silver-root>/_patch_logs/ with --apply).
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.common.constants import PARENT_LENGTH_M, TABLE_SPEC, lane_1000m_table, normalize_lane
from src.common.fs_utils import list_dir, path_exists, read_file_bytes, write_file_bytes
from src.hierarchy.lane_builder import child_breach_lane_bins
from src.hierarchy.parent_builder import child_breach_bins

DEFAULT_SILVER_ROOT = "s3://dhms-new-stgae/DHMS/NSV/Silver"
DEFECTS = ("roughness", "rutting")             # the defects with LHS / RHS parents
LANE_DEFECTS = ("roughness", "rutting", "cracking", "ravelling")      # the defects with a lane-wise 1000m table
ID_RE = re.compile(r"^D/(\d{6})/N/(\d{5})(?:/\d{4})?$")
REPORT_COLS = ["zone", "phase", "cycle", "upc", "table", "parents", "to_ca_breach", "to_irc_breach",
               "new_display_ids", "status", "note"]


def _read(path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(read_file_bytes(path)), columns=columns)


def _write(df: pd.DataFrame, path: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    write_file_bytes(path, buf.getvalue())


def _up(a: str) -> str:
    return a.rstrip("/")


def _val(entry: str) -> str:
    return entry.split("=", 1)[1] if "=" in entry else entry


def discover(root: str, zone=None, phase=None, cycle=None) -> List[Tuple[str, str, str, str]]:
    """[(zone, phase, cycle, upc)] of every UPC folder under the silver root."""
    found = []
    for z in (e for e in list_dir(root) if e.startswith("Zone=")):
        if zone and _val(z) != str(zone):
            continue
        for p in (e for e in list_dir(f"{root}/{z}") if e.startswith("Phase=")):
            if phase and _val(p) != str(phase):
                continue
            for c in (e for e in list_dir(f"{root}/{z}/{p}") if e.startswith("Cycle=")):
                if cycle and _val(c) != str(cycle):
                    continue
                for u in (e for e in list_dir(f"{root}/{z}/{p}/{c}") if e.startswith("UPC=")):
                    found.append((_val(z), _val(p), _val(c), _val(u)))
    return sorted(found)


def from_file(path: str) -> List[Tuple[str, str, str, str]]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        upc, zone, phase, cycle = line.replace(",", " ").split()[:4]
        out.append((zone, phase, cycle, upc))
    return out


def _parent_bin(df: pd.DataFrame) -> pd.Series:
    if "parent_start_chainage" in df.columns and pd.to_numeric(df["parent_start_chainage"], errors="coerce").notna().all():
        return pd.to_numeric(df["parent_start_chainage"]).astype(int)
    lo = np.minimum(pd.to_numeric(df["start_chainage"], errors="coerce"), pd.to_numeric(df["end_chainage"], errors="coerce"))
    return (np.floor(lo / PARENT_LENGTH_M) * PARENT_LENGTH_M).astype(int)


def _counter_state(folder: str) -> Tuple[int, Optional[str]]:
    """(highest top counter, most used ddmmyy) over the display_defect_id of every table of the UPC."""
    top, dates = 0, {}
    for name in list_dir(folder):
        if not name.endswith(".parquet"):
            continue
        try:
            ids = _read(f"{folder}/{name}", ["display_defect_id"])["display_defect_id"].dropna().astype(str)
        except Exception:
            continue                                  # table without that column
        for v in ids:
            m = ID_RE.match(v)
            if m:
                top = max(top, int(m.group(2)))
                dates[m.group(1)] = dates.get(m.group(1), 0) + 1
    return top, (max(dates, key=dates.get) if dates else None)


def _survey_date(parent: pd.DataFrame) -> Optional[str]:
    for col in ("survey_end_date", "survey_start_date", "run_date", "detected_on"):
        if col in parent.columns:
            d = pd.to_datetime(parent[col], errors="coerce").dropna()
            if len(d):
                return d.iloc[0].strftime("%d%m%y")
    return None


def _patch_measures(raw, ca: bool, irc: bool, was_ca: bool, was_irc: bool, stamp: str):
    try:
        m = json.loads(raw) if isinstance(raw, str) else None
    except Exception:
        m = None
    if not isinstance(m, dict):
        return raw
    if ca:
        m.update(breach_ca=1, breach_from_value_ca=int(was_ca), breach_from_child_ca=1)
    if irc:
        m.update(breach_irc=1, breach_from_value_irc=int(was_irc), breach_from_child_irc=1)
    m["patched_parent_breach"] = stamp
    return json.dumps(m)


def _child_10m_table(defect: str) -> str:
    return next(t for t, kind in TABLE_SPEC[defect]["children"] if kind == "10m")


def _patch_lanes(root: str, key: Tuple[str, str, str, str], apply: bool, backup_root: str, stamp: str) -> List[Dict]:
    """master_<defect>_1000m_lane: breach also when any 10m child of the lane in those 1000m breaches."""
    zone, phase, cycle, upc = key
    base = dict(zone=zone, phase=phase, cycle=cycle, upc=upc)
    folder = f"{_up(root)}/Zone={zone}/Phase={phase}/Cycle={cycle}/UPC={upc}/dlm_kpi_new"
    rows = []
    for d in LANE_DEFECTS:
        t_lane, t_child = lane_1000m_table(d), _child_10m_table(d)
        l_path, c_path = f"{folder}/{t_lane}.parquet", f"{folder}/{t_child}.parquet"
        if not path_exists(l_path):
            rows.append({**base, "table": t_lane, "status": "skipped",
                         "note": "no lane table in this UPC (processed before the lane tables existed) - run it again"})
            continue
        lane = _read(l_path)
        if lane.empty:
            rows.append({**base, "table": t_lane, "parents": 0, "status": "skipped", "note": "lane table is empty"})
            continue
        if not path_exists(c_path):
            rows.append({**base, "table": t_lane, "parents": len(lane), "status": "skipped",
                         "note": f"{t_child} not found - cannot see the children"})
            continue
        child = _read(c_path)
        ca_hits = child_breach_lane_bins(child, "As per CA", PARENT_LENGTH_M)
        irc_hits = child_breach_lane_bins(child, "As per IRC", PARENT_LENGTH_M)

        bins = _parent_bin(lane)
        lanes = lane["lane"].apply(normalize_lane)
        child_ca = pd.Series([(ln, int(b)) in ca_hits for ln, b in zip(lanes, bins)], index=lane.index)
        child_irc = pd.Series([(ln, int(b)) in irc_hits for ln, b in zip(lanes, bins)], index=lane.index)
        was_ca = pd.to_numeric(lane["breach_flag"], errors="coerce").fillna(0).astype(int) == 1
        was_irc = pd.Series(False, index=lane.index)
        if "affected_length_irc" in lane.columns:
            was_irc |= pd.to_numeric(lane["affected_length_irc"], errors="coerce").fillna(0) > 0
        if "rating_irc" in lane.columns:
            was_irc |= lane["rating_irc"].astype(str) == "Poor"
        to_ca, to_irc = child_ca & ~was_ca, child_irc & ~was_irc
        n_ca, n_irc = int(to_ca.sum()), int(to_irc.sum())
        rec = {**base, "table": t_lane, "parents": len(lane), "to_ca_breach": n_ca, "to_irc_breach": n_irc,
               "new_display_ids": 0}
        if not (n_ca or n_irc):
            rows.append({**rec, "status": "unchanged", "note": ""})
            continue

        out = lane.copy()
        length = pd.to_numeric(out.get("length_m"), errors="coerce")
        length = length.fillna((pd.to_numeric(out["end_chainage"], errors="coerce") -
                                pd.to_numeric(out["start_chainage"], errors="coerce")).abs())
        if n_ca:
            out.loc[to_ca, "breach_flag"] = 1
            out.loc[to_ca, "defect_id"] = out.loc[to_ca, "parent_id"]
            out.loc[to_ca, "road_condition"] = "Bad"
            out.loc[to_ca, "affected_length_ca"] = length[to_ca]
            out.loc[to_ca, "affected_length"] = length[to_ca]
        if n_irc:
            out.loc[to_irc, "affected_length_irc"] = length[to_irc]
            if d in ("roughness", "rutting"):
                out.loc[to_irc, "rating_irc"] = "Poor"
        if "measures" in out.columns:
            for i in out.index[to_ca | to_irc]:
                out.at[i, "measures"] = _patch_measures(out.at[i, "measures"], bool(to_ca[i]), bool(to_irc[i]),
                                                        bool(was_ca[i]), bool(was_irc[i]), stamp)
        if apply:
            rel = l_path[len(_up(root)) + 1:]
            write_file_bytes(f"{backup_root}/{rel}", read_file_bytes(l_path))      # original first
            _write(out, l_path)
            rows.append({**rec, "status": "patched", "note": f"backup: {backup_root}/{rel}"})
        else:
            rows.append({**rec, "status": "would patch", "note": "dry run (use --apply)"})
    return rows


def patch_upc(root: str, key: Tuple[str, str, str, str], apply: bool, backup_root: str, stamp: str) -> List[Dict]:
    """Parents (LHS / RHS) and lane-wise 1000m tables of one UPC."""
    return _patch_parents(root, key, apply, backup_root, stamp) + _patch_lanes(root, key, apply, backup_root, stamp)


def _patch_parents(root: str, key: Tuple[str, str, str, str], apply: bool, backup_root: str, stamp: str) -> List[Dict]:
    zone, phase, cycle, upc = key
    base = dict(zone=zone, phase=phase, cycle=cycle, upc=upc)
    folder = f"{_up(root)}/Zone={zone}/Phase={phase}/Cycle={cycle}/UPC={upc}/dlm_kpi_new"
    rows, work = [], []                              # work: (table, path, parent_df, to_ca mask, to_irc mask)

    for d in DEFECTS:
        t_par, t_child = f"master_{d}_parent", f"master_{d}_child_10m"
        p_path, c_path = f"{folder}/{t_par}.parquet", f"{folder}/{t_child}.parquet"
        if not path_exists(p_path):
            rows.append({**base, "table": t_par, "status": "skipped", "note": "parent table not found"})
            continue
        parent = _read(p_path)
        if parent.empty:
            rows.append({**base, "table": t_par, "parents": 0, "status": "skipped", "note": "parent table is empty"})
            continue
        if not path_exists(c_path):
            rows.append({**base, "table": t_par, "parents": len(parent), "status": "skipped",
                         "note": f"{t_child} not found - cannot see the children"})
            continue
        child = _read(c_path)
        ca_hits = child_breach_bins(child, "As per CA", PARENT_LENGTH_M)
        irc_hits = child_breach_bins(child, "As per IRC", PARENT_LENGTH_M)

        bins = _parent_bin(parent)
        side = parent["side"].astype(str)
        child_ca = pd.Series([(s, int(b)) in ca_hits for s, b in zip(side, bins)], index=parent.index)
        child_irc = pd.Series([(s, int(b)) in irc_hits for s, b in zip(side, bins)], index=parent.index)
        was_ca = pd.to_numeric(parent["breach_flag"], errors="coerce").fillna(0).astype(int) == 1
        was_irc = pd.Series(False, index=parent.index)
        if "affected_length_irc" in parent.columns:
            was_irc |= pd.to_numeric(parent["affected_length_irc"], errors="coerce").fillna(0) > 0
        if "rating_irc" in parent.columns:
            was_irc |= parent["rating_irc"].astype(str) == "Poor"
        work.append((t_par, p_path, parent, child_ca & ~was_ca, child_irc & ~was_irc, was_ca, was_irc))

    # new display IDs for the parents that become a CA breach: continue the UPC's counter, never reuse one
    need = sum(int(w[3].sum()) for w in work)
    next_seq, date = 0, None
    if need:
        top, date = _counter_state(folder)
        date = date or _survey_date(work[0][2])
        next_seq = top + 1
        if date is None:
            for w in work:
                rows.append({**base, "table": w[0], "parents": len(w[2]), "to_ca_breach": int(w[3].sum()),
                             "status": "skipped", "note": "new display IDs needed but the date could not be found"})
            return rows

    for t_par, p_path, parent, to_ca, to_irc, was_ca, was_irc in work:
        n_ca, n_irc = int(to_ca.sum()), int(to_irc.sum())
        rec = {**base, "table": t_par, "parents": len(parent), "to_ca_breach": n_ca, "to_irc_breach": n_irc,
               "new_display_ids": n_ca}
        if not (n_ca or n_irc):
            rows.append({**rec, "status": "unchanged", "note": ""})
            continue

        out = parent.copy()
        total = pd.to_numeric(out.get("total_length"), errors="coerce")
        total = total.fillna(pd.to_numeric(out.get("length_m"), errors="coerce"))
        if n_ca:
            order = out[to_ca].copy()                       # same order as the pipeline: side, lane, chainage, id
            order["_ch"] = pd.to_numeric(order["start_chainage"], errors="coerce").fillna(999999999)
            order = order.sort_values(["side", "lane", "_ch", "id"], kind="stable")
            new_ids = {i: f"D/{date}/N/{next_seq + k:05d}" for k, i in enumerate(order.index)}
            next_seq += n_ca
            out.loc[to_ca, "breach_flag"] = 1
            out.loc[to_ca, "defect_id"] = out.loc[to_ca, "parent_id"]
            out.loc[to_ca, "road_condition"] = "Bad"
            out.loc[to_ca, "affected_length_ca"] = total[to_ca]
            out.loc[to_ca, "affected_length"] = total[to_ca]
            out.loc[to_ca, "display_defect_id"] = pd.Series(new_ids)
        if n_irc:
            out.loc[to_irc, "affected_length_irc"] = total[to_irc]
            out.loc[to_irc, "rating_irc"] = "Poor"
        if "measures" in out.columns:
            for i in out.index[to_ca | to_irc]:
                out.at[i, "measures"] = _patch_measures(out.at[i, "measures"], bool(to_ca[i]), bool(to_irc[i]),
                                                        bool(was_ca[i]), bool(was_irc[i]), stamp)
        if apply:
            rel = p_path[len(_up(root)) + 1:]
            write_file_bytes(f"{backup_root}/{rel}", read_file_bytes(p_path))      # original first
            _write(out, p_path)
            rows.append({**rec, "status": "patched", "note": f"backup: {backup_root}/{rel}"})
        else:
            rows.append({**rec, "status": "would patch", "note": "dry run (use --apply)"})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--silver-root", default=os.environ.get("SILVER_S3_ROOT", DEFAULT_SILVER_ROOT))
    ap.add_argument("--all", action="store_true", help="every UPC under the silver root")
    ap.add_argument("--zone")
    ap.add_argument("--phase")
    ap.add_argument("--cycle")
    ap.add_argument("--upc")
    ap.add_argument("--upc-file", help='lines "UPC ZONE PHASE CYCLE"')
    ap.add_argument("--apply", action="store_true", help="write the patched parquets (default: dry run)")
    ap.add_argument("--workers", type=int, default=4, help="UPCs in parallel (default 4)")
    a = ap.parse_args(argv)

    root = _up(a.silver_root)
    if a.upc_file:
        keys = from_file(a.upc_file)
    elif a.upc:
        if not (a.zone and a.phase and a.cycle):
            ap.error("--upc needs --zone, --phase and --cycle")
        keys = [(a.zone, str(a.phase), str(a.cycle), a.upc.replace("/", "_"))]
    elif a.all:
        keys = discover(root, a.zone, a.phase, a.cycle)
    else:
        ap.error("give --all, --upc (with --zone --phase --cycle) or --upc-file")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = f"{root}/_patch_backup/parent_breach_{stamp}"
    print(f"{'APPLY' if a.apply else 'DRY RUN'}  silver root {root}  {len(keys)} UPC(s)")

    def one(key):
        try:
            return patch_upc(root, key, a.apply, backup_root, stamp)
        except Exception as e:                          # one bad UPC never stops the rest
            return [dict(zone=key[0], phase=key[1], cycle=key[2], upc=key[3], status="ERROR", note=f"{type(e).__name__}: {e}")]

    report = []
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for n, rows in enumerate(pool.map(one, keys), start=1):
            report += rows
            hit = [r for r in rows if r.get("status") in ("patched", "would patch")]
            print(f"[{n}/{len(keys)}] {keys[n - 1][3]:20} " +
                  (", ".join(f"{r['table'].replace('master_', '')}: +{r.get('to_ca_breach', 0)} CA +{r.get('to_irc_breach', 0)} IRC"
                             for r in hit) if hit else "/".join(sorted({r.get('status', '') for r in rows}))))

    os.makedirs(os.path.join(ROOT, "patch", "logs"), exist_ok=True)
    local = os.path.join(ROOT, "patch", "logs", f"parent_breach_{stamp}.csv")
    with open(local, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, REPORT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(report)
    if a.apply:
        write_file_bytes(f"{root}/_patch_logs/parent_breach_{stamp}.csv", open(local, "rb").read())

    st = pd.DataFrame(report).groupby("status").size() if report else pd.Series(dtype=int)
    print("\n" + "\n".join(f"  {n:5}  {s}" for s, n in st.items()))
    print(f"  parents turned CA breach {sum(int(r.get('to_ca_breach') or 0) for r in report)}, "
          f"IRC {sum(int(r.get('to_irc_breach') or 0) for r in report)}, new display IDs "
          f"{sum(int(r.get('new_display_ids') or 0) for r in report if r.get('status') in ('patched', 'would patch'))}")
    print(f"report: {local}" + (f"  and  {root}/_patch_logs/parent_breach_{stamp}.csv" if a.apply else ""))
    if not a.apply:
        print("dry run - nothing was written. Add --apply to patch.")
    return 1 if any(r.get("status") == "ERROR" for r in report) else 0


if __name__ == "__main__":
    sys.exit(main())
