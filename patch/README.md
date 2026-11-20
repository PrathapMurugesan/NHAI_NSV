# Patches for UPCs that are already processed

## `patch_parent_breach.py` - parent breach follows the 10m children

The extraction now flags a LHS / RHS 1000m parent (roughness, rutting) as a breach when its value is above the
limit **or when any one 10m child of that side in those 1000m is a breach** (CA and IRC each from their own 10m
rows). This patch applies that rule to the Silver parquets that are already in S3. The extraction is **not** run
again - it reads `master_<defect>_parent` and `master_<defect>_child_10m` of each UPC and rewrites the parent file.

What changes on a parent that becomes a breach (a row that is already a breach never changes):

| Column | New value |
|---|---|
| `breach_flag` | 1 |
| `defect_id` | the `parent_id` |
| `road_condition` | `Bad` |
| `affected_length_ca`, `affected_length` | the parent's `total_length` |
| `affected_length_irc`, `rating_irc` (IRC child breach) | `total_length`, `Poor` |
| `display_defect_id` (CA) | a new `D/<ddmmyy>/N/<n>`, continuing the UPC's counter |
| `measures` (JSON) | `breach_ca`, `breach_from_child_ca` / `_irc`, `patched_parent_breach` |

No existing display ID of any table changes, so the children and the image upload log keep their IDs. Children
that were breaching without a flagged parent keep their own top-level ID. `rating_ca` (the Good / Fair / Poor band of
the average) is not touched.

```bash
cd /root/nsv/NHAI_NSV && git pull && source .venv/bin/activate

python3 patch/patch_parent_breach.py --all                     # 1. DRY RUN on every processed UPC: report only
python3 patch/patch_parent_breach.py --all --apply             # 2. patch them
```

Other selections (all of them also work as a dry run without `--apply`):

```bash
python3 patch/patch_parent_breach.py --zone B --phase 2 --cycle 1 --upc N_07024_01001_GJ --apply    # one UPC
python3 patch/patch_parent_breach.py --zone A --all --apply                                          # one zone (also --phase, --cycle)
python3 patch/patch_parent_breach.py --upc-file batches/upc_list1.txt --apply                        # a batch list (UPC ZONE PHASE CYCLE)
python3 patch/patch_parent_breach.py --all --silver-root s3://bucket/other/Silver                     # another silver root
```

* UPCs are found under `--silver-root` (default `$SILVER_S3_ROOT`, else `s3://dhms-new-stgae/DHMS/NSV/Silver`):
  `Zone=*/Phase=*/Cycle=*/UPC=*/dlm_kpi_new/`. `--workers N` runs N UPCs at the same time (default 4).
* **Nothing is written without `--apply`.** With `--apply`, every parquet that changes is first copied to
  `<silver-root>/_patch_backup/parent_breach_<time>/Zone=../Phase=../Cycle=../UPC=../dlm_kpi_new/` - copy it back to undo.
* Safe to run twice: a second run reports `unchanged`. A UPC without a parent or child table, or one that fails,
  is reported (`skipped` / `ERROR`) and never stops the others.
* Report per UPC and table (parents, parents turned CA / IRC breach, new IDs, status): `patch/logs/parent_breach_<time>.csv`
  and, with `--apply`, `<silver-root>/_patch_logs/parent_breach_<time>.csv`.
* Run the dry run first and look at the numbers; UPCs processed with the new extraction code already follow the rule
  and show `unchanged`.
