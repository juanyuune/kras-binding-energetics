"""
build_master_table.py  —  Phase 1, Step 1
==========================================
Builds the seven-output regression master table for KRAS PLM-MCNN.

For each of the 3,553 KRAS missense variants, produces one row with:
  - Identity    : hgvs_pro, position, wt_aa, mutant_aa, mutant_sequence
  - 7 targets   : ddG_fold, ddG_RAF1, ddG_PIK3CG, ddG_RALGDS,
                  ddG_SOS1, ddG_K27, ddG_K55  (kcal/mol, continuous)
  - 7 std       : std_fold, std_RAF1, ...      (experimental uncertainty)
  - 7 masks     : mask_fold, mask_RAF1, ...    (1=observed, 0=missing)
  - Fold status : folding_status               (Folding stable/destabilized/etc)
  - Fold assign : partition                    (fold_0 to fold_4,
                                                test_random, test_bio)
  - Derived cls : class_fold, class_RAF1, ...  (for reference only)

Design decisions (following professor's specification):
  1. Missing values are NOT imputed — mask = 0 means excluded from loss
  2. Sign convention: positive ddG = destabilising / weakening
  3. Fold assignment loaded from existing FASTA split files
     (position-exclusive, pre-registered before model training)
  4. Folding ddG is the 7th regression target — NOT a filter criterion

Output:
  /KRAS/data/kras_master_table.csv   — full table (3,553 rows)
  /KRAS/data/kras_master_table.json  — same, for easy inspection

Run:
  python build_master_table.py \
    --xlsx   /srv/jupyterlab/workspace/KRAS/Weng_KRAS_Six_Partner_Mutant_Sequence_Dataset.xlsx \
    --fasta  /srv/jupyterlab/workspace/KRAS/fasta/kras_all.fasta \
    --outdir /srv/jupyterlab/workspace/KRAS/data/
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--xlsx",   required=True,
                    help="Path to Weng_KRAS_Six_Partner_Mutant_Sequence_Dataset.xlsx")
parser.add_argument("--fasta",  required=True,
                    help="Path to kras_all.fasta (contains partition assignments)")
parser.add_argument("--outdir", required=True,
                    help="Output directory for master table CSV and JSON")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── sheet definitions ─────────────────────────────────────────────────────────
PARTNER_SHEETS = {
    'RAF1':   'RAF1',
    'PIK3CG': 'PIK3CG',
    'RALGDS': 'RALGDS',
    'SOS1':   'SOS1',
    'K27':    'DARPin K27',
    'K55':    'DARPin K55',
}

TARGETS   = ['fold'] + list(PARTNER_SHEETS.keys())
N_TARGETS = len(TARGETS)   # 7

# ── Step 1: load folding ddG from RAF1 sheet (same across all sheets) ─────────
logging.info("Loading folding ddG and variant identity from RAF1 sheet...")

base = pd.read_excel(args.xlsx, sheet_name='RAF1')

# keep only needed columns
master = base[[
    'hgvs_pro', 'wt_aa_1', 'position', 'mutant_aa_1', 'mutant_sequence',
    'folding_ddG_kcal_mol', 'folding_std_kcal_mol', 'folding_status_d0.50',
]].copy()

master = master.rename(columns={
    'hgvs_pro':              'variant',
    'wt_aa_1':               'wt_aa',
    'mutant_aa_1':           'mutant_aa',
    'folding_ddG_kcal_mol':  'ddG_fold',
    'folding_std_kcal_mol':  'std_fold',
    'folding_status_d0.50':  'folding_status',
})

master['position'] = master['position'].astype(int)

logging.info(f"  Base table: {len(master)} variants")

# ── Step 2: add mask for folding ddG ─────────────────────────────────────────
master['mask_fold'] = master['ddG_fold'].notna().astype(int)
logging.info(f"  Folding ddG observed: {master['mask_fold'].sum()} / {len(master)}")

# ── Step 3: add binding ddG per partner ───────────────────────────────────────
logging.info("Loading binding ddG for each partner...")

for partner_key, sheet_name in PARTNER_SHEETS.items():
    logging.info(f"  Loading {partner_key} ({sheet_name})...")

    df_p = pd.read_excel(args.xlsx, sheet_name=sheet_name)

    # merge on hgvs_pro
    df_p = df_p.rename(columns={'hgvs_pro': 'variant'})
    df_p = df_p[['variant', 'ddG_kcal_mol', 'std_kcal_mol', 'class_4']].copy()
    df_p = df_p.rename(columns={
        'ddG_kcal_mol': f'ddG_{partner_key}',
        'std_kcal_mol': f'std_{partner_key}',
        'class_4':      f'class_{partner_key}',
    })

    master = master.merge(df_p, on='variant', how='left')

    # mask: 1 if ddG observed (not NaN)
    master[f'mask_{partner_key}'] = master[f'ddG_{partner_key}'].notna().astype(int)
    obs = master[f'mask_{partner_key}'].sum()
    logging.info(f"    {partner_key}: {obs} / {len(master)} observed")

# ── Step 4: load partition assignments from FASTA ────────────────────────────
logging.info("Loading partition assignments from FASTA...")

def parse_partition_map(fasta_path):
    """Returns {variant_name: partition_string}"""
    part_map = {}
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line.strip().lstrip('>')
            parts  = header.split('|')
            fields = {'variant': parts[0]}
            for p in parts[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    fields[k] = v
            var  = fields.get('variant', '')
            part = fields.get('partition', 'unknown')
            if var and var not in part_map:
                part_map[var] = part
    return part_map

part_map = parse_partition_map(args.fasta)
logging.info(f"  Loaded {len(part_map)} partition assignments")

master['partition'] = master['variant'].map(part_map)

# check for unmapped variants
unmapped = master['partition'].isna().sum()
if unmapped > 0:
    logging.warning(f"  {unmapped} variants not found in FASTA — "
                    f"partition set to 'unknown'")
    master['partition'] = master['partition'].fillna('unknown')

# partition distribution
logging.info("  Partition distribution:")
for part, count in master['partition'].value_counts().items():
    logging.info(f"    {part:<15}: {count}")

# ── Step 5: compute summary columns ──────────────────────────────────────────
logging.info("Computing summary columns...")

ddg_cols  = [f'ddG_{t}'  for t in TARGETS]
std_cols  = [f'std_{t}'  for t in TARGETS]
mask_cols = [f'mask_{t}' for t in TARGETS]

# number of observed targets per variant
master['n_observed'] = master[mask_cols].sum(axis=1)

# mean ddG across observed binding partners (not fold)
bind_ddg_cols  = [f'ddG_{p}'  for p in PARTNER_SHEETS]
bind_mask_cols = [f'mask_{p}' for p in PARTNER_SHEETS]

def masked_mean(row, val_cols, mask_cols):
    vals  = [row[v] for v, m in zip(val_cols, mask_cols) if row[m] == 1]
    return float(np.mean(vals)) if vals else np.nan

master['mean_ddG_binding'] = master.apply(
    lambda r: masked_mean(r, bind_ddg_cols, bind_mask_cols), axis=1
)

# ── Step 6: derive impact class from folding ──────────────────────────────────
# Folding stable = folding_ddG ≤ 0.50 kcal/mol (professor's d0.50 threshold)
# This is used for the folding-conditioned model analysis
master['is_folding_stable'] = (
    master['folding_status'] == 'Folding stable'
).astype(int)

# ── Step 7: final column order ────────────────────────────────────────────────
col_order = (
    ['variant', 'wt_aa', 'position', 'mutant_aa', 'mutant_sequence']
    + ['partition', 'folding_status', 'is_folding_stable']
    + ['n_observed', 'mean_ddG_binding']
    # targets
    + [f'ddG_{t}'  for t in TARGETS]
    # standard deviations
    + [f'std_{t}'  for t in TARGETS]
    # masks
    + [f'mask_{t}' for t in TARGETS]
    # derived binding classes (for reference)
    + [f'class_{p}' for p in PARTNER_SHEETS]
)
# only keep columns that exist
col_order = [c for c in col_order if c in master.columns]
master = master[col_order]

# ── Step 8: save ──────────────────────────────────────────────────────────────
csv_path  = os.path.join(args.outdir, 'kras_master_table.csv')
json_path = os.path.join(args.outdir, 'kras_master_table.json')

master.to_csv(csv_path,  index=False)
master.to_json(json_path, orient='records', indent=2)

logging.info(f"\nSaved: {csv_path}")
logging.info(f"Saved: {json_path}")

# ── Step 9: verification printout ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("MASTER TABLE VERIFICATION")
print("=" * 65)

print(f"\nShape: {master.shape}  ({master.shape[0]} variants × {master.shape[1]} columns)")

print(f"\n--- TARGET COVERAGE ---")
print(f"{'Target':<12} {'Observed':>10} {'Missing':>8} {'%':>8}")
print("-" * 42)
for t in TARGETS:
    obs     = master[f'mask_{t}'].sum()
    missing = len(master) - obs
    pct     = 100 * obs / len(master)
    print(f"  {t:<10} {obs:>10} {missing:>8} {pct:>7.1f}%")

print(f"\n--- VARIANTS BY N_OBSERVED ---")
for n in sorted(master['n_observed'].unique(), reverse=True):
    count = (master['n_observed'] == n).sum()
    print(f"  n_observed={n}: {count:4d} variants")

print(f"\n--- PARTITION DISTRIBUTION ---")
print(f"{'Partition':<15} {'N':>6} {'%':>8}")
print("-" * 32)
for part, count in master['partition'].value_counts().items():
    pct = 100 * count / len(master)
    print(f"  {part:<15} {count:>6} {pct:>7.1f}%")

print(f"\n--- FOLDING STATUS ---")
for status, count in master['folding_status'].value_counts().items():
    print(f"  {status:<30}: {count:4d}")

print(f"\n--- ddG RANGES (observed values) ---")
for t in TARGETS:
    col   = f'ddG_{t}'
    obs   = master[col].dropna()
    print(f"  {t:<10}: min={obs.min():+.3f}  "
          f"max={obs.max():+.3f}  "
          f"mean={obs.mean():+.3f}  "
          f"std={obs.std():.3f}")

print(f"\n--- SAMPLE ROW (G12D) ---")
g12d = master[master['variant'] == 'p.Gly12Asp']
if len(g12d):
    row = g12d.iloc[0]
    print(f"  variant   : {row['variant']}")
    print(f"  partition : {row['partition']}")
    for t in TARGETS:
        obs = row[f'mask_{t}']
        val = row[f'ddG_{t}']
        val_str = f"{val:+.4f}" if pd.notna(val) else "   NaN"
        print(f"  ddG_{t:<8}: {val_str}  mask={int(obs)}")

print(f"\n{'='*65}")
print(f"Master table complete. Output: {args.outdir}")
print(f"{'='*65}")