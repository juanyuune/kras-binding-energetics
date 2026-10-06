#!/usr/bin/env python3
"""
Build the master regression table for KRAS PLM-MCNN.

One row per variant (3,553 total), with:
  - identity: variant, wt_aa, position, mutant_aa, mutant_sequence
  - 7 ddG targets: fold, RAF1, PIK3CG, RALGDS, SOS1, K27, K55 (kcal/mol)
  - 7 std columns: experimental uncertainty per target
  - 7 mask columns: 1=observed, 0=missing (never imputed)
  - partition: fold_0..fold_4, test_random, test_curated (position-exclusive)

Sign convention: positive ddG = destabilising / binding weakened.
Partition assignments are loaded from a pre-registered FASTA file —
positions were assigned before any model training to prevent leakage.

Usage:
    python build_master_table.py \
        --xlsx   data/Weng_KRAS_Six_Partner_Mutant_Sequence_Dataset.xlsx \
        --fasta  data/kras_all.fasta \
        --outdir data/
"""

import os
import argparse

import numpy as np
import pandas as pd


PARTNERS = {
    'RAF1':   'RAF1',
    'PIK3CG': 'PIK3CG',
    'RALGDS': 'RALGDS',
    'SOS1':   'SOS1',
    'K27':    'DARPin K27',
    'K55':    'DARPin K55',
}
TARGETS = ['fold'] + list(PARTNERS.keys())


def parse_fasta_partitions(fasta_path):
    """Read partition assignments from FASTA headers (key=value fields)."""
    parts = {}
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith('>'):
                continue
            fields = {}
            for tok in line.strip().lstrip('>').split('|'):
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    fields[k] = v
                else:
                    fields['variant'] = tok
            var = fields.get('variant', '')
            if var and var not in parts:
                parts[var] = fields.get('partition', 'unknown')
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx',   required=True)
    parser.add_argument('--fasta',  required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load folding ddG and variant identity from RAF1 sheet
    print('loading folding ddG from RAF1 sheet...')
    base = pd.read_excel(args.xlsx, sheet_name='RAF1')
    master = base[[
        'hgvs_pro', 'wt_aa_1', 'position', 'mutant_aa_1', 'mutant_sequence',
        'folding_ddG_kcal_mol', 'folding_std_kcal_mol', 'folding_status_d0.50',
    ]].copy().rename(columns={
        'hgvs_pro':             'variant',
        'wt_aa_1':              'wt_aa',
        'mutant_aa_1':          'mutant_aa',
        'folding_ddG_kcal_mol': 'ddG_fold',
        'folding_std_kcal_mol': 'std_fold',
        'folding_status_d0.50': 'folding_status',
    })
    master['position']  = master['position'].astype(int)
    master['mask_fold'] = master['ddG_fold'].notna().astype(int)
    print(f'  {len(master)} variants  fold observed: {master["mask_fold"].sum()}')

    # merge binding ddG per partner
    for key, sheet in PARTNERS.items():
        print(f'  loading {key} ({sheet})...')
        df = pd.read_excel(args.xlsx, sheet_name=sheet)
        df = df.rename(columns={
            'hgvs_pro':     'variant',
            'ddG_kcal_mol': f'ddG_{key}',
            'std_kcal_mol': f'std_{key}',
            'class_4':      f'class_{key}',
        })[['variant', f'ddG_{key}', f'std_{key}', f'class_{key}']]
        master = master.merge(df, on='variant', how='left')
        master[f'mask_{key}'] = master[f'ddG_{key}'].notna().astype(int)
        print(f'    {master[f"mask_{key}"].sum()} / {len(master)} observed')

    # partition assignments from pre-registered FASTA
    print(f'\nloading partitions from {args.fasta}...')
    part_map = parse_fasta_partitions(args.fasta)
    master['partition'] = master['variant'].map(part_map).fillna('unknown')
    print(master['partition'].value_counts().to_string())

    # summary columns
    mask_cols      = [f'mask_{t}' for t in TARGETS]
    bind_ddg_cols  = [f'ddG_{p}'  for p in PARTNERS]
    bind_mask_cols = [f'mask_{p}' for p in PARTNERS]

    master['n_observed'] = master[mask_cols].sum(1)
    master['mean_ddG_binding'] = master.apply(
        lambda r: float(np.mean([r[v] for v, m in zip(bind_ddg_cols, bind_mask_cols)
                                 if r[m] == 1])) if r[bind_mask_cols].sum() > 0 else np.nan,
        axis=1,
    )
    master['is_folding_stable'] = (master['folding_status'] == 'Folding stable').astype(int)

    # column order
    cols = (['variant', 'wt_aa', 'position', 'mutant_aa', 'mutant_sequence',
              'partition', 'folding_status', 'is_folding_stable',
              'n_observed', 'mean_ddG_binding'] +
            [f'ddG_{t}'   for t in TARGETS] +
            [f'std_{t}'   for t in TARGETS] +
            [f'mask_{t}'  for t in TARGETS] +
            [f'class_{p}' for p in PARTNERS])
    master = master[[c for c in cols if c in master.columns]]

    # save
    csv_path  = os.path.join(args.outdir, 'kras_master_table.csv')
    json_path = os.path.join(args.outdir, 'kras_master_table.json')
    master.to_csv(csv_path, index=False)
    master.to_json(json_path, orient='records', indent=2)
    print(f'\nsaved: {csv_path}  ({master.shape[0]} rows x {master.shape[1]} cols)')
    print(f'saved: {json_path}')

    # verification
    print('\ncoverage per target:')
    for t in TARGETS:
        obs = master[f'mask_{t}'].sum()
        print(f'  {t:<10}: {obs}/{len(master)} ({100*obs/len(master):.1f}%)')

    print('\nddG ranges:')
    for t in TARGETS:
        v = master[f'ddG_{t}'].dropna()
        print(f'  {t:<10}: [{v.min():+.3f}, {v.max():+.3f}]  mean={v.mean():+.3f}')

    g12d = master[master['variant'] == 'p.Gly12Asp']
    if len(g12d):
        row = g12d.iloc[0]
        print(f'\nG12D check: partition={row["partition"]}')
        for t in TARGETS:
            v = row[f'ddG_{t}']
            print(f'  ddG_{t:<8}: {v:+.4f}' if pd.notna(v) else f'  ddG_{t:<8}: NaN')


if __name__ == '__main__':
    main()
