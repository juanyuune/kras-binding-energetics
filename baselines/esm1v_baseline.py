#!/usr/bin/env python3
"""
ESM-1v zero-shot baseline on test_bio.
Masked marginal scoring: log p(mut|context) - log p(wt|context),
averaged across the five ESM-1v ensemble models.

Sign convention: ESM-1v score is negated before correlation with ddG
(positive ESM-1v = tolerated; positive ddG = binding weakened).

Usage:
    python esm1v_baseline.py \
        --master_csv data/kras_master_table.csv \
        --output     results/
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import esm
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')


def reconstruct_wt(master):
    """Recover WT KRAS sequence from the mutant sequences in the master table."""
    sample  = master.dropna(subset=['mutant_sequence']).iloc[0]
    seq_len = len(sample['mutant_sequence'])
    wt      = list(sample['mutant_sequence'])  # fill unmutated positions from any sequence

    for _, row in master.dropna(subset=['mutant_sequence', 'wt_aa', 'position']).iterrows():
        pos = int(row['position']) - 1
        if 0 <= pos < seq_len:
            wt[pos] = row['wt_aa']

    seq = ''.join(wt)
    assert 'X' not in seq, "WT reconstruction incomplete — X still present"
    return seq


def score_model(model, alphabet, wt_seq, test_bio, device):
    bc = alphabet.get_batch_converter()
    scores = []

    for i, (_, row) in enumerate(test_bio.iterrows()):
        try:
            pos    = int(row['position']) - 1
            wt_aa  = str(row['wt_aa'])
            mut_aa = str(row['mutant_aa'])

            masked = wt_seq[:pos] + '<mask>' + wt_seq[pos+1:]
            _, _, tokens = bc([('protein', masked)])
            with torch.no_grad():
                out = model(tokens.to(device), repr_layers=[], return_contacts=False)

            lp      = torch.log_softmax(out['logits'][0, pos+1], dim=-1)
            wt_idx  = alphabet.tok_to_idx[wt_aa]
            mut_idx = alphabet.tok_to_idx[mut_aa]
            scores.append((lp[mut_idx] - lp[wt_idx]).item())

            if i % 50 == 0:
                print(f'    {i}/{len(test_bio)}  {row["variant"]}  {scores[-1]:.3f}')
        except Exception as e:
            print(f'    error {row.get("variant","?")}: {e}')
            scores.append(float('nan'))

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_csv', required=True)
    parser.add_argument('--output',     required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    master   = pd.read_csv(args.master_csv)
    test_bio = master[master['partition'] == 'test_curated'].copy().reset_index(drop=True)
    ddg_cols = [c for c in master.columns if c.startswith('ddG_')]
    print(f'test_bio: {len(test_bio)} variants  targets: {ddg_cols}')

    wt_seq = reconstruct_wt(master)
    print(f'WT sequence ({len(wt_seq)} aa): {wt_seq[:30]}...')

    # score one model at a time to stay within GPU memory
    all_scores = []
    for i in range(1, 6):
        name = f'esm1v_t33_650M_UR90S_{i}'
        print(f'\nloading {name}...')
        model, alphabet = esm.pretrained.load_model_and_alphabet(name)
        model = model.eval().to(device)
        all_scores.append(score_model(model, alphabet, wt_seq, test_bio, device))
        del model
        torch.cuda.empty_cache()
        print(f'{name} done')

    # sign-correct and average: negate so positive = binding weakened
    test_bio['esm1v_score'] = -np.nanmean(all_scores, axis=0)
    n_nan = int(np.isnan(test_bio['esm1v_score']).sum())
    print(f'\nscoring done  NaN: {n_nan}/{len(test_bio)}')

    rows = []
    for col in ddg_cols:
        t   = col.replace('ddG_', '')
        obs = test_bio[col].notna() & test_bio['esm1v_score'].notna()
        if obs.sum() < 2:
            continue
        yt, yp = test_bio.loc[obs, col].values, test_bio.loc[obs, 'esm1v_score'].values
        r,  _  = pearsonr(yt, yp)
        sr, _  = spearmanr(yt, yp)
        print(f'  {t:<10}  r={r:+.4f}  rho={sr:+.4f}  N={obs.sum()}')
        rows.append({
            'Model': 'ESM1v_zero_shot', 'Split': 'test_bio', 'Target': t,
            'Pearson_r': round(r, 4), 'Spearman_rho': round(sr, 4),
            'N': int(obs.sum()),
        })

    macro = float(np.nanmean([r['Pearson_r'] for r in rows]))
    print(f'\nmacro Pearson r (test_bio): {macro:.4f}')

    pd.DataFrame(rows).to_csv(os.path.join(args.output, 'results_esm1v.csv'), index=False)

    keep = ['variant', 'wt_aa', 'position', 'mutant_aa', 'partition', 'esm1v_score'] + ddg_cols
    test_bio[[c for c in keep if c in test_bio.columns]].to_csv(
        os.path.join(args.output, 'predictions_esm1v_testbio.csv'), index=False
    )
    print(f'saved to {args.output}')


if __name__ == '__main__':
    main()
