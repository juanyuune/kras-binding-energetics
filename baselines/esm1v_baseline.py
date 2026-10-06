"""
esm1v_baseline.py
=================
Run ESM-1v zero-shot mutation effect prediction on test_bio variants.
Uses masked marginal scoring: log p(mut|context) - log p(wt|context)
averaged across all five ESM-1v ensemble models.

Usage:
  python esm1v_baseline.py \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --output     /srv/jupyterlab/workspace/KRAS/code/results/
"""

import os, argparse, logging
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument('--master_csv', required=True)
parser.add_argument('--output',     required=True)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")

# ── load master table ─────────────────────────────────────────────────────────
master = pd.read_csv(args.master_csv)
logging.info(f"Columns: {list(master.columns)}")

# Identify ΔΔG target columns
all_cols = list(master.columns)
ddg_cols = [c for c in all_cols if c.startswith('ddG_')]
logging.info(f"ΔΔG columns found: {ddg_cols}")

# test_bio partition
test_bio = master[master['partition']=='test_curated'].copy().reset_index(drop=True)
logging.info(f"test_bio variants: {len(test_bio)}")
logging.info(f"Sample:\n{test_bio[['variant','wt_aa','position','mutant_aa']].head(5)}")

# ── reconstruct WT sequence from data ────────────────────────────────────────
# Use mutant_sequence and position to recover WT
# mutant_sequence has mutant_aa at position-1 (0-indexed), rest is WT
sample = master.dropna(subset=['mutant_sequence']).iloc[0]
seq_len = len(sample['mutant_sequence'])
wt_seq_arr = ['X'] * seq_len

for _, row in master.dropna(subset=['mutant_sequence','wt_aa','position']).iterrows():
    pos_0 = int(row['position']) - 1
    if 0 <= pos_0 < seq_len:
        wt_seq_arr[pos_0] = row['wt_aa']

# Fill any remaining X with the mutant sequence of a nearby variant
# (X positions were never mutated — fill from any mutant seq at non-mutated positions)
sample_seq = sample['mutant_sequence']
for i, aa in enumerate(wt_seq_arr):
    if aa == 'X':
        wt_seq_arr[i] = sample_seq[i]

wt_seq = ''.join(wt_seq_arr)
logging.info(f"Reconstructed WT sequence length: {len(wt_seq)}")
logging.info(f"WT seq (first 30): {wt_seq[:30]}")
logging.info(f"X remaining: {wt_seq.count('X')}")

# Verify: at mutated positions, WT seq should match wt_aa
errors = 0
for _, row in test_bio.dropna(subset=['wt_aa','position']).head(20).iterrows():
    pos_0 = int(row['position']) - 1
    if wt_seq[pos_0] != row['wt_aa']:
        logging.warning(f"  WT mismatch pos {int(row['position'])}: seq={wt_seq[pos_0]} vs wt_aa={row['wt_aa']}")
        errors += 1
logging.info(f"WT verification errors (first 20): {errors}")


# ── load and score one model at a time to save GPU memory ────────────────────
import esm

logging.info(f"\nScoring {len(test_bio)} variants (one ESM-1v model at a time)...")
all_model_scores = []

for model_i in range(1, 6):
    model_name = f"esm1v_t33_650M_UR90S_{model_i}"
    logging.info(f"  Loading {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    logging.info(f"  Scoring with {model_name}...")

    bc = alphabet.get_batch_converter()
    model_scores = []
    for i, (idx, row) in enumerate(test_bio.iterrows()):
        try:
            pos_0 = int(row['position']) - 1
            wt_aa = str(row['wt_aa'])
            mut_aa = str(row['mutant_aa'])
            masked_seq = wt_seq[:pos_0] + '<mask>' + wt_seq[pos_0+1:]
            _, _, tokens = bc([("protein", masked_seq)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[], return_contacts=False)
            logits = out['logits'][0, pos_0+1]
            lp = torch.log_softmax(logits, dim=-1)
            wt_idx  = alphabet.tok_to_idx[wt_aa]
            mut_idx = alphabet.tok_to_idx[mut_aa]
            model_scores.append((lp[mut_idx] - lp[wt_idx]).item())
            if i % 50 == 0:
                logging.info(f"    {i}/171 {row['variant']} score={model_scores[-1]:.4f}")
        except Exception as e:
            logging.error(f"    Error {row.get('variant','?')}: {e}")
            model_scores.append(float('nan'))

    all_model_scores.append(model_scores)
    del model
    torch.cuda.empty_cache()
    logging.info(f"  {model_name} done, unloaded")

import numpy as np
esm_scores = list(np.nanmean(all_model_scores, axis=0))


test_bio = test_bio.copy()
test_bio['esm1v_score'] = esm_scores
n_nan = int(np.sum(np.isnan(esm_scores)))
logging.info(f"\nScoring complete. NaN: {n_nan}/{len(test_bio)}")

# ── evaluate ──────────────────────────────────────────────────────────────────
logging.info("\n=== ESM-1v Zero-Shot vs Experimental ΔΔG ===\n")
results = []
for col in ddg_cols:
    t = col.replace('ddG_', '')
    obs = test_bio[col].notna() & test_bio['esm1v_score'].notna()
    if obs.sum() < 2:
        continue
    yt = test_bio.loc[obs, col].values
    yp = test_bio.loc[obs, 'esm1v_score'].values
    r,  _ = pearsonr(yt, yp)
    sr, _ = spearmanr(yt, yp)
    mae   = float(np.mean(np.abs(yt - yp)))
    logging.info(f"  {t:<10}: r={r:+.4f}  rho={sr:+.4f}  MAE={mae:.4f}  N={obs.sum()}")
    results.append({'Model':'ESM1v_zero_shot','Split':'test_bio',
                    'Target':t,'Pearson_r':round(r,4),
                    'Spearman_rho':round(sr,4),'MAE':round(mae,4),
                    'N':int(obs.sum())})

if results:
    macro_r = float(np.nanmean([r['Pearson_r'] for r in results]))
    logging.info(f"\n  Macro Pearson r (test_bio): {macro_r:.4f}")

    out_csv = os.path.join(args.output, 'results_esm1v.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)

    pred_csv = os.path.join(args.output, 'predictions_esm1v_testbio.csv')
    save_cols = ['variant','wt_aa','position','mutant_aa','partition','esm1v_score'] + ddg_cols
    test_bio[[c for c in save_cols if c in test_bio.columns]].to_csv(pred_csv, index=False)

    logging.info(f"Saved: {out_csv}")
    logging.info(f"Saved: {pred_csv}")
    print(f"\nDone. Macro Pearson r = {macro_r:.4f}")
else:
    logging.error("No results computed — check ddg_cols")