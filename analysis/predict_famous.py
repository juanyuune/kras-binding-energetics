"""
predict_famous.py
=================
Extracts PLM-MCNN predictions for the 8 canonical oncogenic KRAS mutations:
  G12D, G12V, G12R, G12C, G13D, Q61H, Q61L, Q61R

All 8 are in test_curated (biological challenge set) — the model never saw
these positions during training. Compares model predictions to experimental
ΔΔG values and clinical expectations.

Run:
  python predict_famous.py \
    --tensor_dir /srv/jupyterlab/workspace/KRAS/tensors/ \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --model_pt   /srv/jupyterlab/workspace/KRAS/code/results/mcnn_k1_4_8_16_p256_f256.pt \
    --output     /srv/jupyterlab/workspace/KRAS/code/results/
"""

import os, argparse, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

from dataset import (
    KRASRegressionDataset, compute_target_stats,
    TARGET_NAMES, N_TARGETS, SPLIT_PARTITIONS, MASK_COLS,
)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tensor_dir", required=True)
parser.add_argument("--master_csv", required=True)
parser.add_argument("--model_pt",   required=True,
                    help="Path to saved model .pt file")
parser.add_argument("--output",     required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

# ── famous mutations ───────────────────────────────────────────────────────────
FAMOUS = {
    'G12D': {'position': 12, 'wt': 'G', 'mut': 'D'},
    'G12V': {'position': 12, 'wt': 'G', 'mut': 'V'},
    'G12R': {'position': 12, 'wt': 'G', 'mut': 'R'},
    'G12C': {'position': 12, 'wt': 'G', 'mut': 'C'},
    'G13D': {'position': 13, 'wt': 'G', 'mut': 'D'},
    'Q61H': {'position': 61, 'wt': 'Q', 'mut': 'H'},
    'Q61L': {'position': 61, 'wt': 'Q', 'mut': 'L'},
    'Q61R': {'position': 61, 'wt': 'Q', 'mut': 'R'},
}

# ── clinical context (from literature) ────────────────────────────────────────
CLINICAL = {
    'G12D': 'Most common KRAS mutation; impairs GTP hydrolysis; near-neutral binding profile',
    'G12V': 'Strong GTPase impairment; broad effector engagement; activating',
    'G12R': 'Specificity modulator; SOS1 exchange strongly impaired; effector-selective',
    'G12C': 'Covalent inhibitor target (sotorasib, adagrasib); SOS1 partially impaired',
    'G13D': 'Near-neutral across partners; GEF exchange partially retained',
    'Q61H': 'Moderate GTPase impairment; near-neutral binding profile',
    'Q61L': 'Strong activating; SOS1 weakened; conformation-selective',
    'Q61R': 'Strong activating; SOS1 strongly weakened; K27 conformation shift',
}

DDG_DELTA = 0.25

def classify(ddg):
    if np.isnan(ddg): return 'NaN'
    if ddg < -DDG_DELTA: return 'Enhanced'
    if ddg >  DDG_DELTA: return 'Weakened'
    return 'Neutral'


# ── reconstruct model architecture (must match training) ─────────────────────
class MutationCenteredPool(nn.Module):
    def __init__(self, half_win=5):
        super().__init__()
        self.half_win = half_win

    def forward(self, conv_out, mask_ch):
        B, F, L = conv_out.shape
        if mask_ch is None:
            pos = L // 2
            lo  = max(0, pos - self.half_win)
            hi  = min(L, pos + self.half_win + 1)
            return conv_out[:, :, lo:hi].mean(dim=-1)
        out = torch.zeros(B, F, device=conv_out.device)
        for b in range(B):
            pos = int(mask_ch[b].argmax().item())
            lo  = max(0, pos - self.half_win)
            hi  = min(L, pos + self.half_win + 1)
            out[b] = conv_out[b, :, lo:hi].mean(dim=-1)
        return out


class KRAS_MCNN(nn.Module):
    def __init__(self, n_channels=5, emb_dim=1280, proj_dim=256,
                 kernels=None, n_filters=256, local_win=5,
                 dropout=0.30, n_targets=N_TARGETS):
        super().__init__()
        if kernels is None: kernels = [1,4,8,16]
        self.kernels  = kernels
        self.proj_dim = proj_dim
        self.n_ch     = n_channels

        self.projection = nn.Linear(emb_dim, proj_dim, bias=False)
        in_channels     = n_channels * proj_dim

        self.conv_branches = nn.ModuleList()
        for k in kernels:
            self.conv_branches.append(nn.Sequential(
                nn.Conv1d(in_channels, n_filters, kernel_size=k, padding=k//2),
                nn.GELU(),
            ))

        self.mut_pool  = MutationCenteredPool(half_win=local_win)
        branch_dim     = n_filters * 3
        concat_dim     = branch_dim * len(kernels)

        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(n_targets)])

    def forward(self, x_full):
        B, _, L, D = x_full.shape
        # select channels to match n_channels from checkpoint
        if self.n_ch == 4:
            ch_idx = [0,1,2,3]   # A4: no mask channel
        else:
            ch_idx = [0,1,2,3,4] # full model: all 5 channels
        x        = x_full[:, ch_idx, :, :]
        mask_ch  = x_full[:, 4, :, 0] if self.n_ch == 5 else None
        x_proj   = self.projection(x)
        x_flat   = x_proj.permute(0,1,3,2)
        x_flat   = x_flat.reshape(B, self.n_ch * self.proj_dim, L)

        branch_outs = []
        for conv in self.conv_branches:
            feat = conv(x_flat)
            if feat.shape[2] != L: feat = feat[:,:,:L]
            g_max  = feat.max(dim=2).values
            g_mean = feat.mean(dim=2)
            m_pool = self.mut_pool(feat, mask_ch)
            branch_outs.append(torch.cat([g_max, g_mean, m_pool], dim=1))

        h     = torch.cat(branch_outs, dim=1)
        h     = self.trunk(h)
        preds = torch.cat([head(h) for head in self.heads], dim=1)
        return preds


def main():
    print("=" * 70)
    print("Famous Mutation Analysis — KRAS PLM-MCNN")
    print("=" * 70)

    # ── load master table ──────────────────────────────────────────────────────
    master = pd.read_csv(args.master_csv)

    # ── normalisation stats ────────────────────────────────────────────────────
    train_ds = KRASRegressionDataset(
        split='train', tensor_dir=args.tensor_dir, master_csv=args.master_csv)
    means, stds = compute_target_stats(train_ds)

    # ── find famous mutations ─────────────────────────────────────────────────
    results = []
    for name, info in FAMOUS.items():
        row = master[
            (master['position'] == info['position']) &
            (master['mutant_aa'] == info['mut'])
        ]
        if len(row) == 0:
            logging.warning(f"  {name} not found in master table")
            continue
        row = row.iloc[0]
        logging.info(f"  {name}: variant={row['variant']}  partition={row['partition']}")
        results.append({'name': name, 'variant': row['variant'],
                        'position': row['position'], 'mut': info['mut'],
                        'partition': row['partition'], 'row': row})

    # ── load model ─────────────────────────────────────────────────────────────
    # detect n_channels from checkpoint to handle any saved model
    state       = torch.load(args.model_pt, map_location=device)
    conv_weight = state.get('conv_branches.0.0.weight', None)
    proj_weight = state.get('projection.weight', None)
    if conv_weight is not None and proj_weight is not None:
        proj_dim   = proj_weight.shape[0]      # e.g. 256
        conv_in    = conv_weight.shape[1]      # e.g. 1024 or 1280
        n_channels = conv_in // proj_dim       # 4 or 5
        logging.info(f"Detected n_channels={n_channels} proj_dim={proj_dim} from checkpoint")
    else:
        n_channels = 5
        proj_dim   = 256
        logging.warning("Could not detect channels from checkpoint — assuming 5")

    model = KRAS_MCNN(n_channels=n_channels, proj_dim=proj_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    logging.info(f"Model loaded: {args.model_pt}")

    # ── predict ────────────────────────────────────────────────────────────────
    TENSOR_DIR_BIO = os.path.join(args.tensor_dir, 'test_bio')

    print(f"\n{'Mutation':<8} {'Target':<10} {'Experimental':>13} {'Predicted':>11} {'Exp class':<11} {'Pred class':<11} {'Match'}")
    print("-" * 75)

    all_rows = []
    for entry in results:
        name    = entry['name']
        variant = entry['variant']
        row     = entry['row']

        # load tensor
        tpath = os.path.join(TENSOR_DIR_BIO, variant + '.npy')
        if not os.path.exists(tpath):
            logging.warning(f"  Tensor not found: {tpath}")
            continue

        tensor = torch.from_numpy(np.load(tpath)).unsqueeze(0).to(device)  # (1,5,188,1280)

        with torch.no_grad():
            pred_norm = model(tensor).cpu().numpy()[0]   # (7,)

        # denormalise
        pred_raw = np.array([
            pred_norm[j] * stds[t] + means[t]
            if stds[t] > 1e-6 else pred_norm[j]
            for j, t in enumerate(TARGET_NAMES)
        ])

        for j, t in enumerate(TARGET_NAMES):
            exp_val  = float(row.get(f'ddG_{t}', np.nan))
            pred_val = float(pred_raw[j])
            exp_cls  = classify(exp_val)
            pred_cls = classify(pred_val)
            match    = '✓' if exp_cls == pred_cls else '✗'

            if not np.isnan(exp_val):
                print(f"  {name:<8} {t:<10} {exp_val:>13.3f} {pred_val:>11.3f} "
                      f"{exp_cls:<11} {pred_cls:<11} {match}")

            all_rows.append({
                'Mutation': name, 'Target': t,
                'Exp_ddG':  round(exp_val,  4) if not np.isnan(exp_val) else np.nan,
                'Pred_ddG': round(pred_val, 4),
                'Exp_class':  exp_cls,
                'Pred_class': pred_cls,
                'Class_match': exp_cls == pred_cls,
                'Clinical':   CLINICAL[name],
            })
        print()

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    fpath = os.path.join(args.output, 'results_famous_mutations.csv')
    df.to_csv(fpath, index=False)
    logging.info(f"\nSaved: {fpath}")

    # ── summary ────────────────────────────────────────────────────────────────
    df_obs = df[~df['Exp_ddG'].isna()]
    acc = df_obs['Class_match'].mean()
    print(f"\n{'='*60}")
    print(f"Overall class accuracy: {acc:.1%} ({df_obs['Class_match'].sum()}/{len(df_obs)})")
    print(f"\nPer-target class accuracy:")
    for t in TARGET_NAMES:
        sub = df_obs[df_obs['Target']==t]
        if len(sub) > 0:
            a = sub['Class_match'].mean()
            print(f"  {t:<10}: {a:.1%} ({sub['Class_match'].sum()}/{len(sub)})")


if __name__ == "__main__":
    main()