"""
predict_all_testbio.py
======================
Generate per-variant predictions for ALL test_bio variants across
three primary model seeds, then produce the predicted vs experimental
scatter plot (Figure 10) for the manuscript.

Usage:
  python predict_all_testbio.py \
    --tensor_dir /srv/jupyterlab/workspace/KRAS/tensors/ \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --model_pts  /srv/jupyterlab/workspace/KRAS/code/results/MCNN_seed1.pt \
                 /srv/jupyterlab/workspace/KRAS/code/results/MCNN_seed2.pt \
                 /srv/jupyterlab/workspace/KRAS/code/results/MCNN_seed3.pt \
    --output     /srv/jupyterlab/workspace/KRAS/code/results/
"""

import os, argparse, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

from dataset import (
    KRASRegressionDataset, compute_target_stats,
    TARGET_NAMES, N_TARGETS, MASK_COLS,
)
from dataset import KRASRegressionDataset

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--tensor_dir', required=True)
parser.add_argument('--master_csv', required=True)
parser.add_argument('--model_pts',  required=True, nargs='+',
                    help='One or more .pt checkpoint files (seeds)')
parser.add_argument('--output',     required=True)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")

# ── normalisation stats from training ────────────────────────────────────────
master = pd.read_csv(args.master_csv)
train_ds = KRASRegressionDataset(
    split='train',
    tensor_dir=args.tensor_dir,
    master_csv=args.master_csv,
    min_observed=1,
)
means, stds = compute_target_stats(train_ds)

# ── dataset ───────────────────────────────────────────────────────────────────
ds_bio = KRASRegressionDataset(
    split='test_bio',
    tensor_dir=args.tensor_dir,
    master_csv=args.master_csv,
    target_means=means,
    target_stds=stds,
    min_observed=1,
)
loader_bio = DataLoader(ds_bio, batch_size=32, shuffle=False,
                        num_workers=2, pin_memory=True)
logging.info(f"test_bio variants: {len(ds_bio)}")

# ── model (same architecture as primary MCNN) ─────────────────────────────────
class MutationCenteredPool(nn.Module):
    def __init__(self, half_win=5):
        super().__init__()
        self.half_win = half_win
    def forward(self, conv_out, mask_ch):
        B, F, L = conv_out.shape
        out = torch.zeros(B, F, device=conv_out.device)
        if mask_ch is None:
            pos = L // 2
            lo = max(0, pos - self.half_win)
            hi = min(L, pos + self.half_win + 1)
            return conv_out[:, :, lo:hi].mean(dim=-1)
        for b in range(B):
            pos = mask_ch[b].argmax().item()
            lo = max(0, pos - self.half_win)
            hi = min(L, pos + self.half_win + 1)
            out[b] = conv_out[b, :, lo:hi].mean(dim=-1)
        return out

class KRAS_MCNN(nn.Module):
    """Architecture matched exactly to checkpoint key shapes."""
    def __init__(self, n_channels=5, seq_len=188, emb_dim=1280,
                 proj_dim=256, kernels=None, n_filters=256,
                 local_win=5, dropout=0.30, pooling='hybrid',
                 n_targets=N_TARGETS):
        super().__init__()
        if kernels is None: kernels = [1,4,8,16]
        self.kernels = kernels; self.pooling = pooling
        self.n_ch = n_channels; self.proj_dim = proj_dim
        # projection: (1280 -> 256) applied per channel per position
        self.projection = nn.Linear(emb_dim, proj_dim, bias=False)
        # conv input = proj_dim * n_channels = 1280 (matches checkpoint)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(proj_dim * n_channels, n_filters, k, padding=k//2))
            for k in kernels
        ])
        # trunk matches: 0=Linear(3072->512) 2=LayerNorm(512) 4=Linear(512->256)
        # total pooled dim = n_filters*3 * 4 kernels = 256*3*4 = 3072
        self.mut_pool = MutationCenteredPool(local_win)
        self.trunk = nn.Sequential(
            nn.Linear(n_filters * 3 * len(kernels), 512),  # 0
            nn.GELU(),                                       # 1
            nn.LayerNorm(512),                               # 2
            nn.Dropout(dropout),                             # 3
            nn.Linear(512, 256),                             # 4
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(n_targets)])

    def forward(self, x):
        B, C, L, D = x.shape
        # project each channel: (B, C, L, 1280) -> (B, C, L, 256)
        h = self.projection(x)
        # reshape to (B, C*256, L) for conv
        h = h.permute(0, 1, 3, 2).reshape(B, C * self.proj_dim, L)
        # mutation mask from channel 4
        mask_ch = x[:, 4, :, 0] if self.n_ch == 5 else None
        branch_out = []
        for conv in self.conv_branches:
            c = F.gelu(conv(h))
            # ensure length matches L
            if c.shape[-1] != L:
                c = c[:, :, :L] if c.shape[-1] > L else F.pad(c, (0, L - c.shape[-1]))
            g_max  = c.max(dim=-1).values
            g_mean = c.mean(dim=-1)
            m      = self.mut_pool(c, mask_ch)
            branch_out.append(torch.cat([g_max, g_mean, m], dim=1))
        feat = torch.cat(branch_out, dim=1)
        h = self.trunk(feat)
        return torch.stack([head(h).squeeze(1) for head in self.heads], dim=1)

# ── load checkpoints and predict ─────────────────────────────────────────────
all_preds = []
for pt_path in args.model_pts:
    logging.info(f"Loading: {pt_path}")
    state = torch.load(pt_path, map_location=device)
    model = KRAS_MCNN(n_channels=5, proj_dim=256, n_filters=256,
                      kernels=[1,4,8,16], pooling='hybrid', local_win=5,
                      dropout=0.30, n_targets=N_TARGETS)
    model.load_state_dict(state)
    model.to(device).eval()

    preds_list, true_list, mask_list = [], [], []
    with torch.no_grad():
        for batch in loader_bio:
            Z, y, m = batch
            Z = Z.to(device)
            out = model(Z)
            preds_list.append(out.cpu().numpy())
            true_list.append(y.numpy())
            mask_list.append(m.numpy())

    preds = np.vstack(preds_list)
    true  = np.vstack(true_list)
    masks = np.vstack(mask_list)
    all_preds.append(preds)

# 3-seed mean
mean_preds = np.mean(all_preds, axis=0)
true_arr   = true
mask_arr   = masks

# denormalise
pred_raw = mean_preds.copy()
true_raw = true_arr.copy()
for j, t in enumerate(TARGET_NAMES):
    sig = stds[t]; mu = means[t]
    if sig > 1e-6:
        pred_raw[:, j] = mean_preds[:, j] * sig + mu
        true_raw[:, j] = true_arr[:, j] * sig + mu

logging.info("Predictions computed. Generating scatter plot...")

# ── scatter plot ──────────────────────────────────────────────────────────────
TARGET_COLORS = {
    'fold':   '#1f77b4',
    'RAF1':   '#ff7f0e',
    'PIK3CG': '#2ca02c',
    'RALGDS': '#d62728',
    'SOS1':   '#9467bd',
    'K27':    '#8c564b',
    'K55':    '#e377c2',
}
TARGET_LABELS = {
    'fold': '\u0394\u0394G_fold',
    'RAF1': '\u0394\u0394G_RAF1',
    'PIK3CG': '\u0394\u0394G_PIK3CG',
    'RALGDS': '\u0394\u0394G_RALGDS',
    'SOS1': '\u0394\u0394G_SOS1',
    'K27': '\u0394\u0394G_K27',
    'K55': '\u0394\u0394G_K55',
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    'Predicted vs Experimental \u0394\u0394G on Biological Challenge Set (test_bio)\n'
    'Primary PLM-MCNN (three-seed mean)',
    fontsize=13, fontweight='bold', y=1.01
)

axes_flat = axes.flatten()
all_r_values = []

for j, t in enumerate(TARGET_NAMES):
    ax = axes_flat[j]
    obs = mask_arr[:, j] == 1
    if obs.sum() < 2:
        ax.set_visible(False)
        continue

    yt = true_raw[obs, j]
    yp = pred_raw[obs, j]
    r, _ = pearsonr(yt, yp)
    sr, _ = spearmanr(yt, yp)
    all_r_values.append(r)

    color = TARGET_COLORS.get(t, '#333333')
    ax.scatter(yt, yp, alpha=0.4, s=18, color=color, edgecolors='none')

    # identity line
    lim_min = min(yt.min(), yp.min()) - 0.1
    lim_max = max(yt.max(), yp.max()) + 0.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', linewidth=0.8, alpha=0.5)
    ax.axhline(0, color='gray', linewidth=0.4, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.4, alpha=0.4)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Experimental \u0394\u0394G (kcal/mol)', fontsize=9)
    ax.set_ylabel('Predicted \u0394\u0394G (kcal/mol)', fontsize=9)
    ax.set_title(TARGET_LABELS[t], fontsize=11, fontweight='bold')
    ax.annotate(f'r = {r:.3f}\n\u03c1 = {sr:.3f}\nN = {obs.sum()}',
                xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # K27 highlight
    if t == 'K27':
        ax.set_facecolor('#fff5f5')
        ax.annotate('Epistemic\nlimit', xy=(0.05, 0.72),
                    xycoords='axes fraction', fontsize=8,
                    color='#cc0000', style='italic')

# hide last unused panel
axes_flat[7].set_visible(False)

plt.tight_layout()
out_path = os.path.join(args.output, 'fig_pred_vs_exp_testbio.png')
plt.savefig(out_path, dpi=180, bbox_inches='tight')
plt.close()
logging.info(f"Scatter plot saved: {out_path}")

# ── save CSV ──────────────────────────────────────────────────────────────────
rows = []
for j, t in enumerate(TARGET_NAMES):
    obs = mask_arr[:, j] == 1
    if obs.sum() < 2: continue
    yt = true_raw[obs, j]; yp = pred_raw[obs, j]
    r, _ = pearsonr(yt, yp)
    sr, _ = spearmanr(yt, yp)
    mae = float(np.mean(np.abs(yt - yp)))
    rows.append({'Target': t, 'N': int(obs.sum()),
                 'Pearson_r': round(r, 4),
                 'Spearman_rho': round(sr, 4),
                 'MAE': round(mae, 4)})
df_out = pd.DataFrame(rows)
csv_path = os.path.join(args.output, 'results_pred_vs_exp_testbio.csv')
df_out.to_csv(csv_path, index=False)
logging.info(f"\nPer-target summary:")
logging.info(df_out.to_string(index=False))
logging.info(f"\nCSV saved: {csv_path}")
print(f"\nDone. Figure: {out_path}")