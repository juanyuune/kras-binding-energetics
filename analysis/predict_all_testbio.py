#!/usr/bin/env python3
"""
Scatter plot of predicted vs experimental ΔΔG on test_bio.
Averages across seeds before plotting.

Usage:
    python predict_all_testbio.py \
        --tensor_dir tensors/ \
        --master_csv data/kras_master_table.csv \
        --model_pts  results/MCNN_seed1.pt results/MCNN_seed2.pt \
        --output     results/
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from dataset import KRASRegressionDataset, compute_target_stats, TARGET_NAMES, N_TARGETS

warnings.filterwarnings("ignore")



class MutationCenteredPool(nn.Module):
    def __init__(self, half_win=5):
        super().__init__()
        self.hw = half_win

    def forward(self, x, mask):
        B, F, L = x.shape
        out = torch.zeros(B, F, device=x.device)
        for b in range(B):
            pos = mask[b].argmax().item() if mask is not None else L // 2
            lo, hi = max(0, pos - self.hw), min(L, pos + self.hw + 1)
            out[b] = x[b, :, lo:hi].mean(-1)
        return out


class KRAS_MCNN(nn.Module):
    def __init__(self, n_ch=5, proj=256, filters=256,
                 kernels=None, win=5, drop=0.30, n_out=N_TARGETS):
        super().__init__()
        kernels = kernels or [1, 4, 8, 16]
        self.n_ch, self.proj = n_ch, proj
        self.projection   = nn.Linear(1280, proj, bias=False)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(proj * n_ch, filters, k, padding=k // 2))
            for k in kernels
        ])
        self.mut_pool = MutationCenteredPool(win)
        self.trunk = nn.Sequential(
            nn.Linear(filters * 3 * len(kernels), 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(drop),
            nn.Linear(512, 256),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(n_out)])

    def forward(self, x):
        B, C, L, _ = x.shape
        h    = self.projection(x).permute(0, 1, 3, 2).reshape(B, C * self.proj, L)
        mask = x[:, 4, :, 0] if self.n_ch == 5 else None
        outs = []
        for conv in self.conv_branches:
            c = F.gelu(conv(h))
            if c.shape[-1] != L:
                c = c[:, :, :L] if c.shape[-1] > L else F.pad(c, (0, L - c.shape[-1]))
            outs.append(torch.cat([c.max(-1).values, c.mean(-1),
                                   self.mut_pool(c, mask)], dim=1))
        h = self.trunk(torch.cat(outs, dim=1))
        return torch.stack([head(h).squeeze(1) for head in self.heads], dim=1)



COLORS = {
    "fold": "#1f77b4", "RAF1": "#ff7f0e", "PIK3CG": "#2ca02c",
    "RALGDS": "#d62728", "SOS1": "#9467bd", "K27": "#8c564b", "K55": "#e377c2",
}
LABELS = {t: f"ΔΔG_{t}" for t in TARGET_NAMES}


def scatter_plot(true_raw, pred_raw, mask_arr, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Predicted vs Experimental ΔΔG — test_bio (three-seed mean)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for j, t in enumerate(TARGET_NAMES):
        ax  = axes.flat[j]
        obs = mask_arr[:, j] == 1
        if obs.sum() < 2:
            ax.set_visible(False)
            continue

        yt, yp = true_raw[obs, j], pred_raw[obs, j]
        r,  _  = pearsonr(yt, yp)
        sr, _  = spearmanr(yt, yp)

        ax.scatter(yt, yp, alpha=0.4, s=18, color=COLORS[t], edgecolors="none")

        lo = min(yt.min(), yp.min()) - 0.1
        hi = max(yt.max(), yp.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

        ax.set_xlabel("Experimental ΔΔG (kcal/mol)", fontsize=9)
        ax.set_ylabel("Predicted ΔΔG (kcal/mol)", fontsize=9)
        ax.set_title(LABELS[t], fontsize=11, fontweight="bold")
        ax.annotate(f"r = {r:.3f}\nρ = {sr:.3f}\nN = {obs.sum()}",
                    xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        if t == "K27":
            ax.set_facecolor("#fff5f5")
            ax.annotate("Epistemic\nlimit", xy=(0.05, 0.72),
                        xycoords="axes fraction", fontsize=8,
                        color="#cc0000", style="italic")

    axes.flat[7].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir", required=True)
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--model_pts",  required=True, nargs="+")
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = KRASRegressionDataset(
        split="train", tensor_dir=args.tensor_dir,
        master_csv=args.master_csv, min_observed=1,
    )
    means, stds = compute_target_stats(train_ds)

    test_ds = KRASRegressionDataset(
        split="test_bio", tensor_dir=args.tensor_dir,
        master_csv=args.master_csv, target_means=means,
        target_stds=stds, min_observed=1,
    )
    loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"test_bio: {len(test_ds)} variants")

    all_preds = []
    for pt in args.model_pts:
        print(f"loading {pt}")
        model = KRAS_MCNN()
        model.load_state_dict(torch.load(pt, map_location=device))
        model.to(device).eval()

        preds, trues, masks = [], [], []
        with torch.no_grad():
            for Z, y, m in loader:
                preds.append(model(Z.to(device)).cpu().numpy())
                trues.append(y.numpy())
                masks.append(m.numpy())

        all_preds.append(np.vstack(preds))
        true_arr = np.vstack(trues)
        mask_arr = np.vstack(masks)

    mean_preds = np.mean(all_preds, axis=0)

    # denormalise
    pred_raw = mean_preds.copy()
    true_raw = true_arr.copy()
    for j, t in enumerate(TARGET_NAMES):
        if stds[t] > 1e-6:
            pred_raw[:, j] = mean_preds[:, j] * stds[t] + means[t]
            true_raw[:, j] = true_arr[:, j]  * stds[t] + means[t]

    fig_path = os.path.join(args.output, "fig_pred_vs_exp_testbio.png")
    scatter_plot(true_raw, pred_raw, mask_arr, fig_path)
    print(f"figure saved: {fig_path}")

    rows = []
    for j, t in enumerate(TARGET_NAMES):
        obs = mask_arr[:, j] == 1
        if obs.sum() < 2:
            continue
        yt, yp = true_raw[obs, j], pred_raw[obs, j]
        r,  _  = pearsonr(yt, yp)
        sr, _  = spearmanr(yt, yp)
        rows.append({
            "Target": t, "N": int(obs.sum()),
            "Pearson_r": round(r, 4),
            "Spearman_rho": round(sr, 4),
            "MAE": round(float(np.mean(np.abs(yt - yp))), 4),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, "results_pred_vs_exp_testbio.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
