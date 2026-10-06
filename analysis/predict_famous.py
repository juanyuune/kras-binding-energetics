#!/usr/bin/env python3
"""
Predictions and classification accuracy for 8 canonical oncogenic KRAS mutations.
All positions are in test_curated — held out from training.

Usage:
    python predict_famous.py \
        --tensor_dir tensors/ \
        --master_csv data/kras_master_table.csv \
        --model_pt   results/MCNN_seed1.pt \
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

from dataset import KRASRegressionDataset, compute_target_stats, TARGET_NAMES, N_TARGETS

warnings.filterwarnings("ignore")

DDG_DELTA = 0.25

FAMOUS = {
    "G12D": (12, "G", "D"),
    "G12V": (12, "G", "V"),
    "G12R": (12, "G", "R"),
    "G12C": (12, "G", "C"),
    "G13D": (13, "G", "D"),
    "Q61H": (61, "Q", "H"),
    "Q61L": (61, "Q", "L"),
    "Q61R": (61, "Q", "R"),
}

CLINICAL = {
    "G12D": "impairs GAP hydrolysis; near-neutral effector binding (Fell et al. 2020)",
    "G12V": "strong GTPase impairment; broad effector engagement",
    "G12R": "SOS1 exchange strongly impaired; effector-selective (Golan et al. 2026)",
    "G12C": "covalent inhibitor target (sotorasib, adagrasib); SOS1 partially impaired",
    "G13D": "near-neutral across partners; GEF exchange partially retained",
    "Q61H": "moderate GTPase impairment; near-neutral binding profile",
    "Q61L": "strong activating; SOS1 weakened; conformation-selective",
    "Q61R": "strong activating; SOS1 strongly weakened; K27 conformation shift (Lu et al. 2016)",
}


def classify(v):
    if np.isnan(v):    return "NaN"
    if v < -DDG_DELTA: return "Enhanced"
    if v >  DDG_DELTA: return "Weakened"
    return "Neutral"


class MutationCenteredPool(nn.Module):
    def __init__(self, hw=5):
        super().__init__()
        self.hw = hw

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
        self.projection    = nn.Linear(1280, proj, bias=False)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(proj * n_ch, filters, k, padding=k // 2))
            for k in kernels
        ])
        self.mut_pool = MutationCenteredPool(win)
        self.trunk = nn.Sequential(
            nn.Linear(filters * 3 * len(kernels), 512),
            nn.GELU(), nn.LayerNorm(512), nn.Dropout(drop),
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
        h = self.trunk(torch.cat(outs, 1))
        return torch.stack([head(h).squeeze(1) for head in self.heads], dim=1)


def load_model(pt_path, device):
    state = torch.load(pt_path, map_location=device)
    # infer architecture from checkpoint weight shapes
    proj_dim = state["projection.weight"].shape[0]
    conv_in  = state["conv_branches.0.0.weight"].shape[1]
    n_ch     = conv_in // proj_dim
    print(f"  checkpoint: n_channels={n_ch}, proj_dim={proj_dim}")
    model = KRAS_MCNN(n_ch=n_ch, proj=proj_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir", required=True)
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--model_pt",   required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master = pd.read_csv(args.master_csv)

    train_ds = KRASRegressionDataset(
        split="train", tensor_dir=args.tensor_dir,
        master_csv=args.master_csv, min_observed=1,
    )
    means, stds = compute_target_stats(train_ds)

    print(f"{'[train]':10} {len(train_ds)} variants")
    for t in TARGET_NAMES:
        print(f"  {t:<10} N={train_ds.targets_raw[t].notna().sum():<6} "
              f"mean={means[t]:.4f}  std={stds[t]:.4f}")

    found = []
    for name, (pos, wt, mut) in FAMOUS.items():
        row = master[(master["position"] == pos) & (master["mutant_aa"] == mut)]
        if row.empty:
            print(f"  {name}: not found")
            continue
        row = row.iloc[0]
        print(f"  {name}: variant={row['variant']}  partition={row['partition']}")
        found.append((name, row))

    print(f"\nloading {args.model_pt}")
    model = load_model(args.model_pt, device)

    tensor_dir_bio = os.path.join(args.tensor_dir, "test_bio")
    rows = []

    print(f"\n{'Mutation':<8} {'Target':<10} {'Experimental':>13} "
          f"{'Predicted':>11} {'Exp':>10} {'Pred':>10} {'Match'}")
    print("-" * 70)

    for name, row in found:
        variant = row["variant"]
        tpath   = os.path.join(tensor_dir_bio, variant + ".npy")
        if not os.path.exists(tpath):
            print(f"  tensor missing: {tpath}")
            continue

        Z = torch.from_numpy(np.load(tpath)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(Z).cpu().numpy()[0]

        pred_raw = np.array([
            pred_norm[j] * stds[t] + means[t] if stds[t] > 1e-6 else pred_norm[j]
            for j, t in enumerate(TARGET_NAMES)
        ])

        for j, t in enumerate(TARGET_NAMES):
            exp  = float(row.get(f"ddG_{t}", np.nan))
            pred = float(pred_raw[j])
            ec, pc = classify(exp), classify(pred)
            match  = "✓" if ec == pc else "✗"
            if not np.isnan(exp):
                print(f"  {name:<8} {t:<10} {exp:>13.3f} {pred:>11.3f} "
                      f"{ec:>10} {pc:>10} {match}")
            rows.append({
                "Mutation": name, "Target": t,
                "Exp_ddG": round(exp, 4) if not np.isnan(exp) else np.nan,
                "Pred_ddG": round(pred, 4),
                "Exp_class": ec, "Pred_class": pc,
                "Class_match": ec == pc,
                "Clinical": CLINICAL[name],
            })
        print()

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output, "results_famous_mutations.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    obs = df[df["Exp_ddG"].notna()]
    acc = obs["Class_match"].mean()
    print(f"\nOverall class accuracy: {acc:.1%} "
          f"({obs['Class_match'].sum()}/{len(obs)})")
    print("\nPer-target class accuracy:")
    for t in TARGET_NAMES:
        sub = obs[obs["Target"] == t]
        if len(sub):
            print(f"  {t:<10}: {sub['Class_match'].mean():.1%} "
                  f"({sub['Class_match'].sum()}/{len(sub)})")


if __name__ == "__main__":
    main()
