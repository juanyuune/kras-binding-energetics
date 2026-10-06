#!/usr/bin/env python3
"""
PLM-MCNN training script for KRAS 7-target ΔΔG regression.
Supports all ablations A1-A8 via command-line flags.

Channel ablations (--channels):
    0 1 2 3 4  full model (default)
    1          A1: mutant only
    0 1        A2: WT + mutant, no displacement
    2 3        A3: displacement only
    0 1 2 3    A4: no mask channel

Pooling ablations (--pooling):
    hybrid     full model — global max + global mean + mutation-centred (default)
    global     A5: no local pooling
    local      A6: local pooling only

Kernel ablation (--kernels):
    1 4 8 16   full model (default)
    8          A7: single kernel

Single-task (--single_task N):
    A8: train on one target only (0=fold 1=RAF1 ... 6=K55)

Usage:
    python mcnn.py \
        --tensor_dir tensors/ \
        --master_csv data/kras_master_table.csv \
        --output     results/ \
        --seed 1
"""

import os
import csv
import random
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score,
    precision_recall_fscore_support,
)

from dataset import (
    KRASRegressionDataset, compute_target_stats,
    TARGET_NAMES, N_TARGETS, SPLIT_PARTITIONS, MASK_COLS,
)

warnings.filterwarnings("ignore")

DDG_DELTA = 0.25



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tensor_dir",   required=True)
    p.add_argument("--master_csv",   required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--proj_dim",     type=int,   default=256)
    p.add_argument("--filters",      type=int,   default=256)
    p.add_argument("--kernels",      type=int,   nargs="+", default=[1,4,8,16])
    p.add_argument("--local_win",    type=int,   default=5)
    p.add_argument("--dropout",      type=float, default=0.30)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--patience",     type=int,   default=15)
    p.add_argument("--delta",        type=float, default=1.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save_model",   action="store_true")
    p.add_argument("--single_task",  type=int,   default=None,
                   help="Train on one target: 0=fold 1=RAF1 2=PIK3CG 3=RALGDS 4=SOS1 5=K27 6=K55")
    p.add_argument("--channels",     type=int,   nargs="+", default=[0,1,2,3,4])
    p.add_argument("--pooling",      type=str,   default="hybrid",
                   choices=["hybrid", "global", "local"])
    return p.parse_args()


def model_name(args):
    ch_map = {
        (0,1,2,3,4): "MCNN",
        (1,):        "MCNN_A1_mutant_only",
        (0,1):       "MCNN_A2_WT_Mut",
        (2,3):       "MCNN_A3_diff_only",
        (0,1,2,3):   "MCNN_A4_no_mask",
    }
    name = ch_map.get(tuple(sorted(args.channels)),
                      "MCNN_ch" + "".join(map(str, sorted(args.channels))))
    if args.pooling == "global": name += "_A5_global_pool"
    if args.pooling == "local":  name += "_A6_local_pool"
    if sorted(args.kernels) != [1, 4, 8, 16]:
        name += "_A7_k" + "_".join(map(str, sorted(args.kernels)))
    if args.single_task is not None:
        name = f"MCNN_A8_{TARGET_NAMES[args.single_task]}"
    return name



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
    def __init__(self, channels, pooling, kernels=None, proj=256,
                 filters=256, win=5, drop=0.30, n_out=N_TARGETS):
        super().__init__()
        kernels      = kernels or [1, 4, 8, 16]
        n_ch         = len(channels)
        self.channels = channels
        self.pooling  = pooling
        self.proj     = proj
        self.n_ch     = n_ch
        self.use_mask = 4 in channels

        self.projection    = nn.Linear(1280, proj, bias=False)
        self.conv_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(proj * n_ch, filters, k, padding=k // 2))
            for k in kernels
        ])
        self.mut_pool = MutationCenteredPool(win)

        branch_dim = filters * (3 if pooling == "hybrid" else
                                2 if pooling == "global" else 1)
        self.trunk = nn.Sequential(
            nn.Linear(branch_dim * len(kernels), 512),
            nn.GELU(), nn.LayerNorm(512), nn.Dropout(drop),
            nn.Linear(512, 256),
        )
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(n_out)])

    def forward(self, x_full):
        B, _, L, _ = x_full.shape
        x    = x_full[:, self.channels, :, :]
        mask = x_full[:, 4, :, 0] if self.use_mask else None
        h    = self.projection(x).permute(0, 1, 3, 2).reshape(B, self.n_ch * self.proj, L)

        outs = []
        for conv in self.conv_branches:
            c = F.gelu(conv(h))
            if c.shape[-1] != L:
                c = c[:, :, :L] if c.shape[-1] > L else F.pad(c, (0, L - c.shape[-1]))
            if self.pooling == "hybrid":
                outs.append(torch.cat([c.max(-1).values, c.mean(-1),
                                       self.mut_pool(c, mask)], 1))
            elif self.pooling == "global":
                outs.append(torch.cat([c.max(-1).values, c.mean(-1)], 1))
            else:
                outs.append(self.mut_pool(c, mask))

        h = self.trunk(torch.cat(outs, 1))
        return torch.stack([head(h).squeeze(1) for head in self.heads], dim=1)



def masked_huber(pred, target, mask, active, delta=1.0):
    loss  = F.huber_loss(pred, target, reduction="none", delta=delta)
    tasks = [(loss[:, j] * mask[:, j]).sum() / mask[:, j].sum()
             for j in active if mask[:, j].sum() > 0]
    return torch.stack(tasks).mean() if tasks else pred.sum() * 0


def label(v):
    if np.isnan(v): return -1
    return 0 if v < -DDG_DELTA else 2 if v > DDG_DELTA else 1


def reg_metrics(yt, yp, mask):
    obs = mask == 1
    if obs.sum() < 2:
        return dict(MAE=np.nan, RMSE=np.nan, Pearson_r=np.nan, Spearman_rho=np.nan, N=int(obs.sum()))
    yt, yp = yt[obs], yp[obs]
    pr, _  = pearsonr(yt, yp)
    sr, _  = spearmanr(yt, yp)
    return dict(MAE   =round(float(np.mean(np.abs(yt-yp))), 4),
                RMSE  =round(float(np.sqrt(np.mean((yt-yp)**2))), 4),
                Pearson_r   =round(float(pr), 4),
                Spearman_rho=round(float(sr), 4),
                N=int(obs.sum()))


def cls_metrics(yt, yp, mask):
    obs = mask == 1
    if obs.sum() < 3:
        return dict(MCC=np.nan, Bal_Acc=np.nan, Macro_F1=np.nan)
    lt = np.array([label(v) for v in yt[obs]])
    lp = np.array([label(v) for v in yp[obs]])
    ok = (lt >= 0) & (lp >= 0)
    if ok.sum() < 3:
        return dict(MCC=np.nan, Bal_Acc=np.nan, Macro_F1=np.nan)
    lt, lp = lt[ok], lp[ok]
    _, _, f, _ = precision_recall_fscore_support(lt, lp, labels=[0,1,2],
                                                  average=None, zero_division=0)
    return dict(MCC     =round(float(matthews_corrcoef(lt, lp)), 4),
                Bal_Acc =round(float(balanced_accuracy_score(lt, lp)), 4),
                Macro_F1=round(float(f.mean()), 4))


def denorm(arr, means, stds):
    out = arr.copy()
    for j, t in enumerate(TARGET_NAMES):
        if stds[t] > 1e-6:
            out[:, j] = arr[:, j] * stds[t] + means[t]
    return out


def evaluate(pred, true, mask, name, split, means, stds):
    pr = denorm(pred, means, stds)
    tr = denorm(true, means, stds)
    rows = []
    for j, t in enumerate(TARGET_NAMES):
        r = {"Model": name, "Split": split, "Target": t}
        r.update(reg_metrics(tr[:, j], pr[:, j], mask[:, j]))
        r.update(cls_metrics(tr[:, j], pr[:, j], mask[:, j]))
        rows.append(r)
        print(f"  {t:<10} N={r['N']:<5} r={r['Pearson_r']:.4f}  "
              f"rho={r['Spearman_rho']:.4f}  MCC={r['MCC']:.4f}")
    macro_r = {k: round(float(np.nanmean([r[k] for r in rows])), 4)
               for k in ["MAE", "RMSE", "Pearson_r", "Spearman_rho", "MCC", "Bal_Acc", "Macro_F1"]}
    macro_r["N"] = int(np.nansum([r["N"] for r in rows]))
    macro_row = {"Model": name, "Split": split, "Target": "MACRO", **macro_r}
    rows.append(macro_row)
    print(f"  {'MACRO':<10} N={macro_r['N']:<5} r={macro_r['Pearson_r']:.4f}  "
          f"rho={macro_r['Spearman_rho']:.4f}")
    return rows


def pos_ranking(split_df, pred, true, mask, name, split, means, stds):
    pr = denorm(pred, means, stds)
    tr = denorm(true, means, stds)
    df = split_df.copy().reset_index(drop=True)
    df["pred_mag"] = np.nanmean(np.abs(pr) * mask, 1)
    df["true_mag"] = np.nanmean(np.abs(tr) * mask, 1)
    g  = df.groupby("position")
    pp, tp = g["pred_mag"].mean(), g["true_mag"].mean()
    pos = pp.index.intersection(tp.index)
    if len(pos) < 3:
        return None
    sp, _ = spearmanr(tp[pos], pp[pos])
    kt, _ = kendalltau(tp[pos], pp[pos])
    print(f"  position ranking [{split}]: Spearman={sp:.4f}  Kendall={kt:.4f}  N={len(pos)}")
    return dict(Model=name, Split=split, Target="M_pos_ranking",
                N_positions=len(pos),
                Spearman_Mpos=round(float(sp), 4),
                Kendall_tau  =round(float(kt), 4))



@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds, trues, masks = [], [], []
    for Z, y, m in loader:
        preds.append(model(Z.to(device)).cpu().numpy())
        trues.append(y.numpy())
        masks.append(m.numpy())
    return (np.vstack(preds), np.vstack(trues), np.vstack(masks))


def val_signal(pred, true, mask, means, stds, active):
    pr = denorm(pred, means, stds)
    tr = denorm(true, means, stds)
    rs = []
    for j in active:
        obs = mask[:, j] == 1
        if obs.sum() >= 2:
            r, _ = pearsonr(tr[obs, j], pr[obs, j])
            rs.append(float(r))
    return float(np.mean(rs)) if rs else -1.0


def main():
    args   = parse_args()
    name   = model_name(args)
    active = [args.single_task] if args.single_task is not None else list(range(N_TARGETS))

    os.makedirs(args.output, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{name}  seed={args.seed}  device={device}")

    master   = pd.read_csv(args.master_csv)
    train_ds = KRASRegressionDataset("train", args.tensor_dir, args.master_csv)
    means, stds = compute_target_stats(train_ds)

    datasets = {s: KRASRegressionDataset(s, args.tensor_dir, args.master_csv,
                                          target_means=means, target_stds=stds)
                for s in ["train", "val", "test_random", "test_bio"]}

    kw = dict(batch_size=args.batch, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(datasets["train"], shuffle=True,  **kw)
    val_loader   = DataLoader(datasets["val"],   shuffle=False, **kw)

    model = KRAS_MCNN(
        channels=args.channels, pooling=args.pooling,
        kernels=args.kernels,   proj=args.proj_dim,
        filters=args.filters,   win=args.local_win,
        drop=args.dropout,
    ).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, patience_count, best_state = -999.0, 0, None

    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, len(train_loader)
        for step, (Z, y, m) in enumerate(train_loader, 1):
            Z, y, m = Z.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            loss = masked_huber(model(Z), y, m, active, args.delta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            bar = "█" * int(30 * step/n) + "░" * (30 - int(30 * step/n))
            print(f"\rEpoch {ep:3d}/{args.epochs} [{bar}] {step}/{n} "
                  f"loss={total/step:.4f}", end="", flush=True)
        print()
        sched.step()

        if ep % 5 == 0 or ep == 1:
            vp, vt, vm = run_eval(model, val_loader, device)
            sig = val_signal(vp, vt, vm, means, stds, active)
            print(f"  val Pearson={sig:.4f}")
            if sig > best:
                best        = sig
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 5
                if patience_count >= args.patience:
                    print(f"  early stop at epoch {ep}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    if args.save_model and best_state:
        ckpt = os.path.join(args.output, f"{name}_seed{args.seed}.pt")
        torch.save(best_state, ckpt)
        print(f"saved: {ckpt}")

    # evaluate on all splits
    split_parts = {
        "val":         {"fold_4"},
        "test_random": {"test_random"},
        "test_bio":    {"test_curated"},
    }
    all_rows, pos_rows = [], []

    for sn in ["val", "test_random", "test_bio"]:
        loader = DataLoader(datasets[sn], shuffle=False, **kw)
        pred, true, mask = run_eval(model, loader, device)
        print(f"\n[{sn}]")
        all_rows += evaluate(pred, true, mask, name, sn, means, stds)

        df_split = master[master["partition"].isin(split_parts[sn])].copy()
        df_split = df_split[df_split[MASK_COLS].sum(1) >= 1].reset_index(drop=True)
        pr = pos_ranking(df_split, pred, true, mask, name, sn, means, stds)
        if pr:
            pos_rows.append(pr)

    # append to results CSV
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = dict(channels=str(args.channels), pooling=args.pooling,
                kernels=str(args.kernels), proj_dim=args.proj_dim,
                filters=args.filters, dropout=args.dropout,
                lr=args.lr, batch=args.batch, seed=args.seed,
                best_val_Pearson=round(best, 4), timestamp=ts)
    for r in all_rows:
        r.update(meta)

    keys = ["Model", "Split", "Target", "N",
            "MAE", "RMSE", "Pearson_r", "Spearman_rho",
            "MCC", "Bal_Acc", "Macro_F1",
            "channels", "pooling", "kernels",
            "proj_dim", "filters", "dropout", "lr",
            "batch", "seed", "best_val_Pearson", "timestamp"]

    fpath      = os.path.join(args.output, "results_ablations.csv")
    first_write = not (os.path.exists(fpath) and os.path.getsize(fpath) > 0)
    with open(fpath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if first_write:
            w.writeheader()
        w.writerows(all_rows)
    print(f"\nresults -> {fpath}")

    if pos_rows:
        ppath      = os.path.join(args.output, "results_ablations_position.csv")
        first_pos  = not (os.path.exists(ppath) and os.path.getsize(ppath) > 0)
        pkeys      = ["Model", "Split", "Target", "N_positions",
                      "Spearman_Mpos", "Kendall_tau"]
        with open(ppath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pkeys, extrasaction="ignore")
            if first_pos:
                w.writeheader()
            w.writerows(pos_rows)


if __name__ == "__main__":
    main()
