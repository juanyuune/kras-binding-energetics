#!/usr/bin/env python3
"""
Three baselines for KRAS ddG prediction.

  1. Training-fold mean predictor (trivial lower bound)
  2. Physicochemical features + MLP (17 features: hydrophobicity, volume,
     polarity, charge, mol weight — WT, mut, delta — plus position and
     Grantham distance)
  3. Mean-pooled PLM displacement + ridge regression (channel 2 of tensor)

References:
  Kyte & Doolittle (1982) J Mol Biol 157:105
  Zimmerman et al. (1968) J Theor Biol 21:170
  Grantham (1974) Science 185:862

Usage:
    python baselines.py \
        --tensor_dir tensors/ \
        --master_csv data/kras_master_table.csv \
        --output     results/
"""

import os
import csv
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

from dataset import (
    KRASRegressionDataset, compute_target_stats,
    TARGET_NAMES, N_TARGETS, SPLIT_PARTITIONS, MASK_COLS,
)

warnings.filterwarnings("ignore")


# physicochemical property scales
HYDROPHOBICITY = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
    'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
    'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3,
}
VOLUME = {
    'A':67,'C':86,'D':91,'E':109,'F':135,'G':48,'H':118,'I':124,'K':135,
    'L':124,'M':124,'N':96,'P':90,'Q':114,'R':148,'S':73,'T':93,'V':105,
    'W':163,'Y':141,
}
POLARITY = {
    'A':0.00,'C':1.48,'D':49.70,'E':49.90,'F':0.00,'G':0.00,'H':51.60,
    'I':0.00,'K':49.50,'L':0.00,'M':1.43,'N':3.38,'P':1.58,'Q':3.53,
    'R':52.00,'S':1.67,'T':1.66,'V':0.00,'W':2.10,'Y':1.61,
}
CHARGE = {
    'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,
    'M':0,'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0,
}
MOLWT = {
    'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.0,'H':155.2,
    'I':131.2,'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.2,
    'R':174.2,'S':105.1,'T':119.1,'V':117.1,'W':204.2,'Y':181.2,
}

SCALES = [HYDROPHOBICITY, VOLUME, POLARITY, CHARGE, MOLWT]
N_PROPS = len(SCALES)
N_PHYSCHEM = 17  # 5 WT + 5 mut + 5 delta + position + Grantham


def physchem_features(df):
    X = np.zeros((len(df), N_PHYSCHEM), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        wt, mut, pos = str(row['wt_aa']), str(row['mutant_aa']), int(row['position'])
        for j, sc in enumerate(SCALES):
            X[i, j]            = sc.get(wt,  0.0)
            X[i, j + N_PROPS]  = sc.get(mut, 0.0)
            X[i, j + 2*N_PROPS] = sc.get(mut, 0.0) - sc.get(wt, 0.0)
        X[i, 15] = (pos - 2) / (188 - 2)
        X[i, 16] = float(np.sqrt(
            (VOLUME.get(mut,0)   - VOLUME.get(wt,0))**2 +
            (POLARITY.get(mut,0) - POLARITY.get(wt,0))**2
        ))
    return X


def masked_huber(pred, target, mask, delta=1.0):
    loss = nn.functional.huber_loss(pred, target, reduction='none', delta=delta)
    tasks = []
    for j in range(N_TARGETS):
        n = mask[:, j].sum()
        if n > 0:
            tasks.append((loss[:, j] * mask[:, j]).sum() / n)
    return torch.stack(tasks).mean() if tasks else torch.tensor(0.0)


def metrics(yt, yp, mask):
    obs = mask == 1
    if obs.sum() < 2:
        return dict(MAE=np.nan, RMSE=np.nan, Pearson_r=np.nan, Spearman_rho=np.nan, N=int(obs.sum()))
    yt, yp = yt[obs], yp[obs]
    pr, _  = pearsonr(yt, yp)
    sr, _  = spearmanr(yt, yp)
    return dict(
        MAE=round(float(np.mean(np.abs(yt-yp))), 4),
        RMSE=round(float(np.sqrt(np.mean((yt-yp)**2))), 4),
        Pearson_r=round(float(pr), 4),
        Spearman_rho=round(float(sr), 4),
        N=int(obs.sum()),
    )


def evaluate(true, pred, mask, model, split):
    rows = []
    for j, t in enumerate(TARGET_NAMES):
        m = metrics(true[:, j], pred[:, j], mask[:, j])
        rows.append({'Model': model, 'Split': split, 'Target': t, **m})
    macro = {k: round(float(np.nanmean([r[k] for r in rows])), 4)
             for k in ['MAE', 'RMSE', 'Pearson_r', 'Spearman_rho']}
    macro['N'] = int(np.nansum([r['N'] for r in rows]))
    rows.append({'Model': model, 'Split': split, 'Target': 'MACRO', **macro})
    print(f"  {model} / {split}  macro r={macro['Pearson_r']:.4f}")
    return rows


def get_df(master, split):
    parts = SPLIT_PARTITIONS[split]
    df = master[master['partition'].isin(parts)].copy().reset_index(drop=True)
    return df[df[MASK_COLS].sum(1) >= 1].reset_index(drop=True)


def standardise(Xtr, *rest):
    mu  = Xtr.mean(0, keepdims=True)
    sig = Xtr.std(0,  keepdims=True)
    sig[sig < 1e-6] = 1.0
    return tuple((X - mu) / sig for X in [Xtr, *rest])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_dir', required=True)
    parser.add_argument('--master_csv', required=True)
    parser.add_argument('--output',     required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    master  = pd.read_csv(args.master_csv)
    splits  = ['train', 'val', 'test_random', 'test_bio']
    eval_sp = ['val', 'test_random', 'test_bio']

    train_ds = KRASRegressionDataset(
        split='train', tensor_dir=args.tensor_dir,
        master_csv=args.master_csv, min_observed=1,
    )
    means, stds = compute_target_stats(train_ds)
    dfs = {s: get_df(master, s) for s in splits}

    def yt(s): return dfs[s][[f'ddG_{t}' for t in TARGET_NAMES]].values.astype(np.float32)
    def mt(s): return dfs[s][[f'mask_{t}' for t in TARGET_NAMES]].values.astype(np.float32)

    all_results = []

    # -- baseline 1: training mean predictor --
    print('\nbaseline 1: training mean predictor')
    tr_means = {
        t: float(dfs['train'].loc[dfs['train'][f'mask_{t}']==1, f'ddG_{t}'].mean())
        for t in TARGET_NAMES
    }
    for s in eval_sp:
        n = len(dfs[s])
        pred = np.array([[tr_means[t] for t in TARGET_NAMES]] * n, dtype=np.float32)
        all_results += evaluate(yt(s), pred, mt(s), 'Mean_predictor', s)

    # -- baseline 2: physicochemical MLP --
    print('\nbaseline 2: physicochemical MLP')
    Xph = {s: physchem_features(dfs[s]) for s in splits}
    Xph_tr, Xph_v, Xph_r, Xph_b = standardise(
        Xph['train'], Xph['val'], Xph['test_random'], Xph['test_bio']
    )
    Xph = {'train': Xph_tr, 'val': Xph_v, 'test_random': Xph_r, 'test_bio': Xph_b}

    Ytr_n = yt('train').copy()
    for j, t in enumerate(TARGET_NAMES):
        obs = mt('train')[:, j] == 1
        if stds[t] > 1e-6:
            Ytr_n[obs, j] = (yt('train')[obs, j] - means[t]) / stds[t]
    Ytr_n = np.nan_to_num(Ytr_n, nan=0.0)

    class PhyschemMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(N_PHYSCHEM, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, N_TARGETS),
            )
        def forward(self, x): return self.net(x)

    mlp = PhyschemMLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt  = torch.from_numpy(Xph['train']).to(device)
    Yt  = torch.from_numpy(Ytr_n).to(device)
    Mt  = torch.from_numpy(mt('train')).to(device)

    for ep in range(200):
        mlp.train(); opt.zero_grad()
        masked_huber(mlp(Xt), Yt, Mt).backward()
        opt.step()
        if (ep+1) % 50 == 0:
            with torch.no_grad():
                l = masked_huber(mlp(Xt), Yt, Mt).item()
            print(f'  epoch {ep+1}/200  loss={l:.4f}')

    mlp.eval()
    with torch.no_grad():
        for s in eval_sp:
            pn = mlp(torch.from_numpy(Xph[s]).to(device)).cpu().numpy()
            pr = pn.copy()
            for j, t in enumerate(TARGET_NAMES):
                if stds[t] > 1e-6:
                    pr[:, j] = pn[:, j] * stds[t] + means[t]
            all_results += evaluate(yt(s), pr, mt(s), 'Physchem_MLP', s)

    # -- baseline 3: PLM displacement + ridge --
    print('\nbaseline 3: PLM displacement + ridge regression')
    Xplm = {}
    for s in splits:
        df   = dfs[s]
        tdir = os.path.join(args.tensor_dir, s)
        X    = np.zeros((len(df), 1280), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            tp = os.path.join(tdir, row['variant'] + '.npy')
            if os.path.exists(tp):
                X[i] = np.load(tp)[2].mean(0)  # channel 2: signed displacement, mean over L
        Xplm[s] = X
        print(f'  [{s}] {X.shape}')

    Xp_tr, Xp_v, Xp_r, Xp_b = standardise(
        Xplm['train'], Xplm['val'], Xplm['test_random'], Xplm['test_bio']
    )
    Xplm = {'train': Xp_tr, 'val': Xp_v, 'test_random': Xp_r, 'test_bio': Xp_b}

    ridge_preds = {s: np.full((len(dfs[s]), N_TARGETS), np.nan) for s in eval_sp}
    for j, t in enumerate(TARGET_NAMES):
        obs = mt('train')[:, j] == 1
        if obs.sum() < 10:
            continue
        r = Ridge(alpha=1.0)
        r.fit(Xplm['train'][obs], yt('train')[obs, j])
        for s in eval_sp:
            ridge_preds[s][:, j] = r.predict(Xplm[s])

    for s in eval_sp:
        all_results += evaluate(yt(s), ridge_preds[s], mt(s), 'PLM_Ridge', s)

    # save
    out_path = os.path.join(args.output, 'results_baselines.csv')
    keys = ['Model', 'Split', 'Target', 'N', 'MAE', 'RMSE', 'Pearson_r', 'Spearman_rho']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(all_results)
    print(f'\nsaved: {out_path}')


if __name__ == '__main__':
    main()
