#!/usr/bin/env python3
"""
PyTorch Dataset for KRAS 7-target ΔΔG regression.

Each sample returns:
    tensor  : FloatTensor (5, 188, 1280)
              ch0=H_WT  ch1=H_Mut  ch2=H_Mut-H_WT  ch3=|H_Mut-H_WT|  ch4=mask
    targets : FloatTensor (7,)  normalised ΔΔG [fold, RAF1, PIK3CG, RALGDS, SOS1, K27, K55]
    masks   : FloatTensor (7,)  1=observed  0=missing

Run as a sanity check:
    python dataset.py \
        --tensor_dir tensors/ \
        --master_csv data/kras_master_table.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


SEQ_LEN    = 188
EMB_DIM    = 1280
N_CHANNELS = 5
N_TARGETS  = 7

TARGET_NAMES = ['fold', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'K27', 'K55']
TARGET_COLS  = [f'ddG_{t}'  for t in TARGET_NAMES]
MASK_COLS    = [f'mask_{t}' for t in TARGET_NAMES]
STD_COLS     = [f'std_{t}'  for t in TARGET_NAMES]

SPLIT_DIRS = {
    'train':       'train',
    'val':         'val',
    'test_random': 'test_random',
    'test_bio':    'test_bio',
}

SPLIT_PARTITIONS = {
    'train':       {'fold_0', 'fold_1', 'fold_2', 'fold_3'},
    'val':         {'fold_4'},
    'test_random': {'test_random'},
    'test_bio':    {'test_curated'},
}


class KRASRegressionDataset(Dataset):
    def __init__(self, split, tensor_dir, master_csv,
                 target_means=None, target_stds=None, min_observed=1):
        if split not in SPLIT_DIRS:
            raise ValueError(f"Unknown split '{split}'. Valid: {list(SPLIT_DIRS.keys())}")

        self.tensor_dir   = os.path.join(tensor_dir, SPLIT_DIRS[split])
        self.target_means = target_means
        self.target_stds  = target_stds

        if not os.path.exists(self.tensor_dir):
            raise FileNotFoundError(
                f"Tensor directory not found: {self.tensor_dir}\n"
                "Run precompute_paired_tensor.py first."
            )

        master = pd.read_csv(master_csv)
        df     = master[master['partition'].isin(SPLIT_PARTITIONS[split])].copy()
        df     = df[df[MASK_COLS].sum(1) >= min_observed].reset_index(drop=True)

        # skip variants whose tensor file is missing
        missing = [v for v in df['variant'] if
                   not os.path.exists(os.path.join(self.tensor_dir, v + '.npy'))]
        if missing:
            print(f'[{split}] {len(missing)} missing tensors — skipping')
            df = df[~df['variant'].isin(missing)].reset_index(drop=True)

        self.df          = df
        self.variants    = df['variant'].tolist()
        self.targets_raw = df[TARGET_COLS].values.astype(np.float32)
        self.masks       = df[MASK_COLS].values.astype(np.float32)

        n_obs = self.masks.sum(1)
        print(f'[{split}] {len(df)} variants '
              f'(partitions: {sorted(SPLIT_PARTITIONS[split])}, min_observed={min_observed})')
        print(f'[{split}] n_observed per variant: '
              f'mean={n_obs.mean():.2f}  min={int(n_obs.min())}  max={int(n_obs.max())}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tensor = torch.from_numpy(
            np.load(os.path.join(self.tensor_dir, self.variants[idx] + '.npy'))
        )
        targets = self.targets_raw[idx].copy()
        masks   = self.masks[idx].copy()

        if self.target_means is not None and self.target_stds is not None:
            for i, t in enumerate(TARGET_NAMES):
                if masks[i] == 1:
                    mu, sig = self.target_means[t], self.target_stds[t]
                    if sig > 1e-6:
                        targets[i] = (targets[i] - mu) / sig

        targets = np.nan_to_num(targets, nan=0.0)
        return tensor, torch.from_numpy(targets), torch.from_numpy(masks)


def compute_target_stats(train_ds):
    """
    Per-target mean and std from observed training values only.
    Must be computed from the training set and passed to all other splits.
    """
    means, stds = {}, {}
    print('\nnormalisation stats (training set):')
    print(f"  {'Target':<10} {'N_obs':>6} {'Mean':>10} {'Std':>10}")
    for i, t in enumerate(TARGET_NAMES):
        obs      = train_ds.targets_raw[train_ds.masks[:, i] == 1, i]
        means[t] = float(np.nanmean(obs)) if len(obs) > 0 else 0.0
        stds[t]  = float(np.nanstd(obs))  if len(obs) > 0 else 1.0
        if stds[t] < 1e-6:
            stds[t] = 1.0
        print(f"  {t:<10} {len(obs):>6} {means[t]:>10.4f} {stds[t]:>10.4f}")
    return means, stds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_dir', required=True)
    parser.add_argument('--master_csv', required=True)
    parser.add_argument('--batch',      type=int, default=32)
    args = parser.parse_args()

    train_ds    = KRASRegressionDataset('train', args.tensor_dir, args.master_csv)
    means, stds = compute_target_stats(train_ds)

    all_ok = True
    for split in ['train', 'val', 'test_random', 'test_bio']:
        try:
            ds = KRASRegressionDataset(split, args.tensor_dir, args.master_csv,
                                       target_means=means, target_stds=stds)
        except FileNotFoundError as e:
            print(f'[{split}] skipped: {e}')
            continue

        Z, y, m = next(iter(DataLoader(ds, batch_size=args.batch,
                                       shuffle=False, num_workers=0)))
        n_nan_Z = torch.isnan(Z).sum().item()
        n_nan_y = torch.isnan(y).sum().item()
        masked_zero = (y[m == 0] == 0).all().item()

        print(f'\n[{split}]')
        print(f'  size={len(ds)}  tensor={tuple(Z.shape)}')
        print(f'  ch0 range: {Z[:,0].min():.3f} / {Z[:,0].max():.3f}')
        print(f'  ch2 range: {Z[:,2].min():.3f} / {Z[:,2].max():.3f}')
        print(f'  ch4 unique: {Z[:,4].unique().tolist()}  (expected [0.0, 1.0])')
        print(f'  obs targets range: {y[m==1].min():.3f} / {y[m==1].max():.3f}')
        print(f'  NaN in Z: {n_nan_Z}  NaN in y: {n_nan_y}  masked=0: {masked_zero}')

        if n_nan_Z > 0 or n_nan_y > 0 or not masked_zero:
            print('  FAIL')
            all_ok = False
        else:
            print('  OK')

    print(f'\n{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
