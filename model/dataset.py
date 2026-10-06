"""
dataset.py  —  Phase 2
=======================
PyTorch Dataset for KRAS PLM-MCNN regression.

Each example:
  tensor  : FloatTensor (5, 188, 1280)
             ch0=E_WT  ch1=E_Mut  ch2=Mut-WT  ch3=|Mut-WT|  ch4=mask
  targets : FloatTensor (7,)  ddG values [fold,RAF1,PIK3CG,RALGDS,SOS1,K27,K55]
  masks   : FloatTensor (7,)  1=observed  0=missing

Run:
  python dataset.py \
    --tensor_dir /srv/jupyterlab/workspace/KRAS/tensors/ \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --batch      32
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── constants ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 188
EMB_DIM    = 1280
N_CHANNELS = 5
N_TARGETS  = 7

TARGET_NAMES = ['fold', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'K27', 'K55']
TARGET_COLS  = [f'ddG_{t}'  for t in TARGET_NAMES]
STD_COLS     = [f'std_{t}'  for t in TARGET_NAMES]
MASK_COLS    = [f'mask_{t}' for t in TARGET_NAMES]

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


# ── Dataset ───────────────────────────────────────────────────────────────────
class KRASRegressionDataset(Dataset):
    """
    PyTorch Dataset for KRAS 7-target regression.

    Parameters
    ----------
    split        : 'train' | 'val' | 'test_random' | 'test_bio'
    tensor_dir   : base dir containing split subfolders with .npy tensors
    master_csv   : path to kras_master_table.csv
    target_means : dict {target_name: mean}  for normalisation (optional)
    target_stds  : dict {target_name: std}   for normalisation (optional)
    min_observed : minimum observed targets to include a variant (default 1)
    """

    def __init__(
        self,
        split:        str,
        tensor_dir:   str,
        master_csv:   str,
        target_means: dict = None,
        target_stds:  dict = None,
        min_observed: int  = 1,
    ):
        if split not in SPLIT_DIRS:
            raise ValueError(
                f"Unknown split '{split}'. Valid: {list(SPLIT_DIRS.keys())}"
            )

        self.split        = split
        self.tensor_dir   = os.path.join(tensor_dir, SPLIT_DIRS[split])
        self.target_means = target_means
        self.target_stds  = target_stds
        self.min_observed = min_observed

        if not os.path.exists(self.tensor_dir):
            raise FileNotFoundError(
                f"Tensor directory not found: {self.tensor_dir}\n"
                f"Run precompute_paired_tensor.py first."
            )

        # load and filter master table
        master     = pd.read_csv(master_csv)
        valid_parts = SPLIT_PARTITIONS[split]
        split_df   = master[master['partition'].isin(valid_parts)].copy()
        split_df   = split_df.reset_index(drop=True)

        # filter by minimum observed targets
        mask_sum = split_df[[f'mask_{t}' for t in TARGET_NAMES]].sum(axis=1)
        split_df = split_df[mask_sum >= min_observed].reset_index(drop=True)

        logging.info(
            f"[{split}] {len(split_df)} variants "
            f"(partitions: {sorted(valid_parts)}, "
            f"min_observed={min_observed})"
        )

        # verify tensor files exist — skip missing
        missing = [
            row['variant']
            for _, row in split_df.iterrows()
            if not os.path.exists(
                os.path.join(self.tensor_dir, row['variant'] + '.npy')
            )
        ]
        if missing:
            logging.warning(
                f"[{split}] {len(missing)} tensor files not found — skipping"
            )
            split_df = split_df[
                ~split_df['variant'].isin(missing)
            ].reset_index(drop=True)

        self.df = split_df

        # pre-load target arrays
        self.targets_raw = split_df[TARGET_COLS].values.astype(np.float32)
        self.stds_raw    = split_df[STD_COLS].values.astype(np.float32)
        self.masks       = split_df[MASK_COLS].values.astype(np.float32)
        self.variants    = split_df['variant'].tolist()

        n_obs = self.masks.sum(axis=1)
        logging.info(
            f"[{split}] n_observed per variant: "
            f"mean={n_obs.mean():.2f}  "
            f"min={int(n_obs.min())}  "
            f"max={int(n_obs.max())}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        variant = self.variants[idx]

        # load 5-channel tensor — (5, 188, 1280)
        tensor = np.load(
            os.path.join(self.tensor_dir, variant + '.npy')
        )
        tensor = torch.from_numpy(tensor)   # FloatTensor (5, 188, 1280)

        # targets and masks
        targets_raw = self.targets_raw[idx].copy()   # (7,)
        masks       = self.masks[idx].copy()          # (7,)

        # apply per-target normalisation if provided
        if self.target_means is not None and self.target_stds is not None:
            for i, t in enumerate(TARGET_NAMES):
                if masks[i] == 1:
                    mu  = self.target_means.get(t, 0.0)
                    sig = self.target_stds.get(t, 1.0)
                    if sig > 1e-6:
                        targets_raw[i] = (targets_raw[i] - mu) / sig

        # replace NaN with 0.0 — mask ensures excluded from loss
        targets_raw = np.nan_to_num(targets_raw, nan=0.0)

        targets = torch.from_numpy(targets_raw)   # FloatTensor (7,)
        masks   = torch.from_numpy(masks)          # FloatTensor (7,)

        return tensor, targets, masks

    def get_variant_info(self, idx):
        """Return metadata for one variant."""
        row = self.df.iloc[idx]
        return {
            'variant':    row['variant'],
            'position':   int(row['position']),
            'wt_aa':      row['wt_aa'],
            'mutant_aa':  row['mutant_aa'],
            'partition':  row['partition'],
            'n_observed': int(self.masks[idx].sum()),
        }


# ── normalisation statistics ──────────────────────────────────────────────────
def compute_target_stats(train_dataset: KRASRegressionDataset):
    """
    Compute per-target mean and std from OBSERVED training values only.
    Returns means and stds dicts — pass to all other splits.

    Must be computed from training set only to prevent test leakage.
    """
    means = {}
    stds  = {}

    logging.info("Computing normalisation statistics from training set:")
    logging.info(f"  {'Target':<10} {'N_obs':>6} {'Mean':>10} {'Std':>10}")

    for i, t in enumerate(TARGET_NAMES):
        observed = train_dataset.targets_raw[
            train_dataset.masks[:, i] == 1, i
        ]
        n_obs    = len(observed)
        means[t] = float(np.nanmean(observed)) if n_obs > 0 else 0.0
        stds[t]  = float(np.nanstd(observed))  if n_obs > 0 else 1.0
        if stds[t] < 1e-6:
            stds[t] = 1.0

        logging.info(
            f"  {t:<10} {n_obs:>6} {means[t]:>10.4f} {stds[t]:>10.4f}"
        )

    return means, stds


# ── sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir", required=True)
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--batch",      type=int, default=32)
    args = parser.parse_args()

    print("=" * 65)
    print("KRAS Regression Dataset — Sanity Check")
    print("=" * 65)

    # build train first to get normalisation stats
    train_ds = KRASRegressionDataset(
        split      = "train",
        tensor_dir = args.tensor_dir,
        master_csv = args.master_csv,
    )

    means, stds = compute_target_stats(train_ds)

    # build all splits with normalisation
    print()
    all_ok = True
    for split_name in ["train", "val", "test_random", "test_bio"]:
        try:
            ds = KRASRegressionDataset(
                split        = split_name,
                tensor_dir   = args.tensor_dir,
                master_csv   = args.master_csv,
                target_means = means,
                target_stds  = stds,
            )
        except FileNotFoundError as e:
            print(f"[{split_name}] SKIPPED: {e}")
            continue

        loader = DataLoader(
            ds, batch_size=args.batch,
            shuffle=False, num_workers=0
        )
        tensor, targets, masks = next(iter(loader))

        n_obs = masks.sum(dim=1)
        n_nan_tensor  = torch.isnan(tensor).sum().item()
        n_nan_targets = torch.isnan(targets).sum().item()
        masked_out    = targets[masks == 0]
        all_zero      = (masked_out == 0).all().item() if len(masked_out) > 0 else True
        ch4_vals      = tensor[:, 4].unique().tolist()

        # compute expected size dynamically from master table
        # so this check is always correct regardless of min_observed threshold
        master_check  = pd.read_csv(args.master_csv)
        valid_parts   = SPLIT_PARTITIONS[split_name]
        split_check   = master_check[master_check['partition'].isin(valid_parts)]
        mask_sum_chk  = split_check[[f'mask_{t}' for t in TARGET_NAMES]].sum(axis=1)
        total_in_part = len(split_check)
        after_filter  = int((mask_sum_chk >= 1).sum())
        n_excluded    = total_in_part - after_filter
        exp_size      = {split_name: after_filter}
        size_ok  = len(ds) == exp_size[split_name]

        print(f"[{split_name}]")
        print(f"  Total in partition : {total_in_part}")
        print(f"  Zero-observed excl : {n_excluded}  (no labels — correctly excluded)")
        print(f"  Dataset size       : {len(ds)}  "
              f"(expected {exp_size[split_name]}): "
              f"{'OK' if size_ok else 'MISMATCH'}")
        print(f"  tensor shape      : {tuple(tensor.shape)}")
        print(f"  targets shape     : {tuple(targets.shape)}")
        print(f"  masks shape       : {tuple(masks.shape)}")
        print(f"  ch0 range (E_WT)  : "
              f"{tensor[:,0].min():.4f} / {tensor[:,0].max():.4f}")
        print(f"  ch2 range (disp)  : "
              f"{tensor[:,2].min():.4f} / {tensor[:,2].max():.4f}")
        print(f"  ch4 unique (mask) : {ch4_vals}  (expected [0.0, 1.0])")
        print(f"  targets (obs only): "
              f"{targets[masks==1].min():.4f} / {targets[masks==1].max():.4f}")
        print(f"  mask obs/batch    : "
              f"mean={n_obs.float().mean():.2f}  "
              f"min={n_obs.min().item()}  max={n_obs.max().item()}")
        print(f"  NaN in tensor     : {n_nan_tensor}  (expected 0)")
        print(f"  NaN in targets    : {n_nan_targets}  (expected 0)")
        print(f"  Missing targets=0 : {all_zero}  (expected True)")

        if not size_ok:
            all_ok = False
        if n_nan_tensor > 0 or n_nan_targets > 0 or not all_zero:
            all_ok = False
        print()

    print("--- Normalisation statistics ---")
    print(f"  {'Target':<10} {'Mean':>10} {'Std':>10}")
    for t in TARGET_NAMES:
        print(f"  {t:<10} {means[t]:>10.4f} {stds[t]:>10.4f}")

    print(f"\n{'='*65}")
    print(f"{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"{'='*65}")