#!/usr/bin/env python3
"""
Build 5-channel paired perturbation tensors from ESM-2 embeddings.

For each variant, loads WT and mutant ESM-2 embeddings and saves:
    ch0: H_WT              wild-type embedding       (188, 1280)
    ch1: H_Mut             mutant embedding          (188, 1280)
    ch2: H_Mut - H_WT      signed displacement       (188, 1280)
    ch3: |H_Mut - H_WT|    absolute displacement     (188, 1280)
    ch4: M                 mutation mask, 1 at pos k (188, 1280)

Output: variant_name.npy  shape (5, 188, 1280)  float32  ~4.6 MB each

Usage:
    python precompute_paired_tensor.py \
        --emb_base   emb/ \
        --master_csv data/kras_master_table.csv \
        --out_dir    tensors/
"""

import os
import pickle
import argparse

import numpy as np
import pandas as pd


SEQ_LEN    = 188
EMB_DIM    = 1280
N_CHANNELS = 5

SPLIT_MAP = {
    'fold_0':       'train',
    'fold_1':       'train',
    'fold_2':       'train',
    'fold_3':       'train',
    'fold_4':       'val',
    'test_random':  'test_random',
    'test_curated': 'test_bio',
}


def load_esm2(path):
    """Load one ESM-2 .esm2 file, return (SEQ_LEN, EMB_DIM) float32."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = next(iter(data.values()))
    data = np.array(data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1: data = data[0]
    if data.ndim == 3 and data.shape[1] == 1: data = data[:, 0, :]
    # pad or truncate to SEQ_LEN
    out = np.zeros((SEQ_LEN, data.shape[1]), dtype=np.float32)
    n   = min(data.shape[0], SEQ_LEN)
    out[:n] = data[:n]
    return out


def build_tensor(e_wt, e_mut, position):
    diff = e_mut - e_wt
    mask = np.zeros_like(e_wt)
    idx  = int(position) - 1  # 1-indexed to 0-indexed
    if 0 <= idx < SEQ_LEN:
        mask[idx] = 1.0
    return np.stack([e_wt, e_mut, diff, np.abs(diff), mask], axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_base',   required=True)
    parser.add_argument('--master_csv', required=True)
    parser.add_argument('--out_dir',    required=True)
    args = parser.parse_args()

    master = pd.read_csv(args.master_csv)
    print(f'{len(master)} variants')

    for d in set(SPLIT_MAP.values()):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    # WT embedding is shared across all variants — load once
    wt_path = os.path.join(args.emb_base, 'WT_KRAS.esm2')
    if not os.path.exists(wt_path):
        raise FileNotFoundError(f'WT embedding not found: {wt_path}')
    e_wt = load_esm2(wt_path)
    print(f'WT loaded: {e_wt.shape}')
    np.save(os.path.join(args.out_dir, 'WT_KRAS.npy'), e_wt)

    total_ok = total_fail = 0

    for split_dir in set(SPLIT_MAP.values()):
        parts    = [k for k, v in SPLIT_MAP.items() if v == split_dir]
        variants = master[master['partition'].isin(parts)]
        emb_dir  = os.path.join(args.emb_base, split_dir)
        out_dir  = os.path.join(args.out_dir,  split_dir)

        if not os.path.exists(emb_dir):
            print(f'[{split_dir}] emb dir missing — skipping')
            continue

        print(f'[{split_dir}] {len(variants)} variants...')
        ok = fail = 0

        for _, row in variants.iterrows():
            src = os.path.join(emb_dir, row['variant'] + '.esm2')
            dst = os.path.join(out_dir, row['variant'] + '.npy')
            if not os.path.exists(src):
                fail += 1
                continue
            try:
                tensor = build_tensor(e_wt, load_esm2(src), row['position'])
                np.save(dst, tensor)
                ok += 1
            except Exception as e:
                print(f'  error {row["variant"]}: {e}')
                fail += 1

        print(f'  written={ok}  failed={fail}')
        total_ok   += ok
        total_fail += fail

    print(f'\ntotal: {total_ok} written  {total_fail} failed')

    # spot check
    train_dir = os.path.join(args.out_dir, 'train')
    files = [f for f in os.listdir(train_dir) if f.endswith('.npy')]
    if files:
        arr = np.load(os.path.join(train_dir, files[0]))
        print(f'\nspot check {files[0]}:')
        print(f'  shape={arr.shape}  dtype={arr.dtype}')
        print(f'  |ch2|==ch3: {np.allclose(np.abs(arr[2]), arr[3], atol=1e-5)}')
        print(f'  ch0+ch2==ch1: {np.allclose(arr[0]+arr[2], arr[1], atol=1e-5)}')
        print(f'  mask binary: {np.all((arr[4]==0)|(arr[4]==1))}')
        print(f'  mask rows==1: {(arr[4]==1).all(1).sum()} (expected 1)')

    mb_each = N_CHANNELS * SEQ_LEN * EMB_DIM * 4 / 1024**2
    print(f'\n{mb_each:.2f} MB per file  |  {mb_each*total_ok/1024:.1f} GB total')


if __name__ == '__main__':
    main()
