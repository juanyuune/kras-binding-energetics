"""
precompute_paired_tensor.py  —  Phase 1, Step 2
================================================
Builds the 5-channel paired perturbation tensor for each KRAS variant.

For each variant, loads WT and mutant ESM-2 embeddings and produces:

  Channel 0: E_WT           — wild-type embedding        (188, 1280)
  Channel 1: E_Mut          — mutant embedding            (188, 1280)
  Channel 2: E_Mut - E_WT   — signed displacement         (188, 1280)
  Channel 3: |E_Mut - E_WT| — absolute displacement       (188, 1280)
  Channel 4: M              — mutation mask               (188, 1280)
               M[k, :] = 1.0 at the mutated residue position k
               M[k, :] = 0.0 everywhere else

Saved as: variant_name.npy   shape (5, 188, 1280)  float32

The projection (1280 → 256) is a learned layer inside the MCNN.
Precomputing at full 1280 dimensions keeps the projection trainable.

Output structure:
  /KRAS/tensors/
    train/      p.Gly12Asp.npy  shape (5, 188, 1280)
    val/        ...
    test_random/...
    test_bio/   ...
    WT_KRAS.npy — WT embedding only (188, 1280) for reference

Memory per file: 5 × 188 × 1280 × 4 bytes = 4.82 MB
Total disk: 3,553 × 4.82 MB ≈ 17.1 GB

Run:
  python precompute_paired_tensor.py \
    --emb_base   /srv/jupyterlab/workspace/KRAS/emb/ \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --out_dir    /srv/jupyterlab/workspace/KRAS/tensors/
"""

import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb_base",   required=True,
                    help="Base dir containing WT_KRAS.esm2 and split subfolders "
                         "(train/ val/ test_random/ test_bio/)")
parser.add_argument("--master_csv", required=True,
                    help="Path to kras_master_table.csv from build_master_table.py")
parser.add_argument("--out_dir",    required=True,
                    help="Output dir for .npy tensor files")
args = parser.parse_args()

SEQ_LEN = 188
EMB_DIM = 1280
N_CHANNELS = 5

# split subfolder names — must match emb_base structure
SPLIT_MAP = {
    'fold_0':       'train',
    'fold_1':       'train',
    'fold_2':       'train',
    'fold_3':       'train',
    'fold_4':       'val',
    'test_random':  'test_random',
    'test_curated': 'test_bio',
}

# ── helpers ───────────────────────────────────────────────────────────────────
def load_esm2(path: str) -> np.ndarray:
    """Load one .esm2 file. Returns (SEQ_LEN, EMB_DIM) float32."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        data = next(iter(data.values()))
    data = np.array(data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim == 3 and data.shape[1] == 1:
        data = data[:, 0, :]
    # pad or truncate to SEQ_LEN
    L, D = data.shape
    out  = np.zeros((SEQ_LEN, D), dtype=np.float32)
    keep = min(L, SEQ_LEN)
    out[:keep, :] = data[:keep, :]
    return out


def make_mutation_mask(position: int, seq_len: int = SEQ_LEN,
                       emb_dim: int = EMB_DIM) -> np.ndarray:
    """
    Binary mutation mask. Shape (seq_len, emb_dim).
    Row at (position-1) is all 1.0; all other rows are 0.0.
    Position is 1-indexed (KRAS residue numbering).
    """
    mask = np.zeros((seq_len, emb_dim), dtype=np.float32)
    idx  = int(position) - 1   # convert to 0-indexed
    if 0 <= idx < seq_len:
        mask[idx, :] = 1.0
    return mask


def build_tensor(e_wt: np.ndarray,
                 e_mut: np.ndarray,
                 position: int) -> np.ndarray:
    """
    Build 5-channel paired perturbation tensor.

    Z = [E_WT, E_Mut, E_Mut-E_WT, |E_Mut-E_WT|, M]

    Returns shape (5, SEQ_LEN, EMB_DIM) float32.
    """
    signed_diff = e_mut - e_wt
    abs_diff    = np.abs(signed_diff)
    mask        = make_mutation_mask(position)

    tensor = np.stack([
        e_wt,        # channel 0
        e_mut,       # channel 1
        signed_diff, # channel 2
        abs_diff,    # channel 3
        mask,        # channel 4
    ], axis=0)       # (5, 188, 1280)

    return tensor.astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("Phase 1 Step 2 — Precompute 5-Channel Paired Tensors")
    print("=" * 65)

    # load master table
    logging.info(f"Loading master table from {args.master_csv}")
    master = pd.read_csv(args.master_csv)
    logging.info(f"  {len(master)} variants loaded")

    # create output directories
    for split_dir in set(SPLIT_MAP.values()):
        os.makedirs(os.path.join(args.out_dir, split_dir), exist_ok=True)

    # load WT embedding once — shared across all variants
    wt_path = os.path.join(args.emb_base, 'WT_KRAS.esm2')
    if not os.path.exists(wt_path):
        raise FileNotFoundError(f"WT embedding not found: {wt_path}")

    logging.info(f"Loading WT embedding from {wt_path}")
    e_wt = load_esm2(wt_path)
    logging.info(f"  WT shape: {e_wt.shape}")

    # save WT embedding separately for reference
    wt_out = os.path.join(args.out_dir, 'WT_KRAS.npy')
    np.save(wt_out, e_wt)
    logging.info(f"  WT saved: {wt_out}")

    # process each split
    total_written = 0
    total_failed  = 0
    split_counts  = {v: 0 for v in set(SPLIT_MAP.values())}

    for split_name in set(SPLIT_MAP.values()):
        # get variants for this split
        partition_keys = [k for k, v in SPLIT_MAP.items() if v == split_name]
        split_variants = master[master['partition'].isin(partition_keys)].copy()

        emb_split_dir = os.path.join(args.emb_base, split_name)
        out_split_dir = os.path.join(args.out_dir,  split_name)

        if not os.path.exists(emb_split_dir):
            logging.warning(f"EMB dir not found: {emb_split_dir} — skipping")
            continue

        logging.info(f"\n[{split_name}] {len(split_variants)} variants "
                     f"from partitions: {partition_keys}")

        written = 0
        failed  = 0

        for _, row in split_variants.iterrows():
            variant  = row['variant']
            position = int(row['position'])

            # find mutant embedding file
            emb_fname = variant + '.esm2'
            emb_path  = os.path.join(emb_split_dir, emb_fname)
            out_path  = os.path.join(out_split_dir, variant + '.npy')

            if not os.path.exists(emb_path):
                logging.warning(f"  Not found: {emb_path}")
                failed += 1
                continue

            try:
                e_mut  = load_esm2(emb_path)
                tensor = build_tensor(e_wt, e_mut, position)
                np.save(out_path, tensor)
                written += 1
            except Exception as e:
                logging.error(f"  Failed on {variant}: {e}")
                failed += 1

        logging.info(f"  Written: {written}  Failed: {failed}")
        split_counts[split_name] = written
        total_written += written
        total_failed  += failed

    # ── verification ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("VERIFICATION")
    print("=" * 65)

    expected = {
        'train':       2565,
        'val':         627,
        'test_random': 190,
        'test_bio':    171,
    }

    all_ok = True
    for split_name, exp in expected.items():
        out_dir = os.path.join(args.out_dir, split_name)
        if not os.path.exists(out_dir):
            print(f"  {split_name:<15}: MISSING directory")
            all_ok = False
            continue
        n = len([f for f in os.listdir(out_dir) if f.endswith('.npy')])
        status = 'OK' if n == exp else f'MISMATCH (expected {exp})'
        print(f"  {split_name:<15}: {n:5d} .npy files  {status}")
        if n != exp:
            all_ok = False

    # spot check shape and values
    print(f"\n  Spot check — loading one tensor:")
    sample_dir = os.path.join(args.out_dir, 'train')
    samples    = [f for f in os.listdir(sample_dir) if f.endswith('.npy')]
    if samples:
        arr = np.load(os.path.join(sample_dir, samples[0]))
        print(f"    File     : {samples[0]}")
        print(f"    Shape    : {arr.shape}  (expected (5, 188, 1280))")
        print(f"    dtype    : {arr.dtype}  (expected float32)")
        print(f"    Shape OK : {arr.shape == (N_CHANNELS, SEQ_LEN, EMB_DIM)}")

        # check channel 4 (mask) is binary
        mask_ch   = arr[4]
        is_binary = np.all((mask_ch == 0) | (mask_ch == 1))
        print(f"    Mask binary: {is_binary}")

        # check that exactly one position row is all 1s in mask
        rows_all_ones = (mask_ch == 1).all(axis=1).sum()
        print(f"    Mask has exactly 1 mutated row: {rows_all_ones == 1} "
              f"(found {rows_all_ones})")

        # check channel 2 and 3 relationship
        signed = arr[2]
        absval = arr[3]
        abs_ok = np.allclose(absval, np.abs(signed), atol=1e-5)
        print(f"    |signed| == absolute: {abs_ok}")

        # check channel 0 + 2 = channel 1
        recon_mut = arr[0] + arr[2]
        recon_ok  = np.allclose(recon_mut, arr[1], atol=1e-5)
        print(f"    E_WT + (E_Mut-E_WT) == E_Mut: {recon_ok}")

    print(f"\n  Total written : {total_written}")
    print(f"  Total failed  : {total_failed}")

    if all_ok and total_failed == 0:
        print("\nAll tensors computed and saved successfully.")
    else:
        print("\nWARNING: check output above.")

    print(f"\nOutput: {args.out_dir}")
    print(f"{'='*65}")

    # memory estimate
    size_per_file_mb = (N_CHANNELS * SEQ_LEN * EMB_DIM * 4) / (1024**2)
    total_gb = (size_per_file_mb * total_written) / 1024
    print(f"\nDisk usage estimate:")
    print(f"  Per file : {size_per_file_mb:.2f} MB")
    print(f"  Total    : {total_gb:.1f} GB for {total_written} files")


if __name__ == "__main__":
    main()