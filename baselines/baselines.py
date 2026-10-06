"""
baselines.py  —  Phase 3
=========================
Three required baselines before MCNN (professor's specification).

Baseline 1 — Training-fold mean predictor
  Predict the per-target mean ddG from training data.
  Trivial baseline — any model must beat this.

Baseline 2 — Physicochemical features + MLP
  17 features: WT props, Mut props, delta, position, Grantham distance.
  Two-layer MLP with masked Huber loss.
  Tests whether PLM adds value beyond substitution descriptors.

Baseline 3 — PLM displacement + ridge regression
  1280-dim mean-pooled signed displacement (channel 2 of tensor).
  Ridge regression per target with masking.
  Tests how informative frozen PLM is without MCNN.

References:
  Kyte & Doolittle (1982) J Mol Biol 157:105
  Zimmerman et al. (1968) J Theor Biol 21:170
  Grantham (1974) Science 185:862
  Notin et al. ProteinGym (NeurIPS 2022)

Run:
  python baselines.py \
    --tensor_dir /srv/jupyterlab/workspace/KRAS/tensors/ \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --output     /srv/jupyterlab/workspace/KRAS/code/results/
"""

import os, csv, logging, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
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
parser.add_argument("--output",     required=True)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

# ── physicochemical tables (published scales) ─────────────────────────────────
# Kyte & Doolittle 1982
HYDROPHOBICITY = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,
    'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,
    'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
# Zimmerman et al. 1968 — van der Waals volume (Å³)
VOLUME = {
    'A':67,'C':86,'D':91,'E':109,'F':135,'G':48,'H':118,'I':124,'K':135,
    'L':124,'M':124,'N':96,'P':90,'Q':114,'R':148,'S':73,'T':93,'V':105,
    'W':163,'Y':141}
# Zimmerman et al. 1968 — polarity
POLARITY = {
    'A':0.00,'C':1.48,'D':49.70,'E':49.90,'F':0.00,'G':0.00,'H':51.60,
    'I':0.00,'K':49.50,'L':0.00,'M':1.43,'N':3.38,'P':1.58,'Q':3.53,
    'R':52.00,'S':1.67,'T':1.66,'V':0.00,'W':2.10,'Y':1.61}
# Approximate charge at pH 7
CHARGE = {
    'A':0,'C':0,'D':-1,'E':-1,'F':0,'G':0,'H':0.1,'I':0,'K':1,'L':0,
    'M':0,'N':0,'P':0,'Q':0,'R':1,'S':0,'T':0,'V':0,'W':0,'Y':0}
# Residue molecular weight (Da)
MOLWT = {
    'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.0,'H':155.2,
    'I':131.2,'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.2,
    'R':174.2,'S':105.1,'T':119.1,'V':117.1,'W':204.2,'Y':181.2}

SCALES      = [HYDROPHOBICITY, VOLUME, POLARITY, CHARGE, MOLWT]
N_PROPS     = len(SCALES)   # 5
N_PHYSCHEM  = 17            # 5 WT + 5 Mut + 5 delta + position + Grantham


def make_physchem_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build 17-dim physicochemical feature vector per variant.
    [0:5]  WT properties   [5:10] Mut properties   [10:15] delta
    [15]   position /186   [16]   Grantham distance
    """
    X = np.zeros((len(df), N_PHYSCHEM), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        wt  = str(row['wt_aa'])
        mut = str(row['mutant_aa'])
        pos = int(row['position'])
        for j, sc in enumerate(SCALES):
            X[i, j]           = sc.get(wt,  0.0)
            X[i, j + N_PROPS] = sc.get(mut, 0.0)
            X[i, j + 2*N_PROPS] = sc.get(mut, 0.0) - sc.get(wt, 0.0)
        X[i, 15] = (pos - 2) / (188 - 2)   # normalise to [0,1]
        vol_d = VOLUME.get(mut, 0) - VOLUME.get(wt, 0)
        pol_d = POLARITY.get(mut, 0) - POLARITY.get(wt, 0)
        X[i, 16] = float(np.sqrt(vol_d**2 + pol_d**2))
    return X


# ── masked Huber loss ─────────────────────────────────────────────────────────
def masked_huber_loss(preds, targets, masks, delta=1.0):
    """Macro-averaged masked Huber loss across 7 tasks."""
    huber  = nn.functional.huber_loss(preds, targets,
                                       reduction='none', delta=delta)
    losses = []
    for j in range(N_TARGETS):
        n_obs = masks[:, j].sum()
        if n_obs > 0:
            losses.append((huber[:, j] * masks[:, j]).sum() / n_obs)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, mask):
    obs = mask == 1
    if obs.sum() < 2:
        return {'MAE': np.nan, 'RMSE': np.nan,
                'Pearson_r': np.nan, 'Spearman_rho': np.nan,
                'N': int(obs.sum())}
    yt = y_true[obs]; yp = y_pred[obs]
    pr, _ = pearsonr(yt, yp)
    sr, _ = spearmanr(yt, yp)
    return {
        'MAE':          round(float(np.mean(np.abs(yt-yp))), 4),
        'RMSE':         round(float(np.sqrt(np.mean((yt-yp)**2))), 4),
        'Pearson_r':    round(float(pr), 4),
        'Spearman_rho': round(float(sr), 4),
        'N':            int(obs.sum()),
    }


def evaluate_predictions(all_true, all_pred, all_mask, model_name, split_name):
    results = []
    logging.info(f"\n  [{model_name}] [{split_name}]")
    logging.info(f"  {'Target':<10} {'N':>5} {'MAE':>8} {'RMSE':>8} "
                 f"{'Pearson':>9} {'Spearman':>10}")
    for j, t in enumerate(TARGET_NAMES):
        m = compute_metrics(all_true[:,j], all_pred[:,j], all_mask[:,j])
        logging.info(f"  {t:<10} {m['N']:>5} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} "
                     f"{m['Pearson_r']:>9.4f} {m['Spearman_rho']:>10.4f}")
        results.append({'Model':model_name,'Split':split_name,'Target':t,**m})
    # macro average
    macro = {k: round(float(np.nanmean([r[k] for r in results])), 4)
             for k in ['MAE','RMSE','Pearson_r','Spearman_rho']}
    macro['N'] = int(np.nansum([r['N'] for r in results]))
    logging.info(f"  {'MACRO':<10} {macro['N']:>5} {macro['MAE']:>8.4f} "
                 f"{macro['RMSE']:>8.4f} {macro['Pearson_r']:>9.4f} "
                 f"{macro['Spearman_rho']:>10.4f}")
    results.append({'Model':model_name,'Split':split_name,'Target':'MACRO',**macro})
    return results


# ── helper: get split dataframe ───────────────────────────────────────────────
def get_split_df(master, split_name):
    sp = SPLIT_PARTITIONS[split_name]
    df = master[master['partition'].isin(sp)].copy().reset_index(drop=True)
    ms = df[MASK_COLS].sum(axis=1)
    return df[ms >= 1].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("KRAS Baselines — Phase 3")
    print("=" * 65)

    master = pd.read_csv(args.master_csv)

    # normalisation stats from training set
    train_ds = KRASRegressionDataset(
        split='train', tensor_dir=args.tensor_dir, master_csv=args.master_csv)
    means, stds = compute_target_stats(train_ds)

    train_df = get_split_df(master, 'train')
    all_results = []

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE 1: Training-fold mean predictor
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}\nBASELINE 1 — Training-fold mean predictor\n{'='*65}")

    train_means = {}
    for t in TARGET_NAMES:
        obs = train_df.loc[train_df[f'mask_{t}']==1, f'ddG_{t}']
        train_means[t] = float(obs.mean()) if len(obs) > 0 else 0.0
        logging.info(f"  Training mean {t}: {train_means[t]:.4f}")

    for split_name in ['val','test_random','test_bio']:
        df = get_split_df(master, split_name)
        n  = len(df)
        Y  = df[[f'ddG_{t}' for t in TARGET_NAMES]].values.astype(np.float32)
        M  = df[[f'mask_{t}' for t in TARGET_NAMES]].values.astype(np.float32)
        P  = np.array([[train_means[t] for t in TARGET_NAMES]] * n, dtype=np.float32)
        all_results.extend(evaluate_predictions(Y, P, M, 'Mean_predictor', split_name))

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE 2: Physicochemical features + MLP
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}\nBASELINE 2 — Physicochemical features + MLP\n{'='*65}")

    # build features and targets for all splits
    Xs, Ys, Ms = {}, {}, {}
    for sn in ['train','val','test_random','test_bio']:
        df     = get_split_df(master, sn)
        Xs[sn] = make_physchem_features(df)
        Ys[sn] = df[[f'ddG_{t}' for t in TARGET_NAMES]].values.astype(np.float32)
        Ms[sn] = df[[f'mask_{t}' for t in TARGET_NAMES]].values.astype(np.float32)

    # standardise features on training set only
    feat_mu  = Xs['train'].mean(0, keepdims=True)
    feat_sig = Xs['train'].std(0,  keepdims=True)
    feat_sig[feat_sig < 1e-6] = 1.0
    for sn in Xs: Xs[sn] = (Xs[sn] - feat_mu) / feat_sig

    # normalised targets for training
    Y_tr_norm = Ys['train'].copy()
    for j, t in enumerate(TARGET_NAMES):
        sig = stds[t]; mu = means[t]
        obs = Ms['train'][:,j] == 1
        if sig > 1e-6: Y_tr_norm[obs, j] = (Ys['train'][obs, j] - mu) / sig
    Y_tr_norm = np.nan_to_num(Y_tr_norm, nan=0.0)

    class PhyschemMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(N_PHYSCHEM, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 64),         nn.ReLU(),
                nn.Linear(64, N_TARGETS),
            )
        def forward(self, x): return self.net(x)

    mlp = PhyschemMLP().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt  = torch.from_numpy(Xs['train']).to(device)
    Yt  = torch.from_numpy(Y_tr_norm).to(device)
    Mt  = torch.from_numpy(Ms['train']).to(device)

    logging.info("Training PhyschemMLP (200 epochs)...")
    for ep in range(200):
        mlp.train(); opt.zero_grad()
        loss = masked_huber_loss(mlp(Xt), Yt, Mt)
        loss.backward(); opt.step()
        if (ep+1) % 50 == 0:
            logging.info(f"  epoch {ep+1}/200  loss={loss.item():.4f}")

    mlp.eval()
    with torch.no_grad():
        for sn in ['val','test_random','test_bio']:
            pred_norm = mlp(torch.from_numpy(Xs[sn]).to(device)).cpu().numpy()
            pred_raw  = pred_norm.copy()
            for j, t in enumerate(TARGET_NAMES):
                if stds[t] > 1e-6:
                    pred_raw[:,j] = pred_norm[:,j] * stds[t] + means[t]
            all_results.extend(
                evaluate_predictions(Ys[sn], pred_raw, Ms[sn], 'Physchem_MLP', sn))

    # ═══════════════════════════════════════════════════════════════════════
    # BASELINE 3: PLM displacement + ridge regression
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}\nBASELINE 3 — PLM displacement + ridge regression\n{'='*65}")

    from sklearn.linear_model import Ridge

    TENSOR_SUBDIRS = {
        'train':'train','val':'val',
        'test_random':'test_random','test_bio':'test_bio'
    }

    # load mean-pooled displacement features (channel 2 = signed Mut-WT)
    Xp, Yp, Mp = {}, {}, {}
    for sn in ['train','val','test_random','test_bio']:
        df  = get_split_df(master, sn)
        n   = len(df)
        Xft = np.zeros((n, 1280), dtype=np.float32)
        tdir = os.path.join(args.tensor_dir, TENSOR_SUBDIRS[sn])
        for i, (_, row) in enumerate(df.iterrows()):
            tp = os.path.join(tdir, row['variant'] + '.npy')
            if os.path.exists(tp):
                Xft[i] = np.load(tp)[2].mean(axis=0)   # ch2: (188,1280) → mean → (1280,)
            else:
                logging.warning(f"  Missing tensor: {row['variant']}")
        Xp[sn] = Xft
        Yp[sn] = df[[f'ddG_{t}' for t in TARGET_NAMES]].values.astype(np.float32)
        Mp[sn] = df[[f'mask_{t}' for t in TARGET_NAMES]].values.astype(np.float32)
        logging.info(f"  [{sn}] PLM features: {Xft.shape}")

    # standardise on training set
    plm_mu  = Xp['train'].mean(0, keepdims=True)
    plm_sig = Xp['train'].std(0,  keepdims=True)
    plm_sig[plm_sig < 1e-6] = 1.0
    for sn in Xp: Xp[sn] = (Xp[sn] - plm_mu) / plm_sig

    # fit one ridge per target
    ridge_pred = {sn: np.full((len(Xp[sn]), N_TARGETS), np.nan)
                  for sn in ['val','test_random','test_bio']}

    for j, t in enumerate(TARGET_NAMES):
        obs = Mp['train'][:,j] == 1
        if obs.sum() < 10:
            logging.warning(f"  {t}: too few observed — skipping")
            continue
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xp['train'][obs], Yp['train'][obs, j])
        logging.info(f"  Ridge {t}: {obs.sum()} training samples")
        for sn in ['val','test_random','test_bio']:
            ridge_pred[sn][:,j] = ridge.predict(Xp[sn])

    for sn in ['val','test_random','test_bio']:
        all_results.extend(
            evaluate_predictions(Yp[sn], ridge_pred[sn], Mp[sn], 'PLM_Ridge', sn))

    # ── save CSV ──────────────────────────────────────────────────────────────
    keys  = ['Model','Split','Target','N','MAE','RMSE','Pearson_r','Spearman_rho']
    fpath = os.path.join(args.output, 'results_baselines.csv')
    with open(fpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    logging.info(f"\nResults saved: {fpath}")

    # ── summary ───────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(all_results)
    macro  = df_res[df_res['Target']=='MACRO']

    print(f"\n{'='*75}")
    print("SUMMARY — MACRO AVERAGES ACROSS 7 TARGETS")
    print(f"{'='*75}")
    print(f"{'Model':<20} {'Split':<15} {'MAE':>8} {'RMSE':>8} "
          f"{'Pearson':>9} {'Spearman':>10}")
    print("-"*75)
    for _, row in macro.iterrows():
        print(f"{row['Model']:<20} {row['Split']:<15} "
              f"{row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['Pearson_r']:>9.4f} {row['Spearman_rho']:>10.4f}")
    print(f"\nMCNN must outperform all three baselines.")


if __name__ == "__main__":
    main()