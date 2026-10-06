"""
mcnn.py  —  Ablation-ready PLM-MCNN for KRAS 7-target regression
=================================================================
Supports all ablations A1–A7 via command-line arguments.

Channel ablations (--channels):
  0 1 2 3 4  : full model — E_WT, E_Mut, Mut-WT, |Mut-WT|, mask (default)
  1          : A1 — mutant embedding only
  0 1        : A2 — WT + Mutant, no difference channels
  2 3        : A3 — difference channels only
  0 1 2 3   : A4 — no mutation mask

Pooling ablations (--pooling):
  hybrid     : global max + global mean + mutation-centred (default)
  global     : A5 — global max + global mean only
  local      : A6 — mutation-centred only

Kernel ablation (--kernels):
  1 4 8 16   : full model (default)
  8          : A7 — single kernel

Model is named automatically from ablation config for CSV traceability.

Run (full model):
  python mcnn.py \
    --tensor_dir /srv/jupyterlab/workspace/KRAS/tensors/ \
    --master_csv /srv/jupyterlab/workspace/KRAS/data/kras_master_table.csv \
    --output     /srv/jupyterlab/workspace/KRAS/code/results/

Run (A1 — mutant only):
  python mcnn.py ... --channels 1

Run (A7 — kernel 8 only):
  python mcnn.py ... --kernels 8
"""

import os, csv, logging, argparse, datetime, random
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
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from dataset import (
    KRASRegressionDataset, compute_target_stats,
    TARGET_NAMES, N_TARGETS, SPLIT_PARTITIONS, MASK_COLS,
)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tensor_dir",   required=True)
parser.add_argument("--master_csv",   required=True)
parser.add_argument("--output",       required=True)
parser.add_argument("--proj_dim",     type=int,   default=256)
parser.add_argument("--filters",      type=int,   default=256)
parser.add_argument("--kernels",      type=int,   nargs="+", default=[1,4,8,16])
parser.add_argument("--local_win",    type=int,   default=5)
parser.add_argument("--dropout",      type=float, default=0.30)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--batch",        type=int,   default=32)
parser.add_argument("--epochs",       type=int,   default=150)
parser.add_argument("--patience",     type=int,   default=15)
parser.add_argument("--delta",        type=float, default=1.0)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--save_model",   action="store_true", default=False,
                    help="Save best checkpoint to output/<MODEL_NAME>_seed<SEED>.pt")
parser.add_argument("--single_task",  type=int,   default=None,
                    help="Train on single target only: 0=fold 1=RAF1 2=PIK3CG 3=RALGDS 4=SOS1 5=K27 6=K55")
parser.add_argument("--channels",     type=int,   nargs="+", default=[0,1,2,3,4],
                    help="Tensor channels to use: 0=E_WT 1=E_Mut 2=Mut-WT 3=|Mut-WT| 4=mask")
parser.add_argument("--pooling",      type=str,   default="hybrid",
                    choices=["hybrid","global","local"],
                    help="Pooling: hybrid=max+mean+mut-centred global=max+mean local=mut-centred")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── auto model name from config ───────────────────────────────────────────────
def make_model_name():
    ch_map = {
        (0,1,2,3,4): 'MCNN',
        (1,):        'MCNN_A1_mutant_only',
        (0,1):       'MCNN_A2_WT_Mut',
        (2,3):       'MCNN_A3_diff_only',
        (0,1,2,3):   'MCNN_A4_no_mask',
    }
    ch_key = tuple(sorted(args.channels))
    name   = ch_map.get(ch_key, f'MCNN_ch{"".join(map(str,sorted(args.channels)))}')
    if args.pooling == 'global': name += '_A5_global_pool'
    if args.pooling == 'local':  name += '_A6_local_pool'
    if args.kernels != [1,4,8,16]:
        ks = "_".join(map(str, sorted(args.kernels)))
        name += f'_A7_k{ks}'
    if args.single_task is not None:
        tname = TARGET_NAMES[args.single_task]
        name = f'MCNN_A8_{tname}'
    return name

MODEL_NAME   = make_model_name()
N_CHANNELS   = len(args.channels)
# A8: single-task mode — only one target active
ACTIVE_TARGETS = list(range(N_TARGETS))
if args.single_task is not None:
    ACTIVE_TARGETS = [args.single_task]
USE_MASK_CH  = 4 in args.channels   # channel 4 is the mutation mask

logging.info(f"Model name    : {MODEL_NAME}")
logging.info(f"Device        : {device}")
logging.info(f"Seed          : {args.seed}")
logging.info(f"Channels      : {args.channels}  ({N_CHANNELS} active)")
logging.info(f"Pooling       : {args.pooling}")
logging.info(f"kernels       : {args.kernels}")
logging.info(f"proj_dim      : {args.proj_dim}")
logging.info(f"filters       : {args.filters}")
logging.info(f"dropout       : {args.dropout}")


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION-CENTRED POOLING
# ═══════════════════════════════════════════════════════════════════════════════

class MutationCenteredPool(nn.Module):
    def __init__(self, half_win=5):
        super().__init__()
        self.half_win = half_win

    def forward(self, conv_out, mask_ch):
        """
        conv_out : (B, F, L)
        mask_ch  : (B, L) — 1 at mutation position (or None if no mask channel)
        Returns  : (B, F)
        """
        B, F, L = conv_out.shape
        out = torch.zeros(B, F, device=conv_out.device)
        if mask_ch is None:
            # no mask channel available — fall back to centre of sequence
            pos = L // 2
            lo  = max(0, pos - self.half_win)
            hi  = min(L, pos + self.half_win + 1)
            out = conv_out[:, :, lo:hi].mean(dim=-1)
            return out
        for b in range(B):
            pos = mask_ch[b].argmax().item()
            lo  = max(0, pos - self.half_win)
            hi  = min(L, pos + self.half_win + 1)
            out[b] = conv_out[b, :, lo:hi].mean(dim=-1)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class KRAS_MCNN(nn.Module):
    """
    Ablation-ready PLM-MCNN.
    Supports arbitrary channel subsets and pooling strategies.
    """

    def __init__(
        self,
        n_channels  = 5,
        seq_len     = 188,
        emb_dim     = 1280,
        proj_dim    = 256,
        kernels     = None,
        n_filters   = 256,
        local_win   = 5,
        dropout     = 0.30,
        pooling     = "hybrid",
        n_targets   = N_TARGETS,
    ):
        super().__init__()
        if kernels is None: kernels = [1,4,8,16]
        self.kernels  = kernels
        self.pooling  = pooling
        self.seq_len  = seq_len
        self.proj_dim = proj_dim
        self.n_ch     = n_channels

        # projection: shared across channels
        self.projection = nn.Linear(emb_dim, proj_dim, bias=False)

        in_channels = n_channels * proj_dim

        # conv branches
        self.conv_branches = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.conv_branches.append(nn.Sequential(
                nn.Conv1d(in_channels, n_filters, kernel_size=k, padding=pad),
                nn.GELU(),
            ))

        # mutation-centred pooling
        self.mut_pool = MutationCenteredPool(half_win=local_win)

        # branch output dim depends on pooling strategy
        if pooling == "hybrid":
            branch_dim = n_filters * 3   # max + mean + local
        elif pooling == "global":
            branch_dim = n_filters * 2   # max + mean
        elif pooling == "local":
            branch_dim = n_filters * 1   # local only

        concat_dim = branch_dim * len(kernels)

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 7 independent regression heads
        self.heads = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(n_targets)
        ])

    def forward(self, x_full):
        """
        x_full  : (B, 5, 188, 1280) — full 5-channel tensor
        Selects active channels, then projects, convoles, pools.
        Returns : (B, 7) continuous ddG predictions
        """
        B, _, L, D = x_full.shape

        # select active channels
        x = x_full[:, args.channels, :, :]   # (B, n_ch, L, D)

        # extract mutation mask if channel 4 is active
        mask_ch = None
        if USE_MASK_CH:
            mask_idx = args.channels.index(4)
            mask_ch  = x[:, mask_idx, :, 0]   # (B, L)

        # project: (B, n_ch, L, D) → (B, n_ch, L, proj_dim)
        x_proj = self.projection(x)

        # reshape to (B, n_ch*proj_dim, L)
        x_flat = x_proj.permute(0, 1, 3, 2)              # (B, n_ch, proj_dim, L)
        x_flat = x_flat.reshape(B, self.n_ch * self.proj_dim, L)

        # conv branches with selected pooling
        branch_outs = []
        for conv in self.conv_branches:
            feat = conv(x_flat)
            if feat.shape[2] != L:
                feat = feat[:, :, :L]

            if self.pooling == "hybrid":
                g_max  = feat.max(dim=2).values
                g_mean = feat.mean(dim=2)
                m_pool = self.mut_pool(feat, mask_ch)
                branch_outs.append(torch.cat([g_max, g_mean, m_pool], dim=1))
            elif self.pooling == "global":
                g_max  = feat.max(dim=2).values
                g_mean = feat.mean(dim=2)
                branch_outs.append(torch.cat([g_max, g_mean], dim=1))
            elif self.pooling == "local":
                m_pool = self.mut_pool(feat, mask_ch)
                branch_outs.append(m_pool)

        h = torch.cat(branch_outs, dim=1)
        h = self.trunk(h)
        preds = torch.cat([head(h) for head in self.heads], dim=1)  # (B, 7)
        return preds


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def masked_huber_loss(preds, targets, masks, delta=1.0):
    """
    Masked Huber macro-averaged across 7 tasks (professor Section 4.3).
    L = (1/7) * sum_j [ sum_i m_ij * Huber(y_ij, yhat_ij) / sum_i m_ij ]
    """
    huber  = F.huber_loss(preds, targets, reduction='none', delta=delta)
    losses = []
    for j in ACTIVE_TARGETS:
        n_obs = masks[:, j].sum()
        if n_obs > 0:
            losses.append((huber[:, j] * masks[:, j]).sum() / n_obs)
    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=preds.device)
    return torch.stack(losses).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════
DDG_DELTA = 0.25

def derive_class(ddg):
    if ddg < -DDG_DELTA: return 0
    if ddg >  DDG_DELTA: return 2
    return 1


def compute_regression_metrics(y_true, y_pred, mask):
    obs = mask == 1
    if obs.sum() < 2:
        return dict(MAE=np.nan, RMSE=np.nan, Pearson_r=np.nan,
                    Spearman_rho=np.nan, N=int(obs.sum()))
    yt = y_true[obs]; yp = y_pred[obs]
    pr, _ = pearsonr(yt, yp)
    sr, _ = spearmanr(yt, yp)
    return dict(
        MAE         = round(float(np.mean(np.abs(yt-yp))), 4),
        RMSE        = round(float(np.sqrt(np.mean((yt-yp)**2))), 4),
        Pearson_r   = round(float(pr), 4),
        Spearman_rho= round(float(sr), 4),
        N           = int(obs.sum()),
    )


def compute_derived_class_metrics(y_true, y_pred, mask):
    obs = mask == 1
    if obs.sum() < 3:
        return dict(MCC=np.nan, Bal_Acc=np.nan, Macro_F1=np.nan)
    yt = np.array([derive_class(v) for v in y_true[obs]])
    yp = np.array([derive_class(v) for v in y_pred[obs]])
    mcc     = matthews_corrcoef(yt, yp)
    bal_acc = balanced_accuracy_score(yt, yp)
    _, _, f, _ = precision_recall_fscore_support(
        yt, yp, labels=[0,1,2], average=None, zero_division=0)
    return dict(MCC=round(float(mcc),4), Bal_Acc=round(float(bal_acc),4),
                Macro_F1=round(float(f.mean()),4))


def evaluate_all(all_true, all_pred, all_mask, model_name, split_name, means, stds):
    results = []
    pred_raw = all_pred.copy(); true_raw = all_true.copy()
    for j, t in enumerate(TARGET_NAMES):
        sig = stds[t]; mu = means[t]
        if sig > 1e-6:
            pred_raw[:,j] = all_pred[:,j]*sig+mu
            true_raw[:,j] = all_true[:,j]*sig+mu

    logging.info(f"\n  [{model_name}] [{split_name}]")
    logging.info(f"  {'Target':<10} {'N':>5} {'MAE':>8} {'RMSE':>8} "
                 f"{'Pearson':>9} {'Spearman':>10} {'MCC':>7}")

    for j, t in enumerate(TARGET_NAMES):
        reg = compute_regression_metrics(true_raw[:,j], pred_raw[:,j], all_mask[:,j])
        cls = compute_derived_class_metrics(true_raw[:,j], pred_raw[:,j], all_mask[:,j])
        logging.info(f"  {t:<10} {reg['N']:>5} {reg['MAE']:>8.4f} {reg['RMSE']:>8.4f} "
                     f"{reg['Pearson_r']:>9.4f} {reg['Spearman_rho']:>10.4f} "
                     f"{cls['MCC']:>7.4f}")
        row = {'Model':model_name,'Split':split_name,'Target':t}
        row.update(reg); row.update(cls)
        results.append(row)

    macro_reg = {k:round(float(np.nanmean([r[k] for r in results])),4)
                 for k in ['MAE','RMSE','Pearson_r','Spearman_rho']}
    macro_reg['N'] = int(np.nansum([r['N'] for r in results]))
    macro_cls = {k:round(float(np.nanmean([r[k] for r in results])),4)
                 for k in ['MCC','Bal_Acc','Macro_F1']}
    logging.info(f"  {'MACRO':<10} {macro_reg['N']:>5} {macro_reg['MAE']:>8.4f} "
                 f"{macro_reg['RMSE']:>8.4f} {macro_reg['Pearson_r']:>9.4f} "
                 f"{macro_reg['Spearman_rho']:>10.4f} {macro_cls['MCC']:>7.4f}")
    macro_row = {'Model':model_name,'Split':split_name,'Target':'MACRO'}
    macro_row.update(macro_reg); macro_row.update(macro_cls)
    results.append(macro_row)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION RANKING
# ═══════════════════════════════════════════════════════════════════════════════

def position_level_evaluation(master, all_true, all_pred, all_mask,
                               split_df, model_name, split_name, means, stds):
    pred_raw = all_pred.copy(); true_raw = all_true.copy()
    for j, t in enumerate(TARGET_NAMES):
        sig = stds[t]; mu = means[t]
        if sig > 1e-6:
            pred_raw[:,j] = all_pred[:,j]*sig+mu
            true_raw[:,j] = all_true[:,j]*sig+mu

    split_df = split_df.copy().reset_index(drop=True)
    split_df['pred_mag'] = np.nanmean(np.abs(pred_raw)*all_mask, axis=1)
    split_df['true_mag'] = np.nanmean(np.abs(true_raw)*all_mask, axis=1)

    pos_groups     = split_df.groupby('position')
    pos_pred_mpos  = pos_groups['pred_mag'].mean()
    pos_true_mpos  = pos_groups['true_mag'].mean()
    common_pos     = pos_pred_mpos.index.intersection(pos_true_mpos.index)
    if len(common_pos) < 3: return None

    sp, _ = spearmanr(pos_true_mpos[common_pos], pos_pred_mpos[common_pos])
    kt, _ = kendalltau(pos_true_mpos[common_pos], pos_pred_mpos[common_pos])

    logging.info(f"\n  Position ranking [{split_name}]: "
                 f"Spearman={sp:.4f}  Kendall={kt:.4f}  N={len(common_pos)}")
    return {
        'Model': model_name,'Split': split_name,'Target': 'M_pos_ranking',
        'N_positions': len(common_pos),
        'Spearman_Mpos': round(float(sp),4),
        'Kendall_tau':   round(float(kt),4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, epoch, max_epochs):
    model.train()
    total_loss = 0.0
    n_steps    = len(loader)
    for step, (xb, tb, mb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        tb = tb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(xb)
        loss = masked_huber_loss(pred, tb, mb, delta=args.delta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pct    = step/n_steps
        filled = int(30*pct)
        bar    = "█"*filled + "░"*(30-filled)
        print(f"\rEpoch {epoch:3d}/{max_epochs} [{bar}] {step}/{n_steps} "
              f"loss={total_loss/step:.4f}", end="", flush=True)
    print()
    return total_loss/max(n_steps,1)


@torch.no_grad()
def predict(model, loader, split_name=""):
    model.eval()
    all_preds=[]; all_targets=[]; all_masks=[]
    n_steps = len(loader)
    for step, (xb, tb, mb) in enumerate(loader, 1):
        pred = model(xb.to(device, non_blocking=True)).cpu().numpy()
        all_preds.append(pred); all_targets.append(tb.numpy()); all_masks.append(mb.numpy())
        pct=step/n_steps; filled=int(20*pct); bar="█"*filled+"░"*(20-filled)
        print(f"\r  Evaluating {split_name} [{bar}] {step}/{n_steps}", end="", flush=True)
    print()
    return (np.concatenate(all_preds,axis=0),
            np.concatenate(all_targets,axis=0),
            np.concatenate(all_masks,axis=0))


def stopping_signal(all_pred, all_true, all_mask, means, stds):
    pred_raw=all_pred.copy(); true_raw=all_true.copy()
    for j,t in enumerate(TARGET_NAMES):
        sig=stds[t]; mu=means[t]
        if sig>1e-6:
            pred_raw[:,j]=all_pred[:,j]*sig+mu
            true_raw[:,j]=all_true[:,j]*sig+mu
    prs=[]
    for j in ACTIVE_TARGETS:
        obs=all_mask[:,j]==1
        if obs.sum()>=2:
            pr,_=pearsonr(true_raw[obs,j],pred_raw[obs,j])
            prs.append(float(pr))
    return float(np.mean(prs)) if prs else -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print(f"KRAS PLM-MCNN — {MODEL_NAME}")
    print("="*65)

    master = pd.read_csv(args.master_csv)

    train_ds = KRASRegressionDataset(
        split='train', tensor_dir=args.tensor_dir, master_csv=args.master_csv)
    means, stds = compute_target_stats(train_ds)

    ds = {}
    for sn in ['train','val','test_random','test_bio']:
        ds[sn] = KRASRegressionDataset(
            split=sn, tensor_dir=args.tensor_dir, master_csv=args.master_csv,
            target_means=means, target_stds=stds)
        logging.info(f"  {sn:<15}: {len(ds[sn])} variants")

    train_loader = DataLoader(ds['train'], batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(ds['val'],   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model = KRAS_MCNN(
        n_channels = N_CHANNELS,
        proj_dim   = args.proj_dim,
        kernels    = args.kernels,
        n_filters  = args.filters,
        local_win  = args.local_win,
        dropout    = args.dropout,
        pooling    = args.pooling,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_signal=-999.0; patience_cnt=0; best_state=None

    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, optimizer, epoch, args.epochs)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            vp, vt, vm = predict(model, val_loader, "val")
            signal      = stopping_signal(vp, vt, vm, means, stds)
            logging.info(f"Epoch {epoch:3d}/{args.epochs}  "
                         f"train_loss={loss:.4f}  val_mean_Pearson={signal:.4f}")
            if signal > best_signal:
                best_signal  = signal
                best_state   = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                patience_cnt = 0
                logging.info(f"  → Best val Pearson={best_signal:.4f}")
            else:
                patience_cnt += 5
                if patience_cnt >= args.patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    # save checkpoint if requested
    if args.save_model and best_state:
        import torch as _torch
        ckpt_name = f"{MODEL_NAME}_seed{args.seed}.pt"
        ckpt_path = os.path.join(args.output, ckpt_name)
        _torch.save(best_state, ckpt_path)
        logging.info(f"Checkpoint saved: {ckpt_path}")

    # evaluation
    all_results=[]; pos_results=[]
    SPLIT_PART_MAP = {'val':{'fold_4'},'test_random':{'test_random'},'test_bio':{'test_curated'}}

    for sn in ['val','test_random','test_bio']:
        loader = DataLoader(ds[sn], batch_size=args.batch, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
        pred, true, mask = predict(model, loader, sn)
        results = evaluate_all(pred, true, mask, MODEL_NAME, sn, means, stds)
        all_results.extend(results)

        split_df = master[master['partition'].isin(SPLIT_PART_MAP[sn])].copy()
        mask_sum = split_df[MASK_COLS].sum(axis=1)
        split_df = split_df[mask_sum>=1].reset_index(drop=True)
        pos_r = position_level_evaluation(
            master, true, pred, mask, split_df, MODEL_NAME, sn, means, stds)
        if pos_r: pos_results.append(pos_r)

    # save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for r in all_results:
        r['timestamp']        = timestamp
        r['kernels']          = str(args.kernels)
        r['channels']         = str(args.channels)
        r['pooling']          = args.pooling
        r['proj_dim']         = args.proj_dim
        r['filters']          = args.filters
        r['dropout']          = args.dropout
        r['lr']               = args.lr
        r['batch']            = args.batch
        r['seed']             = args.seed
        r['best_val_Pearson'] = round(best_signal, 4)

    keys = ['Model','Split','Target','N',
            'MAE','RMSE','Pearson_r','Spearman_rho',
            'MCC','Bal_Acc','Macro_F1',
            'channels','pooling','kernels',
            'proj_dim','filters','dropout','lr','batch',
            'seed','best_val_Pearson','timestamp']

    fpath = os.path.join(args.output, 'results_ablations.csv')
    file_exists = os.path.exists(fpath) and os.path.getsize(fpath) > 0
    with open(fpath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        if not file_exists: writer.writeheader()
        writer.writerows(all_results)
    logging.info(f"\nResults saved: {fpath}")

    if pos_results:
        ppath = os.path.join(args.output, 'results_ablations_position.csv')
        pfile_exists = os.path.exists(ppath) and os.path.getsize(ppath) > 0
        pkeys = ['Model','Split','Target','N_positions','Spearman_Mpos','Kendall_tau']
        with open(ppath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pkeys, extrasaction='ignore')
            if not pfile_exists: writer.writeheader()
            writer.writerows(pos_results)

    # summary
    df_res = pd.DataFrame(all_results)
    macro  = df_res[df_res['Target']=='MACRO']
    print(f"\n{'='*75}")
    print(f"SUMMARY — {MODEL_NAME}")
    print(f"{'='*75}")
    print(f"{'Split':<15} {'MAE':>8} {'RMSE':>8} {'Pearson':>9} {'Spearman':>10} {'MCC':>8}")
    print("-"*62)
    for _, row in macro.iterrows():
        print(f"{row['Split']:<15} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['Pearson_r']:>9.4f} {row['Spearman_rho']:>10.4f} "
              f"{row['MCC']:>8.4f}")

    if pos_results:
        print(f"\n{'='*55}")
        print("POSITION RANKING (Spearman on M_pos)")
        for r in pos_results:
            print(f"  {r['Split']:<15}: {r['Spearman_Mpos']:.4f}  "
                  f"Kendall={r['Kendall_tau']:.4f}  N={r['N_positions']}")


if __name__ == "__main__":
    main()