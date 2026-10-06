# KRAS PLM-MCNN: Paired Perturbation Embeddings for Multidimensional Binding Energetics Prediction

This repository contains the code for the paper:

> **Paired Perturbation Embeddings from a Protein Language Model Enable Multidimensional Prediction of KRAS Variant Binding Energetics and Reveal Differential Sequence Accessibility of Active- and Inactive-State Conformations**

## Overview

We present a paired protein language model–multi-window convolutional neural network (PLM-MCNN) that predicts seven-dimensional binding free energy change (ΔΔG) profiles for 3,553 KRAS variants across six interaction partners and folding stability, using only sequence-derived ESM-2 embeddings without structural input.

**Key findings:**
- The explicit perturbation channel H_Mut−H_WT is more informative than the absolute wild-type embedding (ablation A2 Δ = −0.411, statistically significant)
- Primary model achieves macro-averaged Pearson r = 0.576 on held-out oncogenic positions (G12, G13, Q61)
- GDP-bound inactive-state prediction (K27, r = 0.302) is substantially less sequence-accessible than GTP-bound active-state prediction (K55, r = 0.734) — confirmed epistemic by three mechanistic experiments
- 9 of 10 top-impact positions are Specificity Modulators with partner-selective ΔΔG profiles

---

## Repository Structure

```
kras-plm-mcnn/
├── model/
│   ├── mcnn.py               # Primary PLM-MCNN model and training loop
│   └── dataset.py            # PyTorch dataset class for paired tensors
├── precompute/
│   ├── build_master_table.py # Merge raw DMS data into master table
│   └── precompute_paired_tensor.py  # Generate ESM-2 paired perturbation tensors
├── baselines/
│   ├── baselines.py          # Physicochemical MLP and PLM ridge regression
│   └── esm1v_baseline.py     # ESM-1v zero-shot masked marginal scoring
├── analysis/
│   ├── predict_all_testbio.py    # Generate scatter plot predictions
│   ├── predict_famous.py         # Famous mutation analysis
│   └── calculate_vulnerability.py # Position-level impact ranking
├── figures/
│   ├── figure_ablation_comparison.py
│   ├── figure_model_vs_baseline_per_target.py
│   └── figure_famous_mutation_heatmap.py
├── data/
│   └── README.md             # Data download instructions
├── requirements.txt
└── README.md
```

---

## Data

The experimental dataset is from:

> Weng C, Faure AJ, Escobedo A, et al. The energetic and allosteric landscape for KRAS inhibition. *Nature*. 2024;626:643–652.

**MaveDB accession:** `urn:mavedb:00000115`

Download the dataset from [MaveDB](https://www.mavedb.org/scoresets/urn:mavedb:00000115/) and place the files in the `data/` directory. Then run:

```bash
python precompute/build_master_table.py \
  --input_dir data/raw/ \
  --output    data/kras_master_table.csv
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/juanyuune/kras-plm-mcnn.git
cd kras-plm-mcnn

# Install dependencies
pip install -r requirements.txt

# ESM-2 and ESM-1v models are downloaded automatically on first use
# Requires ~3 GB disk space for ESM-2 650M
```

**Requirements:** Python 3.9+, CUDA GPU recommended (tested on CUDA 13.0)

---

## Reproducing Results

### Step 1 — Precompute paired perturbation tensors

```bash
python precompute/precompute_paired_tensor.py \
  --master_csv data/kras_master_table.csv \
  --output_dir tensors/
```

This generates 3,553 float32 tensor files (15.9 GB total). Requires ESM-2 650M (~2.5 GB VRAM).

### Step 2 — Train the primary model (5 seeds)

```bash
for SEED in 1 2 3 4 5; do
  python model/mcnn.py \
    --tensor_dir tensors/ \
    --master_csv data/kras_master_table.csv \
    --output     results/ \
    --proj_dim 256 --filters 256 \
    --kernels 1 4 8 16 \
    --channels 0 1 2 3 4 \
    --pooling hybrid --local_win 5 \
    --dropout 0.30 --lr 1e-4 \
    --weight_decay 1e-4 \
    --batch 32 --epochs 150 \
    --patience 15 \
    --seed $SEED \
    --save_model
done
```

### Step 3 — Run baselines

```bash
# Physicochemical MLP and PLM ridge regression
python baselines/baselines.py \
  --tensor_dir tensors/ \
  --master_csv data/kras_master_table.csv \
  --output     results/

# ESM-1v zero-shot
python baselines/esm1v_baseline.py \
  --master_csv data/kras_master_table.csv \
  --output     results/
```

### Step 4 — Famous mutation analysis

```bash
python analysis/predict_famous.py \
  --tensor_dir tensors/ \
  --master_csv data/kras_master_table.csv \
  --model_pt   results/MCNN_seed1.pt \
  --output     results/
```

### Step 5 — Generate scatter plot (Figure 10)

```bash
python analysis/predict_all_testbio.py \
  --tensor_dir tensors/ \
  --master_csv data/kras_master_table.csv \
  --model_pts  results/MCNN_seed1.pt \
               results/MCNN_seed2.pt \
               results/MCNN_seed3.pt \
  --output     results/
```

---

## Ablation Study

Run all eight ablation configurations:

```bash
# A1 — mutant only
python model/mcnn.py [base args] --channels 1

# A2 — WT + Mut without displacement
python model/mcnn.py [base args] --channels 0 1

# A3 — displacement only
python model/mcnn.py [base args] --channels 2 3

# A4 — no mask channel
python model/mcnn.py [base args] --channels 0 1 2 3

# A5 — global pooling only
python model/mcnn.py [base args] --pooling global

# A6 — local pooling only
python model/mcnn.py [base args] --pooling local

# A7 — single kernel k=8
python model/mcnn.py [base args] --kernels 8

# A8 — single-task models (one per target)
for TARGET in 0 1 2 3 4 5 6; do
  python model/mcnn.py [base args] --single_task $TARGET
done
```

---

## Model Architecture

The PLM-MCNN processes a five-channel paired perturbation tensor:

```
Z = [H_WT, H_Mut, H_Mut−H_WT, |H_Mut−H_WT|, M]
Shape: (5, 188, 1280)
```

Five forward-pass stages:
1. **Shared projection** — Linear(1280→256) per channel per position
2. **Multi-window Conv1d** — kernels {1, 4, 8, 16}, 256 filters each
3. **Hybrid pooling** — global max + global mean + mutation-centred (±5 residues)
4. **Shared trunk** — Linear(3072→512)→GELU→LayerNorm→Dropout(0.30)→Linear(512→256)
5. **Independent heads** — 7 × Linear(256→1)

**Total trainable parameters:** 11,538,951 (ESM-2 650M frozen)

---

## Results Summary

| Model | test_bio Macro Pearson r |
|-------|------------------------|
| Physicochemical MLP | −0.041 |
| PLM Ridge Regression | 0.359 |
| ESM-1v Zero-Shot | 0.196 |
| **PLM-MCNN (ours)** | **0.576** |

Per-target on biological challenge set:

| Target | Pearson r | Notes |
|--------|-----------|-------|
| ΔΔG_fold | 0.692 | Folding stability |
| ΔΔG_RAF1 | 0.558 | MEK/ERK pathway |
| ΔΔG_PIK3CG | 0.627 | PI3K/AKT pathway |
| ΔΔG_RALGDS | 0.641 | RAL pathway |
| ΔΔG_SOS1 | 0.477 | Nucleotide exchange |
| ΔΔG_K27 | 0.302 | GDP-bound sensor (epistemic limit) |
| ΔΔG_K55 | 0.734 | GTP-bound sensor |

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{kras_plm_mcnn_2026,
  title   = {Paired Perturbation Embeddings from a Protein Language Model Enable
             Multidimensional Prediction of KRAS Variant Binding Energetics and
             Reveal Differential Sequence Accessibility of Active- and
             Inactive-State Conformations},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2026}
}
```

Also cite the experimental dataset:

```bibtex
@article{weng2024kras,
  title   = {The energetic and allosteric landscape for KRAS inhibition},
  author  = {Weng, Chenchun and Faure, Alejandro J and Escobedo, Albert and others},
  journal = {Nature},
  volume  = {626},
  pages   = {643--652},
  year    = {2024}
}
```

---

## License

MIT License. See `LICENSE` for details.
