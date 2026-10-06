import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

abl = pd.read_csv('/srv/jupyterlab/workspace/KRAS/code/results/results_ablations.csv')

MODEL_LABELS = {
    'MCNN':                'Full\nmodel',
    'MCNN_A1_mutant_only': 'A1\nMut only',
    'MCNN_A2_WT_Mut':      'A2\nWT+Mut',
    'MCNN_A3_diff_only':   'A3\nDiff only',
    'MCNN_A4_no_mask':     'A4\nNo mask',
    'MCNN_A5_global_pool': 'A5\nGlobal',
    'MCNN_A6_local_pool':  'A6\nLocal',
    'MCNN_A7_k8':          'A7\nKernel 8',
}
MODEL_ORDER = list(MODEL_LABELS.keys())
COLORS = ['#4878CF','#6ACC65','#D65F5F','#B47CC7','#C4AD66','#77BEDB','#E09F3E','#9B5DE5']
SPLITS = [('val','Validation'), ('test_random','test_random'), ('test_bio','Biological challenge')]

macro = abl[abl['Target']=='MACRO']

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
fig.suptitle('Ablation Study — Macro-Averaged Pearson r Across Seven Targets',
             fontsize=13, fontweight='bold', y=1.01)

for ax, (split, split_label) in zip(axes, SPLITS):
    means, errs = [], []
    for m in MODEL_ORDER:
        vals = macro[(macro['Model']==m)&(macro['Split']==split)]['Pearson_r']
        means.append(vals.mean())
        n = len(vals)
        t = stats.t.ppf(0.975, df=max(n-1,1))
        errs.append(t * vals.std() / np.sqrt(n) if n > 1 else 0)

    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(x, means, yerr=errs, capsize=4, color=COLORS,
                  edgecolor='white', linewidth=0.5,
                  error_kw=dict(elinewidth=1, ecolor='#333333'))
    ax.axhline(means[0], color='#333333', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
    ax.set_title(split_label, fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel('Pearson r' if ax == axes[0] else '', fontsize=10)
    ax.set_ylim(0.15, 0.72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, color='#333333')

plt.tight_layout()
plt.savefig('/srv/jupyterlab/workspace/KRAS/figures/fig_ablation_comparison.png',
            dpi=180, bbox_inches='tight')
plt.show()
print("Saved")