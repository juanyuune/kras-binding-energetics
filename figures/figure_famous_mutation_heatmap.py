import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fam = pd.read_csv('/srv/jupyterlab/workspace/KRAS/code/results/results_famous_mutations.csv')

MUTATIONS = ['G12D','G12V','G12R','G12C','G13D','Q61H','Q61L','Q61R']
TARGETS   = ['fold','RAF1','PIK3CG','RALGDS','SOS1','K27','K55']

exp_mat   = np.zeros((len(MUTATIONS), len(TARGETS)))
pred_mat  = np.zeros((len(MUTATIONS), len(TARGETS)))
match_mat = np.ones((len(MUTATIONS), len(TARGETS)), dtype=bool)

for i, m in enumerate(MUTATIONS):
    for j, t in enumerate(TARGETS):
        row = fam[(fam['Mutation']==m)&(fam['Target']==t)]
        if len(row) > 0:
            exp_mat[i,j]   = row.iloc[0]['Exp_ddG']
            pred_mat[i,j]  = row.iloc[0]['Pred_ddG']
            match_mat[i,j] = row.iloc[0]['Class_match']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Famous Oncogenic KRAS Mutations — Experimental vs Predicted ΔΔG Profiles',
             fontsize=12, fontweight='bold', y=1.02)

vmin, vmax = -0.5, 1.4

for ax, mat, title in zip(axes,
    [exp_mat, pred_mat],
    ['Experimental ΔΔG (kcal/mol)', 'Predicted ΔΔG (kcal/mol)']):

    im = ax.imshow(mat, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(TARGETS, fontsize=10)
    ax.set_yticks(range(len(MUTATIONS)))
    ax.set_yticklabels(MUTATIONS, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    for i in range(len(MUTATIONS)):
        for j in range(len(TARGETS)):
            v = mat[i,j]
            color = 'white' if abs(v) > 0.6 else '#333333'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')
            if ax == axes[1] and not match_mat[i,j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                    fill=False, edgecolor='black', linewidth=2.5))

    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔΔG (kcal/mol)')

axes[1].text(0.5, -0.10,
    'Bold border = classification error (δ = 0.25 kcal/mol)',
    ha='center', transform=axes[1].transAxes,
    fontsize=9, style='italic', color='#555555')

plt.tight_layout()
plt.savefig('/srv/jupyterlab/workspace/KRAS/figures/fig_famous_mutations_heatmap.png',
            dpi=180, bbox_inches='tight')
plt.show()
print("Saved")