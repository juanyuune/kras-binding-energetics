import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

base = pd.read_csv('/srv/jupyterlab/workspace/KRAS/code/results/results_baselines.csv')
mcnn = pd.read_csv('/srv/jupyterlab/workspace/KRAS/code/results/results_mcnn.csv')

TARGET_NAMES = ['fold','RAF1','PIK3CG','RALGDS','SOS1','K27','K55']

bio_base  = base[(base['Split']=='test_bio')&(base['Target'].isin(TARGET_NAMES))]
plm_ridge = bio_base[bio_base['Model']=='PLM_Ridge'].set_index('Target')['Pearson_r']
physchem  = bio_base[bio_base['Model']=='Physchem_MLP'].set_index('Target')['Pearson_r']

fc5 = mcnn[(mcnn['Model']=='MCNN_FC')&(mcnn['seed'].isin([1,2,3,4,5]))]
bio_fc    = fc5[(fc5['Split']=='test_bio')&(fc5['Target'].isin(TARGET_NAMES))]
mcnn_mean = bio_fc.groupby('Target')['Pearson_r'].mean()
mcnn_std  = bio_fc.groupby('Target')['Pearson_r'].std()
t5 = stats.t.ppf(0.975, df=4)
mcnn_ci   = t5 * mcnn_std / np.sqrt(5)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(TARGET_NAMES))
w = 0.25

b1 = ax.bar(x - w, [physchem.get(t, np.nan) for t in TARGET_NAMES],
            w, label='Physchem + MLP', color='#9ecae1', edgecolor='white')
b2 = ax.bar(x,     [plm_ridge.get(t, np.nan) for t in TARGET_NAMES],
            w, label='PLM + Ridge',    color='#fc8d59', edgecolor='white')
b3 = ax.bar(x + w, [mcnn_mean.get(t, np.nan) for t in TARGET_NAMES],
            w, label='PLM-MCNN',       color='#2c7bb6', edgecolor='white',
            yerr=[mcnn_ci.get(t, 0) for t in TARGET_NAMES],
            capsize=4, error_kw=dict(elinewidth=1, ecolor='#1a1a1a'))

ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(0.25, color='gray', linewidth=0.6, linestyle=':', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'ΔΔG {t}' for t in TARGET_NAMES], fontsize=10)
ax.set_ylabel('Pearson r', fontsize=11)
ax.set_title('Per-Target Pearson r on Biological Challenge Set (test_bio)\n'
             'Baselines vs PLM-MCNN (five-seed mean ± 95% CI)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.set_ylim(-0.40, 0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.axvspan(5.5, 6.5, alpha=0.07, color='red')
ax.text(6, 0.78, 'K27\n(hardest)', ha='center', fontsize=8,
        color='#cc0000', style='italic')

plt.tight_layout()
plt.savefig('/srv/jupyterlab/workspace/KRAS/figures/fig_model_comparison.png',
            dpi=180, bbox_inches='tight')
plt.show()
print("Saved")