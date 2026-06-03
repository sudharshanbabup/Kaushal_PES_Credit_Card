import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import COL_W, ieee_fig, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    astats = state['cross_seed_ablation']
    
    fig, ax = ieee_fig(w=COL_W + 0.3, h=2.6)
    
    an = ['AADNN\nOnly', 'AE\nOnly', 'AADNN+AE\n(Static)', 'AADNN+AE\n(Dynamic)']
    ac = ['AADNN_only', 'AE_only', 'Hybrid_static', 'Hybrid_dynamic']
    
    af1 = [astats[c]['F1'][0] for c in ac]
    af1s = [astats[c]['F1'][1] for c in ac]
    aau = [astats[c]['AUC'][0] for c in ac]
    aaus = [astats[c]['AUC'][1] for c in ac]
    arec = [astats[c]['Rec'][0] for c in ac]
    arecs = [astats[c]['Rec'][1] for c in ac]
    
    x = np.arange(4)
    w = 0.22
    
    ax.bar(x - w, af1, w, yerr=af1s, label='$F_1$', color='#2c7bb6', edgecolor='none', capsize=2, error_kw={'lw': 0.6})
    ax.bar(x, aau, w, yerr=aaus, label='AUC', color='#fdae61', edgecolor='none', capsize=2, error_kw={'lw': 0.6})
    ax.bar(x + w, arec, w, yerr=arecs, label='Recall', color='#2ca02c', edgecolor='none', capsize=2, error_kw={'lw': 0.6})
    
    ax.set_xticks(x)
    ax.set_xticklabels(an, fontsize=6)
    ax.set(ylabel='Score')
    ax.set_ylim(0, 1.1)
    ax.legend(ncol=3, loc='upper center', fontsize=6.5)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig09_ablation.png'))
    plt.close()
    print("fig09_ablation.png generated.")

if __name__ == '__main__':
    main()
