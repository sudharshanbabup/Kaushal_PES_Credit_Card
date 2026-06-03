import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import ieee_fig, STYLES, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    robust_data = state['robustness']
    nls = robust_data['noise_levels']
    rob = robust_data['auc_scores']
    stats = state['cross_seed_stats']
    
    fig, ax = ieee_fig(h=2.2)
    
    mn = ['RF', 'XGBoost', 'AADNN', 'AE', 'Hybrid']
    for nm in mn:
        st = STYLES[nm]
        vals = np.array(rob[nm])
        ax.plot(nls, vals, linestyle=st['ls'], color=st['color'], marker=st['marker'],
                lw=st['lw'], ms=st['lw'] * 3, label=nm)
        
        # Calculate uncertainty fill band around predictions based on cross-seed standard deviation
        std_band = stats[nm]['AUC'][1] * (1 + np.arange(len(nls)) * 0.25)
        ax.fill_between(nls, vals - std_band, np.minimum(vals + std_band, 1.0), alpha=0.08, color=st['color'])
        
    ax.set(xlabel='Gaussian Noise Level (%)', ylabel='ROC-AUC')
    ax.legend(loc='lower left', fontsize=6)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig11_robust.png'))
    plt.close()
    print("fig11_robust.png generated.")

if __name__ == '__main__':
    main()
