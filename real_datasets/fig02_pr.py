import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from plot_utils import ieee_fig, STYLES, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    primary = state['primary_run']
    y_te = primary['y_test']
    preds = primary['predictions']
    stats = state['cross_seed_stats']
    
    fig, ax = ieee_fig(h=2.6)
    
    for nm in ['RF', 'XGBoost', 'AADNN', 'AE', 'Hybrid']:
        sc_ = preds[nm]
        pr, re, _ = precision_recall_curve(y_te, sc_)
        st = STYLES[nm]
        ax.plot(re, pr, linestyle=st['ls'], color=st['color'], marker=st['marker'],
                markevery=max(1, len(re) // 8), lw=st['lw'], ms=st['lw'] * 3.5,
                label=f"{nm} ({stats[nm]['PRAUC'][0]:.3f})")
                
    ax.set(xlabel='Recall', ylabel='Precision')
    ax.legend(loc='lower left', fontsize=6)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig02_pr.png'))
    plt.close()
    print("fig02_pr.png generated.")

if __name__ == '__main__':
    main()
