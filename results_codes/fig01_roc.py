import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
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
        fpr, tpr, _ = roc_curve(y_te, sc_)
        st = STYLES[nm]
        ax.plot(fpr, tpr, linestyle=st['ls'], color=st['color'], marker=st['marker'],
                markevery=max(1, len(fpr) // 8), lw=st['lw'], ms=st['lw'] * 3.5,
                label=f"{nm} ({stats[nm]['AUC'][0]:.3f})")
                
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3, lw=0.5)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig01_roc.png'))
    plt.close()
    print("fig01_roc.png generated.")

if __name__ == '__main__':
    main()
