import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from plot_utils import ieee_fig, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    primary = state['primary_run']
    y_te = primary['y_test']
    nn_p = primary['predictions']['AADNN']
    ae_sc = primary['predictions']['AE']
    
    fig, ax = ieee_fig(h=2.2)
    alphas = np.arange(0.30, 0.95, 0.01)
    aucs_a = [roc_auc_score(y_te, a * nn_p + (1.0 - a) * ae_sc) for a in alphas]
    
    ax.plot(alphas, aucs_a, '-', color='k', lw=1.0, marker='o', ms=2, markevery=3)
    bi = np.argmax(aucs_a)
    ax.axvline(alphas[bi], color='#d62728', ls='--', lw=0.8)
    ax.annotate(f'$\\alpha^*={alphas[bi]:.2f}$', xy=(alphas[bi], aucs_a[bi]),
                xytext=(alphas[bi] - 0.15, aucs_a[bi] - 0.002), fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.7))
    ax.set(xlabel='$\\alpha$ (AADNN weight)', ylabel='ROC-AUC')
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig03_alpha.png'))
    plt.close()
    print("fig03_alpha.png generated.")

if __name__ == '__main__':
    main()
