import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot_utils import ieee_fig, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    primary = state['primary_run']
    y_te = primary['y_test']
    ae_err = primary['ae_errors']
    
    fig, ax = ieee_fig(h=2.4)
    
    cl97 = np.percentile(ae_err, 97)
    ae_l = np.clip(ae_err[y_te == 0], 0, cl97)
    ae_f = np.clip(ae_err[y_te == 1], 0, cl97)
    
    bins = np.linspace(0, cl97, 45)
    ax.hist(ae_l, bins=bins, alpha=0.35, density=True, color='#2c7bb6', edgecolor='none', label='Legitimate')
    ax.hist(ae_f, bins=bins, alpha=0.35, density=True, color='#d7191c', edgecolor='none', label='Fraud')
    
    xs = np.linspace(0, cl97, 300)
    if len(ae_l) > 10:
        ax.plot(xs, gaussian_kde(ae_l)(xs), color='#2c7bb6', lw=1.2)
    if len(ae_f) > 10:
        ax.plot(xs, gaussian_kde(ae_f)(xs), color='#d7191c', lw=1.2)
        
    ax.set(xlabel='Reconstruction Error (MSE)', ylabel='Density')
    ax.legend(loc='upper right')
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig04_ae_dist.png'))
    plt.close()
    print("fig04_ae_dist.png generated.")

if __name__ == '__main__':
    main()
