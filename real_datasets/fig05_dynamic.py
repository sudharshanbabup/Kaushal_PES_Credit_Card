import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import COL_W, set_ieee_style, load_pipeline_state, get_figures_dir
from data_loader import feat_cols

def main():
    state = load_pipeline_state()
    primary = state['primary_run']
    X_te = primary['X_test']
    y_te = primary['y_test']
    actual_dts = primary['dynamic_thresholds']
    st_t = primary['static_threshold']
    
    set_ieee_style()
    fig, ax1 = plt.subplots(figsize=(COL_W + 0.4, 2.5))
    
    # Compute actual fraud rates per window
    idx_s = np.argsort(X_te[:, feat_cols.index('Hour')])
    ws = len(y_te) // 8
    actual_fr = []
    for i in range(8):
        s, e = i * ws, min((i + 1) * ws, len(y_te))
        ii = idx_s[s:e]
        actual_fr.append(y_te[ii].mean() * 100)
        
    xw = np.arange(8)
    lw_lab = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8']
    
    ax1.bar(xw, actual_dts, color='#2c7bb6', alpha=0.7, edgecolor='white', width=0.55, label='Dynamic $\\tau$', zorder=2)
    ax1.axhline(st_t, color='k', ls='--', lw=0.8, label=f'Static $\\tau$={st_t:.3f}', zorder=3)
    
    for b, v in zip(ax1.patches, actual_dts):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.2f}', ha='center', va='bottom', fontsize=5.5)
        
    ax1.set_xticks(xw)
    ax1.set_xticklabels(lw_lab, fontsize=6.5)
    ax1.set(xlabel='Temporal Window', ylabel='Threshold ($\\tau$)')
    
    ax2 = ax1.twinx()
    ax2.plot(xw, actual_fr, 's-', color='#d7191c', ms=4, lw=0.9, label='Fraud rate (%)', zorder=4)
    ax2.set_ylabel('Fraud Rate (%)', color='#d7191c', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='#d7191c')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=5.5)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig05_dynamic.png'))
    plt.close()
    print("fig05_dynamic.png generated.")

if __name__ == '__main__':
    main()
