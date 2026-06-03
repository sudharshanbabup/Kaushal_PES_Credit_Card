import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from plot_utils import COL_W, set_ieee_style, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    primary = state['primary_run']
    y_te = primary['y_test']
    y_st = primary['y_static']
    y_dy = primary['y_dynamic']
    
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2 + 0.3, 2.4))
    
    cm_st = confusion_matrix(y_te, y_st)
    cm_dy = confusion_matrix(y_te, y_dy)
    
    fn_st = cm_st[1, 0]
    fn_dy = cm_dy[1, 0]
    pct = ((fn_st - fn_dy) / fn_st * 100) if fn_st > 0 else 0
    
    for ax, (cm, lab) in zip(axes, [(cm_st, f'(a) Static $\\tau$'), (cm_dy, f'(b) Dynamic $\\tau$ (FN $\\downarrow${pct:.0f}%)')]):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'],
                    cbar=False, annot_kws={'size': 8}, linewidths=0.5, linecolor='white')
        ax.set(xlabel='Predicted', ylabel='Actual')
        ax.set_title(lab, fontsize=8, pad=4)
        
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig06_confusion.png'))
    plt.close()
    print("fig06_confusion.png generated.")

if __name__ == '__main__':
    main()
