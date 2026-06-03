import os
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import ieee_fig, load_pipeline_state, get_figures_dir

def main():
    state = load_pipeline_state()
    fi = state['shap']['feature_importance']
    
    fig, ax = ieee_fig(h=2.8)
    top = fi.head(12)
    ax.barh(range(12), top.SHAP.values[::-1], color='#2c7bb6', edgecolor='none', height=0.65)
    ax.set_yticks(range(12))
    ax.set_yticklabels(top.Feature.values[::-1], fontsize=6.5)
    ax.set(xlabel='Mean |SHAP Value|')
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig07_shap.png'))
    plt.close()
    print("fig07_shap.png generated.")

if __name__ == '__main__':
    main()
