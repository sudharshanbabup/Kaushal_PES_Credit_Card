import os
import matplotlib.pyplot as plt
import shap
from plot_utils import COL_W, load_pipeline_state, get_figures_dir
from data_loader import feat_cols

def main():
    state = load_pipeline_state()
    shap_data = state['shap']
    sv = shap_data['shap_values']
    X_te_subset = shap_data['X_test_subset']
    fi = shap_data['feature_importance']
    
    fig = plt.figure(figsize=(COL_W + 0.8, 3.2))
    t10 = [feat_cols.index(f) for f in fi.head(10).Feature.values]
    
    shap.summary_plot(
        sv[:, t10], 
        X_te_subset[:, t10], 
        feature_names=fi.head(10).Feature.tolist(),
        show=False, 
        plot_size=None, 
        max_display=10
    )
    plt.xlabel('SHAP Value', fontsize=8)
    
    fig_dir = get_figures_dir()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig08_shap_summary.png'))
    plt.close()
    print("fig08_shap_summary.png generated.")

if __name__ == '__main__':
    main()
