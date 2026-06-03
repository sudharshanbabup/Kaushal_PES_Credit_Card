import os
import json
import numpy as np

# Import custom utilities and individual figure scripts
from plot_utils import load_pipeline_state, get_figures_dir
from fig01_roc import main as plot_fig01
from fig02_pr import main as plot_fig02
from fig03_alpha import main as plot_fig03
from fig04_ae_dist import main as plot_fig04
from fig05_dynamic import main as plot_fig05
from fig06_confusion import main as plot_fig06
from fig07_shap import main as plot_fig07
from fig08_shap_summary import main as plot_fig08
from fig09_ablation import main as plot_fig09
from fig10_novel import main as plot_fig10
from fig11_robust import main as plot_fig11
from fig12_cost import main as plot_fig12

# Set up paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def cv(o):
    """
    Helper function to convert numpy types to standard python types for JSON serialization.
    """
    if isinstance(o, (np.floating, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: cv(v) for k, v in o.items()}
    if isinstance(o, list):
        return [cv(v) for v in o]
    return o

def main():
    print("="*60)
    print("  Master Script: Running Real Credit Card Dataset Suite")
    print("="*60)
    
    # 1. Load pipeline state (this automatically runs pipeline.py if not cached)
    state = load_pipeline_state()
    
    # 2. Run all figure plotting scripts
    print("\n--- Generating Figures ---")
    plot_fig01()
    plot_fig02()
    plot_fig03()
    plot_fig04()
    plot_fig05()
    plot_fig06()
    plot_fig07()
    plot_fig08()
    plot_fig09()
    plot_fig10()
    plot_fig11()
    plot_fig12()
    print("All 12 figures generated successfully inside real_datasets/figures/.")
    
    # 3. Compile and save final_results.json
    print("\n--- Compiling Final Results ---")
    stats = state['cross_seed_stats']
    astats = state['cross_seed_ablation']
    primary = state['primary_run']
    
    mn = ['RF', 'XGBoost', 'AADNN', 'AE', 'Hybrid']
    mets = ['AUC', 'PRAUC', 'Prec', 'Rec', 'F1']
    ac = ['AADNN_only', 'AE_only', 'Hybrid_static', 'Hybrid_dynamic']
    
    final_res = {
        'stats': {nm: {m: {'mean': stats[nm][m][0], 'std': stats[nm][m][1]} for m in mets} for nm in mn},
        'ablation': {c: {m: {'mean': astats[c][m][0], 'std': astats[c][m][1]} for m in ['F1', 'AUC', 'Rec']} for c in ac},
        'alpha': primary['best_alpha'],
        'novel_fraud': state['novel_fraud'],
        'robustness': state['robustness']['auc_scores'],
        'cost_sensitivity': state['cost_sensitivity']['costs'],
        'timing': primary['timings'],
        'static_f1': state['static_f1'],
        'dynamic_f1': state['dynamic_f1']
    }
    
    results_json_path = os.path.join(RESULTS_DIR, 'final_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(cv(final_res), f, indent=2)
        
    print(f"Results saved to {results_json_path}")
    print("\nReal Dataset Experiment Suite run complete!")
    print("="*60)

if __name__ == '__main__':
    main()
