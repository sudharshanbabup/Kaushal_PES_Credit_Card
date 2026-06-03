# Credit Card Fraud Detection - Modular Codebase

This directory holds the modular, publication-quality implementation of the Hybrid AADNN-Autoencoder framework for credit card fraud detection. The codebase has been decoupled from a monolithic experiment script into structured, readable modules and standalone figure generators.

## Codebase Structure

```
results_codes/
├── README.md               # Codebase documentation
├── requirements.txt        # Python package dependencies
├── data_loader.py          # Data generation and feature engineering utilities
├── models.py               # Model definitions (RF, XGBoost, AADNN, AE, Hybrid)
├── pipeline.py             # Pipeline runner (trains models, caches results across seeds)
├── plot_utils.py           # Shared IEEE publication-quality plotting styles & cache loader
├── generate_all.py         # Master controller: Runs the pipeline and generates all results/plots
├── cache/                  # (Auto-generated) Cache directory for intermediate outputs
│   └── pipeline_state.pkl  # Pickled experimental state (predictions, models, scores)
├── figures/                # (Auto-generated) Folder containing the 12 generated plots
│   ├── fig01_roc.png       # ROC Curves
│   ├── fig02_pr.png        # Precision-Recall Curves
│   ├── fig03_alpha.png     # Fusion weight alpha optimization
│   ├── fig04_ae_dist.png   # Autoencoder reconstruction error distribution
│   ├── fig05_dynamic.png   # Dynamic threshold vs. fraud rate per window
│   ├── fig06_confusion.png # Confusion matrices (static vs. dynamic)
│   ├── fig07_shap.png      # SHAP feature importance (bar chart)
│   ├── fig08_shap_summary.png # SHAP summary dot plot
│   ├── fig09_ablation.png  # Ablation study metrics comparison
│   ├── fig10_novel.png     # Detection rate on unseen/novel fraud types
│   ├── fig11_robust.png    # Robustness (noise injection) with standard deviation bands
│   └── fig12_cost.png      # Cost sensitivity line plot
└── results/                # (Auto-generated) Folder containing final aggregated scores
    └── final_results.json  # Cross-seed aggregated metric values in JSON format
```

## Setup and Dependencies

To install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Code

### 1. Generate All Figures and Results (Recommended)
To run the entire pipeline (train models across 3 seeds, evaluate all scenarios, compile `final_results.json`, and output all 12 figures):

```bash
python generate_all.py
```

### 2. Generate/Modify a Specific Figure
Each figure script is fully standalone. They use the cached results in `cache/pipeline_state.pkl`. If the cache does not exist, the script will automatically run the training pipeline first to build it, then plot.

For example, to update or view Figure 1 (ROC curve):

```bash
python fig01_roc.py
```

### 3. Re-run Pipeline and Force Retraining
If you modify the models in `models.py` or the data generation parameters in `data_loader.py`, you can force retraining of the pipeline and update the cache by running:

```bash
python pipeline.py
```
This will retrain the models and overwrite the existing cache in `cache/pipeline_state.pkl`. After that, running `generate_all.py` or any figure script will use the new updated cache.
