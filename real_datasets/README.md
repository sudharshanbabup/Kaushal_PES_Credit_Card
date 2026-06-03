# Real Credit Card Dataset Experimentation Suite

This directory contains the modular pipeline specifically configured to run on the real **European Cardholder credit card dataset** (`creditcard.csv`).

## Prerequisites

1. **Download the Dataset**:
   You need to download the official dataset from Kaggle:
   - [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   
2. **Place the File**:
   Extract and rename the downloaded file to `creditcard.csv` and place it in:
   - The workspace root directory (which contains `experiment_code.py`), OR
   - Directly inside this `real_datasets` folder.

## Structure

The files in this folder are structurally identical to the synthetic version, but are connected to the `load_real_data` pipeline in `data_loader.py` to ingest and process the real CSV data.

```
real_datasets/
├── README.md               # This documentation
├── data_loader.py          # Loads, samples, and engineers real CSV dataset
├── models.py               # Model configurations (RF, XGBoost, AADNN, AE, Hybrid)
├── pipeline.py             # Pipeline runner (runs cross-seed loops, novel fraud simulation)
├── plot_utils.py           # Matplotlib publication settings
├── generate_all.py         # Master controller: Runs pipeline and plots all figures
├── cache/                  # (Auto-generated) Pickled cache outputs
└── figures/                # (Auto-generated) Real dataset figures
```

## How to Run

### Stratified Sampling (Default - Fast)
The real dataset contains 284,807 transactions. Training neural networks and computing SHAP values over multiple seeds on the full dataset can take several minutes on local CPUs. 

By default, the pipeline uses a **stratified sample of 50,000 transactions** (preserving the class imbalance ratio). This runs in approximately 1-2 minutes and accurately represents the model's behavior.

To execute the suite with sampling:
```bash
python generate_all.py
```

### Running on the Full Dataset
If you want to train on the complete 284,807 rows, open [real_datasets/pipeline.py](file:///Users/sudharshanbabupandava/JioCloud/Research_outside%20/Kaushal/Kaushal_PES_Credit_Card%20%282%29/real_datasets/pipeline.py) and change the default `sample_size` to `None` in `run_pipeline(sample_size=None)`. Then execute:
```bash
python generate_all.py
```
*(Warning: This may take several minutes to run all seeds).*
