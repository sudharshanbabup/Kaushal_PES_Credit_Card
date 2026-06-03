import os
import numpy as np
import pandas as pd

# Define the standard list of feature columns
feat_cols = [
    'Time'
] + [f'V{i}' for i in range(1, 29)] + [
    'Amount', 'Hour', 'Is_Night', 'Amount_Log', 'Amount_Zscore',
    'V1_V2_ratio', 'V_magnitude', 'High_amount', 'Amount_deviation'
]

# Cost sensitivity constant parameters
C_FP = 10     # Fixed investigation cost per false positive ($10)
C_FN_m = 0.5  # False negative cost multiplier applied to transaction amount

def load_real_data(csv_path="creditcard.csv", sample_size=50000, seed=42):
    """
    Load the real credit card dataset (Kaggle creditcard.csv).
    
    If the file is not in the current directory, it searches the parent directories
    and workspace root.
    
    For execution efficiency (especially on local machines), it uses a stratified sample
    of 50,000 records by default. Set sample_size=None to use the full 284,807 transactions.
    """
    import os
    
    # Search for the CSV file in common paths
    possible_paths = [
        csv_path,
        os.path.join("..", csv_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", csv_path),
    ]
    
    found = False
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = p
            found = True
            break
            
    if not found:
        raise FileNotFoundError(
            f"Real credit card dataset '{csv_path}' not found!\n"
            f"Please download 'creditcard.csv' from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)\n"
            f"and place it in the workspace root or inside this 'real_datasets' folder."
        )
        
    print(f"Loading real dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # If sample size is requested, perform stratified sampling to preserve fraud ratio
    if sample_size is not None and sample_size < len(df):
        print(f"Taking a stratified sample of {sample_size} transactions (out of {len(df)}) for execution efficiency...")
        fraud_ratio = df.Class.mean()
        # Stratified split using pandas groupby and sample
        df = df.groupby('Class', group_keys=False).apply(
            lambda x: x.sample(int(np.round(sample_size * (fraud_ratio if x.name == 1 else 1-fraud_ratio))), random_state=seed)
        ).reset_index(drop=True)
    else:
        print(f"Using full dataset of {len(df)} transactions. Note: training may take longer...")
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
    return engineer_features(df)

def engineer_features(df):
    """
    Compute engineered features on top of a raw credit card transactions DataFrame
    which contains columns: 'Time', 'Amount', and 'V1' through 'V28'.
    """
    df = df.copy()
    df['Hour'] = (df.Time / 3600) % 24
    df['Is_Night'] = ((df.Hour >= 22) | (df.Hour <= 5)).astype(int)
    df['Amount_Log'] = np.log1p(df.Amount)
    df['Amount_Zscore'] = (df.Amount - df.Amount.mean()) / (df.Amount.std() + 1e-10)
    df['V1_V2_ratio'] = df.V1 / (np.abs(df.V2) + 1e-6)
    df['V_magnitude'] = np.sqrt(sum(df[f'V{i}']**2 for i in range(1, 29)))
    df['High_amount'] = (df.Amount > df.Amount.quantile(0.95)).astype(int)
    df['Amount_rolling_mean'] = df.Amount.rolling(100, min_periods=1).mean()
    df['Amount_deviation'] = df.Amount - df.Amount_rolling_mean
    return df

def attn_f(X):
    """
    Generate cross-group attention-inspired nonlinear features.
    """
    F = np.zeros((len(X), 8))
    F[:, 0] = np.mean(X[:, :10] * X[:, 10:20], 1)
    F[:, 1] = np.mean(X[:, 5:15] * X[:, 15:25], 1)
    F[:, 2] = np.std(X, 1)
    F[:, 3] = np.max(np.abs(X), 1)
    F[:, 4] = np.sum(X[:, :14]**2, 1)
    F[:, 5] = np.sum(X[:, 14:28]**2, 1)
    F[:, 6] = np.median(X, 1)
    F[:, 7] = np.sum(np.abs(np.diff(X, axis=1)), 1)
    return F
