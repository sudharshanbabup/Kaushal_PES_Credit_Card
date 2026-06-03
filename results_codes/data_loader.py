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

def gen_data(n=40000, seed=42):
    """
    Generate synthetic credit card transaction data.
    Replicates key statistical properties of credit card transaction distributions.
    """
    np.random.seed(seed)
    
    # Calculate fraud and legitimate counts (1.7% fraud rate)
    nf = int(n * 0.017)
    nl = n - nf
    
    # Generate PCA features (V1 to V28)
    V_l = np.random.randn(nl, 28)
    V_f = np.random.randn(nf, 28) * 1.3
    
    # Inject specific mean shifts for fraud cases
    shifts = {
        0: -2.0, 1: 1.2, 2: -2.5, 3: 1.5, 4: -1.0, 
        6: -1.5, 9: -2.0, 11: -2.2, 13: -1.3, 16: -1.0
    }
    for k, v in shifts.items():
        V_f[:, k] += v
        
    # Inject subclass pattern into fraud (e.g., hybrid / unseen structure)
    nh = int(nf * 0.12)
    V_f[:nh] = np.random.randn(nh, 28) * 1.1
    V_f[:nh, 2] -= 0.5
    
    cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Legit transactions df
    dl = pd.DataFrame(
        np.column_stack([
            np.sort(np.random.uniform(0, 172800, nl)),
            V_l,
            np.abs(np.random.lognormal(3.5, 1.5, nl))
        ]),
        columns=cols
    )
    dl['Class'] = 0
    
    # Fraud transactions df
    df_ = pd.DataFrame(
        np.column_stack([
            np.random.uniform(0, 172800, nf),
            V_f,
            np.abs(np.random.lognormal(4.2, 1.8, nf))
        ]),
        columns=cols
    )
    df_['Class'] = 1
    
    # Combine and shuffle
    df = pd.concat([dl, df_], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Hour'] = (df.Time / 3600) % 24
    df['Is_Night'] = ((df.Hour >= 22) | (df.Hour <= 5)).astype(int)
    df['Amount_Log'] = np.log1p(df.Amount)
    df['Amount_Zscore'] = (df.Amount - df.Amount.mean()) / df.Amount.std()
    df['V1_V2_ratio'] = df.V1 / (np.abs(df.V2) + 1e-6)
    df['V_magnitude'] = np.sqrt(sum(df[f'V{i}']**2 for i in range(1, 29)))
    df['High_amount'] = (df.Amount > df.Amount.quantile(0.95)).astype(int)
    df['Amount_rolling_mean'] = df.Amount.rolling(100, min_periods=1).mean()
    df['Amount_deviation'] = df.Amount - df.Amount_rolling_mean
    return df

def attn_f(X):
    """
    Generate cross-group attention-inspired nonlinear features.
    Computes interactions and summaries over subsets of features.
    """
    F = np.zeros((len(X), 8))
    # Nonlinear interactions between different subsets of features
    F[:, 0] = np.mean(X[:, :10] * X[:, 10:20], 1)
    F[:, 1] = np.mean(X[:, 5:15] * X[:, 15:25], 1)
    # Statistical measures across features for each sample
    F[:, 2] = np.std(X, 1)
    F[:, 3] = np.max(np.abs(X), 1)
    # Energy partitions
    F[:, 4] = np.sum(X[:, :14]**2, 1)
    F[:, 5] = np.sum(X[:, 14:28]**2, 1)
    F[:, 6] = np.median(X, 1)
    F[:, 7] = np.sum(np.abs(np.diff(X, axis=1)), 1)
    return F
