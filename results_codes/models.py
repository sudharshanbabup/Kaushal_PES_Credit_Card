import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve

def bf1(y_true, scores):
    """
    Find the threshold that maximizes the F1 score.
    Returns: (best_threshold, max_f1_score)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1)
    
    # If the precision/recall curve returned a single boundary, return default threshold
    if best_idx >= len(thresholds):
        return 0.5, 0.0
    return thresholds[best_idx], f1[best_idx]

def create_rf_model(seed=42):
    """
    Create a Random Forest model with publication-tuned hyperparameters.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        min_samples_split=10,
        random_state=seed,
        n_jobs=-1
    )

def create_xgb_model(seed=42, scale_pos_weight=1.0):
    """
    Create an XGBoost model with publication-tuned hyperparameters.
    """
    return XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=seed,
        tree_method='hist'
    )

def create_aadnn_model(seed=42):
    """
    Create an Attention Augmented Deep Neural Network (AADNN) MLP classifier.
    """
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=150,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15
    )

def create_ae_model(seed=42):
    """
    Create a bottleneck Deep Autoencoder (AE) MLP regressor.
    """
    return MLPRegressor(
        hidden_layer_sizes=(64, 16, 4, 16, 64),
        activation='relu',
        solver='adam',
        alpha=1e-5,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )

def get_hybrid_score(nn_p, ae_sc, alpha):
    """
    Compute hybrid fused risk score.
    """
    return alpha * nn_p + (1.0 - alpha) * ae_sc
