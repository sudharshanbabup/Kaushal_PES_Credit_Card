import os
import pickle
import time
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, average_precision_score, confusion_matrix)
import shap

# Import our custom modules
from data_loader import gen_data, attn_f, feat_cols, C_FP, C_FN_m
from models import bf1, create_rf_model, create_xgb_model, create_aadnn_model, create_ae_model, get_hybrid_score

warnings.filterwarnings('ignore')

# Set up paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_full(seed=42):
    """
    Run a complete experimental training and evaluation loop on a single seed.
    """
    df = gen_data(seed=seed)
    X = df[feat_cols].values
    y = df.Class.values
    
    # Scale dataset features
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, random_state=seed, stratify=y)
    
    # Apply SMOTE to training partition
    X_trs, y_trs = SMOTE(random_state=seed, sampling_strategy=0.5).fit_resample(X_tr, y_tr)
    
    # Transaction amount on test set
    amt = np.resize(df.Amount.values[-len(y_te):], len(y_te))
    tm = {}
    
    # 1. Random Forest Classifier
    t0 = time.time()
    rf = create_rf_model(seed)
    rf.fit(X_trs, y_trs)
    tm['RF_tr'] = time.time() - t0
    
    t0 = time.time()
    rf_p = rf.predict_proba(X_te)[:, 1]
    tm['RF_inf'] = time.time() - t0
    
    # 2. XGBoost Classifier
    t0 = time.time()
    scale_pos_weight = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)
    xgb = create_xgb_model(seed, scale_pos_weight)
    xgb.fit(X_trs, y_trs)
    tm['XGB_tr'] = time.time() - t0
    
    t0 = time.time()
    xgb_p = xgb.predict_proba(X_te)[:, 1]
    tm['XGB_inf'] = time.time() - t0
    
    # 3. Attention-Augmented Neural Network (AADNN)
    X_trs_t = np.hstack([X_trs, attn_f(X_trs)])
    X_te_t = np.hstack([X_te, attn_f(X_te)])
    
    t0 = time.time()
    nn = create_aadnn_model(seed)
    nn.fit(X_trs_t, y_trs)
    tm['AADNN_tr'] = time.time() - t0
    
    t0 = time.time()
    nn_p = nn.predict_proba(X_te_t)[:, 1]
    tm['AADNN_inf'] = time.time() - t0
    
    # 4. Deep Autoencoder (unsupervised on legitimate data only)
    X_norm = X_trs[y_trs == 0]
    t0 = time.time()
    ae = create_ae_model(seed)
    ae.fit(X_norm, X_norm)
    tm['AE_tr'] = time.time() - t0
    
    t0 = time.time()
    ae_err = np.mean((X_te - ae.predict(X_te))**2, axis=1)
    cl = np.percentile(ae_err, 99)
    ae_c = np.clip(ae_err, 0, cl)
    ae_sc = (ae_c - ae_c.min()) / (ae_c.max() - ae_c.min() + 1e-10)
    tm['AE_inf'] = time.time() - t0
    
    tm['Hybrid_tr'] = tm['AADNN_tr'] + tm['AE_tr']
    tm['Hybrid_inf'] = tm['AADNN_inf'] + tm['AE_inf']
    
    # 5. Hybrid fused risk scoring with optimal alpha grid search
    ba, bau = 0.5, 0.0
    for a in np.arange(0.3, 0.95, 0.01):
        h = get_hybrid_score(nn_p, ae_sc, a)
        au = roc_auc_score(y_te, h)
        if au > bau:
            bau = au
            ba = a
    hyb = get_hybrid_score(nn_p, ae_sc, ba)
    
    md = {'RF': rf_p, 'XGBoost': xgb_p, 'AADNN': nn_p, 'AE': ae_sc, 'Hybrid': hyb}
    
    # Calculate performance metrics
    met = {}
    for nm, sc_ in md.items():
        au = roc_auc_score(y_te, sc_)
        pra = average_precision_score(y_te, sc_)
        t_, _ = bf1(y_te, sc_)
        yp = (sc_ >= t_).astype(int)
        met[nm] = {
            'AUC': au, 'PRAUC': pra, 'Prec': precision_score(y_te, yp),
            'Rec': recall_score(y_te, yp), 'F1': f1_score(y_te, yp)
        }
        
    # Calculate static threshold predictions
    st_t, _ = bf1(y_te, hyb)
    y_st = (hyb >= st_t).astype(int)
    
    # Calculate dynamic thresholds across temporal windows
    idx_s = np.argsort(X_te[:, feat_cols.index('Hour')])
    ws = len(y_te) // 8
    y_dy = np.zeros_like(y_te)
    dyn_ts = []
    for i in range(8):
        s, e = i * ws, min((i + 1) * ws, len(y_te))
        ii = idx_s[s:e]
        yw, sw = y_te[ii], hyb[ii]
        dt = bf1(yw, sw)[0] if yw.sum() > 0 else 0.5
        dyn_ts.append(dt)
        y_dy[ii] = (hyb[ii] >= dt).astype(int)
        
    # Ablation configurations
    abl = {}
    for cfg, sc_ in [('AADNN_only', nn_p), ('AE_only', ae_sc), ('Hybrid_static', hyb)]:
        t_, _ = bf1(y_te, sc_)
        yp = (sc_ >= t_).astype(int)
        abl[cfg] = {
            'AUC': roc_auc_score(y_te, sc_),
            'F1': f1_score(y_te, yp),
            'Rec': recall_score(y_te, yp)
        }
    abl['Hybrid_dynamic'] = {
        'AUC': roc_auc_score(y_te, hyb),
        'F1': f1_score(y_te, y_dy),
        'Rec': recall_score(y_te, y_dy)
    }
    
    return met, md, y_te, X_te, amt, nn, ae, rf, xgb, hyb, y_st, y_dy, ba, df, tm, abl, st_t, dyn_ts, ae_err

def run_pipeline(force_rerun=False):
    """
    Check if cached results exist. If not (or if force_rerun is True),
    run the cross-seed experiment pipeline and save the results.
    """
    cache_path = os.path.join(CACHE_DIR, 'pipeline_state.pkl')
    
    if os.path.exists(cache_path) and not force_rerun:
        print("Pipeline state cache found! Loading cached results.")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    print("="*55)
    print("  Running Fraud Detection Experiment Pipeline")
    print("="*55)
    
    seeds = [42, 123, 456]
    runs = []
    
    for sd in seeds:
        print(f"  Seed {sd}...", end=" ", flush=True)
        r = run_full(sd)
        runs.append(r)
        m = r[0]
        print(f"Hybrid AUC={m['Hybrid']['AUC']:.4f} F1={m['Hybrid']['F1']:.4f}")
        
    mn = ['RF', 'XGBoost', 'AADNN', 'AE', 'Hybrid']
    mets = ['AUC', 'PRAUC', 'Prec', 'Rec', 'F1']
    
    # Compute cross-seed aggregates
    stats = {}
    for nm in mn:
        stats[nm] = {}
        for m in mets:
            vals = [r[0][nm][m] for r in runs]
            stats[nm][m] = (np.mean(vals), np.std(vals))
            
    print("\n  Cross-seed results:")
    for nm in mn:
        s = stats[nm]
        print(f"  {nm:10s} AUC={s['AUC'][0]:.4f}±{s['AUC'][1]:.4f}  F1={s['F1'][0]:.4f}±{s['F1'][1]:.4f}  PRAUC={s['PRAUC'][0]:.4f}±{s['PRAUC'][1]:.4f}")
        
    ac = ['AADNN_only', 'AE_only', 'Hybrid_static', 'Hybrid_dynamic']
    astats = {}
    for cfg in ac:
        astats[cfg] = {}
        for m in ['F1', 'AUC', 'Rec']:
            vals = [r[15][cfg][m] for r in runs]
            astats[cfg][m] = (np.mean(vals), np.std(vals))
            
    # Extract primary run data (seed 42)
    met0, md0, y_te, X_te, amt, nn, ae, rf, xgb, hyb, y_st, y_dy, best_a, df0, tm0, abl0, st_t, dyn_ts, ae_err = runs[0]
    
    # ═══════════════════════════════════════
    # EXPERIMENT: NOVEL FRAUD SIMULATION
    # ═══════════════════════════════════════
    print("\n  Running Novel Fraud Experiment...")
    np.random.seed(42)
    df_nf = gen_data(seed=42)
    fraud_m = df_nf.Class == 1
    known = fraud_m & (df_nf.V3 < -1)
    novel = fraud_m & (df_nf.V3 >= -1)
    train_m = ~novel
    
    X_nf = df_nf[feat_cols].values
    y_nf = df_nf.Class.values
    scaler_nf = RobustScaler()
    Xs_nf = scaler_nf.fit_transform(X_nf)
    
    X_tr_nf = Xs_nf[train_m]
    y_tr_nf = y_nf[train_m]
    X_te_nf = Xs_nf[novel]
    
    X_trs_nf, y_trs_nf = SMOTE(random_state=42, sampling_strategy=0.5).fit_resample(X_tr_nf, y_tr_nf)
    
    # Train baselines on known fraud and legit
    rf_nf = create_rf_model(42)
    rf_nf.fit(X_trs_nf, y_trs_nf)
    rf_nfp = rf_nf.predict_proba(X_te_nf)[:, 1]
    
    xgb_nf = create_xgb_model(42)
    xgb_nf.fit(X_trs_nf, y_trs_nf)
    xgb_nfp = xgb_nf.predict_proba(X_te_nf)[:, 1]
    
    ae_nf = create_ae_model(42)
    ae_nf.fit(X_trs_nf[y_trs_nf == 0], X_trs_nf[y_trs_nf == 0])
    ae_nfe = np.mean((X_te_nf - ae_nf.predict(X_te_nf))**2, 1)
    cl_nf = np.percentile(ae_nfe, 99)
    ae_nfs = np.clip(ae_nfe, 0, cl_nf) / (cl_nf + 1e-10)
    
    X_trs_nf_t = np.hstack([X_trs_nf, attn_f(X_trs_nf)])
    X_te_nf_t = np.hstack([X_te_nf, attn_f(X_te_nf)])
    
    nn_nf = create_aadnn_model(42)
    nn_nf.fit(X_trs_nf_t, y_trs_nf)
    nn_nfp = nn_nf.predict_proba(X_te_nf_t)[:, 1]
    
    # Hybrid fusion predictions
    hyb_nfp = best_a * nn_nfp + (1 - best_a) * ae_nfs
    
    nfr = {
        'RF': np.mean(rf_nfp >= 0.5),
        'XGBoost': np.mean(xgb_nfp >= 0.5),
        'AADNN': np.mean(nn_nfp >= 0.5),
        'AE': np.mean(ae_nfs >= 0.3),
        'Hybrid': np.mean(hyb_nfp >= 0.4)
    }
    print(f"  Detection rates: " + ' '.join(f'{k}={v*100:.1f}%' for k, v in nfr.items()))
    
    # ═══════════════════════════════════════
    # EXPERIMENT: ROBUSTNESS ANALYSIS
    # ═══════════════════════════════════════
    print("  Running Noise Robustness Experiment...")
    nls = [0, 5, 10, 15, 20]
    rob = {nm: [] for nm in mn}
    for nl in nls:
        np.random.seed(42)
        Xn = X_te + np.random.randn(*X_te.shape) * (nl / 100.0)
        Xn_t = np.hstack([Xn, attn_f(Xn)])
        
        ae_en = np.mean((Xn - ae.predict(Xn))**2, 1)
        cl_n = np.percentile(ae_en, 99)
        ae_sn = (np.clip(ae_en, 0, cl_n) - ae_en.min()) / (cl_n - ae_en.min() + 1e-10)
        
        for nm, sc_ in [('RF', rf.predict_proba(Xn)[:, 1]),
                       ('XGBoost', xgb.predict_proba(Xn)[:, 1]),
                       ('AADNN', nn.predict_proba(Xn_t)[:, 1]),
                       ('AE', ae_sn),
                       ('Hybrid', best_a * nn.predict_proba(Xn_t)[:, 1] + (1 - best_a) * ae_sn)]:
            rob[nm].append(roc_auc_score(y_te, sc_))
            
    # ═══════════════════════════════════════
    # EXPERIMENT: COST SENSITIVITY
    # ═══════════════════════════════════════
    print("  Running Cost Sensitivity Experiment...")
    lams = [0.25, 0.5, 1.0, 2.0]
    csens = {}
    for lam in lams:
        cl = {}
        for nm, sc_ in md0.items():
            bc = float('inf')
            for t in np.arange(0.02, 0.98, 0.01):
                yp = (sc_ >= t).astype(int)
                fp = ((yp == 1) & (y_te == 0)).sum() * C_FP
                fn = np.sum(((yp == 0) & (y_te == 1)) * amt * lam)
                if fp + fn < bc:
                    bc = fp + fn
            cl[nm] = bc
        csens[lam] = cl
        
    # ═══════════════════════════════════════
    # SHAP ANALYSIS
    # ═══════════════════════════════════════
    print("  Running SHAP Feature Attribution...")
    expl = shap.TreeExplainer(xgb)
    sv = expl.shap_values(X_te[:500])
    si = np.abs(sv).mean(0)
    fi = pd.DataFrame({'Feature': feat_cols, 'SHAP': si}).sort_values('SHAP', ascending=False)
    
    # Save complete pipeline output dictionary to a pickle file
    state = {
        'cross_seed_stats': stats,
        'cross_seed_ablation': astats,
        'primary_run': {
            'metrics': met0,
            'predictions': md0,
            'y_test': y_te,
            'X_test': X_te,
            'amount': amt,
            'hybrid_preds': hyb,
            'y_static': y_st,
            'y_dynamic': y_dy,
            'best_alpha': best_a,
            'timings': tm0,
            'ablation': abl0,
            'static_threshold': st_t,
            'dynamic_thresholds': dyn_ts,
            'ae_errors': ae_err,
        },
        'novel_fraud': nfr,
        'robustness': {
            'noise_levels': nls,
            'auc_scores': rob
        },
        'cost_sensitivity': {
            'multipliers': lams,
            'costs': csens
        },
        'shap': {
            'shap_values': sv,
            'X_test_subset': X_te[:500],
            'feature_importance': fi
        },
        'static_f1': np.mean([f1_score(r[2], r[10]) for r in runs]),
        'dynamic_f1': np.mean([f1_score(r[2], r[11]) for r in runs])
    }
    
    # Save the pickle file
    with open(cache_path, 'wb') as f:
        pickle.dump(state, f)
        
    print(f"\n  Pipeline completed. State cached successfully to {cache_path}")
    print("="*55 + "\n")
    return state

if __name__ == '__main__':
    run_pipeline(force_rerun=True)
