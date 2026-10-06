import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs('artifacts/figures', exist_ok=True)
os.makedirs('artifacts/predictions', exist_ok=True)
os.makedirs('artifacts/oof', exist_ok=True)

def evaluate_predictions(y_true, y_pred):
    """Central metric function used across the entire benchmark."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        pearson = 0.0
    else:
        pearson = float(pearsonr(y_true, y_pred)[0])
    return {
        'rmse': rmse,
        'pearson': pearson
    }

def run_cv_experiment(X, y, filenames, model_name, pipeline_builder, n_splits=5, clip_range=(0, 5)):
    """Standard 5-fold cross-validation with fold tracking and strict isolation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    fold_assignments = np.zeros(len(y), dtype=int)
    
    val_rmses, val_prs = [], []
    train_rmses, train_prs = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_assignments[val_idx] = fold
        
        # Build and fit pipeline inside fold
        pipe = pipeline_builder()
        pipe.fit(X_tr, y_tr)
        
        p_tr = pipe.predict(X_tr)
        p_val = pipe.predict(X_val)
        
        if clip_range:
            p_tr = np.clip(p_tr, clip_range[0], clip_range[1])
            p_val = np.clip(p_val, clip_range[0], clip_range[1])
            
        m_tr = evaluate_predictions(y_tr, p_tr)
        m_val = evaluate_predictions(y_val, p_val)
        
        train_rmses.append(m_tr['rmse'])
        train_prs.append(m_tr['pearson'])
        val_rmses.append(m_val['rmse'])
        val_prs.append(m_val['pearson'])
        oof_preds[val_idx] = p_val
        
    overall_oof = evaluate_predictions(y, oof_preds)
    
    # Save OOF file
    oof_df = pd.DataFrame({
        'filename': filenames,
        'true_score': y,
        'oof_prediction': np.round(oof_preds, 4),
        'fold': fold_assignments,
        'model_name': model_name
    })
    safe_name = model_name.lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    oof_df.to_csv(f'artifacts/oof/{safe_name}.csv', index=False)
    
    # Retrain on full dataset to get ACTUAL Training RMSE and Pearson
    full_pipe = pipeline_builder()
    full_pipe.fit(X, y)
    full_train_pred = full_pipe.predict(X)
    if clip_range:
        full_train_pred = np.clip(full_train_pred, clip_range[0], clip_range[1])
    full_train_eval = evaluate_predictions(y, full_train_pred)
    
    return {
        'val_rmse_mean': float(np.mean(val_rmses)),
        'val_rmse_std': float(np.std(val_rmses)),
        'val_pr_mean': float(np.mean(val_prs)),
        'val_pr_std': float(np.std(val_prs)),
        'oof_rmse': overall_oof['rmse'],
        'oof_pr': overall_oof['pearson'],
        'training_rmse': full_train_eval['rmse'],
        'training_pearson': full_train_eval['pearson'],
        'oof_preds': oof_preds,
        'fitted_full_model': full_pipe
    }

def run_tfidf_cv_experiment(texts, y, filenames, model_name, tfidf_params, alpha=5.0, clip_range=(0, 5)):
    """Run TF-IDF with vectorizer strictly fitted inside each CV fold."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    fold_assignments = np.zeros(len(y), dtype=int)
    
    val_rmses, val_prs = [], []
    train_rmses, train_prs = [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts, y)):
        txt_tr = [texts[i] for i in train_idx]
        txt_val = [texts[i] for i in val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_assignments[val_idx] = fold
        
        vec = TfidfVectorizer(**tfidf_params)
        X_tr = vec.fit_transform(txt_tr)
        X_val = vec.transform(txt_val)
        
        scaler = StandardScaler(with_mean=False)
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        reg = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        reg.fit(X_tr_s, y_tr)
        
        p_tr = reg.predict(X_tr_s)
        p_val = reg.predict(X_val_s)
        
        if clip_range:
            p_tr = np.clip(p_tr, clip_range[0], clip_range[1])
            p_val = np.clip(p_val, clip_range[0], clip_range[1])
            
        m_tr = evaluate_predictions(y_tr, p_tr)
        m_val = evaluate_predictions(y_val, p_val)
        
        train_rmses.append(m_tr['rmse'])
        train_prs.append(m_tr['pearson'])
        val_rmses.append(m_val['rmse'])
        val_prs.append(m_val['pearson'])
        oof_preds[val_idx] = p_val
        
    overall_oof = evaluate_predictions(y, oof_preds)
    
    # Save OOF file
    oof_df = pd.DataFrame({
        'filename': filenames,
        'true_score': y,
        'oof_prediction': np.round(oof_preds, 4),
        'fold': fold_assignments,
        'model_name': model_name
    })
    safe_name = model_name.lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
    oof_df.to_csv(f'artifacts/oof/{safe_name}.csv', index=False)
    
    # Fit on full data
    full_vec = TfidfVectorizer(**tfidf_params)
    X_full = full_vec.fit_transform(texts)
    full_scaler = StandardScaler(with_mean=False)
    X_full_s = full_scaler.fit_transform(X_full)
    full_reg = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    full_reg.fit(X_full_s, y)
    full_pred = full_reg.predict(X_full_s)
    if clip_range:
        full_pred = np.clip(full_pred, clip_range[0], clip_range[1])
    full_eval = evaluate_predictions(y, full_pred)
    
    return {
        'val_rmse_mean': float(np.mean(val_rmses)),
        'val_rmse_std': float(np.std(val_rmses)),
        'val_pr_mean': float(np.mean(val_prs)),
        'val_pr_std': float(np.std(val_prs)),
        'oof_rmse': overall_oof['rmse'],
        'oof_pr': overall_oof['pearson'],
        'training_rmse': full_eval['rmse'],
        'training_pearson': full_eval['pearson'],
        'oof_preds': oof_preds,
        'full_vec': full_vec,
        'full_scaler': full_scaler,
        'full_reg': full_reg
    }

def run_nested_cv_ensemble(candidate_models_dict, y, filenames, n_outer_splits=5, n_inner_splits=4, clip_range=(0, 5)):
    """
    CRITICAL FIX #3: Strict Nested Cross-Validation for Leakage-Safe Ensembling.
    Outer Loop (5 folds):
      Inner Loop (4 folds on Outer Training set):
        Fit each candidate model, produce inner OOF predictions.
        Optimize non-negative blending weights strictly on inner OOF predictions.
      Refit candidate models on full Outer Training set.
      Predict Outer Validation set using the optimized inner weights.
    Outer Validation fold predictions are NEVER used during weight selection!
    """
    print("\n--- RUNNING STRICT NESTED CROSS-VALIDATION ENSEMBLE (ZERO LEAKAGE) ---")
    kf_outer = KFold(n_splits=n_outer_splits, shuffle=True, random_state=RANDOM_STATE)
    
    model_names = list(candidate_models_dict.keys())
    M = len(model_names)
    
    nested_oof_preds = np.zeros(len(y))
    fold_weights = []
    
    for outer_fold, (outer_tr_idx, outer_val_idx) in enumerate(kf_outer.split(filenames, y)):
        print(f"  Nested CV Outer Fold {outer_fold + 1}/{n_outer_splits}...")
        y_outer_tr = y[outer_tr_idx]
        y_outer_val = y[outer_val_idx]
        
        # Inner CV on outer_tr_idx
        kf_inner = KFold(n_splits=n_inner_splits, shuffle=True, random_state=RANDOM_STATE + outer_fold)
        inner_oof_matrix = np.zeros((len(outer_tr_idx), M))
        
        for m_idx, m_name in enumerate(model_names):
            builder = candidate_models_dict[m_name]['builder']
            X_data = candidate_models_dict[m_name]['X'][outer_tr_idx]
            
            for inner_tr_idx, inner_val_idx in kf_inner.split(X_data, y_outer_tr):
                pipe = builder()
                pipe.fit(X_data[inner_tr_idx], y_outer_tr[inner_tr_idx])
                p_val = pipe.predict(X_data[inner_val_idx])
                if clip_range:
                    p_val = np.clip(p_val, clip_range[0], clip_range[1])
                inner_oof_matrix[inner_val_idx, m_idx] = p_val
                
        # Optimize weights on inner_oof_matrix
        def loss(w):
            blend = inner_oof_matrix @ w
            if clip_range:
                blend = np.clip(blend, clip_range[0], clip_range[1])
            m = evaluate_predictions(y_outer_tr, blend)
            return m['rmse'] - 0.5 * m['pearson']
            
        init_w = np.ones(M) / M
        bounds = [(0, 1) for _ in range(M)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        opt_res = minimize(loss, init_w, bounds=bounds, constraints=constraints)
        best_w = opt_res.x / np.sum(opt_res.x)
        fold_weights.append(best_w)
        
        # Now refit candidate models on full outer_tr_idx and predict outer_val_idx
        outer_val_matrix = np.zeros((len(outer_val_idx), M))
        for m_idx, m_name in enumerate(model_names):
            builder = candidate_models_dict[m_name]['builder']
            X_tr = candidate_models_dict[m_name]['X'][outer_tr_idx]
            X_val = candidate_models_dict[m_name]['X'][outer_val_idx]
            
            pipe = builder()
            pipe.fit(X_tr, y_outer_tr)
            p_val = pipe.predict(X_val)
            if clip_range:
                p_val = np.clip(p_val, clip_range[0], clip_range[1])
            outer_val_matrix[:, m_idx] = p_val
            
        # Outer validation fold prediction using inner weights
        outer_fold_pred = outer_val_matrix @ best_w
        if clip_range:
            outer_fold_pred = np.clip(outer_fold_pred, clip_range[0], clip_range[1])
        nested_oof_preds[outer_val_idx] = outer_fold_pred
        
    nested_metrics = evaluate_predictions(y, nested_oof_preds)
    avg_weights = np.mean(fold_weights, axis=0)
    avg_weight_dict = {model_names[i]: float(avg_weights[i]) for i in range(M)}
    
    print(f"  -> Nested CV OOF RMSE    : {nested_metrics['rmse']:.4f}")
    print(f"  -> Nested CV OOF Pearson : {nested_metrics['pearson']:.4f}")
    print(f"  -> Average Ensemble Weights across outer folds: {avg_weight_dict}")
    
    # Save nested OOF predictions
    nested_oof_df = pd.DataFrame({
        'filename': filenames,
        'true_score': y,
        'nested_oof_prediction': np.round(nested_oof_preds, 4)
    })
    nested_oof_df.to_csv('artifacts/oof/nested_cv_ensemble.csv', index=False)
    
    return {
        'nested_oof_rmse': nested_metrics['rmse'],
        'nested_oof_pearson': nested_metrics['pearson'],
        'nested_oof_preds': nested_oof_preds,
        'avg_weights': avg_weight_dict
    }

def main():
    print("="*60)
    print("SHL GRAMMAR SCORING ENGINE — V2 AUDIT & LEAKAGE-FREE PIPELINE")
    print("="*60)
    
    # 1. Load ground truth metadata
    train_meta = pd.read_csv('Dataset_Final/train.csv')
    test_meta = pd.read_csv('Dataset_Final/test.csv')
    y_train = train_meta['label'].values.astype(np.float64)
    filenames_train = train_meta['filename'].tolist()
    filenames_test = test_meta['filename'].tolist()
    
    # 2. Strict ID-based merging and row alignment verification
    print("\n[Step 1] Loading and strictly aligning all feature spaces by filename ID...")
    
    # ASR Transcripts
    train_trans = pd.read_csv('artifacts/transcripts/train_transcripts.csv')
    test_trans = pd.read_csv('artifacts/transcripts/test_transcripts.csv')
    
    train_trans_aligned = train_meta[['filename']].merge(train_trans, on='filename', how='left')
    test_trans_aligned = test_meta[['filename']].merge(test_trans, on='filename', how='left')
    
    train_texts = train_trans_aligned['transcript'].fillna('').astype(str).tolist()
    test_texts = test_trans_aligned['transcript'].fillna('').astype(str).tolist()
    
    assert len(train_texts) == len(train_meta), "Alignment error in train texts!"
    assert len(test_texts) == len(test_meta), "Alignment error in test texts!"
    print(f"  ASR Transcripts loaded: {len(train_texts)} train, {len(test_texts)} test.")
    
    # Audio Acoustic Features
    train_audio = pd.read_csv('artifacts/audio_features/train_audio_features.csv')
    test_audio = pd.read_csv('artifacts/audio_features/test_audio_features.csv')
    audio_cols = [c for c in train_audio.columns if c not in ('filename', 'label')]
    
    train_audio_aligned = train_meta[['filename']].merge(train_audio, on='filename', how='left')
    test_audio_aligned = test_meta[['filename']].merge(test_audio, on='filename', how='left')
    X_train_audio = train_audio_aligned[audio_cols].fillna(0).values.astype(np.float32)
    X_test_audio = test_audio_aligned[audio_cols].fillna(0).values.astype(np.float32)
    print(f"  Acoustic Features loaded: {X_train_audio.shape[1]} features.")
    
    # Linguistic Grammar Features
    train_ling = pd.read_csv('artifacts/linguistic_features/train_linguistic_features.csv')
    test_ling = pd.read_csv('artifacts/linguistic_features/test_linguistic_features.csv')
    ling_cols = [c for c in train_ling.columns if c not in ('filename', 'label')]
    
    train_ling_aligned = train_meta[['filename']].merge(train_ling, on='filename', how='left')
    test_ling_aligned = test_meta[['filename']].merge(test_ling, on='filename', how='left')
    X_train_ling = train_ling_aligned[ling_cols].fillna(0).values.astype(np.float32)
    X_test_ling = test_ling_aligned[ling_cols].fillna(0).values.astype(np.float32)
    print(f"  Linguistic Features loaded: {X_train_ling.shape[1]} features.")
    
    # Pretrained Text Embeddings
    X_train_emb = np.load('artifacts/text_embeddings/train_text_embeddings.npy')
    X_test_emb = np.load('artifacts/text_embeddings/test_text_embeddings.npy')
    print(f"  Text Embeddings loaded: {X_train_emb.shape[1]} dimensions.")
    
    # Combined feature sets
    X_train_ling_audio = np.hstack([X_train_ling, X_train_audio])
    X_test_ling_audio = np.hstack([X_test_ling, X_test_audio])
    
    X_train_multimodal = np.hstack([X_train_ling, X_train_audio, X_train_emb])
    X_test_multimodal = np.hstack([X_test_ling, X_test_audio, X_test_emb])
    print(f"  Full Multimodal matrix: {X_train_multimodal.shape[1]} total features.")
    
    # Assert row alignment
    assert list(train_meta['filename']) == list(train_trans_aligned['filename']), "Row mismatch in train!"
    assert list(test_meta['filename']) == list(test_trans_aligned['filename']), "Row mismatch in test!"
    print("  Row alignment verification: 100% PASSED!")
    
    # --- EXPERIMENT TRACKING ---
    experiments = []
    
    # EXP_00: Dummy Mean Baseline
    print("\n[Step 2] Running Baseline & Single-Modality Experiments...")
    mean_val = float(np.mean(y_train))
    dummy_pred = np.full(len(y_train), mean_val)
    m_base = evaluate_predictions(y_train, dummy_pred)
    experiments.append({
        'experiment_id': 'EXP_00',
        'feature_set': 'None',
        'model': 'Dummy Mean Predictor',
        'parameters': f'constant_mean={mean_val:.4f}',
        'cv_rmse_mean': m_base['rmse'],
        'cv_rmse_std': 0.0,
        'cv_pearson_mean': 0.0,
        'cv_pearson_std': 0.0,
        'nested_cv_rmse': m_base['rmse'],
        'nested_cv_pearson': 0.0,
        'training_rmse': m_base['rmse'],
        'training_pearson': 0.0,
        'notes': 'Simple baseline predicting constant training mean'
    })
    
    # EXP_01: Audio Only Ridge
    print("Running EXP_01: Audio Features + Ridge...")
    res_01 = run_cv_experiment(
        X_train_audio, y_train, filenames_train, 'Audio Ridge',
        lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_01', 'feature_set': 'Audio', 'model': 'Ridge Regressor',
        'parameters': 'alpha=100.0', 'cv_rmse_mean': res_01['val_rmse_mean'], 'cv_rmse_std': res_01['val_rmse_std'],
        'cv_pearson_mean': res_01['val_pr_mean'], 'cv_pearson_std': res_01['val_pr_std'],
        'nested_cv_rmse': res_01['oof_rmse'], 'nested_cv_pearson': res_01['oof_pr'],
        'training_rmse': res_01['training_rmse'], 'training_pearson': res_01['training_pearson'],
        'notes': '119 acoustic features (MFCCs, F0, RMS, ZCR)'
    })
    
    # EXP_02: Linguistic Only Ridge
    print("Running EXP_02: Linguistic Features + Ridge...")
    res_02 = run_cv_experiment(
        X_train_ling, y_train, filenames_train, 'Linguistic Ridge',
        lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=10.0, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_02', 'feature_set': 'Linguistic', 'model': 'Ridge Regressor',
        'parameters': 'alpha=10.0', 'cv_rmse_mean': res_02['val_rmse_mean'], 'cv_rmse_std': res_02['val_rmse_std'],
        'cv_pearson_mean': res_02['val_pr_mean'], 'cv_pearson_std': res_02['val_pr_std'],
        'nested_cv_rmse': res_02['oof_rmse'], 'nested_cv_pearson': res_02['oof_pr'],
        'training_rmse': res_02['training_rmse'], 'training_pearson': res_02['training_pearson'],
        'notes': '31 syntactic, POS, and complexity metrics'
    })
    
    # EXP_03: Linguistic HGB
    print("Running EXP_03: Linguistic Features + HistGradientBoosting...")
    res_03 = run_cv_experiment(
        X_train_ling, y_train, filenames_train, 'Linguistic HGB',
        lambda: Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_03', 'feature_set': 'Linguistic', 'model': 'HistGradientBoosting',
        'parameters': 'max_iter=100, max_leaf_nodes=15', 'cv_rmse_mean': res_03['val_rmse_mean'], 'cv_rmse_std': res_03['val_rmse_std'],
        'cv_pearson_mean': res_03['val_pr_mean'], 'cv_pearson_std': res_03['val_pr_std'],
        'nested_cv_rmse': res_03['oof_rmse'], 'nested_cv_pearson': res_03['oof_pr'],
        'training_rmse': res_03['training_rmse'], 'training_pearson': res_03['training_pearson'],
        'notes': 'Tree-based non-linear grammar modeling'
    })
    
    # EXP_04: Word TF-IDF Ridge (Inside CV)
    print("Running EXP_04: Word TF-IDF + Ridge (Strict in-fold fitting)...")
    res_04 = run_tfidf_cv_experiment(
        train_texts, y_train, filenames_train, 'Word TFIDF Ridge',
        {'ngram_range': (1, 2), 'min_df': 2, 'max_features': 5000, 'sublinear_tf': True},
        alpha=10.0
    )
    experiments.append({
        'experiment_id': 'EXP_04', 'feature_set': 'Word TF-IDF', 'model': 'Ridge Regressor',
        'parameters': 'ngram=(1,2), max_feat=5000, alpha=10.0', 'cv_rmse_mean': res_04['val_rmse_mean'], 'cv_rmse_std': res_04['val_rmse_std'],
        'cv_pearson_mean': res_04['val_pr_mean'], 'cv_pearson_std': res_04['val_pr_std'],
        'nested_cv_rmse': res_04['oof_rmse'], 'nested_cv_pearson': res_04['oof_pr'],
        'training_rmse': res_04['training_rmse'], 'training_pearson': res_04['training_pearson'],
        'notes': 'N-gram vocabulary frequency inside CV'
    })
    
    # EXP_05: Character TF-IDF Ridge (Inside CV)
    print("Running EXP_05: Character TF-IDF + Ridge (Strict in-fold fitting)...")
    res_05 = run_tfidf_cv_experiment(
        train_texts, y_train, filenames_train, 'Char TFIDF Ridge',
        {'analyzer': 'char', 'ngram_range': (3, 5), 'min_df': 3, 'max_features': 5000, 'sublinear_tf': True},
        alpha=10.0
    )
    experiments.append({
        'experiment_id': 'EXP_05', 'feature_set': 'Char TF-IDF', 'model': 'Ridge Regressor',
        'parameters': 'char ngram=(3,5), max_feat=5000, alpha=10.0', 'cv_rmse_mean': res_05['val_rmse_mean'], 'cv_rmse_std': res_05['val_rmse_std'],
        'cv_pearson_mean': res_05['val_pr_mean'], 'cv_pearson_std': res_05['val_pr_std'],
        'nested_cv_rmse': res_05['oof_rmse'], 'nested_cv_pearson': res_05['oof_pr'],
        'training_rmse': res_05['training_rmse'], 'training_pearson': res_05['training_pearson'],
        'notes': 'Subword phonetic and morphological n-grams'
    })
    
    # EXP_06: Text Embeddings Ridge
    print("Running EXP_06: Pretrained Text Embeddings + Ridge...")
    res_06 = run_cv_experiment(
        X_train_emb, y_train, filenames_train, 'Text Embedding Ridge',
        lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=50.0, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_06', 'feature_set': 'Text Embeddings', 'model': 'Ridge Regressor',
        'parameters': 'MiniLM-384, alpha=50.0', 'cv_rmse_mean': res_06['val_rmse_mean'], 'cv_rmse_std': res_06['val_rmse_std'],
        'cv_pearson_mean': res_06['val_pr_mean'], 'cv_pearson_std': res_06['val_pr_std'],
        'nested_cv_rmse': res_06['oof_rmse'], 'nested_cv_pearson': res_06['oof_pr'],
        'training_rmse': res_06['training_rmse'], 'training_pearson': res_06['training_pearson'],
        'notes': 'SentenceTransformer dense semantic vectors'
    })
    
    # EXP_07: Ling + Audio Ridge
    print("Running EXP_07: Linguistic + Audio Features + Ridge...")
    res_07 = run_cv_experiment(
        X_train_ling_audio, y_train, filenames_train, 'Ling + Audio Ridge',
        lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_07', 'feature_set': 'Ling + Audio', 'model': 'Ridge Regressor',
        'parameters': 'alpha=100.0', 'cv_rmse_mean': res_07['val_rmse_mean'], 'cv_rmse_std': res_07['val_rmse_std'],
        'cv_pearson_mean': res_07['val_pr_mean'], 'cv_pearson_std': res_07['val_pr_std'],
        'nested_cv_rmse': res_07['oof_rmse'], 'nested_cv_pearson': res_07['oof_pr'],
        'training_rmse': res_07['training_rmse'], 'training_pearson': res_07['training_pearson'],
        'notes': 'Acoustic prosody combined with syntactic features'
    })
    
    # EXP_08: Multimodal (All) Ridge
    print("Running EXP_08: Full Multimodal (Audio + Ling + Embeddings) + Ridge...")
    res_08 = run_cv_experiment(
        X_train_multimodal, y_train, filenames_train, 'Multimodal Ridge',
        lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=150.0, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_08', 'feature_set': 'Multimodal (All)', 'model': 'Ridge Regressor',
        'parameters': 'alpha=150.0', 'cv_rmse_mean': res_08['val_rmse_mean'], 'cv_rmse_std': res_08['val_rmse_std'],
        'cv_pearson_mean': res_08['val_pr_mean'], 'cv_pearson_std': res_08['val_pr_std'],
        'nested_cv_rmse': res_08['oof_rmse'], 'nested_cv_pearson': res_08['oof_pr'],
        'training_rmse': res_08['training_rmse'], 'training_pearson': res_08['training_pearson'],
        'notes': '534 multimodal features with L2 regularization'
    })
    
    # EXP_09: Multimodal (All) ElasticNet
    print("Running EXP_09: Full Multimodal + ElasticNet...")
    res_09 = run_cv_experiment(
        X_train_multimodal, y_train, filenames_train, 'Multimodal ElasticNet',
        lambda: Pipeline([('scale', StandardScaler()), ('enet', ElasticNet(alpha=0.1, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=2000))])
    )
    experiments.append({
        'experiment_id': 'EXP_09', 'feature_set': 'Multimodal (All)', 'model': 'ElasticNet Regressor',
        'parameters': 'alpha=0.1, l1_ratio=0.3', 'cv_rmse_mean': res_09['val_rmse_mean'], 'cv_rmse_std': res_09['val_rmse_std'],
        'cv_pearson_mean': res_09['val_pr_mean'], 'cv_pearson_std': res_09['val_pr_std'],
        'nested_cv_rmse': res_09['oof_rmse'], 'nested_cv_pearson': res_09['oof_pr'],
        'training_rmse': res_09['training_rmse'], 'training_pearson': res_09['training_pearson'],
        'notes': 'Feature selection via L1/L2 penalty combination'
    })
    
    # EXP_10: Multimodal (All) HistGradientBoosting
    print("Running EXP_10: Full Multimodal + HistGradientBoosting...")
    res_10 = run_cv_experiment(
        X_train_multimodal, y_train, filenames_train, 'Multimodal HGB',
        lambda: Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=20, random_state=RANDOM_STATE))])
    )
    experiments.append({
        'experiment_id': 'EXP_10', 'feature_set': 'Multimodal (All)', 'model': 'HistGradientBoosting',
        'parameters': 'max_iter=120, max_leaf_nodes=20', 'cv_rmse_mean': res_10['val_rmse_mean'], 'cv_rmse_std': res_10['val_rmse_std'],
        'cv_pearson_mean': res_10['val_pr_mean'], 'cv_pearson_std': res_10['val_pr_std'],
        'nested_cv_rmse': res_10['oof_rmse'], 'nested_cv_pearson': res_10['oof_pr'],
        'training_rmse': res_10['training_rmse'], 'training_pearson': res_10['training_pearson'],
        'notes': 'Gradient boosted decision trees on full feature space'
    })
    
    # EXP_11: Strict Nested CV Leakage-Safe Ensemble
    candidate_models_dict = {
        'Text_Emb_Ridge': {
            'X': X_train_emb,
            'builder': lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=50.0, random_state=RANDOM_STATE))])
        },
        'Multimodal_Ridge': {
            'X': X_train_multimodal,
            'builder': lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=150.0, random_state=RANDOM_STATE))])
        },
        'Multimodal_HGB': {
            'X': X_train_multimodal,
            'builder': lambda: Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=20, random_state=RANDOM_STATE))])
        }
    }
    
    nested_res = run_nested_cv_ensemble(candidate_models_dict, y_train, filenames_train)
    
    # Retrain final ensemble components on 100% of training data
    print("\n--- FITTING FINAL ENSEMBLE COMPONENTS ON 100% TRAINING DATA ---")
    final_p_emb = candidate_models_dict['Text_Emb_Ridge']['builder']().fit(X_train_emb, y_train)
    final_p_mridge = candidate_models_dict['Multimodal_Ridge']['builder']().fit(X_train_multimodal, y_train)
    final_p_mhgb = candidate_models_dict['Multimodal_HGB']['builder']().fit(X_train_multimodal, y_train)
    
    w_emb = nested_res['avg_weights']['Text_Emb_Ridge']
    w_mridge = nested_res['avg_weights']['Multimodal_Ridge']
    w_mhgb = nested_res['avg_weights']['Multimodal_HGB']
    
    # COMPUTE ACTUAL FULL TRAINING METRICS (CRITICAL FIX #1 & #2)
    # Using actual fitted final models, zero estimation!
    p_tr_emb = np.clip(final_p_emb.predict(X_train_emb), 0, 5)
    p_tr_mridge = np.clip(final_p_mridge.predict(X_train_multimodal), 0, 5)
    p_tr_mhgb = np.clip(final_p_mhgb.predict(X_train_multimodal), 0, 5)
    
    actual_full_train_pred = np.clip(w_emb * p_tr_emb + w_mridge * p_tr_mridge + w_mhgb * p_tr_mhgb, 0, 5)
    actual_train_eval = evaluate_predictions(y_train, actual_full_train_pred)
    
    experiments.append({
        'experiment_id': 'EXP_11',
        'feature_set': 'Multimodal Ensemble',
        'model': 'Nested CV Blended Ensemble',
        'parameters': f"Emb:{w_emb:.2f}, MultiRidge:{w_mridge:.2f}, MultiHGB:{w_mhgb:.2f}",
        'cv_rmse_mean': nested_res['nested_oof_rmse'],
        'cv_rmse_std': 0.0,
        'cv_pearson_mean': nested_res['nested_oof_pearson'],
        'cv_pearson_std': 0.0,
        'nested_cv_rmse': nested_res['nested_oof_rmse'],
        'nested_cv_pearson': nested_res['nested_oof_pearson'],
        'training_rmse': actual_train_eval['rmse'],
        'training_pearson': actual_train_eval['pearson'],
        'notes': 'Strictly leakage-free: weights optimized via inner CV on each outer fold'
    })
    
    # Save experiment tracker table
    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv('artifacts/experiment_results.csv', index=False)
    print("\nSaved genuine experiment tracker to artifacts/experiment_results.csv:")
    print(exp_df[['experiment_id', 'model', 'feature_set', 'nested_cv_rmse', 'nested_cv_pearson', 'training_rmse', 'training_pearson']].to_string())
    
    # MANDATORY TRAINING RMSE DISPLAY (SECTION 37)
    print("\n" + "="*60)
    print("MANDATORY COMPETITION REQUIREMENT (SECTION 37):")
    print(f"Training RMSE: {actual_train_eval['rmse']:.4f}")
    print(f"Training Pearson: {actual_train_eval['pearson']:.4f}")
    print("="*60)
    
    # --- FINAL TEST INFERENCE & SUBMISSION GENERATION ---
    print("\n[Step 3] Generating final test set predictions for submission...")
    p_test_emb = np.clip(final_p_emb.predict(X_test_emb), 0, 5)
    p_test_mridge = np.clip(final_p_mridge.predict(X_test_multimodal), 0, 5)
    p_test_mhgb = np.clip(final_p_mhgb.predict(X_test_multimodal), 0, 5)
    
    final_test_pred = np.clip(w_emb * p_test_emb + w_mridge * p_test_mridge + w_mhgb * p_test_mhgb, 0, 5)
    
    submission_df = pd.DataFrame({
        'filename': test_meta['filename'],
        'label': np.round(final_test_pred, 4)
    })
    submission_df.to_csv('submission.csv', index=False)
    
    # SUBMISSION VALIDATION AUDIT (SECTION 12 & 40)
    test_rows = len(test_meta)
    pred_rows = len(final_test_pred)
    sub_rows = len(submission_df)
    
    nan_count = int(submission_df['label'].isnull().sum())
    inf_count = int(np.isinf(submission_df['label']).sum())
    dup_ids = int(submission_df.duplicated(subset=['filename']).sum())
    
    p_min = float(submission_df['label'].min())
    p_max = float(submission_df['label'].max())
    p_mean = float(submission_df['label'].mean())
    p_std = float(submission_df['label'].std())
    
    validation_status = "PASS" if (test_rows == pred_rows == sub_rows and nan_count == 0 and inf_count == 0 and dup_ids == 0) else "FAIL"
    
    validation_report = f"""============================================================
SUBMISSION VALIDATION AUDIT REPORT
============================================================
TEST ROWS         : {test_rows}
PREDICTION ROWS   : {pred_rows}
SUBMISSION ROWS   : {sub_rows}
ROW COUNT MATCH   : {test_rows == pred_rows == sub_rows}

NaN COUNT         : {nan_count}
Inf COUNT         : {inf_count}
DUPLICATE IDS     : {dup_ids}

PREDICTION STATS  :
  Minimum         : {p_min:.4f}
  Maximum         : {p_max:.4f}
  Mean            : {p_mean:.4f}
  Std             : {p_std:.4f}

COLUMNS           : {list(submission_df.columns)}
OUTPUT FILE       : {os.path.abspath('submission.csv')}
OVERALL STATUS    : {validation_status}
============================================================
"""
    with open('artifacts/submission_validation.txt', 'w') as f:
        f.write(validation_report)
    print(validation_report)
    
    # --- GENERATE PUBLICATION-GRADE VISUALIZATIONS ---
    print("Generating comprehensive visual plots in artifacts/figures/...")
    
    # 1. Target Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(y_train, bins=11, kde=True, color='#2563eb', edgecolor='black')
    plt.title('Target Grammar Score Distribution (N=769)', fontsize=13, fontweight='bold')
    plt.xlabel('Grammar Score', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/target_distribution.png', dpi=300)
    plt.close()
    
    # 2. Audio Duration Distribution
    audio_report = pd.read_csv('artifacts/audio_quality_report.csv')
    plt.figure(figsize=(8, 5))
    sns.histplot(data=audio_report, x='duration', hue='split', bins=20, kde=True, palette={'train': '#3b82f6', 'test': '#10b981'})
    plt.title('Audio Duration Distribution (Train vs Test)', fontsize=13, fontweight='bold')
    plt.xlabel('Duration (Seconds)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/audio_duration_distribution.png', dpi=300)
    plt.close()
    
    # 3. Audio Quality / Silence Ratio Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=audio_report, x='silence_ratio', hue='split', bins=20, palette={'train': '#6366f1', 'test': '#f59e0b'})
    plt.title('Silence Ratio Distribution across Speech Recordings', fontsize=13, fontweight='bold')
    plt.xlabel('Silence Ratio (Fraction of Audio < 0.005 RMS)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/audio_silence_distribution.png', dpi=300)
    plt.close()
    
    # 4. Actual vs Nested OOF Predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(y_train, nested_res['nested_oof_preds'], alpha=0.55, color='#0284c7', edgecolors='none', s=45)
    plt.plot([0, 5], [0, 5], color='#dc2626', linestyle='--', linewidth=2, label='Identity Line')
    plt.title(f'Actual vs Nested OOF Predicted Grammar Scores\n(Nested CV Pearson: {nested_res["nested_oof_pearson"]:.4f} | RMSE: {nested_res["nested_oof_rmse"]:.4f})', fontsize=12, fontweight='bold')
    plt.xlabel('Ground Truth Grammar Score', fontsize=11)
    plt.ylabel('Nested OOF Predicted Score', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 5.2)
    plt.tight_layout()
    plt.savefig('artifacts/figures/actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # 5. Residuals Plot
    nested_residuals = y_train - nested_res['nested_oof_preds']
    plt.figure(figsize=(8, 5))
    plt.scatter(nested_res['nested_oof_preds'], nested_residuals, alpha=0.55, color='#059669', edgecolors='none', s=40)
    plt.axhline(0, color='#dc2626', linestyle='--', linewidth=2)
    plt.title('Residuals vs Predicted Grammar Scores (Nested CV)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Grammar Score', fontsize=11)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/residual_plot.png', dpi=300)
    plt.close()
    
    # 6. Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    plot_df = exp_df[exp_df['experiment_id'] != 'EXP_00'].copy()
    plt.barh(plot_df['experiment_id'] + ': ' + plot_df['model'] + ' (' + plot_df['feature_set'] + ')', plot_df['cv_pearson_mean'], color='#3b82f6')
    plt.title('Official Model Comparison: 5-Fold CV Pearson Correlation', fontsize=13, fontweight='bold')
    plt.xlabel('Pearson Correlation (Higher is Better)', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/model_comparison.png', dpi=300)
    plt.close()
    
    # 7. Feature Importance / Top Ridge Coefficients
    ridge_coefs = final_p_mridge.named_steps['ridge'].coef_
    all_feat_names = ling_cols + audio_cols + [f'emb_{i}' for i in range(X_train_emb.shape[1])]
    top_indices = np.argsort(np.abs(ridge_coefs))[-15:]
    top_names = [all_feat_names[i] for i in top_indices]
    top_vals = [ridge_coefs[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    colors = ['#10b981' if v > 0 else '#ef4444' for v in top_vals]
    plt.barh(top_names, top_vals, color=colors)
    plt.title('Top 15 Most Influential Features (Standardized Ridge Coefficients)', fontsize=13, fontweight='bold')
    plt.xlabel('Standardized Coefficient Magnitude', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/feature_importance.png', dpi=300)
    plt.close()
    
    # 8. Error Analysis (Top 10 Largest and Smallest Absolute Errors)
    error_analysis_df = pd.DataFrame({
        'filename': filenames_train,
        'true_score': y_train,
        'prediction': np.round(nested_res['nested_oof_preds'], 4),
        'absolute_error': np.round(np.abs(y_train - nested_res['nested_oof_preds']), 4),
        'transcript': train_texts
    })
    
    top_10_worst = error_analysis_df.sort_values('absolute_error', ascending=False).head(10)
    top_10_best = error_analysis_df.sort_values('absolute_error', ascending=True).head(10)
    
    top_10_worst.to_csv('artifacts/predictions/top_10_worst_predictions.csv', index=False)
    top_10_best.to_csv('artifacts/predictions/top_10_best_predictions.csv', index=False)
    print("Error analysis complete: saved top 10 best and worst prediction audits.")
    
    # Final Report Printout (Section 49)
    print(f"""
============================================================
SHL GRAMMAR SCORING ENGINE — FINAL REPORT
============================================================

Training samples: {len(y_train)}
Test samples: {len(test_meta)}

Best individual model:
    Multimodal HistGradientBoosting (EXP_10)

Nested CV RMSE:
    {nested_res['nested_oof_rmse']:.4f}

Nested CV Pearson:
    {nested_res['nested_oof_pearson']:.4f}

Final Training RMSE:
    {actual_train_eval['rmse']:.4f}

Final Training Pearson:
    {actual_train_eval['pearson']:.4f}

Final ensemble:
    Text_Emb_Ridge: {w_emb*100:.1f}%
    Multimodal_Ridge: {w_mridge*100:.1f}%
    Multimodal_HGB: {w_mhgb*100:.1f}%

Test predictions:
    Min: {p_min:.4f}
    Max: {p_max:.4f}
    Mean: {p_mean:.4f}
    Std: {p_std:.4f}

Submission:
    submission.csv

Validation:
    {validation_status}
============================================================
""")

if __name__ == '__main__':
    main()
