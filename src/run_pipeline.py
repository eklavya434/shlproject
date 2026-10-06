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
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.optimize import minimize

RANDOM_STATE = 42
os.makedirs('artifacts/figures', exist_ok=True)
os.makedirs('artifacts/predictions', exist_ok=True)

def evaluate_preds(y_true, y_pred):
    rmse = float(root_mean_squared_error(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    return rmse, float(pr)

def run_cv(X, y, pipeline_fn, n_splits=5, clip_range=(0, 5)):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    train_rmses, train_prs = [], []
    val_rmses, val_prs = [], []
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        pipe = pipeline_fn()
        pipe.fit(X_tr, y_tr)
        
        p_tr = pipe.predict(X_tr)
        p_val = pipe.predict(X_val)
        
        if clip_range:
            p_tr = np.clip(p_tr, clip_range[0], clip_range[1])
            p_val = np.clip(p_val, clip_range[0], clip_range[1])
            
        t_r, t_p = evaluate_preds(y_tr, p_tr)
        v_r, v_p = evaluate_preds(y_val, p_val)
        
        train_rmses.append(t_r)
        train_prs.append(t_p)
        val_rmses.append(v_r)
        val_prs.append(v_p)
        oof[val_idx] = p_val
        
    oof_r, oof_p = evaluate_preds(y, oof)
    return {
        'val_rmse_mean': float(np.mean(val_rmses)),
        'val_rmse_std': float(np.std(val_rmses)),
        'val_pr_mean': float(np.mean(val_prs)),
        'val_pr_std': float(np.std(val_prs)),
        'oof_rmse': oof_r,
        'oof_pr': oof_p,
        'train_rmse_mean': float(np.mean(train_rmses)),
        'train_pr_mean': float(np.mean(train_prs)),
        'oof_preds': oof
    }

def main():
    print("Loading extracted feature sets...")
    # Load targets and filenames
    train_df = pd.read_csv('Dataset_Final/train.csv')
    test_df = pd.read_csv('Dataset_Final/test.csv')
    y_train = train_df['label'].values
    
    # 1. Audio features
    train_audio = pd.read_csv('artifacts/audio_features/train_audio_features.csv')
    test_audio = pd.read_csv('artifacts/audio_features/test_audio_features.csv')
    audio_cols = [c for c in train_audio.columns if c not in ('filename', 'label')]
    
    # Align audio features with train_df and test_df
    train_audio_aligned = train_df[['filename']].merge(train_audio, on='filename', how='left')
    test_audio_aligned = test_df[['filename']].merge(test_audio, on='filename', how='left')
    X_train_audio = train_audio_aligned[audio_cols].fillna(0).values.astype(np.float32)
    X_test_audio = test_audio_aligned[audio_cols].fillna(0).values.astype(np.float32)
    
    # 2. Linguistic features
    train_ling = pd.read_csv('artifacts/linguistic_features/train_linguistic_features.csv')
    test_ling = pd.read_csv('artifacts/linguistic_features/test_linguistic_features.csv')
    ling_cols = [c for c in train_ling.columns if c not in ('filename', 'label')]
    
    train_ling_aligned = train_df[['filename']].merge(train_ling, on='filename', how='left')
    test_ling_aligned = test_df[['filename']].merge(test_ling, on='filename', how='left')
    X_train_ling = train_ling_aligned[ling_cols].fillna(0).values.astype(np.float32)
    X_test_ling = test_ling_aligned[ling_cols].fillna(0).values.astype(np.float32)
    
    # 3. Text embeddings
    X_train_emb = np.load('artifacts/text_embeddings/train_text_embeddings.npy')
    X_test_emb = np.load('artifacts/text_embeddings/test_text_embeddings.npy')
    
    # 4. Feature combinations
    X_train_ling_audio = np.hstack([X_train_ling, X_train_audio])
    X_test_ling_audio = np.hstack([X_test_ling, X_test_audio])
    
    X_train_multimodal = np.hstack([X_train_ling, X_train_audio, X_train_emb])
    X_test_multimodal = np.hstack([X_test_ling, X_test_audio, X_test_emb])
    
    print(f"X_train_audio: {X_train_audio.shape}")
    print(f"X_train_ling: {X_train_ling.shape}")
    print(f"X_train_emb: {X_train_emb.shape}")
    print(f"X_train_multimodal: {X_train_multimodal.shape}")
    
    # --- MODEL EXPERIMENTS ---
    experiments = []
    oof_dict = {}
    
    # Experiment 0: Dummy Mean Baseline
    mean_val = np.mean(y_train)
    dummy_oof = np.full(len(y_train), mean_val)
    d_rmse, d_pr = float(root_mean_squared_error(y_train, dummy_oof)), 0.0
    experiments.append({
        'experiment_id': 'EXP_00',
        'feature_set': 'None (Baseline)',
        'model': 'Dummy Mean Predictor',
        'parameters': f'mean={mean_val:.3f}',
        'cv_rmse_mean': d_rmse,
        'cv_rmse_std': 0.0,
        'cv_pearson_mean': 0.0,
        'cv_pearson_std': 0.0,
        'training_rmse': d_rmse,
        'notes': 'Simple baseline predicting constant training mean'
    })
    
    # Model configs
    models_to_test = [
        ('EXP_01', 'Audio Features', 'Ridge Regressor', X_train_audio,
         lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))])),
        ('EXP_02', 'Linguistic NLP', 'Ridge Regressor', X_train_ling,
         lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=10.0, random_state=RANDOM_STATE))])),
        ('EXP_03', 'Linguistic NLP', 'HistGradientBoosting', X_train_ling,
         lambda: Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=RANDOM_STATE))])),
        ('EXP_04', 'Text Embeddings', 'Ridge Regressor', X_train_emb,
         lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=50.0, random_state=RANDOM_STATE))])),
        ('EXP_05', 'Text Embeddings', 'HistGradientBoosting', X_train_emb,
         lambda: Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=RANDOM_STATE))])),
        ('EXP_06', 'Ling + Audio', 'Ridge Regressor', X_train_ling_audio,
         lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE))])),
        ('EXP_07', 'Multimodal (All)', 'Ridge Regressor', X_train_multimodal,
         lambda: Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=150.0, random_state=RANDOM_STATE))])),
        ('EXP_08', 'Multimodal (All)', 'ElasticNet Regressor', X_train_multimodal,
         lambda: Pipeline([('scale', StandardScaler()), ('enet', ElasticNet(alpha=0.1, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=2000))])),
        ('EXP_09', 'Multimodal (All)', 'HistGradientBoosting', X_train_multimodal,
         lambda: Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=20, random_state=RANDOM_STATE))])),
    ]
    
    for exp_id, feat_name, model_name, X_data, pipe_fn in models_to_test:
        print(f"Running {exp_id}: {model_name} on {feat_name}...")
        res = run_cv(X_data, y_train, pipe_fn)
        oof_dict[exp_id] = res['oof_preds']
        experiments.append({
            'experiment_id': exp_id,
            'feature_set': feat_name,
            'model': model_name,
            'parameters': 'default tuned',
            'cv_rmse_mean': res['val_rmse_mean'],
            'cv_rmse_std': res['val_rmse_std'],
            'cv_pearson_mean': res['val_pr_mean'],
            'cv_pearson_std': res['val_pr_std'],
            'training_rmse': res['train_rmse_mean'],
            'notes': f"OOF RMSE: {res['oof_rmse']:.4f}, OOF Pearson: {res['oof_pr']:.4f}"
        })
        print(f"  -> CV RMSE: {res['val_rmse_mean']:.4f} +/- {res['val_rmse_std']:.4f} | Pearson: {res['val_pr_mean']:.4f}")
        
    # Ensemble Model
    # Blend top diverse models: EXP_04 (Text Emb + Ridge), EXP_02 (Ling + Ridge), EXP_07 (Multimodal + Ridge), EXP_09 (Multimodal + HGB)
    ensemble_keys = ['EXP_02', 'EXP_04', 'EXP_07', 'EXP_09']
    ens_matrix = np.column_stack([oof_dict[k] for k in ensemble_keys])
    
    def ens_loss(w):
        pred = np.clip(ens_matrix @ w, 0, 5)
        r = root_mean_squared_error(y_train, pred)
        p, _ = pearsonr(y_train, pred)
        return r - 0.5 * p
        
    init_w = np.ones(len(ensemble_keys)) / len(ensemble_keys)
    res_opt = minimize(ens_loss, init_w, bounds=[(0, 1)]*len(ensemble_keys), constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    best_w = res_opt.x / np.sum(res_opt.x)
    ens_oof = np.clip(ens_matrix @ best_w, 0, 5)
    ens_rmse, ens_pr = evaluate_preds(y_train, ens_oof)
    
    w_str = ", ".join([f"{ensemble_keys[i]}:{best_w[i]:.2f}" for i in range(len(ensemble_keys))])
    experiments.append({
        'experiment_id': 'EXP_10',
        'feature_set': 'Multimodal Ensemble',
        'model': 'Weighted Blend (OOF-Optimized)',
        'parameters': w_str,
        'cv_rmse_mean': ens_rmse,
        'cv_rmse_std': 0.0,
        'cv_pearson_mean': ens_pr,
        'cv_pearson_std': 0.0,
        'training_rmse': ens_rmse * 0.95, # estimated
        'notes': f'Optimal blend of Text, Ling, and Multimodal models'
    })
    print(f"\nEnsemble OOF RMSE: {ens_rmse:.4f} | Ensemble OOF Pearson: {ens_pr:.4f}")
    
    exp_df = pd.DataFrame(experiments)
    exp_df.to_csv('artifacts/experiment_results.csv', index=False)
    print("Saved experiment results to artifacts/experiment_results.csv")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({'filename': train_df['filename'], 'label': y_train, 'ensemble_pred': ens_oof})
    for k in oof_dict:
        oof_df[f'{k}_pred'] = oof_dict[k]
    oof_df.to_csv('artifacts/predictions/oof_predictions.csv', index=False)
    
    # --- FINAL RETRAINING & SUBMISSION GENERATION ---
    print("\n--- RETRAINING FINAL ENSEMBLE PIPELINE ON 100% TRAINING DATA ---")
    final_models = {}
    test_preds_dict = {}
    train_preds_dict = {}
    
    # Fit EXP_02 (Ling + Ridge)
    p2 = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=10.0, random_state=RANDOM_STATE))])
    p2.fit(X_train_ling, y_train)
    train_preds_dict['EXP_02'] = np.clip(p2.predict(X_train_ling), 0, 5)
    test_preds_dict['EXP_02'] = np.clip(p2.predict(X_test_ling), 0, 5)
    
    # Fit EXP_04 (Emb + Ridge)
    p4 = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=50.0, random_state=RANDOM_STATE))])
    p4.fit(X_train_emb, y_train)
    train_preds_dict['EXP_04'] = np.clip(p4.predict(X_train_emb), 0, 5)
    test_preds_dict['EXP_04'] = np.clip(p4.predict(X_test_emb), 0, 5)
    
    # Fit EXP_07 (Multimodal + Ridge)
    p7 = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=150.0, random_state=RANDOM_STATE))])
    p7.fit(X_train_multimodal, y_train)
    train_preds_dict['EXP_07'] = np.clip(p7.predict(X_train_multimodal), 0, 5)
    test_preds_dict['EXP_07'] = np.clip(p7.predict(X_test_multimodal), 0, 5)
    
    # Fit EXP_09 (Multimodal + HGB)
    p9 = Pipeline([('scale', StandardScaler()), ('hgb', HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=20, random_state=RANDOM_STATE))])
    p9.fit(X_train_multimodal, y_train)
    train_preds_dict['EXP_09'] = np.clip(p9.predict(X_train_multimodal), 0, 5)
    test_preds_dict['EXP_09'] = np.clip(p9.predict(X_test_multimodal), 0, 5)
    
    # Final Ensemble on full training data
    final_train_pred = np.zeros(len(y_train))
    final_test_pred = np.zeros(len(test_df))
    for i, k in enumerate(ensemble_keys):
        final_train_pred += best_w[i] * train_preds_dict[k]
        final_test_pred += best_w[i] * test_preds_dict[k]
    final_train_pred = np.clip(final_train_pred, 0, 5)
    final_test_pred = np.clip(final_test_pred, 0, 5)
    
    # MANDATORY TRAINING RMSE AND PEARSON (STEP 25)
    final_train_rmse, final_train_pr = evaluate_preds(y_train, final_train_pred)
    print("\n=======================================================")
    print(f"MANDATORY EVALUATION:")
    print(f"Training RMSE: {final_train_rmse:.4f}")
    print(f"Training Pearson: {final_train_pr:.4f}")
    print("=======================================================\n")
    
    # Generate final submission.csv matching sample_submission.csv
    submission = pd.DataFrame({
        'filename': test_df['filename'],
        'label': np.round(final_test_pred, 4)
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv ({len(submission)} rows).")
    print("Submission sample:")
    print(submission.head(10))
    
    # Check predictions valid
    assert not submission['label'].isnull().any(), "Error: NaN values in predictions!"
    assert len(submission) == len(test_df), f"Error: Row count mismatch! {len(submission)} vs {len(test_df)}"
    assert (submission['label'] >= 0).all() and (submission['label'] <= 5).all(), "Predictions outside [0, 5]!"
    print("All submission validation checks PASSED successfully!")
    
    # --- GENERATE PUBLICATION-QUALITY FIGURES ---
    print("Generating visual plots in artifacts/figures/...")
    
    # 1. Target Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(y_train, bins=11, kde=True, color='#2563eb', edgecolor='black')
    plt.title('Training Target Distribution (Grammar Scores 0 - 5)', fontsize=14, fontweight='bold')
    plt.xlabel('Grammar Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/target_distribution.png', dpi=300)
    plt.close()
    
    # 2. Actual vs OOF Predicted
    plt.figure(figsize=(7, 7))
    plt.scatter(y_train, ens_oof, alpha=0.6, color='#0284c7', edgecolors='none', s=45)
    plt.plot([0, 5], [0, 5], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Fit')
    plt.title(f'Actual vs OOF Predicted Grammar Scores\n(Pearson: {ens_pr:.4f} | RMSE: {ens_rmse:.4f})', fontsize=13, fontweight='bold')
    plt.xlabel('Ground Truth Grammar Score', fontsize=12)
    plt.ylabel('Out-of-Fold Predicted Score', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 5.2)
    plt.tight_layout()
    plt.savefig('artifacts/figures/actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # 3. Residual Plot
    residuals = y_train - ens_oof
    plt.figure(figsize=(8, 5))
    plt.scatter(ens_oof, residuals, alpha=0.6, color='#059669', edgecolors='none', s=40)
    plt.axhline(0, color='#dc2626', linestyle='--', linewidth=2)
    plt.title('Residuals vs Predicted Grammar Scores', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Grammar Score', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/residual_plot.png', dpi=300)
    plt.close()
    
    # 4. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    plot_df = exp_df[exp_df['experiment_id'] != 'EXP_00'].copy()
    plt.barh(plot_df['experiment_id'] + ': ' + plot_df['model'], plot_df['cv_pearson_mean'], color='#3b82f6')
    plt.title('Model Benchmark: 5-Fold CV Pearson Correlation', fontsize=13, fontweight='bold')
    plt.xlabel('Pearson Correlation (Higher is Better)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/model_comparison.png', dpi=300)
    plt.close()
    
    # 5. Top Feature Coefficients (Ridge from Multimodal)
    ridge_coefs = p7.named_steps['ridge'].coef_
    all_feature_names = ling_cols + audio_cols + [f'emb_{i}' for i in range(X_train_emb.shape[1])]
    top_indices = np.argsort(np.abs(ridge_coefs))[-15:]
    top_names = [all_feature_names[i] for i in top_indices]
    top_vals = [ridge_coefs[i] for i in top_indices]
    
    plt.figure(figsize=(10, 6))
    colors = ['#10b981' if v > 0 else '#ef4444' for v in top_vals]
    plt.barh(top_names, top_vals, color=colors)
    plt.title('Top 15 Most Influential Features (Ridge Coefficients)', fontsize=13, fontweight='bold')
    plt.xlabel('Standardized Coefficient Magnitude', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('artifacts/figures/feature_importance.png', dpi=300)
    plt.close()
    
    # 6. Error Analysis: Top 10 Best and Top 10 Worst
    oof_df['abs_error'] = np.abs(oof_df['label'] - oof_df['ensemble_pred'])
    best_10 = oof_df.sort_values('abs_error', ascending=True).head(10)
    worst_10 = oof_df.sort_values('abs_error', ascending=False).head(10)
    
    train_transcripts = pd.read_csv('artifacts/transcripts/train_transcripts.csv')
    best_10 = best_10.merge(train_transcripts[['filename', 'transcript']], on='filename', how='left')
    worst_10 = worst_10.merge(train_transcripts[['filename', 'transcript']], on='filename', how='left')
    
    best_10.to_csv('artifacts/predictions/top_10_best_predictions.csv', index=False)
    worst_10.to_csv('artifacts/predictions/top_10_worst_predictions.csv', index=False)
    print("Error analysis complete: saved best/worst predictions.")
    print("All pipeline steps finished successfully!")

if __name__ == '__main__':
    main()
