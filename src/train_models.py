import os
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

RANDOM_STATE = 42

def evaluate_predictions(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    return rmse, pr

def run_cross_validation(X, y, model_builder, n_splits=5, clip_range=(0, 5)):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y))
    train_rmse_list = []
    train_pr_list = []
    val_rmse_list = []
    val_pr_list = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        pipeline = model_builder()
        pipeline.fit(X_train, y_train)
        
        train_p = pipeline.predict(X_train)
        val_p = pipeline.predict(X_val)
        
        if clip_range:
            train_p = np.clip(train_p, clip_range[0], clip_range[1])
            val_p = np.clip(val_p, clip_range[0], clip_range[1])
            
        t_rmse, t_pr = evaluate_predictions(y_train, train_p)
        v_rmse, v_pr = evaluate_predictions(y_val, val_p)
        
        train_rmse_list.append(t_rmse)
        train_pr_list.append(t_pr)
        val_rmse_list.append(v_rmse)
        val_pr_list.append(v_pr)
        oof_preds[val_idx] = val_p
        
    overall_oof_rmse, overall_oof_pr = evaluate_predictions(y, oof_preds)
    
    return {
        'val_rmse_mean': np.mean(val_rmse_list),
        'val_rmse_std': np.std(val_rmse_list),
        'val_pr_mean': np.mean(val_pr_list),
        'val_pr_std': np.std(val_pr_list),
        'oof_rmse': overall_oof_rmse,
        'oof_pr': overall_oof_pr,
        'train_rmse_mean': np.mean(train_rmse_list),
        'train_pr_mean': np.mean(train_pr_list),
        'oof_preds': oof_preds
    }

def optimize_ensemble(oof_dict, y, clip_range=(0, 5)):
    """Find optimal non-negative weights for linear combination of models."""
    from scipy.optimize import minimize
    
    names = list(oof_dict.keys())
    matrix = np.column_stack([oof_dict[name] for name in names])
    
    def loss(weights):
        pred = matrix @ weights
        if clip_range:
            pred = np.clip(pred, clip_range[0], clip_range[1])
        # Minimize RMSE and maximize negative correlation
        rmse = root_mean_squared_error(y, pred)
        pr, _ = pearsonr(y, pred)
        return rmse - 0.5 * pr
        
    init_w = np.ones(len(names)) / len(names)
    bounds = [(0, 1) for _ in names]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    res = minimize(loss, init_w, bounds=bounds, constraints=constraints)
    best_weights = res.x / np.sum(res.x)
    ensemble_oof = matrix @ best_weights
    if clip_range:
        ensemble_oof = np.clip(ensemble_oof, clip_range[0], clip_range[1])
    e_rmse, e_pr = evaluate_predictions(y, ensemble_oof)
    
    weight_dict = {names[i]: float(best_weights[i]) for i in range(len(names))}
    return weight_dict, ensemble_oof, e_rmse, e_pr
