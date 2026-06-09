import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Default hyperparameters verified in CV
DEFAULT_LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 1800,
    'learning_rate': 0.009466191175439549,
    'max_depth': 10,
    'num_leaves': 15,
    'min_child_samples': 5,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    'min_split_gain': 0.05,
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': -1
}

def calculate_recency_weights(df):
    """Calculates sample weights to apply a recency bias to training data."""
    df = df.copy()
    df['weight'] = 1.0 + (df['Date'].dt.year - 2012) / 10.0
    df.loc[df['Date'].dt.year >= 2019, 'weight'] *= 1.5
    return df['weight']

def train_and_validate_fold(train_features, features_list, fold_year, params=None):
    """Runs training on data before fold_year, and validates on fold_year."""
    if params is None:
        params = DEFAULT_LGB_PARAMS
        
    train_idx = train_features['Date'].dt.year < fold_year
    val_idx = train_features['Date'].dt.year == fold_year
    
    X_tr = train_features.loc[train_idx, features_list]
    y_tr = train_features.loc[train_idx, 'rev_norm']
    w_tr = train_features.loc[train_idx, 'weight']
    
    X_va = train_features.loc[val_idx, features_list]
    y_va = train_features.loc[val_idx, 'rev_norm']
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr, 
        sample_weight=w_tr, 
        eval_set=[(X_va, y_va)], 
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    # Predict and de-normalize
    pred_norm = model.predict(X_va)
    trend = train_features.loc[val_idx, 'trend_rev']
    pred_actual = pred_norm * trend
    
    actual = train_features.loc[val_idx, 'Revenue']
    mae = mean_absolute_error(actual, pred_actual)
    return mae, model

def run_walk_forward_validation(train_features, features_list, cv_folds=[2021, 2022], params=None):
    """Executes walk-forward time-series cross validation."""
    print("Starting Walk-Forward Cross Validation...")
    results = []
    for fold in cv_folds:
        mae, _ = train_and_validate_fold(train_features, features_list, fold, params)
        results.append(mae)
        print(f"  Fold Year {fold} - MAE: {mae:,.2f}")
    
    mean_cv = np.mean(results)
    print(f"Average CV MAE across folds: {mean_cv:,.2f}")
    return mean_cv

def evaluate_predictions(y_true, y_pred):
    """Computes MAE, RMSE, and R2 regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def apply_domain_constraints(revenue_pred, cogs_pred):
    """Enforces logical business constraints: non-negativity and COGS limit."""
    cogs_pred = np.maximum(cogs_pred, 0)
    cogs_pred = np.minimum(cogs_pred, revenue_pred * 1.30)
    return cogs_pred
