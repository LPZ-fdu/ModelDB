import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== New: redirect all prints to a txt file while keeping console output =====
import sys

class Tee(object):
    """Write stdout to multiple streams (e.g., console + file)."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# =========================================================
# 0) Global random seed
# =========================================================
SEED_MASTER = 42
np.random.seed(SEED_MASTER)
rng = np.random.RandomState(SEED_MASTER)

# =========================================================
# 1) Paths and files
# =========================================================
target_path = 'XXX/sample_data.xlsx'  # contains YEAR, DOY, R2_Kfold
meta_tpl = 'XXX/MetaFeatures_{}.xlsx'  # 2018/2019/2020

model_save_dir = r'XXX'
os.makedirs(model_save_dir, exist_ok=True)

# New: log file path
log_path = os.path.join(model_save_dir, 'run_log.txt')

# Annual national benchmark R2 thresholds (baseline)
year_thresholds = {2018: 0.80, 2019: 0.82, 2020: 0.81}

# =========================================================
# 2) Load and merge: target R2 + second-stage features
# =========================================================
def load_meta_features(year: int) -> pd.DataFrame:
    f = meta_tpl.format(year)
    if not os.path.exists(f):
        raise FileNotFoundError(f"Second-stage feature file not found: {f}")
    df_ = pd.read_excel(f)
    df_['YEAR'] = year
    return df_

target_df = pd.read_excel(target_path)
need_cols = {'YEAR', 'DOY', 'R2_Kfold'}
missing = need_cols - set(target_df.columns)
if missing:
    raise ValueError(f"Target file is missing required columns: {missing}. Please check: {target_path}")

meta_all = pd.concat([load_meta_features(y) for y in [2018, 2019, 2020]], ignore_index=True)

# Merge by YEAR + DOY
df = target_df.merge(meta_all, on=['YEAR', 'DOY'], how='left')

# =========================================================
# 3) Step 1: explainability-driven pre-screening (candidate features)
# =========================================================
def pick_features_step1(df_: pd.DataFrame):
    cols = set(df_.columns)

    # ---- A) PM2.5 distribution statistics (core)
    pm_core = [
        'pm25_min', 'pm25_median', 'pm25_IQR',
        'pm25_max', 'pm25_mean', 'pm25_std'
    ]
    pm_feats = [c for c in pm_core if c in cols]

    # ---- B) Meteorological regime features: lag01 only
    met_vars = ['WindSpeed10m', 'CloudCover24h', 'VaporPressure', 'SnowDepth', 'Tmean_24h']
    met_stats = ['mean', 'std', 'p50', 'min', 'max', 'range']

    met_feats = []
    for v in met_vars:
        for s in met_stats:
            c = f"Met_{v}_{s}_lag01"
            if c in cols:
                met_feats.append(c)

    step1 = sorted(set(pm_feats + met_feats))
    return step1, {'pm_feats': pm_feats, 'met_feats': met_feats}

step1_features, step1_detail = pick_features_step1(df)

# Save Step1 feature list
pd.Series(step1_features, name='feature').to_csv(
    os.path.join(model_save_dir, 'features_step1.csv'),
    index=False, encoding='utf-8-sig'
)

# Save grouping info
pd.DataFrame({
    'group': list(step1_detail.keys()),
    'n_features': [len(step1_detail[k]) for k in step1_detail.keys()]
}).to_csv(os.path.join(model_save_dir, 'features_step1_groups.csv'),
          index=False, encoding='utf-8-sig')

print(f"[INFO] Step1 candidate feature count: {len(step1_features)}. Saved: features_step1.csv")

# =========================================================
# 4) Step 2: quick denoising (drop missing >20%; drop near-zero variance)
# =========================================================
def filter_features_step2(df_: pd.DataFrame, features: list,
                          missing_thr: float = 0.20,
                          var_thr: float = 1e-4,
                          use_iqr: bool = True):
    X = df_[features].copy()

    # Force numeric (non-numeric -> NaN)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    # (1) Missing rate filter
    miss_rate = X.isna().mean()
    keep1 = miss_rate[miss_rate <= missing_thr].index.tolist()

    # (2) Near-zero variance filter: IQR==0 (robust) + var threshold fallback
    X1 = X[keep1]
    if use_iqr:
        q75 = X1.quantile(0.75, numeric_only=True)
        q25 = X1.quantile(0.25, numeric_only=True)
        iqr = q75 - q25
        keep2 = iqr[iqr > 0].index.tolist()
    else:
        keep2 = keep1

    X2 = X1[keep2]
    vari = X2.var(numeric_only=True)
    keep3 = vari[vari > var_thr].index.tolist()

    dropped_missing = sorted(set(features) - set(keep1))
    dropped_iqr = sorted(set(keep1) - set(keep2))
    dropped_var = sorted(set(keep2) - set(keep3))

    return keep3, {
        'missing_rate': miss_rate,
        'dropped_missing': dropped_missing,
        'dropped_iqr0': dropped_iqr,
        'dropped_var': dropped_var
    }

final_features, fs_detail = filter_features_step2(df, step1_features)

# Save Step2 results
pd.Series(final_features, name='feature').to_csv(
    os.path.join(model_save_dir, 'features_final.csv'),
    index=False, encoding='utf-8-sig'
)
fs_detail['missing_rate'].to_csv(
    os.path.join(model_save_dir, 'features_missing_rate.csv'),
    encoding='utf-8-sig'
)
pd.Series(fs_detail['dropped_missing'], name='dropped').to_csv(
    os.path.join(model_save_dir, 'dropped_missing_gt20pct.csv'),
    index=False, encoding='utf-8-sig'
)
pd.Series(fs_detail['dropped_iqr0'], name='dropped').to_csv(
    os.path.join(model_save_dir, 'dropped_iqr0.csv'),
    index=False, encoding='utf-8-sig'
)
pd.Series(fs_detail['dropped_var'], name='dropped').to_csv(
    os.path.join(model_save_dir, 'dropped_var_small.csv'),
    index=False, encoding='utf-8-sig'
)

print(f"[INFO] Step2 final feature count: {len(final_features)}. Saved: features_final.csv")

# =========================================================
# 5) Year-stratified split (reproducible)
# =========================================================
def stratified_year_split(df_: pd.DataFrame, year_col: str = 'YEAR', test_size: float = 0.2, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED_MASTER)

    test_indices = []
    train_indices = []

    for year in sorted(df_[year_col].unique()):
        year_indices = df_[df_[year_col] == year].index.values
        test_size_year = int(len(year_indices) * test_size)
        year_indices = rng.permutation(year_indices)

        test_indices.extend(year_indices[:test_size_year])
        train_indices.extend(year_indices[test_size_year:])

    return np.array(train_indices), np.array(test_indices)

# =========================================================
# 6) Parameter strategy: toggle tuning (default off)
# =========================================================
USE_TUNING = False

RF_PRESET_PARAMS = dict(
    n_estimators=500,
    max_features=0.6,
    max_depth=25,
    min_samples_split=10,
    bootstrap=True,
    random_state=SEED_MASTER,
    n_jobs=-1
)

def parameter_search(model, x, y):
    print('  Starting coarse random search...')
    n_estimators_coarse = [int(v) for v in np.linspace(50, 1000, num=20)]
    max_features_coarse = [int(v) for v in np.linspace(1, x.shape[1], num=20)]
    max_depth_coarse = [None] + list(range(3, 51, 2))
    min_samples_split_coarse = [2, 5, 10, 15, 20]
    param_dist_coarse = {
        'n_estimators': n_estimators_coarse,
        'max_features': max_features_coarse,
        'max_depth': max_depth_coarse,
        'min_samples_split': min_samples_split_coarse
    }
    model_random = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist_coarse,
        scoring='neg_mean_squared_error',
        cv=5,
        n_iter=100,
        random_state=SEED_MASTER,
        n_jobs=-1
    )
    start = time.time()
    model_random.fit(x, y)
    print(f'  Best coarse params: {model_random.best_params_}, time: {time.time() - start:.2f}s')
    return model_random.best_params_

def parameter_search_fine(model, x, y, best_coarse):
    print('  Starting fine grid search...')
    n_estimators_fine = np.linspace(
        int(best_coarse['n_estimators'] * 0.8),
        int(best_coarse['n_estimators'] * 1.2),
        num=10
    ).astype(int)

    max_features_fine = np.unique(np.linspace(
        max(1, best_coarse['max_features'] - int((x.shape[1] - 1) * 0.2)),
        min(x.shape[1], best_coarse['max_features'] + int((x.shape[1] - 1) * 0.2)),
        num=5
    ).astype(int))

    if best_coarse['max_depth'] is None:
        max_depth_fine = [None, 3, 5, 7, 10, 15]
    else:
        low = max(1, best_coarse['max_depth'] - int(48 * 0.2))
        high = best_coarse['max_depth'] + int(48 * 0.2)
        max_depth_fine = list(range(low, high + 1))

    min_split_low = max(2, int(best_coarse['min_samples_split'] * 0.8))
    min_split_high = int(best_coarse['min_samples_split'] * 1.2)
    min_samples_split_fine = list(range(min_split_low, min_split_high + 1))

    param_grid_fine = {
        'n_estimators': n_estimators_fine.tolist(),
        'max_features': max_features_fine.tolist(),
        'max_depth': max_depth_fine,
        'min_samples_split': min_samples_split_fine
    }

    model_grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid_fine,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    start = time.time()
    model_grid.fit(x, y)
    print(f'  Best fine params: {model_grid.best_params_}, time: {time.time() - start:.2f}s')
    return model_grid.best_estimator_

# =========================================================
# 7) Prepare training data (features + outcome)
# =========================================================
work_df = df[['YEAR', 'DOY', 'R2_Kfold'] + final_features].copy()

for c in final_features:
    work_df[c] = pd.to_numeric(work_df[c], errors='coerce')

# Export: raw analysis dataset (before imputation)
analysis_raw_path = os.path.join(model_save_dir, 'analysis_dataset_raw_with_na.csv')
work_df.to_csv(analysis_raw_path, index=False, encoding='utf-8-sig')
print(f"[INFO] Exported raw analysis dataset (with NA): {analysis_raw_path}")

# Impute missing values with column means
work_df[final_features] = work_df[final_features].fillna(
    work_df[final_features].mean(numeric_only=True)
)

# Export: imputed final analysis dataset
analysis_imp_path = os.path.join(model_save_dir, 'analysis_dataset_imputed.csv')
work_df.to_csv(analysis_imp_path, index=False, encoding='utf-8-sig')
print(f"[INFO] Exported imputed analysis dataset: {analysis_imp_path}")

X = work_df[final_features].values
y = work_df['R2_Kfold'].values

# =========================================================
# 8) Main loop
# =========================================================
def check_correct(row):
    thresh = year_thresholds[int(row['YEAR'])]
    pred = row['Predicted_R2_Kfold']
    actual = row['Actual_R2_Kfold']
    return 1 if (pred >= thresh and actual >= thresh) or (pred < thresh and actual < thresh) else 0

n_repeats = 999
metrics_list = []

best_accuracy = 0.0
best_model = None
best_iter = -1
best_fold_results = None
best_params = None

split_rng = np.random.RandomState(SEED_MASTER)

# =========================================================
# New: write all prints to txt (and keep console output)
# =========================================================
orig_stdout = sys.stdout
with open(log_path, 'w', encoding='utf-8') as f_log:
    sys.stdout = Tee(orig_stdout, f_log)

    print(f"[INFO] Logging all prints to: {log_path}")
    print(f"[INFO] SEED_MASTER={SEED_MASTER}, n_repeats={n_repeats}, USE_TUNING={USE_TUNING}")
    print(f"[INFO] year_thresholds={year_thresholds}")
    print(f"[INFO] n_final_features={len(final_features)}")

    for iteration in range(1, n_repeats + 1):
        print(f"\n===== Repeat {iteration} / {n_repeats} =====")

        train_idx, test_idx = stratified_year_split(work_df, year_col='YEAR', test_size=0.2, rng=split_rng)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Core: tuning or preset parameters
        if USE_TUNING:
            base_model = RandomForestRegressor(random_state=SEED_MASTER, n_jobs=-1)
            best_coarse = parameter_search(base_model, X_train, y_train)
            best_model_iter = parameter_search_fine(base_model, X_train, y_train, best_coarse)

            final_model = RandomForestRegressor(
                n_estimators=best_model_iter.n_estimators,
                max_features=best_model_iter.max_features,
                max_depth=best_model_iter.max_depth,
                min_samples_split=best_model_iter.min_samples_split,
                bootstrap=best_model_iter.bootstrap,
                random_state=SEED_MASTER,
                n_jobs=-1
            )
        else:
            final_model = RandomForestRegressor(**RF_PRESET_PARAMS)

        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        fold_results = pd.DataFrame({
            'Predicted_R2_Kfold': y_pred,
            'Actual_R2_Kfold': y_test,
            'YEAR': work_df.loc[test_idx, 'YEAR'].values,
            'DOY': work_df.loc[test_idx, 'DOY'].values
        }, index=test_idx)

        fold_results['Correct'] = fold_results.apply(check_correct, axis=1)
        accuracy = fold_results['Correct'].mean()
        accuracy_by_year = fold_results.groupby('YEAR')['Correct'].mean()

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics_list.append({
            'iteration': iteration,
            'accuracy': accuracy,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'accuracy_2018': float(accuracy_by_year.get(2018, np.nan)),
            'accuracy_2019': float(accuracy_by_year.get(2019, np.nan)),
            'accuracy_2020': float(accuracy_by_year.get(2020, np.nan)),
            'n_features': len(final_features)
        })

        print(f"  Results for repeat {iteration}:")
        print(f"    Overall model-adequacy accuracy: {accuracy:.2%}")
        for yy, acc in accuracy_by_year.items():
            print(f"    Accuracy in {int(yy)}: {acc:.2%}")
        print(f"    MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = final_model
            best_iter = iteration
            best_fold_results = fold_results.copy()
            best_params = {
                'n_estimators': final_model.n_estimators,
                'max_features': final_model.max_features,
                'max_depth': final_model.max_depth,
                'min_samples_split': final_model.min_samples_split
            }

        print(f"  Current best model-adequacy accuracy: {best_accuracy:.2%}")

    # =========================================================
    # 9) Outputs: predictions / metrics / best model / feature importance / params
    # =========================================================
    best_pred_path = os.path.join(model_save_dir, 'best_model_predictions.csv')
    best_fold_results.to_csv(best_pred_path, index=False, encoding='utf-8-sig')

    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(model_save_dir, 'cv_repeats_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')

    print("\nMetrics summary preview (first 5 rows):")
    print(metrics_df.head())

    print(f"\nBest model was obtained at repeat {best_iter}, with model-adequacy accuracy: {best_accuracy:.2%}")
    print(f"Best model parameters: {best_params}")

    best_model_path = os.path.join(model_save_dir, 'rf_best_model_final.joblib')
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved to: {best_model_path}")

    imp = pd.DataFrame({
        'feature': final_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    imp_path = os.path.join(model_save_dir, 'best_model_feature_importance.csv')
    imp.to_csv(imp_path, index=False, encoding='utf-8-sig')

    pd.Series(final_features, name='feature').to_csv(
        os.path.join(model_save_dir, 'best_model_features_used.csv'),
        index=False, encoding='utf-8-sig'
    )

    pd.Series(best_params).to_csv(
        os.path.join(model_save_dir, 'best_model_params.csv'),
        encoding='utf-8-sig'
    )

    # =========================================================
    # 10) New: descriptive summary table across 999 repeats
    #     For each metric: min / max / median / mean / IQR, etc.
    # =========================================================
    def metrics_summary_table(metrics_df_: pd.DataFrame,
                              metrics_cols: list,
                              out_path: str):
        """Compute descriptive statistics for repeated-run metrics and export to Excel."""
        rows = []
        for c in metrics_cols:
            s = pd.to_numeric(metrics_df_[c], errors='coerce').dropna()
            if s.empty:
                rows.append({
                    'metric': c,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'mean': np.nan,
                    'std': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                    'IQR': np.nan
                })
                continue

            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            rows.append({
                'metric': c,
                'min': float(s.min()),
                'max': float(s.max()),
                'median': float(s.median()),
                'mean': float(s.mean()),
                'std': float(s.std(ddof=1)),
                'q25': float(q25),
                'q75': float(q75),
                'IQR': float(q75 - q25)
            })

        out_df = pd.DataFrame(rows)
        out_df.to_excel(out_path, index=False)
        return out_df

    metric_cols_to_summarise = [
        'accuracy', 'mae', 'mse', 'r2',
        'accuracy_2018', 'accuracy_2019', 'accuracy_2020'
    ]

    summary_path = os.path.join(model_save_dir, 'cv_repeats_metrics_summary.xlsx')
    summary_df = metrics_summary_table(metrics_df, metric_cols_to_summarise, summary_path)

    print("\n[INFO] Exported descriptive summary table for 999 repeats:")
    print(summary_df)

    print(f"[DONE] Output files:\n"
          f"- {best_pred_path}\n"
          f"- {metrics_path}\n"
          f"- {imp_path}\n"
          f"- {best_model_path}\n"
          f"- {summary_path}")

# Restore stdout
sys.stdout = orig_stdout
print("All prints were saved to:", log_path)
