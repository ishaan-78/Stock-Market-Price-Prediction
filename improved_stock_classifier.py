# # improved_stock_classifier.py
# """
# Improved end-to-end script:
#  - safe date parsing and chronological split (no random leakage)
#  - richer feature engineering (returns, lags, MAs, vol, RSI, MACD)
#  - proper scaler fitted on training set only
#  - class imbalance handling (class weights + XGBoost scale_pos_weight)
#  - model training: LogisticRegression, SVC, XGBoost (with optional CV)
#  - visualizations: price+MA, histograms, corr heatmap, ROC & PR curves, confusion matrices
#  - simple backtest translating predictions to daily P&L (with optional transaction cost)
# """
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
#
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.metrics import (
#     roc_auc_score, roc_curve, auc,
#     precision_recall_curve, average_precision_score,
#     classification_report, confusion_matrix
# )
# from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier
# import warnings
# warnings.filterwarnings('ignore')
#
# # -------------------------
# # Utility feature functions
# # -------------------------
# def add_technical_features(df):
#     """Add a set of common features derived only from past data (no leakage)."""
#     # daily simple return
#     df['return_1'] = df['Close'].pct_change()
#
#     # lagged returns (1,2,5)
#     df['r_lag_1'] = df['return_1'].shift(1)
#     df['r_lag_2'] = df['return_1'].shift(2)
#     df['r_lag_5'] = df['return_1'].shift(5)
#
#     # moving averages and their ratios
#     df['ma5'] = df['Close'].rolling(window=5).mean()
#     df['ma10'] = df['Close'].rolling(window=10).mean()
#     df['ma20'] = df['Close'].rolling(window=20).mean()
#     df['ma5_ma20_ratio'] = df['ma5'] / df['ma20']
#
#     # volatility (rolling std of returns)
#     df['vol_10'] = df['return_1'].rolling(window=10).std()
#     df['vol_20'] = df['return_1'].rolling(window=20).std()
#
#     # intraday range features
#     df['open_close'] = df['Open'] - df['Close']        # positive if opened higher than closed
#     df['high_low'] = df['High'] - df['Low']            # intraday range (positive)
#
#     # volume-based features (ratio vs rolling mean)
#     df['vol_ma5'] = df['Volume'].rolling(window=5).mean()
#     df['vol_ratio'] = df['Volume'] / (df['vol_ma5'] + 1e-9)
#
#     # RSI (14)
#     window = 14
#     delta = df['Close'].diff()
#     up = delta.clip(lower=0)
#     down = -1 * delta.clip(upper=0)
#     ma_up = up.rolling(window=window, min_periods=window).mean()
#     ma_down = down.rolling(window=window, min_periods=window).mean()
#     rs = ma_up / (ma_down + 1e-9)
#     df['rsi_14'] = 100 - (100 / (1 + rs))
#
#     # MACD (12-26-9)
#     ema12 = df['Close'].ewm(span=12, adjust=False).mean()
#     ema26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['macd'] = ema12 - ema26
#     df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
#     df['macd_hist'] = df['macd'] - df['macd_signal']
#
#     return df
#
# # -------------------------
# # Load & basic cleaning
# # -------------------------
# col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
# df = pd.read_csv("prices.csv", header=None, names=col_names)   # change if your CSV has headers
#
# # parse dates robustly
# df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
#
# # If Adj Close equals Close for all rows, drop it to avoid redundancy.
# if 'Adj Close' in df.columns:
#     try:
#         if df['Close'].equals(df['Adj Close']):
#             df.drop(columns=['Adj Close'], inplace=True)
#     except Exception:
#         pass
#
# # set Date as index (nice for plotting)
# df.set_index('Date', inplace=True)
#
# # -------------------------
# # Feature engineering & target
# # -------------------------
# df = add_technical_features(df)
#
# # is_quarter_end (simple month%3==0), day-of-week, month
# df['month'] = df.index.month
# df['day_of_week'] = df.index.dayofweek    # 0=Monday .. 6=Sunday
# df['is_quarter_month'] = (df['month'] % 3 == 0).astype(int)
#
# # create target: 1 if next day's close > today's close, else 0
# df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
# # drop rows that now have NaNs (rolling windows + last row with no next day)
# df = df.dropna().copy()
#
# # -------------------------
# # Quick EDA / visuals
# # -------------------------
# plt.figure(figsize=(14,6))
# plt.plot(df.index, df['Close'], label='Close')
# plt.plot(df.index, df['ma5'], label='MA5', linestyle='--')
# plt.plot(df.index, df['ma20'], label='MA20', linestyle=':')
# plt.title('Close price with moving averages')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # class balance
# print("Target class distribution:\n", df['target'].value_counts(normalize=False))
# plt.figure(figsize=(5,5))
# plt.pie(df['target'].value_counts().values, labels=['Down/No', 'Up/Yes'], autopct='%1.1f%%')
# plt.title('Next-day up vs down')
# plt.show()
#
# # distributions (histograms)
# plot_cols = ['return_1', 'vol_ratio', 'rsi_14', 'ma5_ma20_ratio']
# plt.figure(figsize=(12,8))
# for i, col in enumerate(plot_cols):
#     plt.subplot(2,2,i+1)
#     sb.histplot(df[col], kde=False)
#     plt.title(col)
# plt.tight_layout()
# plt.show()
#
# # correlation heatmap (full numeric)
# plt.figure(figsize=(12,10))
# num_cols = df.select_dtypes(include=[np.number]).columns
# corr = df[num_cols].corr()
# sb.heatmap(corr, annot=False, cmap='coolwarm', center=0)
# plt.title('Correlation matrix (numeric columns)')
# plt.show()
#
# # -------------------------
# # Train/Validation split (chronological)
# # -------------------------
# # Use last 10% as validation (you can change to 20% if preferred)
# valid_frac = 0.1
# split_ix = int(len(df) * (1 - valid_frac))
# train_df = df.iloc[:split_ix].copy()
# valid_df = df.iloc[split_ix:].copy()
#
# print(f"Train rows: {len(train_df)}, Validation rows: {len(valid_df)}")
#
# # features to use — pick a mix of technicals; avoid including future info
# feature_cols = [
#     'r_lag_1','r_lag_2','r_lag_5',
#     'ma5_ma20_ratio','vol_20',
#     'open_close','high_low',
#     'vol_ratio','rsi_14','macd_hist',
#     'is_quarter_month','day_of_week'
# ]
#
# X_train = train_df[feature_cols].values
# y_train = train_df['target'].values
# X_valid = valid_df[feature_cols].values
# y_valid = valid_df['target'].values
#
# # -------------------------
# # Handle scaling (fit on train only)
# # -------------------------
# scaler = StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_valid_scaled = scaler.transform(X_valid)
#
# # -------------------------
# # Class imbalance handling
# # -------------------------
# # compute pos/neg counts to set scale_pos_weight for XGBoost
# neg = np.sum(y_train == 0)
# pos = np.sum(y_train == 1)
# scale_pos_weight = neg / (pos + 1e-9)
# print(f"Train pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.3f}")
#
# # -------------------------
# # Models
# # -------------------------
# models = {
#     'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
#     'SVC_poly': SVC(kernel='poly', probability=True, class_weight='balanced', random_state=42),
#     'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
#                          scale_pos_weight=scale_pos_weight, random_state=42)
# }
#
# # Train and evaluate
# results = {}
# for name, model in models.items():
#     print(f"\nTraining {name} ...")
#     model.fit(X_train_scaled, y_train)
#     # predicted probabilities
#     if hasattr(model, "predict_proba"):
#         prob_train = model.predict_proba(X_train_scaled)[:,1]
#         prob_valid = model.predict_proba(X_valid_scaled)[:,1]
#     else:
#         # SVC without probability would use decision_function (not the case here since probability=True)
#         prob_train = model.decision_function(X_train_scaled)
#         prob_valid = model.decision_function(X_valid_scaled)
#
#     auc_train = roc_auc_score(y_train, prob_train)
#     auc_valid = roc_auc_score(y_valid, prob_valid)
#     print(f" Train ROC AUC: {auc_train:.4f} | Valid ROC AUC: {auc_valid:.4f}")
#
#     y_pred = model.predict(X_valid_scaled)
#     print("Validation classification report:")
#     print(classification_report(y_valid, y_pred, digits=4))
#     results[name] = {
#         'model': model,
#         'prob_valid': prob_valid,
#         'prob_train': prob_train,
#         'auc_train': auc_train,
#         'auc_valid': auc_valid,
#         'y_pred_valid': y_pred
#     }
#
# # -------------------------
# # ROC and Precision-Recall plots (all models)
# # -------------------------
# plt.figure(figsize=(12,5))
# for name, info in results.items():
#     fpr, tpr, _ = roc_curve(y_valid, info['prob_valid'])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
# plt.plot([0,1],[0,1],'k--', alpha=0.5)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curves (validation)")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# plt.figure(figsize=(12,5))
# for name, info in results.items():
#     precision, recall, _ = precision_recall_curve(y_valid, info['prob_valid'])
#     ap = average_precision_score(y_valid, info['prob_valid'])
#     plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curves (validation)")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # -------------------------
# # Confusion matrices & threshold sensitivity
# # -------------------------
# from sklearn.metrics import ConfusionMatrixDisplay
# for name, info in results.items():
#     print(f"\nConfusion matrix for {name} (threshold=0.5):")
#     cm = confusion_matrix(y_valid, info['y_pred_valid'])
#     disp = ConfusionMatrixDisplay(cm, display_labels=['Down/No', 'Up/Yes'])
#     disp.plot(cmap='Blues')
#     plt.title(f"{name} Confusion Matrix (validation)")
#     plt.show()
#
# # -------------------------
# # Feature importance (XGB) and simple explainability
# # -------------------------
# xgb_model = results['XGB']['model']
# if hasattr(xgb_model, 'feature_importances_'):
#     fi = xgb_model.feature_importances_
#     fi_series = pd.Series(fi, index=feature_cols).sort_values(ascending=False)
#     plt.figure(figsize=(8,5))
#     sb.barplot(x=fi_series.values, y=fi_series.index)
#     plt.title('XGBoost feature importances')
#     plt.tight_layout()
#     plt.show()
#     print("Top features (XGB):")
#     print(fi_series.head(10))
#
# # -------------------------
# # Simple backtest: translate predictions to daily returns
# # -------------------------
# # Strategy: if model predicts next-day up (prob > threshold), go long for next day; else flat.
# # We compute next-day return series and simulate cumulative returns.
# def backtest_from_probs(df_all, valid_index_start, prob_valid, threshold=0.5, tc_per_trade=0.0):
#     """Return a DataFrame with daily strategy returns and cumulative returns for the validation period."""
#     # align predictions with valid_df index
#     valid_index = df_all.index[valid_index_start:]
#     preds = (prob_valid >= threshold).astype(int)
#
#     # next-day simple return used as profit when long today -> next-day open/close assumptions:
#     # we're using close-to-close returns here as a simple proxy (note: in practice you'd use intraday fills).
#     next_return = df_all['Close'].pct_change().shift(-1).loc[valid_index]  # same as earlier return_1 but aligned
#     # Strategy returns: if prediction==1 -> capture next_return, else 0
#     strat_return = preds * next_return.values
#     # subtract transaction cost when a position is opened (simple model: cost applied when prediction changes from 0->1 or 1->0)
#     # Simpler: charge tc_per_trade whenever you take a long position (i.e., preds==1)
#     strat_return_net = strat_return - (preds * tc_per_trade)
#     cum_ret = (1 + strat_return_net).cumprod() - 1
#     result = pd.DataFrame({
#         'pred': preds,
#         'next_return': next_return.values,
#         'strat_return_net': strat_return_net,
#         'cum_ret': cum_ret
#     }, index=valid_index)
#     return result
#
# # choose model to backtest (XGB is a good choice)
# chosen = 'XGB'
# bt = backtest_from_probs(df, split_ix, results[chosen]['prob_valid'], threshold=0.5, tc_per_trade=0.0005)
# plt.figure(figsize=(12,6))
# plt.plot(bt.index, bt['cum_ret'], label=f"{chosen} strategy (threshold=0.5)")
# # buy-and-hold over same validation interval
# bnh = (1 + df['Close'].pct_change().loc[bt.index].fillna(0)).cumprod() - 1
# plt.plot(bnh.index, bnh.values, label='Buy & Hold')
# plt.title(f"Simple backtest: {chosen} vs Buy & Hold (validation)")
# plt.legend()
# plt.show()
# print(bt.tail(10))
#
# # -------------------------
# # OPTIONAL: TimeSeries CV + GridSearch for XGBoost hyperparams (uncomment if you want to run)
# # -------------------------
# # Note: Can be slow. Use TimeSeriesSplit for time-aware CV.
# # tscv = TimeSeriesSplit(n_splits=4)
# # param_grid = {'n_estimators':[50,100,200], 'max_depth':[3,5], 'learning_rate':[0.01,0.1]}
# # xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
# # gs = GridSearchCV(xgb, param_grid, cv=tscv, scoring='roc_auc', n_jobs=-1)
# # gs.fit(X_train_scaled, y_train)
# # print("Best params (cv):", gs.best_params_)
# # best_xgb = gs.best_estimator_








# four_models_with_diagnostics.py
"""
Four-model comparison + diagnostics.

Trains:
1) Direct H-day (full OHLCV features)
2) Direct H-day (close-only features)
3) Recursive 1-day minimal (close-only)
4) Recursive 1-day extended (close-only)

Then:
- Produces recursive daily forecasts for next H business days
- Produces direct H-day forecast points
- Adds diagnostics: baseline comparison, histograms, scatter plots,
  feature importances, mean/std/correlation checks
- Optionally quick-tunes XGBoosts with larger capacity for testing (QUICK_TUNE)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.tseries.offsets import BDay
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from matplotlib.dates import DateFormatter, AutoDateLocator
import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# USER PARAMETERS
# -------------------------
CSV_PATH = "prices.csv"
HORIZON = 30            # change to 60 to predict 60 days ahead
VALIDATION_FRAC = 0.15
TS_CV_SPLITS = 4
RANDOM_STATE = 42
N_HIST_PLOT = 120       # number of historical days to show on plot
QUICK_TUNE = False      # set True to run a quick higher-capacity XGB retrain for direct models

# -------------------------
# Helpers (date parsing + features)
# -------------------------
def robust_read_csv_with_dates(path, col_names):
    df_try = pd.read_csv(path, header=None, names=col_names)
    df_try['Date'] = pd.to_datetime(df_try['Date'], errors='coerce', infer_datetime_format=True)
    def dates_look_reasonable(dt_series):
        if dt_series.isna().all():
            return False
        yrs = dt_series.dropna().dt.year
        if yrs.empty:
            return False
        yr_max = yrs.max()
        yr_min = yrs.min()
        now_year = datetime.datetime.now().year
        return (1900 <= yr_min <= now_year + 1) and (1900 <= yr_max <= now_year + 1)
    if dates_look_reasonable(df_try['Date']):
        return df_try
    formats_to_try = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y', '%Y/%m/%d']
    for fmt in formats_to_try:
        df_try['Date'] = pd.to_datetime(df_try['Date'].astype(str), errors='coerce', format=fmt)
        if dates_look_reasonable(df_try['Date']):
            return df_try
    df_try['Date'] = pd.to_datetime(df_try['Date'].astype(str), errors='coerce', dayfirst=True, infer_datetime_format=True)
    if dates_look_reasonable(df_try['Date']):
        return df_try
    df_try['Date'] = pd.to_datetime(df_try['Date'], errors='coerce')
    return df_try

def add_long_horizon_features(df):
    df['ret_1'] = df['Close'].pct_change()
    for w in [5, 10, 20, 30, 60, 90]:
        df[f'ret_cum_{w}'] = (df['Close'] / df['Close'].shift(w) - 1)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['ma100'] = df['Close'].rolling(100).mean()
    df['ma20_ma100'] = df['ma20'] / (df['ma100'] + 1e-9)
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_60'] = df['ret_1'].rolling(60).std()
    def slope(series):
        x = np.arange(len(series))
        if np.any(np.isnan(series)):
            return np.nan
        return np.polyfit(x, series, 1)[0]
    df['slope_10'] = df['Close'].rolling(10).apply(slope, raw=True)
    df['slope_20'] = df['Close'].rolling(20).apply(slope, raw=True)
    df['slope_60'] = df['Close'].rolling(60).apply(slope, raw=True)
    df['high_low_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['open_close'] = (df['Open'] - df['Close']) / (df['Close'] + 1e-9)
    df['vol_ma_20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio_20'] = df['Volume'] / (df['vol_ma_20'] + 1e-9)
    df['day_of_week'] = df.index.dayofweek
    df['is_quarter_month'] = (df.index.month % 3 == 0).astype(int)
    return df

def make_close_only_features(close_series):
    s = pd.DataFrame({'Close': close_series})
    s['ret_1'] = s['Close'].pct_change()
    s['ret_lag_1'] = s['ret_1'].shift(1)
    s['ma5'] = s['Close'].rolling(5).mean()
    s['ma10'] = s['Close'].rolling(10).mean()
    s['ma20'] = s['Close'].rolling(20).mean()
    s['ma5_ma20'] = s['ma5'] / (s['ma20'] + 1e-9)
    def slope_of_arr(arr):
        x = np.arange(len(arr))
        if np.any(np.isnan(arr)):
            return np.nan
        return np.polyfit(x, arr, 1)[0]
    s['slope_5'] = s['Close'].rolling(5).apply(slope_of_arr, raw=True)
    s['slope_10'] = s['Close'].rolling(10).apply(slope_of_arr, raw=True)
    s['slope_20'] = s['Close'].rolling(20).apply(slope_of_arr, raw=True)
    s['vol_20'] = s['ret_1'].rolling(20).std()
    s['day_of_week'] = s.index.dayofweek
    s['is_quarter_month'] = (s.index.month % 3 == 0).astype(int)
    return s

# -------------------------
# Load and parse CSV
# -------------------------
col_names = ['Date','Open','High','Low','Close','Volume','Adj Close']
df_raw = robust_read_csv_with_dates(CSV_PATH, col_names)
df_raw = df_raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
df_raw = df_raw.dropna(subset=['Date']).reset_index(drop=True)
df_raw.set_index('Date', inplace=True)
yr_min = df_raw.index.year.min() if len(df_raw)>0 else None
yr_max = df_raw.index.year.max() if len(df_raw)>0 else None
print(f"Parsed dates range: {yr_min} to {yr_max} (should match your CSV).")

if 'Adj Close' in df_raw.columns:
    try:
        if df_raw['Close'].equals(df_raw['Adj Close']):
            df_raw.drop(columns=['Adj Close'], inplace=True)
    except Exception:
        pass
df = df_raw.copy()

# -------------------------
# Build datasets (same as previous)
# -------------------------
df_full = add_long_horizon_features(df.copy())
h = HORIZON
df_full[f'target_ret_{h}'] = (df_full['Close'].shift(-h) / df_full['Close']) - 1

feature_cols_full = [
    'ret_cum_5','ret_cum_10','ret_cum_20','ret_cum_30','ret_cum_60',
    'ma20_ma100','vol_20','vol_60','slope_10','slope_20','slope_60',
    'high_low_range','open_close','vol_ratio_20','day_of_week','is_quarter_month'
]
feature_cols_full = [c for c in feature_cols_full if c in df_full.columns]
df_direct_full = df_full.dropna(subset=feature_cols_full + [f'target_ret_{h}']).copy()
if df_direct_full.empty:
    raise ValueError("No usable rows for direct-full after dropna — check data & horizon.")

# split
split_ix_full = int(len(df_direct_full) * (1 - VALIDATION_FRAC))
train_full = df_direct_full.iloc[:split_ix_full].copy()
valid_full = df_direct_full.iloc[split_ix_full:].copy()
Xf_train = train_full[feature_cols_full].values
yf_train = train_full[f'target_ret_{h}'].values
Xf_valid = valid_full[feature_cols_full].values
yf_valid = valid_full[f'target_ret_{h}'].values

scaler_full = StandardScaler().fit(Xf_train)
Xf_train_s = scaler_full.transform(Xf_train)
Xf_valid_s = scaler_full.transform(Xf_valid)

# direct-full models
ridge_full = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge_full.fit(Xf_train_s, yf_train)

tscv = TimeSeriesSplit(n_splits=TS_CV_SPLITS)
xgb_full = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)
param_grid_full = {'n_estimators':[100,300], 'max_depth':[3,5], 'learning_rate':[0.01,0.05]}
pipe_full = Pipeline([('xgb', xgb_full)])
gs_full = GridSearchCV(pipe_full,
                       {'xgb__n_estimators': param_grid_full['n_estimators'],
                        'xgb__max_depth': param_grid_full['max_depth'],
                        'xgb__learning_rate': param_grid_full['learning_rate']},
                       cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
print("Grid search direct-full (may take a few minutes)...")
gs_full.fit(Xf_train_s, yf_train)
best_xgb_full = gs_full.best_estimator_.named_steps['xgb']
pred_xgb_full_valid = best_xgb_full.predict(Xf_valid_s)
pred_ridge_full_valid = ridge_full.predict(Xf_valid_s)
pred_stack_full_valid = 0.5 * pred_ridge_full_valid + 0.5 * pred_xgb_full_valid

# direct-closeonly dataset
df_closeonly_feats = make_close_only_features(df['Close'])
for w in [5,10,20,30,60]:
    df_closeonly_feats[f'ret_cum_{w}'] = (df_closeonly_feats['Close'] / df_closeonly_feats['Close'].shift(w) - 1)
df_closeonly_feats[f'target_ret_{h}'] = (df_closeonly_feats['Close'].shift(-h) / df_closeonly_feats['Close']) - 1

feature_cols_close_direct = [c for c in [
    'ret_cum_5','ret_cum_10','ret_cum_20','ret_cum_30','ret_cum_60',
    'ma5','ma10','ma20','ma5_ma20','vol_20','slope_10','slope_20','day_of_week','is_quarter_month'
] if c in df_closeonly_feats.columns]
df_direct_close = df_closeonly_feats.dropna(subset=feature_cols_close_direct + [f'target_ret_{h}']).copy()
if df_direct_close.empty:
    raise ValueError("No usable rows for direct-closeonly after dropna — check data & horizon.")

split_ix_close = int(len(df_direct_close) * (1 - VALIDATION_FRAC))
train_close = df_direct_close.iloc[:split_ix_close].copy()
valid_close = df_direct_close.iloc[split_ix_close:].copy()
Xc_train = train_close[feature_cols_close_direct].values
yc_train = train_close[f'target_ret_{h}'].values
Xc_valid = valid_close[feature_cols_close_direct].values
yc_valid = valid_close[f'target_ret_{h}'].values

scaler_close_direct = StandardScaler().fit(Xc_train)
Xc_train_s = scaler_close_direct.transform(Xc_train)
Xc_valid_s = scaler_close_direct.transform(Xc_valid)

ridge_close = Ridge(alpha=1.0, random_state=RANDOM_STATE)
ridge_close.fit(Xc_train_s, yc_train)

pipe_close = Pipeline([('xgb', XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1))])
gs_close = GridSearchCV(pipe_close,
                        {'xgb__n_estimators':[100,200], 'xgb__max_depth':[3,4], 'xgb__learning_rate':[0.01,0.05]},
                        cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
print("Grid search direct-closeonly (short)...")
gs_close.fit(Xc_train_s, yc_train)
best_xgb_close = gs_close.best_estimator_.named_steps['xgb']
pred_xgb_close_valid = best_xgb_close.predict(Xc_valid_s)
pred_ridge_close_valid = ridge_close.predict(Xc_valid_s)
pred_stack_close_valid = 0.5 * pred_ridge_close_valid + 0.5 * pred_xgb_close_valid

# recursive minimal
df_1d_min = df_closeonly_feats[['Close','ret_1','ret_lag_1','ma5','ma10','ma20','ma5_ma20','day_of_week','is_quarter_month']].copy()
df_1d_min['target_ret_1'] = (df_1d_min['Close'].shift(-1) / df_1d_min['Close']) - 1
df_1d_min = df_1d_min.dropna().copy()
split_ix_1min = int(len(df_1d_min) * (1 - VALIDATION_FRAC))
train_1min = df_1d_min.iloc[:split_ix_1min].copy()
valid_1min = df_1d_min.iloc[split_ix_1min:].copy()
feat1_min_cols = [c for c in ['ret_lag_1','ma5','ma10','ma20','ma5_ma20','day_of_week','is_quarter_month'] if c in df_1d_min.columns]
X1min_train = train_1min[feat1_min_cols].values
y1min_train = train_1min['target_ret_1'].values
X1min_valid = valid_1min[feat1_min_cols].values
y1min_valid = valid_1min['target_ret_1'].values
scaler_1min = StandardScaler().fit(X1min_train)
X1min_train_s = scaler_1min.transform(X1min_train)
X1min_valid_s = scaler_1min.transform(X1min_valid)
xgb_1_min = XGBRegressor(objective='reg:squarederror', n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
xgb_1_min.fit(X1min_train_s, y1min_train)
pred_1min_valid = xgb_1_min.predict(X1min_valid_s)

# recursive extended
df_1d_ext = df_closeonly_feats.copy()
df_1d_ext['target_ret_1'] = (df_1d_ext['Close'].shift(-1) / df_1d_ext['Close']) - 1
df_1d_ext = df_1d_ext.dropna().copy()
split_ix_1ext = int(len(df_1d_ext) * (1 - VALIDATION_FRAC))
train_1ext = df_1d_ext.iloc[:split_ix_1ext].copy()
valid_1ext = df_1d_ext.iloc[split_ix_1ext:].copy()
feat1_ext_cols = [c for c in ['ret_lag_1','ma5','ma10','ma20','ma5_ma20','slope_10','slope_20','vol_20','day_of_week','is_quarter_month'] if c in df_1d_ext.columns]
X1ext_train = train_1ext[feat1_ext_cols].values
y1ext_train = train_1ext['target_ret_1'].values
X1ext_valid = valid_1ext[feat1_ext_cols].values
y1ext_valid = valid_1ext['target_ret_1'].values
scaler_1ext = StandardScaler().fit(X1ext_train)
X1ext_train_s = scaler_1ext.transform(X1ext_train)
X1ext_valid_s = scaler_1ext.transform(X1ext_valid)
xgb_1_ext = XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
xgb_1_ext.fit(X1ext_train_s, y1ext_train)
pred_1ext_valid = xgb_1_ext.predict(X1ext_valid_s)

# -------------------------
# Diagnostics helpers
# -------------------------
def baseline_zero(y_true):
    y_pred = np.zeros_like(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'mae':mae,'rmse':rmse}

def baseline_mean(y_train, y_valid):
    mean_val = np.nanmean(y_train)
    y_pred = np.full_like(y_valid, fill_value=mean_val, dtype=float)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    return {'mae':mae,'rmse':rmse, 'mean':mean_val}

def show_pred_stats(y_true, y_pred, name):
    print(f"\n{name} stats:")
    print(f"  true mean {np.nanmean(y_true):.6f}, std {np.nanstd(y_true):.6f}")
    print(f"  pred mean {np.nanmean(y_pred):.6f}, std {np.nanstd(y_pred):.6f}")
    corr = np.corrcoef(np.nan_to_num(y_true), np.nan_to_num(y_pred))[0,1]
    print(f"  corr(pred, true) = {corr:.4f}")
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  MAE={mae:.6f}, RMSE={rmse:.6f}")

# -------------------------
# Evaluate validation predictions & baselines
# -------------------------
# direct-full valid preds already computed
show_pred_stats(yf_valid, pred_xgb_full_valid, "Direct-Full XGB (valid)")
print("Baseline (zero) for direct-full:", baseline_zero(yf_valid))
print("Baseline (train mean) for direct-full:", baseline_mean(yf_train, yf_valid))

show_pred_stats(yf_valid, pred_ridge_full_valid, "Direct-Full Ridge (valid)")
show_pred_stats(yf_valid, pred_stack_full_valid, "Direct-Full Stack (valid)")

show_pred_stats(yc_valid, pred_xgb_close_valid, "Direct-Close XGB (valid)")
print("Baseline (zero) for direct-close:", baseline_zero(yc_valid))
print("Baseline (train mean) for direct-close:", baseline_mean(yc_train, yc_valid))

show_pred_stats(yc_valid, pred_ridge_close_valid, "Direct-Close Ridge (valid)")
show_pred_stats(yc_valid, pred_stack_close_valid, "Direct-Close Stack (valid)")

show_pred_stats(y1min_valid, pred_1min_valid, "Recursive 1-day minimal XGB (valid)")
show_pred_stats(y1ext_valid, pred_1ext_valid, "Recursive 1-day extended XGB (valid)")

# histograms of predictions vs actual (for direct models)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sb.histplot(yf_valid, label='actual', kde=False, stat='density')
sb.histplot(pred_xgb_full_valid, label='pred_xgb_full', kde=False, stat='density', color='orange', alpha=0.6)
plt.title('Direct-Full: actual vs pred (valid)')
plt.legend()
plt.subplot(1,2,2)
sb.scatterplot(x=yf_valid, y=pred_xgb_full_valid, alpha=0.6)
plt.plot([yf_valid.min(), yf_valid.max()], [yf_valid.min(), yf_valid.max()], 'k--', linewidth=0.6)
plt.xlabel('actual'); plt.ylabel('predicted'); plt.title('Direct-Full: pred vs actual (valid)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sb.histplot(yc_valid, label='actual', kde=False, stat='density')
sb.histplot(pred_xgb_close_valid, label='pred_xgb_close', kde=False, stat='density', color='purple', alpha=0.6)
plt.title('Direct-Close: actual vs pred (valid)')
plt.legend()
plt.subplot(1,2,2)
sb.scatterplot(x=yc_valid, y=pred_xgb_close_valid, alpha=0.6)
plt.plot([yc_valid.min(), yc_valid.max()], [yc_valid.min(), yc_valid.max()], 'k--', linewidth=0.6)
plt.xlabel('actual'); plt.ylabel('predicted'); plt.title('Direct-Close: pred vs actual (valid)')
plt.tight_layout()
plt.show()

# Feature importances for direct XGBs
def show_feature_importances(xgb_model, feature_names, top_n=12, title="Feature importances"):
    if hasattr(xgb_model, 'feature_importances_'):
        fi = xgb_model.feature_importances_
        fi_s = pd.Series(fi, index=feature_names).sort_values(ascending=False)
        print(f"\n{title} (top {min(top_n,len(fi_s))}):")
        print(fi_s.head(top_n))
        plt.figure(figsize=(6, min(0.4*len(fi_s),6)))
        sb.barplot(x=fi_s.values[:top_n], y=fi_s.index[:top_n])
        plt.title(title)
        plt.tight_layout()
        plt.show()

show_feature_importances(best_xgb_full, feature_cols_full, top_n=12, title="Direct-Full XGB feature importances")
show_feature_importances(best_xgb_close, feature_cols_close_direct, top_n=12, title="Direct-Close XGB feature importances")

# -------------------------
# QUICK TUNE OPTION: retrain direct XGBs with larger capacity to test underfitting
# -------------------------
if QUICK_TUNE:
    print("\nQUICK_TUNE enabled: retraining direct XGBs with larger capacity to check underfitting...")
    # direct-full quick
    xgb_full_q = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1,
                              n_estimators=800, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)
    xgb_full_q.fit(Xf_train_s, yf_train)
    pred_xgb_full_q_valid = xgb_full_q.predict(Xf_valid_s)
    show_pred_stats(yf_valid, pred_xgb_full_q_valid, "Direct-Full XGB (quick_tune valid)")
    show_feature_importances(xgb_full_q, feature_cols_full, top_n=12, title="Direct-Full XGB (quick_tune) feature importances")

    # direct-close quick
    xgb_close_q = XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1,
                               n_estimators=600, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)
    xgb_close_q.fit(Xc_train_s, yc_train)
    pred_xgb_close_q_valid = xgb_close_q.predict(Xc_valid_s)
    show_pred_stats(yc_valid, pred_xgb_close_q_valid, "Direct-Close XGB (quick_tune valid)")
    show_feature_importances(xgb_close_q, feature_cols_close_direct, top_n=12, title="Direct-Close XGB (quick_tune) feature importances")

# -------------------------
# Produce direct points & recursive paths (same as before)
# -------------------------
# Direct points
latest_date = df.index.max()
latest_full_feats = df_full.loc[latest_date, feature_cols_full].copy().fillna(0.0)
X_latest_full_s = scaler_full.transform(latest_full_feats.values.reshape(1, -1))
pred_h_full_xgb = best_xgb_full.predict(X_latest_full_s)[0]
pred_close_h_full_xgb = df.loc[latest_date,'Close'] * (1 + pred_h_full_xgb)

latest_close_feats = df_closeonly_feats.loc[latest_date, feature_cols_close_direct].copy().fillna(0.0)
X_latest_close_s = scaler_close_direct.transform(latest_close_feats.values.reshape(1, -1))
pred_h_close_xgb = best_xgb_close.predict(X_latest_close_s)[0]
pred_close_h_close_xgb = df.loc[latest_date,'Close'] * (1 + pred_h_close_xgb)

# Recursive forecasts
future_dates = pd.bdate_range(start=latest_date + BDay(1), periods=h)
hist_for_min = df['Close'].copy()
hist_for_ext = df['Close'].copy()
rec_min_prices = []
rec_ext_prices = []
for d in future_dates:
    # minimal
    hist = hist_for_min
    ret_lag_1 = hist.pct_change().shift(1).iloc[-1] if len(hist) > 2 else 0.0
    ma5 = hist.rolling(5).mean().iloc[-1] if len(hist) >= 5 else np.nan
    ma10 = hist.rolling(10).mean().iloc[-1] if len(hist) >= 10 else np.nan
    ma20 = hist.rolling(20).mean().iloc[-1] if len(hist) >= 20 else np.nan
    ma5_ma20 = ma5 / (ma20 + 1e-9) if (not np.isnan(ma5) and not np.isnan(ma20)) else 0.0
    dow = d.dayofweek
    is_q = int((d.month % 3) == 0)
    feat_min = np.array([ret_lag_1, ma5 if not np.isnan(ma5) else 0.0, ma10 if not np.isnan(ma10) else 0.0,
                         ma20 if not np.isnan(ma20) else 0.0, ma5_ma20, dow, is_q], dtype=float).reshape(1, -1)
    feat_min = np.nan_to_num(feat_min, nan=0.0, posinf=0.0, neginf=0.0)
    feat_min_s = scaler_1min.transform(feat_min)
    pred_ret_min = xgb_1_min.predict(feat_min_s)[0]
    last_close_min = float(hist.iloc[-1])
    pred_close_min = last_close_min * (1 + pred_ret_min)
    hist_for_min = pd.concat([hist_for_min, pd.Series([pred_close_min], index=[d])])
    rec_min_prices.append(pred_close_min)

    # extended
    hist2 = hist_for_ext
    ret_lag_1_e = hist2.pct_change().shift(1).iloc[-1] if len(hist2) > 2 else 0.0
    ma5_e = hist2.rolling(5).mean().iloc[-1] if len(hist2) >= 5 else np.nan
    ma10_e = hist2.rolling(10).mean().iloc[-1] if len(hist2) >= 10 else np.nan
    ma20_e = hist2.rolling(20).mean().iloc[-1] if len(hist2) >= 20 else np.nan
    ma5_ma20_e = ma5_e / (ma20_e + 1e-9) if (not np.isnan(ma5_e) and not np.isnan(ma20_e)) else 0.0
    slope10_e = np.nan
    slope20_e = np.nan
    if len(hist2) >= 10:
        slope10_e = np.polyfit(np.arange(10), hist2.values[-10:], 1)[0]
    if len(hist2) >= 20:
        slope20_e = np.polyfit(np.arange(20), hist2.values[-20:], 1)[0]
    vol20_e = hist2.pct_change().rolling(20).std().iloc[-1] if len(hist2) >= 20 else 0.0
    feat_ext = np.array([ret_lag_1_e,
                         ma5_e if not np.isnan(ma5_e) else 0.0,
                         ma10_e if not np.isnan(ma10_e) else 0.0,
                         ma20_e if not np.isnan(ma20_e) else 0.0,
                         ma5_ma20_e,
                         slope10_e if not np.isnan(slope10_e) else 0.0,
                         slope20_e if not np.isnan(slope20_e) else 0.0,
                         vol20_e if not np.isnan(vol20_e) else 0.0,
                         dow, is_q], dtype=float).reshape(1, -1)
    feat_ext = np.nan_to_num(feat_ext, nan=0.0, posinf=0.0, neginf=0.0)
    feat_ext_s = scaler_1ext.transform(feat_ext)
    pred_ret_ext = xgb_1_ext.predict(feat_ext_s)[0]
    last_close_ext = float(hist2.iloc[-1])
    pred_close_ext = last_close_ext * (1 + pred_ret_ext)
    hist_for_ext = pd.concat([hist_for_ext, pd.Series([pred_close_ext], index=[d])])
    rec_ext_prices.append(pred_close_ext)

rec_min_df = pd.DataFrame({'pred_close': rec_min_prices}, index=future_dates)
rec_ext_df = pd.DataFrame({'pred_close': rec_ext_prices}, index=future_dates)

# -------------------------
# Plot (same layout) but keep diagnostic plots above
# -------------------------
hist_plot = df['Close'].iloc[-N_HIST_PLOT:]
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(hist_plot.index, hist_plot.values, label='Historical Close', linewidth=1.5)
ax.plot(rec_min_df.index, rec_min_df['pred_close'].values, label=f'Recursive minimal (next {h} bd)', linestyle='-', marker='o')
ax.plot(rec_ext_df.index, rec_ext_df['pred_close'].values, label=f'Recursive extended (next {h} bd)', linestyle='-', marker='s')
# direct points
ax.scatter(rec_min_df.index[-1], pred_close_h_full_xgb, color='red', s=110, zorder=9, label=f'Direct full XGB t+{h} point')
ax.scatter(rec_min_df.index[-1], pred_close_h_close_xgb, color='purple', s=90, zorder=9, label=f'Direct close-only XGB t+{h} point')
# mark recursive endpoints
ax.scatter(rec_min_df.index[-1], rec_min_df['pred_close'].values[-1], color='green', s=80, zorder=8, label=f'Recursive minimal t+{h} point')
ax.scatter(rec_ext_df.index[-1], rec_ext_df['pred_close'].values[-1], color='orange', s=80, zorder=8, label=f'Recursive extended t+{h} point')

ax.axvline(hist_plot.index[-1], color='k', linestyle=':', alpha=0.6, label='Forecast start')
locator = AutoDateLocator()
formatter = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
fig.autofmt_xdate()
ax.set_title(f"Historical close + 4 model forecasts to t+{h} (direct points & recursive paths)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# Print recursive tables & direct values + final diagnostics summary
# -------------------------
pd.set_option('display.float_format', '{:.4f}'.format)
print(f"\nRecursive minimal model day-by-day predictions (next {h} bd):\n")
print(rec_min_df)
print(f"\nRecursive extended model day-by-day predictions (next {h} bd):\n")
print(rec_ext_df)

print("\nDirect model t+H point predictions:")
print(f" Direct full XGB predicted close t+{h}: {pred_close_h_full_xgb:.4f}  (pred return {pred_h_full_xgb:.6f})")
print(f" Direct close-only XGB predicted close t+{h}: {pred_close_h_close_xgb:.4f}  (pred return {pred_h_close_xgb:.6f})")

print("\nSummary diagnostics (validation):")
print(" - If model pred std is near zero and baseline-zero performance is similar, model is likely predicting the mean/zero.")
print(" - If correlation(pred,true) is near 0, model has little predictive signal for that target/horizon.")
print("Check feature importances above to see which features drove the direct models.")
