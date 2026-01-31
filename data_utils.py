import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

def load_yahoo_finance_data(ticker="^NSEI", start="2015-01-01", end="2026-01-30"):
    """
    Loads OHLCV data from Yahoo Finance and cleans it.
    Default: ^NSEI (Nifty 50) as requested by user.
    """
    print(f"Downloading {ticker} data from {start} to {end}...")
    # auto_adjust=True is now default in newer yfinance versions
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex if necessary (often happens with yfinance)
        data.columns = [col[0] for col in data.columns]
    
    # Forward-fill missing values
    data = data.ffill()
    
    print(f"Downloaded {len(data)} rows")
    return data

def engineer_features(data):
    """
    Extracts 13 comprehensive features EXACTLY as specified in KASPER paper Equation 13.
    Implements strict temporal alignment to prevent lookahead bias.
    
    Features (Equation 13):
    1. ln(p_{t-i+1}/p_{t-i}) - Log return of Close
    2. ln(High_{t-i+1}/High_{t-i}) - Log return of High
    3. ln(Low_{t-i+1}/Low_{t-i}) - Log return of Low
    4. ln(Open_{t-i+1}/Open_{t-i}) - Log return of Open
    5. ln(V_{t-i+1}/V_{t-i}) - Log return of Volume
    6. HL_spread - High-Low spread
    7. OC_spread - Open-Close spread
    8. ATR - Average True Range
    9. volatility_ratio - Rolling volatility ratio
    10. price_velocity - First derivative of returns
    11. price_acceleration - Second derivative of returns
    12. volume_change - Volume change rate
    13. volume_state - Binary volume state
    
    Then SelectKBest with k=8 selects the final features.
    """
    df = data.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    initial_rows = len(df)
    print(f"\nInitial data shape: {df.shape}")
    print(f"Implementing Equation 13 feature engineering...")
    
    # ========================================
    # EQUATION 13: 13 FEATURES AS PER PAPER
    # ========================================
    
    # --- Features 1-5: Log-transformed returns (stabilize variance) ---
    # Feature 1: ln(p_{t-i+1}/p_{t-i}) = ln(Close_t / Close_{t-1})
    df['log_ret_close'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-10))
    
    # Feature 2: ln(High_{t-i+1}/High_{t-i})
    df['log_ret_high'] = np.log(df['High'] / (df['High'].shift(1) + 1e-10))
    
    # Feature 3: ln(Low_{t-i+1}/Low_{t-i})
    df['log_ret_low'] = np.log(df['Low'] / (df['Low'].shift(1) + 1e-10))
    
    # Feature 4: ln(Open_{t-i+1}/Open_{t-i})
    df['log_ret_open'] = np.log(df['Open'] / (df['Open'].shift(1) + 1e-10))
    
    # Feature 5: ln(V_{t-i+1}/V_{t-i})
    df['log_ret_volume'] = np.log(df['Volume'] / (df['Volume'].shift(1) + 1e-10))
    
    # --- Feature 6: HL_spread (High-Low spread normalized by Close) ---
    df['HL_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    
    # --- Feature 7: OC_spread (Open-Close spread normalized by Open) ---
    df['OC_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
    
    # --- Feature 8: ATR (Average True Range, 14-period) ---
    # True Range = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Use closed='left' to avoid lookahead bias
    df['ATR'] = true_range.rolling(window=14, closed='left').mean() / (df['Close'] + 1e-10)
    
    # --- Feature 9: volatility_ratio (short-term / long-term volatility) ---
    # Ratio of 20-day to 50-day rolling standard deviation
    vol_short = df['log_ret_close'].rolling(window=20, closed='left').std()
    vol_long = df['log_ret_close'].rolling(window=50, closed='left').std()
    df['volatility_ratio'] = vol_short / (vol_long + 1e-10)
    
    # --- Feature 10: price_velocity (first derivative of log returns) ---
    # Velocity = current log return (already computed as log_ret_close)
    df['price_velocity'] = df['log_ret_close']
    
    # --- Feature 11: price_acceleration (second derivative) ---
    # Acceleration = change in velocity = diff(log_ret_close)
    df['price_acceleration'] = df['log_ret_close'].diff()
    
    # --- Feature 12: volume_change (rate of volume change) ---
    # Current volume / 20-day moving average volume
    vol_ma_20 = df['Volume'].rolling(window=20, closed='left').mean()
    df['volume_change'] = df['Volume'] / (vol_ma_20 + 1e-10)
    
    # --- Feature 13: volume_state (binary: above/below average) ---
    # 1 if current volume > 20-day average, else 0
    df['volume_state'] = (df['Volume'] > vol_ma_20).astype(int)
    
    # ========================================
    # TARGET: Next-day log return (t+1)
    # ========================================
    df['target'] = df['log_ret_close'].shift(-1)
    
    # ========================================
    # CLEAN DATA
    # ========================================
    # Replace infinities and drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # ========================================
    # DEFINE FEATURE SET (13 features from Eq 13)
    # ========================================
    features = [
        'log_ret_close',      # 1
        'log_ret_high',       # 2
        'log_ret_low',        # 3
        'log_ret_open',       # 4
        'log_ret_volume',     # 5
        'HL_spread',          # 6
        'OC_spread',          # 7
        'ATR',                # 8
        'volatility_ratio',   # 9
        'price_velocity',     # 10
        'price_acceleration', # 11
        'volume_change',      # 12
        'volume_state'        # 13
    ]
    
    # Validation
    df_features = df[features].astype(float)
    assert not df_features.isnull().any().any(), "Features contain NaN values!"
    assert not np.isinf(df_features.values).any(), "Features contain infinite values!"
    
    final_rows = len(df)
    print(f"Feature engineering complete: {initial_rows} -> {final_rows} rows")
    print(f"\n13 features engineered as per Equation 13:")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    
    return df_features, df['target']

def prepare_data(X, y, k=8):
    """
    Applies train-val-test split, feature selection, and standardization.
    Split: 70% train, 15% validation, 15% test (temporal order preserved).
    
    As per paper: "These eight features are selected via SelectKBest with f_regression scoring"
    """
    n = len(X)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    
    # Temporal split (NO shuffling across splits)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n{'='*60}")
    print(f"DATA SPLIT (Temporal Order Preserved)")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
    print(f"Val:   {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
    print(f"Test:  {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")
    
    # Feature Selection on Train Set ONLY
    # Paper: "eight features are selected via SelectKBest with f_regression scoring"
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION (SelectKBest, k={k})")
    print(f"{'='*60}")
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X.columns[selector.get_support()].tolist()
    selected_scores = selector.scores_[selector.get_support()]
    
    print(f"Selected {k} features (ranked by f_regression score):")
    for i, (feat, score) in enumerate(sorted(zip(selected_features, selected_scores), 
                                              key=lambda x: x[1], reverse=True), 1):
        print(f"  {i}. {feat:25s} (score: {score:.2f})")
    
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Standardization (fit on train only)
    print(f"\n{'='*60}")
    print(f"STANDARDIZATION (Z-score normalization)")
    print(f"{'='*60}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_selected)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    X_val_scaled = scaler_X.transform(X_val_selected)
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    
    X_test_scaled = scaler_X.transform(X_test_selected)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    print(f"Features standardized: mean=0, std=1")
    print(f"Train X shape: {X_train_scaled.shape}")
    print(f"Train y shape: {y_train_scaled.shape}")
    print(f"{'='*60}\n")
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train_scaled.flatten(),
        'X_val': X_val_scaled, 'y_val': y_val_scaled.flatten(),
        'X_test': X_test_scaled, 'y_test': y_test_scaled.flatten(),
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'selected_features': selected_features
    }
