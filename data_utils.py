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
    Extracts 15 comprehensive features as specified in KASPER paper.
    Implements strict temporal alignment to prevent lookahead bias.
    """
    df = data.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    
    initial_rows = len(df)
    print(f"Initial data shape: {df.shape}")
    
    # Target variable: next-day percentage return
    df['target'] = (df['Close'].shift(-1) - df['Close']) / (df['Close'] + 1e-8)
    # --- 1. Price Dynamics (Eq 13 Alignment) ---
    # Log-transformed returns for High, Low, Close to stabilize variance
    df['log_ret_close'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_ret_high'] = np.log(df['High'] / df['High'].shift(1))
    df['log_ret_low'] = np.log(df['Low'] / df['Low'].shift(1))
    
    # Original spreads
    df['OC_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-8)
    df['HL_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-8)
    
    # --- 2. Velocity & Acceleration ---
    df['price_velocity'] = df['log_ret_close']
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # --- 3. Momentum & Relative Strength ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['momentum_state'] = (df['rsi'] > 50).astype(int)
    
    # --- 4. Volatility & Volume Patterns ---
    df['volatility_ratio'] = df['log_ret_close'].rolling(window=20).std() / (df['log_ret_close'].rolling(window=50).std() + 1e-8)
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14, closed='left').mean() / (df['Close'] + 1e-8)
    
    df['volume_ratio'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-8)
    
    # --- 5. One-Hot Encoding (Fig 2) ---
    # Encode day of the week to capture weekly cycles
    df['day'] = df.index.dayofweek
    day_dummies = pd.get_dummies(df['day'], prefix='day')
    df = pd.concat([df, day_dummies], axis=1)
    
    # Logged high-low spreads as supplementary
    df['log_HL_ratio'] = np.log(df['High'] / df['Low'])
    
    # Lagged features
    df['lag_ret_1'] = df['log_ret_close'].shift(1)
    
    # Rolling averages (Paper mentions moving averages)
    df['ma_5'] = df['Close'].rolling(window=5, closed='left').mean() / df['Close']
    df['ma_20'] = df['Close'].rolling(window=20, closed='left').mean() / df['Close']
    
    # target: t+1 log return
    df['target'] = df['log_ret_close'].shift(-1)
    
    df = df.ffill().replace([np.inf, -np.inf], np.nan).dropna()
    
    # Define features to use
    features = [
        'log_ret_close', 'log_ret_high', 'log_ret_low',
        'OC_spread', 'HL_spread', 'price_velocity', 'price_acceleration',
        'rsi', 'volatility_ratio', 'ATR', 'volume_ratio', 'log_HL_ratio',
        'lag_ret_1', 'ma_5', 'ma_20'
    ] + [col for col in df.columns if col.startswith('day_')]
    
    # Replace infinities and remaining NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # Validation
    df_features = df[features].astype(float)
    assert not df_features.isnull().any().any(), "Features contain NaN values!"
    assert not np.isinf(df_features.values).any(), "Features contain infinite values!"
    
    final_rows = len(df)
    print(f"Feature engineering complete: {initial_rows} -> {final_rows} rows")
    print(f"Features trained: {features}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df_features, df['target']

def prepare_data(X, y, k=8):
    """
    Applies train-val-test split, feature selection, and standardization.
    Split: 70% train, 15% validation, 15% test (temporal order preserved).
    """
    n = len(X)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    
    # Temporal split
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Feature Selection on Train Set ONLY
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X.columns[selector.get_support()].tolist()
    
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    print(f"Selected {k} features: {selected_features}")
    
    # Standardization (fit on train only)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_selected)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    
    X_val_scaled = scaler_X.transform(X_val_selected)
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))
    
    X_test_scaled = scaler_X.transform(X_test_selected)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train_scaled.flatten(),
        'X_val': X_val_scaled, 'y_val': y_val_scaled.flatten(),
        'X_test': X_test_scaled, 'y_test': y_test_scaled.flatten(),
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'selected_features': selected_features
    }
