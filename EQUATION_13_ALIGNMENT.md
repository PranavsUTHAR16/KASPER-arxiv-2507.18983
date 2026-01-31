# Equation 13 Feature Engineering Alignment

## 📋 Overview

The KASPER paper specifies **exactly 13 features** in Equation 13. This document confirms our implementation matches the paper specification precisely.

---

## 🎯 Paper Specification (Equation 13)

```
Φ_{t,i} = [
  ln(p_{t-i+1}/p_{t-i}),              # 1. Close log return
  ln(High_{t-i+1}/High_{t-i}),        # 2. High log return
  ln(Low_{t-i+1}/Low_{t-i}),          # 3. Low log return
  ln(Open_{t-i+1}/Open_{t-i}),        # 4. Open log return
  ln(V_{t-i+1}/V_{t-i}),              # 5. Volume log return
  HL_spread_{t-i},                    # 6. High-Low spread
  OC_spread_{t-i},                    # 7. Open-Close spread
  ATR_{t-i},                          # 8. Average True Range
  volatility_ratio_{t-i},             # 9. Volatility ratio
  price_velocity_{t-i},               # 10. Price velocity
  price_acceleration_{t-i},           # 11. Price acceleration
  volume_change_{t-i},                # 12. Volume change
  volume_state_{t-i}                  # 13. Volume state
]
```

**Then:** SelectKBest with `f_regression` scoring selects the final **8 features** for KASPER.

---

## ✅ Implementation

### Feature 1-5: Log-Transformed Returns

**Purpose:** Stabilize variance and enhance stationarity

```python
# Feature 1: Close price log return
df['log_ret_close'] = np.log(df['Close'] / (df['Close'].shift(1) + 1e-10))

# Feature 2: High price log return
df['log_ret_high'] = np.log(df['High'] / (df['High'].shift(1) + 1e-10))

# Feature 3: Low price log return
df['log_ret_low'] = np.log(df['Low'] / (df['Low'].shift(1) + 1e-10))

# Feature 4: Open price log return
df['log_ret_open'] = np.log(df['Open'] / (df['Open'].shift(1) + 1e-10))

# Feature 5: Volume log return
df['log_ret_volume'] = np.log(df['Volume'] / (df['Volume'].shift(1) + 1e-10))
```

**Why log returns?**
- Additive over time: log(P_t/P_0) = log(P_t/P_{t-1}) + ... + log(P_1/P_0)
- Symmetric: log(2) ≈ -log(0.5)
- Stabilizes variance (important for KAN spline fitting)

---

### Feature 6: HL_spread (High-Low Spread)

**Purpose:** Intraday volatility measure

```python
df['HL_spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
```

**Interpretation:**
- Higher spread → more volatile day
- Normalized by Close for scale-invariance

---

### Feature 7: OC_spread (Open-Close Spread)

**Purpose:** Directional momentum within the day

```python
df['OC_spread'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
```

**Interpretation:**
- Positive → bullish day (close > open)
- Negative → bearish day (close < open)
- Magnitude indicates strength

---

### Feature 8: ATR (Average True Range)

**Purpose:** Multi-day volatility indicator

```python
# True Range = max(H-L, |H-C_prev|, |L-C_prev|)
high_low = df['High'] - df['Low']
high_close = (df['High'] - df['Close'].shift(1)).abs()
low_close = (df['Low'] - df['Close'].shift(1)).abs()
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

# 14-period moving average, normalized by Close
df['ATR'] = true_range.rolling(window=14, closed='left').mean() / (df['Close'] + 1e-10)
```

**Key:** `closed='left'` ensures no lookahead bias (uses only past data)

---

### Feature 9: volatility_ratio

**Purpose:** Detect regime shifts (calm → volatile or vice versa)

```python
vol_short = df['log_ret_close'].rolling(window=20, closed='left').std()
vol_long = df['log_ret_close'].rolling(window=50, closed='left').std()
df['volatility_ratio'] = vol_short / (vol_long + 1e-10)
```

**Interpretation:**
- ratio > 1 → Recent volatility increased (crisis regime?)
- ratio < 1 → Recent volatility decreased (calm regime?)

---

### Feature 10: price_velocity

**Purpose:** First derivative (instantaneous momentum)

```python
df['price_velocity'] = df['log_ret_close']
```

**Note:** Velocity IS the log return itself (rate of price change)

---

### Feature 11: price_acceleration

**Purpose:** Second derivative (momentum change)

```python
df['price_acceleration'] = df['log_ret_close'].diff()
```

**Interpretation:**
- Positive → momentum increasing (trend strengthening)
- Negative → momentum decreasing (trend weakening)
- Captures turning points

---

### Feature 12: volume_change

**Purpose:** Abnormal volume detection

```python
vol_ma_20 = df['Volume'].rolling(window=20, closed='left').mean()
df['volume_change'] = df['Volume'] / (vol_ma_20 + 1e-10)
```

**Interpretation:**
- value > 1.5 → High volume (breakout signal?)
- value < 0.5 → Low volume (consolidation?)

---

### Feature 13: volume_state

**Purpose:** Binary volume regime indicator

```python
df['volume_state'] = (df['Volume'] > vol_ma_20).astype(int)
```

**Interpretation:**
- 1 → Above-average volume period
- 0 → Below-average volume period

---

## 🔍 Feature Selection

**Paper Quote:** 
> "These eight features are selected via SelectKBest with f_regression scoring"

**Implementation:**

```python
selector = SelectKBest(score_func=f_regression, k=8)
X_train_selected = selector.fit_transform(X_train, y_train)
```

**Process:**
1. Engineer all 13 features
2. Fit SelectKBest on **training set only** (no data leakage)
3. Compute f_regression scores (ANOVA F-statistic)
4. Select top 8 features with highest correlation to target
5. Apply same selection to validation and test sets

**Example output:**
```
Selected 8 features (ranked by f_regression score):
  1. log_ret_close          (score: 524.32)
  2. price_velocity         (score: 524.32)  # Same as log_ret_close
  3. price_acceleration     (score: 156.78)
  4. volatility_ratio       (score: 89.45)
  5. ATR                    (score: 67.23)
  6. HL_spread              (score: 45.67)
  7. log_ret_high           (score: 34.12)
  8. volume_change          (score: 28.90)
```

**Note:** Actual selected features may vary by dataset and time period.

---

## 🚫 No Lookahead Bias

### Critical Safeguards:

1. **Rolling windows use `closed='left'`:**
   ```python
   .rolling(window=14, closed='left')  # Excludes current bar
   ```

2. **Target is t+1 (future):**
   ```python
   df['target'] = df['log_ret_close'].shift(-1)  # Next day
   ```

3. **Feature selection on training only:**
   ```python
   selector.fit_transform(X_train, y_train)  # Fit on train
   selector.transform(X_test)                 # Only transform test
   ```

4. **Temporal split (no shuffling):**
   ```python
   Train: [0:70%]    # 2018-2021
   Val:   [70%:85%]  # 2021-2022
   Test:  [85%:100%] # 2022-2023
   ```

---

## 📊 Data Flow

```
Raw OHLCV data (Yahoo Finance)
        ↓
Engineer 13 features (Equation 13)
        ↓
Temporal split (70/15/15)
        ↓
SelectKBest on TRAIN → Select 8 features
        ↓
Apply selection to VAL and TEST
        ↓
Standardize (Z-score) using TRAIN statistics
        ↓
Final input to KASPER: (N, 8)
```

---

## 🔄 Changes from Previous Implementation

### Added:
- ✅ `log_ret_open` (Feature 4)
- ✅ `log_ret_volume` (Feature 5)
- ✅ `volume_state` (Feature 13)

### Removed:
- ❌ Day-of-week one-hot encoding (not in Equation 13)
- ❌ RSI (not in Equation 13)
- ❌ Momentum state (not in Equation 13)
- ❌ Moving average ratios (not in Equation 13)
- ❌ Lagged features (not in Equation 13)

### Modified:
- 🔄 ATR now uses `closed='left'` (no lookahead)
- 🔄 All rolling windows use `closed='left'`
- 🔄 Clearer feature naming matching paper

---

## ✅ Verification

**Run this to confirm implementation:**

```bash
python main.py
```

**Expected output:**

```
Initial data shape: (1500, 6)
Implementing Equation 13 feature engineering...
Feature engineering complete: 1500 -> 1447 rows

13 features engineered as per Equation 13:
  1. log_ret_close
  2. log_ret_high
  3. log_ret_low
  4. log_ret_open
  5. log_ret_volume
  6. HL_spread
  7. OC_spread
  8. ATR
  9. volatility_ratio
  10. price_velocity
  11. price_acceleration
  12. volume_change
  13. volume_state

============================================================
FEATURE SELECTION (SelectKBest, k=8)
============================================================
Selected 8 features (ranked by f_regression score):
  1. log_ret_close          (score: XXX.XX)
  2. price_velocity         (score: XXX.XX)
  ...
```

---

## 📚 Paper References

**Equation 13 (Page X):**
> "The input consists of an n-day window of financial time series data, structured as a state matrix Φt ∈ R^{n×f}, where n denotes the number of historical days and f the number of features per day."

**Section 4.1:**
> "These eight features are selected via SelectKBest with f_regression scoring and form the final input to KASPER."

---

## 🎯 Why This Matters

1. **Exact replication:** Our features now match paper specification 100%
2. **Reproducibility:** Others can replicate our results
3. **No missing features:** Previous implementation was missing log_ret_open, log_ret_volume, volume_state
4. **No extra features:** Removed non-paper features that could confuse results
5. **Proper temporal alignment:** All features use `closed='left'` to prevent lookahead bias

---

## 🔗 Related Files

- **data_utils.py** - Feature engineering implementation
- **main.py** - Pipeline that calls feature engineering
- **kan_layers.py** - Processes the 8 selected features

---

**Last Updated:** January 31, 2026, 5:48 AM IST  
**Commit:** [19fb533](https://github.com/PranavsUTHAR16/KASPER_arxiv_2507.18983/commit/19fb5334f0171f4ef70bec0a908980835b1e6c73)  
**Status:** ✅ Fully aligned with KASPER paper Equation 13
