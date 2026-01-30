# Quick Start: Fixed KASPER Training

## 🚨 What Changed?

### Previous Problem
```
Early stopping → saved model at LOWEST validation loss
                ↓
        But that model had REGIME COLLAPSE!
                ↓
        R² = -1.0, Win Rate = 54%
```

### New Solution
```
During training → Monitor regime health in real-time
                 ↓
        Save TWO models:
        1. best_kasper_healthy.pth ✅ (healthy regimes, may have slightly higher loss)
        2. best_kasper.pth (lowest loss, may be collapsed)
                 ↓
        Load the HEALTHY one for predictions!
```

---

## ⚡ Steps to Run

### 1. Pull Latest Code
```bash
git pull origin main
```

### 2. Clean Old Files
```bash
# Remove old corrupted models
rm -f best_kasper.pth best_kasper_healthy.pth
rm -f *.png
```

### 3. Run Training
```bash
python main.py 2>&1 | tee training.log
```

---

## 🔍 What to Watch During Training

### Epoch Output Format

You'll now see **regime health monitoring**:

```
Epoch 12/200
  Regime dist: [0.28 0.35 0.37] | Temp: 1.850 ✅
  Health score: 0.823 | Healthy: True
  Losses -> huber:0.012 contrastive:0.180 l1:0.0005 orthogonality:0.095 balance:0.035
  💾 Saved HEALTHY model (epoch 12, loss=0.015234)
Epoch 12/200 | Train Loss: 0.015123 | Val Loss: 0.015234 | ✅ HEALTHY
```

**Icons:**
- ✅ = Healthy regimes (all 3+ active, balanced distribution)
- ⚠️ = Unhealthy regimes (collapse detected or imbalanced)

---

### Good Training Indicators

✅ **All regimes active:**
```
Regime dist: [0.28 0.35 0.37]  # All >10%, none >60%
```

✅ **High entropy:**
```
Health score: 0.8+  # Indicates diverse regime usage
```

✅ **Slow temperature decay:**
```
Epoch 10: Temp: 1.850
Epoch 20: Temp: 1.640  
Epoch 50: Temp: 1.220
```

✅ **Healthy models being saved:**
```
💾 Saved HEALTHY model (epoch 23, loss=0.012456)
```

---

### Bad Training Indicators

❌ **Regime collapse:**
```
Regime dist: [0.02 0.95 0.03]  # One regime dominates
Health score: 0.234 | Healthy: False
⚠️ WARNING: Regime collapse detected (2/3 active). Applying 5x penalty.
```

❌ **No healthy models:**
```
Epoch 100/200 | Train Loss: 0.008234 | Val Loss: 0.008456 | ⚠️ UNHEALTHY
# No "💾 Saved HEALTHY model" messages
```

❌ **Loss exploding:**
```
huber:15.234 contrastive:8.456 ...  # All losses increasing rapidly
```

---

## 📊 Expected Final Output

### Training Complete Message

**Best Case** (✅):
```
📊 Training complete. Loading best model...
✅ Loading HEALTHY model from epoch 42

📈 Regime Health History:
  Healthy epochs: 67/100
  Best healthy epoch: 42 (loss=0.012345)
```

**Worst Case** (⚠️):
```
📊 Training complete. Loading best model...
⚠️ No healthy model saved. Loading best loss model (may have collapsed regimes)

📈 Regime Health History:
  ⚠️ NO healthy epochs found during training!
  This indicates severe over-regularization or insufficient model capacity.
  Recommendation: Increase lambda_balance to 0.5+ or reduce lambda_l1 to 0.0
```

---

### Final Metrics (After Fixes)

**Minimum Acceptable:**
```
--- Statistical Metrics (Mean) ---
R²: 0.50 - 0.70       ✅ (positive and > 0.5)
MSE: 0.0001 - 0.0005   ✅

--- Financial Metrics (Mean) ---
Sharpe Ratio: 2.0 - 5.0    ✅ (meaningful returns)
Win Rate: 55% - 62%        ✅ (better than random)
```

**Target Performance:**
```
--- Statistical Metrics (Mean) ---
R²: 0.70 - 0.89       ✅ (strong prediction)
MSE: 0.0001 - 0.0003   ✅

--- Financial Metrics (Mean) ---
Sharpe Ratio: 8.0 - 14.0   ✅ (excellent returns)
Win Rate: 65% - 75%        ✅ (clear skill)
```

---

## 🔧 If Still No Healthy Models

### Option 1: Nuclear - Disable All Regularization

Edit `losses.py` line 128:
```python
def composite_loss(y_pred, y_true, embeddings, regime_probs, model, 
                   delta=1.0, 
                   lambda_contr=0.0,    # Disabled
                   lambda_l1=0.0,       # Disabled
                   lambda_orth=0.0,     # Disabled
                   lambda_balance=0.0): # Disabled
```

This removes ALL penalties except prediction accuracy. If this works (R²>0.5), then over-regularization was the issue.

### Option 2: Extreme Regime Forcing

Edit `losses.py` line 128:
```python
def composite_loss(y_pred, y_true, embeddings, regime_probs, model, 
                   delta=1.0, 
                   lambda_contr=0.10,      # ⬆️ Extreme
                   lambda_l1=0.00001,      # ⬇️ Minimal
                   lambda_orth=0.30,       # ⬆️ Extreme
                   lambda_balance=0.50):   # ⬆️ Extreme
```

### Option 3: Increase Model Capacity

Edit `main.py` line 46:
```python
model = KASPEREnsemble(
    input_dim=input_dim, 
    hidden_dim=512,      # ⬆️ Was 256 or 128
    n_regimes=5,         # ⬆️ Was 3
    n_models=7           # ⬆️ Was 3
)
```

---

## 📁 Files Created

After training completes:

1. **best_kasper_healthy.pth** - Model with healthy regimes (preferred)
2. **best_kasper.pth** - Model with lowest loss (fallback)
3. **performance_plot.png** - Cumulative returns visualization
4. **training.log** - Full training log (if you used `tee`)

---

## 📝 Interpreting Results

### Regime Rules Output

**Healthy** (✅):
```
Regime 0: ma_5 + lag_ret_1 + volatility_ratio -> Future Return
Regime 1: rsi + price_acceleration + volume_ratio -> Future Return
Regime 2: log_ret_high + ATR + ma_20 -> Future Return
```
All 3 regimes have samples and use different features.

**Collapsed** (❌):
```
Regime 0: No samples assigned to this regime -> Future Return
Regime 1: ma_5 + lag_ret_1 + log_ret_high -> Future Return  
Regime 2: ma_5 + lag_ret_1 + price_acceleration -> Future Return
```
Regime 0 unused, regimes 1-2 use similar features (not specialized).

---

## ❓ Troubleshooting

### Q: Training is very slow
**A:** Increase `batch_size` in `main.py`:
```python
trainer.fit(..., batch_size=128)  # Was 64
```

### Q: Loss is NaN
**A:** Learning rate too high. Edit `main.py`:
```python
trainer = ForecasterTrainer(model, lr=0.0001)  # Was 0.0005
```

### Q: GPU out of memory
**A:** Reduce model size:
```python
model = KASPEREnsemble(..., hidden_dim=128, n_models=3)
```

### Q: All epochs are unhealthy
**A:** Try the "Nuclear Option" (disable all regularization) first to confirm model CAN learn, then gradually add back penalties.

---

## 🎯 Success Criteria

### Minimum Success:
- [ ] At least 20% of epochs are marked ✅ HEALTHY
- [ ] `best_kasper_healthy.pth` file exists
- [ ] R² > 0.50 (positive and above half)
- [ ] Sharpe > 1.0 (profitable)
- [ ] Win Rate > 53% (better than random)

### Target Success:
- [ ] At least 50% of epochs are marked ✅ HEALTHY  
- [ ] R² > 0.75 (strong prediction)
- [ ] Sharpe > 5.0 (excellent returns)
- [ ] Win Rate > 60% (clear skill)
- [ ] All 3+ regimes active with 15-50% each

---

## 🚀 Next Steps After Success

1. **Run ablation studies:**
   ```bash
   python ablation_studies.py
   ```

2. **Compare with baselines:**
   ```bash
   python compare_models.py
   ```

3. **Test on different tickers:**
   ```python
   # In main.py, change line 14:
   ticker = "^GSPC"  # S&P 500
   ticker = "^IXIC"  # NASDAQ
   ```

---

**Last Updated:** January 30, 2026, 8:48 PM IST  
**Status:** ✅ All fixes applied, ready for testing

**Key Changes:**
- ✅ Regime health monitoring during training
- ✅ Save healthy models separately  
- ✅ Slower temperature annealing (2.0 → 0.5)
- ✅ Reduced L1 regularization (0.0001)
- ✅ Increased regime forcing (balance=0.30, orth=0.15)
- ✅ Delayed sparsity pruning (after epoch 50)
