import torch
import numpy as np
import pandas as pd
from data_utils import load_yahoo_finance_data, engineer_features, prepare_data
from kasper_model import KASPER
from trainer import ForecasterTrainer
from metrics import calculate_statistical_metrics, calculate_financial_metrics
from interpretability import extract_regime_rules
import matplotlib.pyplot as plt

def main():
    # 1. Data Preparation - USE PAPER'S EXACT SPECIFICATIONS
    ticker = "^DJI"  # Dow Jones (US30) as in paper
    print(f"--- KASPER Model Pipeline for {ticker} ---")
    print("Step 1: Loading and Preprocessing Data...")
    
    # Use EXACT date range from paper
    raw_data = load_yahoo_finance_data(
        ticker=ticker, 
        start="2018-01-01",  # Paper's start date
        end="2023-12-31"     # Paper's end date
    )
    
    X, y = engineer_features(raw_data)
    data_bundles = prepare_data(X, y)
    
    X_train, y_train = data_bundles['X_train'], data_bundles['y_train']
    X_val, y_val = data_bundles['X_val'], data_bundles['y_val']
    X_test, y_test = data_bundles['X_test'], data_bundles['y_test']
    selected_features = data_bundles['selected_features']
    scaler_y = data_bundles['scaler_y']
    
    # 2. Model Initialization
    print("Step 2: Initializing KASPER Model...")
    input_dim = X_train.shape[1]
    model = KASPER(input_dim=input_dim, hidden_dim=64, n_regimes=3)
    model.fit_knots(torch.FloatTensor(X_train))
    
    # 3. Training - FULL 100 EPOCHS
    print("Step 3: Training Model...")
    trainer = ForecasterTrainer(model, patience=25)
    model = trainer.fit(
        X_train, y_train, X_val, y_val, 
        epochs=100,      # Paper specification
        batch_size=32    # Paper specification
    )
    
    # 4. Evaluation
    print("Step 4: Evaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_pred_scaled, test_regime_probs, _ = model(X_test_tensor)
    
    # Unscale predictions
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_scaled.cpu().numpy().reshape(-1, 1)
    ).flatten()
    y_test_true = scaler_y.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
    
    # Metrics
    stats = calculate_statistical_metrics(y_test_true, y_test_pred)
    finance = calculate_financial_metrics(y_test_true, y_test_pred)
    
    print("\n" + "="*50)
    print("PERFORMANCE METRICS vs PAPER BENCHMARK")
    print("="*50)
    print("\n--- Statistical Metrics ---")
    targets = {'MSE': 0.0001, 'RMSE': 0.0046, 'MAE': 0.0033, 'R2': 0.89}
    for k, v in stats.items():
        indicator = "✅" if (k == 'R2' and v > 0.80) or (k == 'MSE' and v < 0.001) else "⚠️"
        print(f"{k}: {v:.6f}  {indicator} (Target: {targets.get(k, 'N/A')})")
    
    print("\n--- Financial Metrics ---")
    f_targets = {
        'Sharpe Ratio': 12.02,
        'Win Rate (%)': 83.17,
        'Max Drawdown (%)': -0.09
    }
    for k, v in finance.items():
        indicator = "✅" if (k == 'Win Rate (%)' and v > 75) or (k == 'Sharpe Ratio' and v > 8.0) else "⚠️"
        print(f"{k}: {v:.4f}  {indicator} (Target: {f_targets.get(k, 'N/A')})")
    
    # 5. Interpretability
    print("\nStep 5: Extracting Regime Rules with Temporal Weighting (Eq 26)...")
    regime_rules = extract_regime_rules(model, X_test, selected_features, gamma=0.9)
    for regime, rules in regime_rules.items():
        print(f"Regime {regime}: {' + '.join(rules)} -> Future Return")
    
    # 6. Visualization
    strategy_returns = np.sign(y_test_pred) * y_test_true
    cum_strategy = np.cumprod(1 + strategy_returns) - 1
    cum_market = np.cumprod(1 + y_test_true) - 1
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_strategy, label='KASPER Strategy', color='blue', linewidth=2)
    plt.plot(cum_market, label='Buy & Hold (Market)', color='gray', 
             linestyle='--', linewidth=2)
    plt.title(f"Cumulative Returns Comparison - {ticker}", fontsize=14)
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_plot.png', dpi=150)
    print("\n✅ Performance plot saved to performance_plot.png")

if __name__ == "__main__":
    main()
