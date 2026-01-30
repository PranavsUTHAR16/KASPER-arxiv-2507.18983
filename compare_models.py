import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kasper_model import KASPEREnsemble
from baselines import PatchTST, DLinear, train_baseline
from trainer import ForecasterTrainer
from data_utils import load_yahoo_finance_data, engineer_features, prepare_data
from metrics import calculate_statistical_metrics, calculate_financial_metrics

def compare_models(ticker="^DJI"):
    print(f"=== Head-to-Head Comparison: {ticker} ===")
    
    # 1. Data Preparation
    raw_data = load_yahoo_finance_data(ticker=ticker, start="2018-01-01", end="2023-12-31")
    X, y = engineer_features(raw_data)
    bundles = prepare_data(X, y)
    
    X_train, y_train = bundles['X_train'], bundles['y_train']
    X_val, y_val = bundles['X_val'], bundles['y_val']
    X_test, y_test = bundles['X_test'], bundles['y_test']
    scaler_y = bundles['scaler_y']
    
    input_dim = X_train.shape[1]
    
    # 2. Initialize Models
    models = {
        'KASPER': KASPEREnsemble(input_dim=input_dim, hidden_dim=128, n_regimes=3, n_models=3),
        'PatchTST': PatchTST(input_dim=input_dim),
        'DLinear': DLinear(input_dim=input_dim)
    }
    
    results = []
    
    # 3. Training & Evaluation
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == 'KASPER':
            model.fit_knots(torch.FloatTensor(X_train))
            trainer = ForecasterTrainer(model, patience=25)
            model = trainer.fit(X_train, y_train, X_val, y_val, epochs=100)
        else:
            model = train_baseline(model, X_train, y_train, X_val, y_val, epochs=100)
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            if name == 'KASPER':
                y_pred_scaled, _, _ = model(X_test_tensor)
            else:
                y_pred_scaled = model(X_test_tensor)
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        stats = calculate_statistical_metrics(y_true, y_pred)
        finance = calculate_financial_metrics(y_true, y_pred)
        
        res = {
            'Model': name,
            'Sharpe': finance['Sharpe Ratio'],
            'R2': stats['R2'],
            'MSE': stats['MSE'],
            'WinRate': finance['Win Rate (%)']
        }
        results.append(res)
        print(f"  {name} Results -> Sharpe: {res['Sharpe']:.4f} | R2: {res['R2']:.4f}")

    # 4. Final Comparison
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS")
    print("="*50)
    print(df.to_string(index=False))
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    df.plot(x='Model', y='Sharpe', kind='bar', ax=ax1, color=['#4285F4', '#EA4335', '#FBBC05'])
    ax1.set_title('Sharpe Ratio Comparison')
    ax1.set_ylabel('Sharpe Ratio')
    
    df.plot(x='Model', y='R2', kind='bar', ax=ax2, color=['#4285F4', '#EA4335', '#FBBC05'])
    ax2.set_title('R2 Score Comparison')
    ax2.set_ylabel('R2 Scale')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("\n✅ Comparison chart saved to comparison_results.png")

if __name__ == "__main__":
    compare_models()
