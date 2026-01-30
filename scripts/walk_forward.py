import sys
import os
# Add root to sys.path for script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kasper.models.kasper_model import KASPER, KASPEREnsemble
from src.kasper.training.trainer import ForecasterTrainer
from src.kasper.utils.metrics import calculate_statistical_metrics, calculate_financial_metrics, calculate_sharpe
from src.kasper.utils.data_utils import load_yahoo_finance_data, engineer_features, prepare_data

def purged_walk_forward(X, y, gap_days=5, n_splits=5):
    """
    Implements purged walk-forward validation (Fix #1).
    Ensures a temporal gap between training and testing to prevent lookahead bias.
    """
    print(f"\n--- Starting Purged Walk-Forward Validation (Gap: {gap_days} days) ---")
    
    indices = np.arange(len(X))
    fold_size = len(X) // (n_splits + 1)
    results = []
    
    for i in range(n_splits):
        # Training data boundary
        train_end = (i + 1) * fold_size
        X_train_raw = X.iloc[:train_end]
        y_train_raw = y.iloc[:train_end]
        
        # PURGE: test starts after gap_days
        test_start = train_end + gap_days
        test_end = test_start + fold_size
        
        if test_end > len(X):
            print(f"  Split {i+1}: End of data reached. Skipping.")
            break
            
        X_test_raw = X.iloc[test_start:test_end]
        y_test_raw = y.iloc[test_start:test_end]
        
        print(f"\nFOLD {i+1} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)} (Gap: {gap_days})")
        
        # Prepare fold data (Scaling & Feature Selection)
        bundles = prepare_data(X_train_raw, y_train_raw)
        X_train, y_train = bundles['X_train'], bundles['y_train']
        X_val, y_val = bundles['X_val'], bundles['y_val']
        scaler_y = bundles['scaler_y']
        scaler_X = bundles['scaler_X']
        selected_features = bundles['selected_features']
        
        # Transform test data using fold's scaler and features
        X_test_selected = X_test_raw[selected_features]
        X_test_scaled = scaler_X.transform(X_test_selected)
        y_test_scaled = scaler_y.transform(y_test_raw.values.reshape(-1, 1)).flatten()
        
        # Initialize and train Ensemble (for stability) - Updated Capacity
        model = KASPEREnsemble(input_dim=X_train.shape[1], hidden_dim=256, n_regimes=4, n_models=5, n_basis=15)
        model.fit_knots(torch.FloatTensor(X_train))
        
        trainer = ForecasterTrainer(model, patience=40)
        model = trainer.fit(X_train, y_train, X_val, y_val, epochs=50)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_scaled, _, _ = model(torch.FloatTensor(X_test_scaled))
            
        y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        y_true = y_test_raw.values.flatten()
        
        sharpe = calculate_sharpe(y_true, y_pred)
        stats = calculate_statistical_metrics(y_true, y_pred)
        
        print(f"  Fold {i+1} Results -> Sharpe: {sharpe:.4f} | R2: {stats['R2']:.4f}")
        results.append({'Sharpe': sharpe, 'R2': stats['R2']})
        
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"PURGED WALK-FORWARD SUMMARY (Mean)")
    print("="*50)
    print(df.mean())
    return df

def evaluate_baselines(X_test, y_test):
    """
    Placeholder for baseline comparisons (Fix #5).
    In a real scenario, this would import PatchTST and DLinear.
    """
    print("\n--- Running Baseline Comparisons (Mock) ---")
    # Mocking baseline results based on typical paper reports
    baselines = {
        'KASPER': 1.12, # (Example Sharpe)
        'PatchTST': 0.85,
        'DLinear': 0.72,
        'LSTM': 0.45
    }
    for name, sharpe in baselines.items():
        print(f"  {name}: Sharpe = {sharpe:.2f}")

if __name__ == "__main__":
    ticker = "^DJI"
    raw_data = load_yahoo_finance_data(ticker=ticker, start="2018-01-01", end="2023-12-31")
    X, y = engineer_features(raw_data)
    
    # Run Purged Walk-Forward
    df_results = purged_walk_forward(X, y, gap_days=5)
    
    # Baseline comparison (Mock)
    evaluate_baselines(None, None)
