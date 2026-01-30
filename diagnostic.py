import torch
import numpy as np
import matplotlib.pyplot as plt
from kasper_model import KASPER
from data_utils import load_yahoo_finance_data, engineer_features, prepare_data

def diagnose_regimes(model, X_train, X_val, X_test):
    """Check regime distribution and embeddings"""
    model.eval()
    device = next(model.parameters()).device
    
    datasets = {
        'Train': X_train,
        'Val': X_val,
        'Test': X_test
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, X) in enumerate(datasets.items()):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            _, regime_probs, _ = model(X_tensor)
            
            regime_assignments = regime_probs.argmax(dim=1).cpu().numpy()
            
            # Plot histogram
            axes[idx].hist(regime_assignments, bins=model.n_regimes, 
                          range=(-0.5, model.n_regimes-0.5), 
                          alpha=0.7, edgecolor='black', color='skyblue')
            axes[idx].set_title(f'{name} Regime Distribution')
            axes[idx].set_xlabel('Regime')
            axes[idx].set_ylabel('Count')
            axes[idx].set_xticks(range(model.n_regimes))
            
            # Print stats
            unique, counts = np.unique(regime_assignments, return_counts=True)
            print(f"\n{name} Set:")
            dist_map = {r: 0 for r in range(model.n_regimes)}
            for r, c in zip(unique, counts):
                dist_map[r] = c
                
            for regime in range(model.n_regimes):
                count = dist_map[regime]
                pct = 100 * count / len(regime_assignments)
                print(f"  Regime {regime}: {count} ({pct:.1f}%)")
    
    plt.tight_layout()
    plt.savefig('regime_diagnostic.png', dpi=150)
    print("\n✅ Saved diagnostic plot to regime_diagnostic.png")

if __name__ == "__main__":
    # Load data for diagnostic
    ticker = "^DJI"
    raw_data = load_yahoo_finance_data(ticker=ticker, start="2018-01-01", end="2023-12-31")
    X, y = engineer_features(raw_data)
    data_bundles = prepare_data(X, y)
    
    # Load best model
    input_dim = data_bundles['X_train'].shape[1]
    from kasper_model import KASPEREnsemble
    model = KASPEREnsemble(input_dim=input_dim, hidden_dim=128, n_regimes=3, n_models=3, n_basis=20)
    try:
        model.load_state_dict(torch.load('best_kasper.pth', map_location='cpu'))
        print("Loaded best_kasper.pth for diagnosis.")
    except Exception as e:
        print(f"Could not load best_kasper.pth: {e}. Using freshly initialized model.")
        
    diagnose_regimes(model, data_bundles['X_train'], data_bundles['X_val'], data_bundles['X_test'])
