"""
Ablation Studies for KASPER (Section 4.6, Pages 16-17)
========================================================
Implementation responding to Reviewer iu4b's request for ablation studies.

As stated in Authors' Response (Dec 30, 2025):
"We have added ablation studies in Section 4.6 (pages 16--17) by disabling the 
orthogonality regularization and the contrastive loss, while keeping the rest of 
KASPER unchanged."

Key Findings from Reviewer Response:
1. **Removing Orthogonality Regularization** → Regime Collapse
   - Samples assigned to a single regime
   - Regime-specific weights converge to near-identical representations
   
2. **Removing Contrastive Loss** → Regime Entanglement  
   - Separation between market states becomes weak
   - Assignments become low-confidence (bullish/bearish proportions nearly identical)
   - Presented in Fig. 8 of revised manuscript

This script implements these ablation experiments and generates diagnostic visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from kasper_model import KASPER, KASPEREnsemble
from trainer import ForecasterTrainer
from data_utils import load_yahoo_finance_data, engineer_features, prepare_data
from metrics import calculate_statistical_metrics, calculate_financial_metrics
from losses import orthogonality_loss, contrastive_loss


class KASPERAblation(nn.Module):
    """
    KASPER variant for ablation studies.
    Allows selective disabling of:
    - Orthogonality regularization (L_orth)
    - Contrastive loss (L_contrast)
    """
    def __init__(self, input_dim, hidden_dim=128, n_regimes=3, n_basis=20, 
                 use_orthogonality=True, use_contrastive=True):
        super(KASPERAblation, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        self.n_basis = n_basis
        
        # Ablation flags
        self.use_orthogonality = use_orthogonality
        self.use_contrastive = use_contrastive
        
        # Initialize base KASPER model
        self.base_model = KASPER(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_regimes=n_regimes,
            n_basis=n_basis
        )
        
    def forward(self, x):
        """Forward pass with regime information."""
        return self.base_model(x)
    
    def compute_loss(self, x, y_true, y_pred, regime_probs, embeddings):
        """
        Custom loss function for ablation studies.
        
        Args:
            x: Input features (batch, input_dim)
            y_true: Ground truth returns (batch,)
            y_pred: Predicted returns (batch,)
            regime_probs: Regime probabilities (batch, n_regimes)
            embeddings: Latent embeddings (batch, hidden_dim)
            
        Returns:
            loss: Total loss (sum of enabled components)
            loss_components: Dictionary of individual loss values
        """
        # Base prediction loss (always enabled)
        mse_loss = torch.nn.functional.mse_loss(y_pred.squeeze(), y_true.squeeze())
        
        loss_components = {
            'mse': mse_loss.item()
        }
        
        total_loss = mse_loss
        
        # Orthogonality regularization (λ_orth = 0.05 from paper)
        if self.use_orthogonality:
            # Pass coefficients directly
            orth_loss = orthogonality_loss(self.base_model.forecasting.coeffs) 
            total_loss = total_loss + 0.05 * orth_loss
            loss_components['orthogonality'] = orth_loss.item()
        else:
            loss_components['orthogonality'] = 0.0
            
        # Contrastive loss (λ_contrast = 0.01 from paper)
        if self.use_contrastive:
            # Use the more advanced contrastive_loss on embeddings
            contr_loss = contrastive_loss(embeddings, regime_probs)
            total_loss = total_loss + 0.01 * contr_loss
            loss_components['contrastive'] = contr_loss.item()
        else:
            loss_components['contrastive'] = 0.0
            
        return total_loss, loss_components
    
    def fit_knots(self, X_train):
        """Initialize knots from training data."""
        self.base_model.fit_knots(X_train)


def train_ablation_model(model, X_train, y_train, X_val, y_val, 
                         epochs=100, batch_size=32, lr=0.001):
    """
    Training loop for ablation models.
    Uses custom loss function that respects ablation settings.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'mse': 0, 'orthogonality': 0, 'contrastive': 0}
        n_batches = 0
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred, regime_probs, embeddings = model(X_batch)
            
            # Compute loss with ablation settings
            loss, loss_components = model.compute_loss(
                X_batch, y_batch, y_pred, regime_probs, embeddings
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses['total'] += loss.item()
            for k, v in loss_components.items():
                epoch_losses[k] += v
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_regime_probs, val_embeddings = model(X_val_t)
            val_loss, _ = model.compute_loss(
                X_val_t, y_val_t, val_pred, val_regime_probs, val_embeddings
            )
            val_loss_value = val_loss.item()
        
        # Early stopping
        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_losses['total']/n_batches:.6f} | "
                  f"Val Loss: {val_loss_value:.6f}")
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def analyze_regime_collapse(model, X_data, name="Dataset"):
    """
    Analyze regime distribution to detect collapse.
    
    Regime collapse indicators:
    - One regime dominates (>95% of samples)
    - Other regimes are rarely/never used
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_data)
        _, regime_probs, _ = model(X_tensor)
        
        regime_assignments = regime_probs.argmax(dim=1).cpu().numpy()
        
        # Calculate distribution
        unique, counts = np.unique(regime_assignments, return_counts=True)
        dist = {r: 0 for r in range(model.n_regimes)}
        for r, c in zip(unique, counts):
            dist[r] = c
            
        print(f"\n{name} Regime Distribution:")
        for regime, count in dist.items():
            pct = 100 * count / len(regime_assignments)
            print(f"  Regime {regime}: {count:5d} ({pct:5.1f}%)")
            
        # Check for collapse (one regime > 95%)
        max_pct = max(dist.values()) / len(regime_assignments)
        is_collapsed = max_pct > 0.95
        
        if is_collapsed:
            print(f"  ⚠️  REGIME COLLAPSE DETECTED! (Max regime: {max_pct*100:.1f}%)")
        
        return dist, is_collapsed


def analyze_regime_entanglement(model, X_data, y_data, name="Dataset"):
    """
    Analyze regime entanglement (weak separation).
    
    Indicators:
    - Low confidence in regime assignments (avg prob < 0.6)
    - Similar proportions of bullish/bearish in each regime
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_data)
        _, regime_probs, _ = model(X_tensor)
        
        # Average confidence (max probability per sample)
        confidences = regime_probs.max(dim=1)[0].cpu().numpy()
        avg_confidence = confidences.mean()
        
        print(f"\n{name} Regime Confidence:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        # Analyze market state separation
        regime_assignments = regime_probs.argmax(dim=1).cpu().numpy()
        
        print(f"\n{name} Market State Separation:")
        for regime in range(model.n_regimes):
            mask = regime_assignments == regime
            if mask.sum() > 0:
                regime_returns = y_data[mask]
                bullish_pct = 100 * (regime_returns > 0).mean()
                bearish_pct = 100 * (regime_returns < 0).mean()
                print(f"  Regime {regime}: Bullish={bullish_pct:.1f}%, Bearish={bearish_pct:.1f}%")
                
                # Check for entanglement (similar proportions)
                if abs(bullish_pct - bearish_pct) < 10:
                    print(f"    ⚠️  WEAK SEPARATION (nearly identical proportions)")
        
        is_entangled = avg_confidence < 0.6
        
        if is_entangled:
            print(f"  ⚠️  REGIME ENTANGLEMENT DETECTED! (Low confidence: {avg_confidence:.3f})")
        
        return avg_confidence, is_entangled


def run_ablation_experiments(ticker="^DJI"):
    """
    Main ablation study function.
    Implements Section 4.6 (pages 16-17) of revised manuscript.
    """
    print("="*70)
    print("KASPER ABLATION STUDIES (Section 4.6)")
    print("Responding to Reviewer iu4b's Request")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    raw_data = load_yahoo_finance_data(ticker=ticker, start="2018-01-01", end="2023-12-31")
    X, y = engineer_features(raw_data)
    bundles = prepare_data(X, y)
    
    X_train, y_train = bundles['X_train'], bundles['y_train']
    X_val, y_val = bundles['X_val'], bundles['y_val']
    X_test, y_test = bundles['X_test'], bundles['y_test']
    scaler_y = bundles['scaler_y']
    
    input_dim = X_train.shape[1]
    
    # Define ablation configurations
    configs = {
        'Full KASPER': {'use_orthogonality': True, 'use_contrastive': True},
        'No Orthogonality': {'use_orthogonality': False, 'use_contrastive': True},
        'No Contrastive': {'use_orthogonality': True, 'use_contrastive': False},
    }
    
    results = []
    regime_stats = {}
    
    # Run experiments
    for config_name, config_params in configs.items():
        print(f"\n{'='*70}")
        print(f"[Experiment] {config_name}")
        print(f"  Orthogonality: {'✓' if config_params['use_orthogonality'] else '✗'}")
        print(f"  Contrastive:   {'✓' if config_params['use_contrastive'] else '✗'}")
        print(f"{'='*70}")
        
        # Initialize model
        model = KASPERAblation(
            input_dim=input_dim,
            hidden_dim=128,
            n_regimes=3,
            **config_params
        )
        
        # Fit knots
        model.fit_knots(torch.FloatTensor(X_train))
        
        # Train
        print("\n[2/5] Training model...")
        model = train_ablation_model(model, X_train, y_train, X_val, y_val, epochs=100)
        
        # Evaluate
        print("\n[3/5] Evaluating on test set...")
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            y_pred_scaled, regime_probs, _ = model(X_test_tensor)
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        stats = calculate_statistical_metrics(y_true, y_pred)
        finance = calculate_financial_metrics(y_true, y_pred)
        
        # Regime analysis
        print("\n[4/5] Analyzing regime behavior...")
        dist, is_collapsed = analyze_regime_collapse(model, X_test, "Test Set")
        confidence, is_entangled = analyze_regime_entanglement(model, X_test, y_test, "Test Set")
        
        regime_stats[config_name] = {
            'distribution': dist,
            'is_collapsed': is_collapsed,
            'avg_confidence': confidence,
            'is_entangled': is_entangled
        }
        
        # Store results
        res = {
            'Configuration': config_name,
            'R²': stats['R2'],
            'Sharpe': finance['Sharpe Ratio'],
            'MSE': stats['MSE'],
            'Win Rate (%)': finance['Win Rate (%)'],
            'Regime Collapse': '✗ Yes' if is_collapsed else '✓ No',
            'Entanglement': '✗ Yes' if is_entangled else '✓ No'
        }
        results.append(res)
        
        print(f"\n[5/5] Results:")
        print(f"  R²: {res['R²']:.4f}")
        print(f"  Sharpe: {res['Sharpe']:.4f}")
        print(f"  Regime Issues: {res['Regime Collapse']} / {res['Entanglement']}")
    
    # Summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Visualizations
    print("\n[Generating Fig. 8 visualization...]")
    plot_ablation_results(df, regime_stats)
    
    print("\n✅ Ablation studies complete!")
    print("   Results saved to 'ablation_results.png'")
    print("\nKey Findings (as stated in reviewer response):")
    print("  1. Removing orthogonality → Regime collapse")
    print("  2. Removing contrastive loss → Regime entanglement")
    print("  3. Full KASPER maintains stable, well-separated regimes")


def plot_ablation_results(df, regime_stats):
    """
    Generate Fig. 8 from revised manuscript.
    Shows impact of ablations on regime quality and performance.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Performance comparison
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(df))
    ax1.bar(x_pos, df['R²'], color=['#4285F4', '#EA4335', '#FBBC05'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['Configuration'], rotation=15, ha='right')
    ax1.set_ylabel('R² Score')
    ax1.set_title('(a) R² Performance')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x_pos, df['Sharpe'], color=['#4285F4', '#EA4335', '#FBBC05'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df['Configuration'], rotation=15, ha='right')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('(b) Sharpe Ratio')
    ax2.grid(axis='y', alpha=0.3)
    
    # Regime distributions
    for idx, (config_name, stats) in enumerate(regime_stats.items()):
        ax = plt.subplot(2, 3, 4 + idx)
        dist = stats['distribution']
        regimes = list(dist.keys())
        counts = [dist[r] for r in regimes]
        
        colors = ['red' if stats['is_collapsed'] or stats['is_entangled'] else 'green']
        ax.bar(regimes, counts, color=colors[0], alpha=0.6, edgecolor='black')
        ax.set_xlabel('Regime')
        ax.set_ylabel('Sample Count')
        ax.set_title(f"({chr(99+idx)}) {config_name}\nConfidence: {stats['avg_confidence']:.2f}")
        ax.set_xticks(regimes)
        
        # Add warning markers
        if stats['is_collapsed']:
            ax.text(0.5, 0.95, '⚠️ Collapse', transform=ax.transAxes,
                   ha='center', va='top', fontsize=10, color='red', weight='bold')
        if stats['is_entangled']:
            ax.text(0.5, 0.85, '⚠️ Entanglement', transform=ax.transAxes,
                   ha='center', va='top', fontsize=10, color='orange', weight='bold')
    
    # Win rate comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(x_pos, df['Win Rate (%)'], color=['#4285F4', '#EA4335', '#FBBC05'])
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df['Configuration'], rotation=15, ha='right')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('(c) Win Rate')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax3.legend()
    
    plt.suptitle('Fig. 8: Ablation Study Results (Section 4.6)\n' + 
                 'Impact of Removing Regularization Components',
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('ablation_results.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    run_ablation_experiments()
