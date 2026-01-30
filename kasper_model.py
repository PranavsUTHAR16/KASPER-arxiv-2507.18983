import torch
import torch.nn as nn
from kan_layers import RegimeDetectionLayer, RegimeAdaptiveForecastingLayer

class KASPER(nn.Module):
    """
    KASPER: Kolmogorov-Arnold Network for Sensitive Prediction and Evolutionary Regime-shifting.
    Integrates two layers of KAN for regime-adaptive stock forecasting.
    """
    def __init__(self, input_dim=8, hidden_dim=64, n_regimes=3, n_basis=10):
        super(KASPER, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        
        # Layer 1: Regime Detection
        self.regime_detection = RegimeDetectionLayer(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            n_regimes=n_regimes
        )
        
        # Layer 2: Adaptive Forecasting
        self.forecasting = RegimeAdaptiveForecastingLayer(
            hidden_dim=hidden_dim, 
            n_regimes=n_regimes, 
            n_basis_functions=n_basis
        )
        
    def fit_knots(self, x):
        """
        Initializes knots for the spline activation layers.
        """
        self.regime_detection.spline_act.fit_knots(x)
        
    def get_orthogonality_loss(self):
        """Compute L_orth for regime weight matrix"""
        return self.regime_detection.orthogonality_loss()
        
    def update_sparsity(self, threshold=0.01):
        """Update sparsity masks in forecasting layer"""
        self.forecasting.update_sparsity_mask(threshold)
        
    def forward(self, x):
        # x shape: (batch, input_dim)
        
        # 1. Regime Detection
        regime_probs, embeddings = self.regime_detection(x)
        
        # 2. Adaptive Forecasting
        prediction = self.forecasting(embeddings, regime_probs)
        
        return prediction, regime_probs, embeddings

    @property
    def temperature(self):
        return self.regime_detection.temperature
        
    @temperature.setter
    def temperature(self, value):
        self.regime_detection.temperature = value

class KASPEREnsemble(nn.Module):
    """
    KASPER Ensemble for variance reduction (Fix 8).
    Averages predictions and regime probabilities across multiple KASPER instances.
    """
    def __init__(self, input_dim=8, hidden_dim=64, n_regimes=3, n_models=3, n_basis=10):
        super(KASPEREnsemble, self).__init__()
        self.n_regimes = n_regimes
        self.models = nn.ModuleList([
            KASPER(input_dim, hidden_dim, n_regimes, n_basis) 
            for _ in range(n_models)
        ])
        
    def fit_knots(self, x):
        for model in self.models:
            model.fit_knots(x)
            
    def update_sparsity(self, threshold=0.01):
        for model in self.models:
            model.update_sparsity(threshold)
            
    def get_orthogonality_loss(self):
        return sum(m.get_orthogonality_loss() for m in self.models) / len(self.models)
        
    @property
    def forecasting(self):
        # Shorthand for trainer compatibility, but coefficients are per-model
        # This is a bit tricky for composite_loss, so we'll need to adapt it 
        # or just train models individually. 
        # For simplicity, we'll return the first model's forecasting for the detail print.
        return self.models[0].forecasting

    def forward(self, x):
        predictions = []
        all_regime_probs = []
        all_embeddings = []
        
        for model in self.models:
            pred, regime_probs, embeddings = model(x)
            predictions.append(pred)
            all_regime_probs.append(regime_probs)
            all_embeddings.append(embeddings)
            
        # Average results
        final_pred = torch.stack(predictions).mean(dim=0)
        final_probs = torch.stack(all_regime_probs).mean(dim=0)
        final_embeddings = torch.stack(all_embeddings).mean(dim=0)
        
        return final_pred, final_probs, final_embeddings

    @property
    def temperature(self):
        return self.models[0].regime_detection.temperature
        
    @temperature.setter
    def temperature(self, value):
        for m in self.models:
            m.regime_detection.temperature = value

if __name__ == "__main__":
    # Test forward pass
    ensemble = KASPEREnsemble(input_dim=8)
    dummy_input = torch.randn(32, 8)
    pred, probs, embed = ensemble(dummy_input)
    print(f"Ensemble Prediction shape: {pred.shape}")
