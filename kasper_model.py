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

if __name__ == "__main__":
    # Test forward pass
    model = KASPER(input_dim=8)
    dummy_input = torch.randn(32, 8)
    pred, probs, embed = model(dummy_input)
    print(f"Prediction shape: {pred.shape}")
    print(f"Regime Probs shape: {probs.shape}")
    print(f"Embeddings shape: {embed.shape}")
