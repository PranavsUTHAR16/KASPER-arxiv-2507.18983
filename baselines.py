"""
Baseline Models for KASPER Comparison (Section 4.7, Table 4)
Implements PatchTST and DLinear as requested by reviewers.
Both models are regime-agnostic (no regime detection capability).
"""

import torch
import torch.nn as nn
import numpy as np


class PatchTST(nn.Module):
    """
    PatchTST Baseline: Patch-based Time Series Transformer
    Regime-agnostic model that uses patching and transformer architecture.
    
    Reference: Nie et al. (2023) "A Time Series is Worth 64 Words"
    As mentioned in Authors' Response to Reviewer iu4b
    """
    def __init__(self, input_dim=8, patch_len=4, hidden_dim=64, n_heads=4, n_layers=2):
        super(PatchTST, self).__init__()
        self.input_dim = input_dim
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(patch_len, hidden_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) - flattened feature vector
        Returns:
            prediction: (batch, 1) - forecasted return
        """
        batch_size = x.shape[0]
        
        # Reshape to patches: (batch, input_dim, patch_len)
        # For simplicity, treat each feature as a patch
        x_patched = x.unsqueeze(-1).expand(-1, -1, self.patch_len)  # (batch, input_dim, patch_len)
        
        # Project patches
        x_proj = self.input_projection(x_patched)  # (batch, input_dim, hidden_dim)
        
        # Add positional encoding
        x_pos = x_proj + self.pos_embedding
        
        # Transformer encoding
        x_encoded = self.transformer(x_pos)  # (batch, input_dim, hidden_dim)
        
        # Flatten and predict
        x_flat = x_encoded.reshape(batch_size, -1)
        prediction = self.output_head(x_flat)
        
        return prediction.squeeze(-1)


class DLinear(nn.Module):
    """
    DLinear Baseline: Decomposition Linear Model
    Regime-agnostic model with trend-seasonality decomposition.
    
    Reference: Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?"
    As mentioned in Authors' Response to Reviewer iu4b
    
    According to reviewers: DLinear achieves R²=0.03, Sharpe=3.40
    (much lower than KASPER's R²=0.89)
    """
    def __init__(self, input_dim=8, seq_len=1, hidden_dim=64):
        super(DLinear, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len  # For single-step, seq_len=1
        
        # Moving average kernel for decomposition
        self.kernel_size = 3
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        
        # Separate linear layers for trend and seasonal components
        self.trend_linear = nn.Linear(input_dim, hidden_dim)
        self.seasonal_linear = nn.Linear(input_dim, hidden_dim)
        
        # Output layer
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
        Returns:
            prediction: (batch,) - forecasted return
        """
        # Decomposition (simplified for feature vector input)
        # Add temporal dimension for decomposition
        x_temp = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Moving average as trend
        trend = self.avg_pool(x_temp.transpose(1, 2)).transpose(1, 2)  # (batch, 1, input_dim)
        seasonal = x_temp - trend
        
        # Remove temporal dimension
        trend = trend.squeeze(1)  # (batch, input_dim)
        seasonal = seasonal.squeeze(1)  # (batch, input_dim)
        
        # Linear transformations
        trend_out = self.trend_linear(trend)  # (batch, hidden_dim)
        seasonal_out = self.seasonal_linear(seasonal)  # (batch, hidden_dim)
        
        # Combine and predict
        combined = torch.cat([trend_out, seasonal_out], dim=-1)  # (batch, hidden_dim*2)
        prediction = self.output(combined)
        
        return prediction.squeeze(-1)


def train_baseline(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001):
    """
    Training function for baseline models.
    Uses same training protocol as KASPER for fair comparison.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/n_batches:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_state)
    return model


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...")
    
    # Test PatchTST
    patch_tst = PatchTST(input_dim=8)
    x_test = torch.randn(32, 8)
    pred_patch = patch_tst(x_test)
    print(f"PatchTST output shape: {pred_patch.shape}")
    
    # Test DLinear
    dlinear = DLinear(input_dim=8)
    pred_dlinear = dlinear(x_test)
    print(f"DLinear output shape: {pred_dlinear.shape}")
    
    print("\n✅ Baseline models initialized successfully!")
    print("Expected performance (from reviewer response):")
    print("  - PatchTST: R²=0.10, Sharpe=2.24")
    print("  - DLinear: R²=0.03, Sharpe=3.40")
    print("  - KASPER: R²=0.89, Sharpe=12.02 (significantly better)")
