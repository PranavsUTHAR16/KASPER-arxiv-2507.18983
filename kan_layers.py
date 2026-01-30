import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SplineActivation(nn.Module):
    """
    Hybrid activation function combining linear and cubic components.
    f(x) = L(x) + C(x)
    """
    def __init__(self, input_dim, n_linear=3, n_cubic=2, initialization='quantile'):
        super(SplineActivation, self).__init__()
        self.input_dim = input_dim
        self.n_linear = n_linear
        self.n_cubic = n_cubic
        
        # Learnable parameters for linear component
        self.w_linear = nn.Parameter(torch.randn(input_dim, n_linear))
        # Learnable parameters for cubic component
        self.v_cubic = nn.Parameter(torch.randn(input_dim, n_cubic))
        
        # Knots for spline (initialized later)
        self.register_buffer('knots', torch.linspace(-1, 1, n_linear + 1))
        
    def fit_knots(self, x):
        """Initializes knots based on quantiles of the input data."""
        with torch.no_grad():
            quantiles = torch.linspace(0, 1, self.n_linear + 1, device=x.device)
            new_knots = torch.quantile(x, quantiles, dim=0).T # (input_dim, n_linear+1)
            self.knots.data = new_knots
        
    def forward(self, x):
        # x: (batch, input_dim)
        x_expanded = x.unsqueeze(-1) # (batch, in, 1)
        knots_expanded = self.knots.unsqueeze(0) # (1, in, n_linear+1)
        
        # Equation 15: Linear Component (Piecewise Linear Ramp)
        # ReLU(x - ki) - ReLU(x - ki+1) creates a linear basis segment
        diffs = F.relu(x_expanded - knots_expanded[:, :, :-1]) - F.relu(x_expanded - knots_expanded[:, :, 1:])
        linear_out = torch.sum(torch.tanh(self.w_linear).unsqueeze(0) * diffs, dim=-1)
        
        # Equation 16: Cubic Component
        # sigma(vi) * x^3
        # Assuming n_cubic is 1 for direct Eq 16 compliance
        x_cubic = x.pow(3).unsqueeze(-1) 
        cubic_out = torch.sigmoid(self.v_cubic).unsqueeze(0) * x_cubic
        cubic_out = cubic_out.sum(dim=-1)
        
        return linear_out + cubic_out

class RegimeDetectionLayer(nn.Module):
    """
    KAN Layer 1: Regime Detection
    Architecture: (L -> BN -> GELU -> Dropout) x 2 -> L -> GS
    """
    def __init__(self, input_dim, hidden_dim=64, n_regimes=3, temperature=0.5):
        super(RegimeDetectionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        self.spline_act = SplineActivation(input_dim)
        
        # Block 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        # Block 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        
        self.regime_logits = nn.Linear(hidden_dim, n_regimes, bias=False)
        self.temperature = temperature
        
    def orthogonality_loss(self):
        """Computes ||W * W^T - I||^2_F on regime logits weights"""
        W = self.regime_logits.weight # (n_regimes, hidden_dim)
        gram = torch.mm(W, W.t())
        identity = torch.eye(W.size(0), device=W.device)
        return torch.norm(gram - identity, p='fro') ** 2
        
    def forward(self, x):
        x_spline = self.spline_act(x)
        
        # Block 1
        h = self.fc1(x_spline)
        h = self.bn1(h)
        h = F.gelu(h)
        h = self.dropout1(h)
        
        # Block 2
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.gelu(h)
        embeddings = self.dropout2(h)
        
        logits = self.regime_logits(embeddings)
        
        if self.training:
            # Using hard=True to force discrete regime selection
            regime_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            regime_probs = F.softmax(logits / self.temperature, dim=-1)
            
        return regime_probs, embeddings

class RegimeAdaptiveForecastingLayer(nn.Module):
    """
    KAN Layer 2: Regime-specific adaptive forecasting using TRUE B-Splines.
    """
    def __init__(self, hidden_dim, n_regimes=3, spline_order=3, n_basis_functions=20):
        super(RegimeAdaptiveForecastingLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_regimes = n_regimes
        self.n_basis = n_basis_functions
        self.spline_order = spline_order
        
        # Forecasting coefficients: (n_regimes, hidden_dim, n_basis)
        self.coeffs = nn.Parameter(torch.randn(n_regimes, hidden_dim, n_basis_functions) * 0.5)
        
        # Knots are REGIME-SPECIFIC and per-feature as per Fig 1
        n_knots = n_basis_functions + spline_order + 1
        self.register_buffer('knots', torch.zeros(n_regimes, hidden_dim, n_knots))
        # Initial uniform knots as fallback
        for r in range(n_regimes):
            for h in range(hidden_dim):
                self.knots.data[r, h] = torch.linspace(-3, 3, n_knots)
        
        # Sparsity mask (for dynamic pruning)
        self.register_buffer('sparsity_mask', torch.ones_like(self.coeffs))

    def fit_knots(self, h, regime_probs):
        """
        Initializes knots for each regime based on the data actually assigned to it.
        h: (N, hidden_dim) - embeddings
        regime_probs: (N, n_regimes) - soft regime assignments
        """
        with torch.no_grad():
            regime_assignments = torch.argmax(regime_probs, dim=1)
            quantiles = torch.linspace(0, 1, self.n_basis + self.spline_order + 1, device=h.device)
            
            for r in range(self.n_regimes):
                mask = (regime_assignments == r)
                if mask.any():
                    h_r = h[mask]
                    # Quantile normalization per feature for this regime
                    new_knots_r = torch.quantile(h_r, quantiles, dim=0).T # (hidden_dim, n_knots)
                    self.knots.data[r] = new_knots_r
                else:
                    # Fallback if no samples (keep existing or use global)
                    pass

    def grid_extension(self, new_n_basis):
        """
        G1 -> G2 extension as mentioned in Fig 1.
        Increases the knot resolution while preserving (as much as possible) the learned shape.
        """
        old_n_basis = self.n_basis
        if new_n_basis <= old_n_basis:
            return
            
        # 1. Update knots with higher resolution
        # For simplicity, we interpolate current knots to the new resolution
        n_knots_new = new_n_basis + self.spline_order + 1
        new_knots = torch.zeros(self.n_regimes, self.hidden_dim, n_knots_new, device=self.knots.device)
        
        for r in range(self.n_regimes):
            for h in range(self.hidden_dim):
                # Basic linear interpolation for the new knot sequence
                old_k = self.knots[r, h]
                indices = torch.linspace(0, len(old_k)-1, n_knots_new, device=self.knots.device)
                new_knots[r, h] = torch.lerp(old_k[indices.long()], old_k[torch.clamp(indices.long() + 1, max=len(old_k)-1)], indices % 1)
        
        self.knots.data = new_knots
        
        # 2. Update coeffs (zero-init or interpolate)
        new_coeffs = torch.zeros(self.n_regimes, self.hidden_dim, new_n_basis, device=self.coeffs.device)
        # Note: True grid extension involves least-squares fitting of coeffs to match old function.
        # Here we'll zero-init the new basis components and keep old ones if they align,
        # or just random. We'll use random * 0.1 for new capacity.
        new_coeffs[:, :, :old_n_basis] = self.coeffs.data
        self.coeffs = nn.Parameter(new_coeffs)
        self.sparsity_mask = torch.ones_like(self.coeffs)
        self.n_basis = new_n_basis

    def update_sparsity_mask(self, threshold=0.01):
        """Update mask based on Equation 22: w' = ReLU(|w| - theta)"""
        with torch.no_grad():
            self.sparsity_mask.data = (torch.abs(self.coeffs) > threshold).float()

    def b_spline_basis(self, x, k):
        """
        Iterative vectorized B-spline basis function calculation.
        Modified to support regime-specific knots.
        x: (batch, hidden_dim, 1)
        returns phi: (n_regimes, batch, hidden_dim, n_basis)
        """
        batch_size, hidden_dim, _ = x.shape
        # x is (batch, hidden_dim) implicitly
        x_val = x.squeeze(-1)
        
        # Output container for all regimes
        all_regime_phi = []
        
        for r in range(self.n_regimes):
            # Knots for this regime: (hidden_dim, n_knots)
            knots_r = self.knots[r]
            n_knots = knots_r.size(1)
            
            # Base case (order 0): (batch, hidden_dim, n_knots - 1)
            # x_val: (batch, hidden_dim), knots_r: (hidden_dim, n_knots)
            # Broadcast comparison
            p_0 = ((x_val.unsqueeze(-1) >= knots_r[:, :-1].unsqueeze(0)) & 
                   (x_val.unsqueeze(-1) < knots_r[:, 1:].unsqueeze(0))).float()
            
            # Current level phi
            phi_current = p_0 
            
            # Recursively build higher degrees
            for deg in range(1, k + 1):
                # deg reduce the number of basis functions by 1 in each step
                # knots_r: (hidden_dim, n_knots)
                # phi_current: (batch, hidden_dim, n_basis_current)
                n_basis_current = phi_current.size(-1)
                new_phi = torch.zeros(batch_size, hidden_dim, n_basis_current - 1, device=x.device)
                
                for i in range(n_basis_current - 1):
                    # Cox-de Boor recursion
                    left_num = x_val - knots_r[:, i].unsqueeze(0)
                    left_denom = (knots_r[:, i + deg] - knots_r[:, i]).unsqueeze(0) + 1e-8
                    left = (left_num / left_denom) * phi_current[:, :, i]
                    
                    right_num = knots_r[:, i + deg + 1].unsqueeze(0) - x_val
                    right_denom = (knots_r[:, i + deg + 1] - knots_r[:, i + 1]).unsqueeze(0) + 1e-8
                    right = (right_num / right_denom) * phi_current[:, :, i + 1]
                    
                    new_phi[:, :, i] = left + right
                phi_current = new_phi
                
            all_regime_phi.append(phi_current)
            
        # Stack to (n_regimes, batch, hidden_dim, n_basis)
        return torch.stack(all_regime_phi, dim=0)

    def forward(self, h, regime_probs):
        # h: (batch, hidden_dim)
        # regime_probs: (batch, n_regimes)
        
        # 1. Compute basis functions for ALL regimes (potentially different knots)
        # phi: (n_regimes, batch, hidden_dim, n_basis)
        phi = self.b_spline_basis(h.unsqueeze(-1), self.spline_order)
        
        # 2. Compute regime-specific predictions
        # coeffs: (r, h, b), phi: (r, n, h, b)
        # masked_coeffs = coeffs * sparsity_mask
        masked_coeffs = self.coeffs * self.sparsity_mask
        
        # Prediction for each regime: sum over hidden_dim and basis
        # Use einsum to sum over h and b for each r and n
        regime_preds = torch.einsum('rhb,rnhb->nr', masked_coeffs, phi)
        
        # 3. Weighted total prediction by regime probabilities
        final_pred = torch.sum(regime_preds * regime_probs, dim=-1, keepdim=True)
        
        return final_pred
