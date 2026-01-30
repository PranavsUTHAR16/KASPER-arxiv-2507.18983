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
        
        # Per-feature knots (Fix 3)
        self.register_buffer(
            'knots', 
            torch.linspace(-1, 1, n_linear + 1).unsqueeze(0).repeat(input_dim, 1)
        )
        
    def fit_knots(self, x):
        """
        Initializes knots PER FEATURE using 1st-99th percentiles (as per author response).
        Ensures robust scaling for financial return distributions.
        """
        with torch.no_grad():
            for i in range(self.input_dim):
                # Use 1st and 99th percentile as bounds
                p_low = torch.quantile(x[:, i], 0.01)   # 1st percentile
                p_high = torch.quantile(x[:, i], 0.99)  # 99th percentile
                
                # Place knots uniformly between bounds (as per Eq 21 clarification)
                knot_positions = torch.linspace(0, 1, self.n_linear + 1, device=x.device)
                self.knots.data[i] = p_low + knot_positions * (p_high - p_low)
                
                # Optional logging for knot health
                if i < 3: # Log first 3 features
                    print(f"  Feature {i}: knots initialized from {p_low:.4f} to {p_high:.4f}")
        
    def forward(self, x):
        # x: (batch, input_dim)
        x_expanded = x.unsqueeze(-1) # (batch, in, 1)
        # knots_expanded: (1, in, n_linear+1) matches self.knots if expanded correctly
        knots_expanded = self.knots.unsqueeze(0) 
        
        # Equation 15: Linear Component (Piecewise Linear Ramp)
        # Using per-feature knots
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
        
        # Initialize regime logits with ORTHOGONAL weights (Diagnosis Fix 7)
        with torch.no_grad():
            nn.init.orthogonal_(self.regime_logits.weight)
            # Scale up to ensure strong initial separation
            self.regime_logits.weight.data *= 3.0
        
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
        # Phase 3: "Middle Ground" scale (0.05) to increase variance
        self.coeffs = nn.Parameter(torch.randn(n_regimes, hidden_dim, n_basis_functions) * 0.05)
        
        # Knots are REGIME-SPECIFIC and per-feature as per Fig 1
        n_knots = n_basis_functions + spline_order + 1
        self.register_buffer('knots', torch.zeros(n_regimes, hidden_dim, n_knots))
        # Initial uniform knots as fallback
        for r in range(n_regimes):
            for h in range(hidden_dim):
                self.knots.data[r, h] = torch.linspace(-3, 3, n_knots)
        
        # Add BatchNorm BEFORE spline transformation (Fix 7)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
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
            n_knots = self.n_basis + self.spline_order + 1
            quantiles = torch.linspace(0, 1, n_knots, device=h.device)
            
            for r in range(self.n_regimes):
                mask = (regime_assignments == r)
                if mask.any() and mask.sum() > n_knots:
                    h_r = h[mask]
                    # Quantile normalization per feature for this regime
                    # Add small epsilon to ensure strictly increasing knots if data is flat
                    new_knots_r = torch.quantile(h_r, quantiles, dim=0).T # (hidden_dim, n_knots)
                    
                    # Ensure strictly increasing
                    diff = new_knots_r[:, 1:] - new_knots_r[:, :-1]
                    if (diff <= 0).any():
                        # Add a small trend to ensure separation
                        epsilon = 1e-4 * torch.arange(n_knots, device=h.device).view(1, -1)
                        new_knots_r = new_knots_r + epsilon
                        
                    self.knots.data[r] = new_knots_r
                else:
                    # Fallback if no samples: keep existing or use global range
                    pass

    def grid_extension(self, new_n_basis, h_samples=None):
        """
        G1 -> G2 extension as mentioned in KAN paper Fig 1.
        Increases the knot resolution while preserving the learned function shape using LSQ.
        """
        old_n_basis = self.n_basis
        if new_n_basis <= old_n_basis:
            return
            
        print(f"  Extending grid from {old_n_basis} to {new_n_basis} basis functions...")
        
        with torch.no_grad():
            # 1. Update knots with higher resolution
            n_knots_new = new_n_basis + self.spline_order + 1
            new_knots = torch.zeros(self.n_regimes, self.hidden_dim, n_knots_new, device=self.knots.device)
            
            for r in range(self.n_regimes):
                for h_idx in range(self.hidden_dim):
                    old_k = self.knots[r, h_idx]
                    # Generate new knots by interpolating old ones (uniform distribution over same range)
                    new_knots[r, h_idx] = torch.linspace(old_k[0], old_k[-1], n_knots_new, device=self.knots.device)
            
            # If h_samples is provided, use LSQ to fit new coefficients
            if h_samples is not None:
                # phi_old: (r, n, h, b_old)
                phi_old = self.b_spline_basis(h_samples.unsqueeze(-1), self.spline_order)
                # target_values: (r, n, h) = sum_b coeff_old * phi_old
                # coeffs: (r, h, b)
                target = torch.einsum('rhb,rnhb->rnh', self.coeffs * self.sparsity_mask, phi_old)
                
                # Temporarily update knots to compute new phi
                original_knots = self.knots.data.clone()
                original_n_basis = self.n_basis
                self.knots.data = new_knots
                self.n_basis = new_n_basis
                
                # phi_new: (r, n, h, b_new)
                phi_new = self.b_spline_basis(h_samples.unsqueeze(-1), self.spline_order)
                
                # Solve LSQ for each regime and hidden dimension
                # phi_new_flat: (r*h, n, b_new), target_flat: (r*h, n)
                phi_new_flat = phi_new.permute(0, 2, 1, 3).reshape(-1, h_samples.size(0), new_n_basis)
                target_flat = target.permute(0, 2, 1).reshape(-1, h_samples.size(0))
                
                # Robust LSQ solution: (phi^T * phi)^-1 * phi^T * target
                # We'll use torch.linalg.lstsq
                new_coeffs_flat = torch.linalg.lstsq(phi_new_flat, target_flat.unsqueeze(-1)).solution
                new_coeffs = new_coeffs_flat.view(self.n_regimes, self.hidden_dim, new_n_basis)
                
                self.coeffs = nn.Parameter(new_coeffs)
            else:
                # Naive extension: interpolate/pad coefficients
                new_coeffs = torch.zeros(self.n_regimes, self.hidden_dim, new_n_basis, device=self.coeffs.device)
                indices = torch.linspace(0, old_n_basis-1, new_n_basis, device=self.coeffs.device).long()
                new_coeffs = self.coeffs.data[:, :, indices]
                self.knots.data = new_knots
                self.coeffs = nn.Parameter(new_coeffs)
                self.n_basis = new_n_basis

            self.sparsity_mask = torch.ones_like(self.coeffs)

    def update_sparsity_mask(self, threshold=0.01):
        """
        Update coefficients and mask based on Equation 22: w' = sign(w) * ReLU(|w| - theta)
        This implements soft-thresholding (proximal operator for L1).
        """
        with torch.no_grad():
            # Update the weights themselves as per Eq 22
            abs_coeffs = torch.abs(self.coeffs)
            new_coeffs = torch.sign(self.coeffs) * F.relu(abs_coeffs - threshold)
            self.coeffs.data = new_coeffs
            
            # Update mask for visualization/pruning
            self.sparsity_mask.data = (torch.abs(self.coeffs) > 0).float()

    def b_spline_basis(self, x, k):
        """
        Fully vectorized B-spline basis function calculation.
        x: (batch, hidden_dim, 1)
        returns phi: (n_regimes, batch, hidden_dim, n_basis)
        """
        # x is (batch, hidden_dim, 1)
        batch_size, hidden_dim, _ = x.shape
        x_val = x # (batch, hidden_dim, 1)
        
        # Knots: (n_regimes, hidden_dim, n_knots)
        knots = self.knots # (R, H, K)
        n_regimes = knots.size(0)
        n_knots = knots.size(2)
        
        # Base case (order 0): phi_0 is 1 if knots[i] <= x < knots[i+1], else 0
        # Broadcast x to (1, batch, hidden_dim, 1) and knots to (n_regimes, 1, hidden_dim, n_knots)
        # Resulting phi: (n_regimes, batch, hidden_dim, n_knots - 1)
        phi = ((x_val.unsqueeze(0) >= knots.unsqueeze(1)[:, :, :, :-1]) & 
               (x_val.unsqueeze(0) < knots.unsqueeze(1)[:, :, :, 1:])).float()
        
        # Cox-de Boor recursion
        for deg in range(1, k + 1):
            n_basis_step = phi.size(-1)
            
            # k_i: t_i for i=0..N-2
            k_i = knots[:, :, :n_basis_step-1].unsqueeze(1)
            # k_id: t_{i+deg} for i=0..N-2
            k_id = knots[:, :, deg:n_basis_step+deg-1].unsqueeze(1)
            # k_i1: t_{i+1} for i=0..N-2
            k_i1 = knots[:, :, 1:n_basis_step].unsqueeze(1)
            # k_id1: t_{i+deg+1} for i=0..N-2
            k_id1 = knots[:, :, deg+1:n_basis_step+deg].unsqueeze(1)
            
            # phi_left: B_{i, d-1} for i=0..N-2
            phi_left = phi[:, :, :, :-1]
            # phi_right: B_{i+1, d-1} for i=0..N-2
            phi_right = phi[:, :, :, 1:]
            
            denom1 = (k_id - k_i)
            term1 = ((x_val.unsqueeze(0) - k_i) / (denom1 + 1e-8)) * phi_left
            
            denom2 = (k_id1 - k_i1)
            term2 = ((k_id1 - x_val.unsqueeze(0)) / (denom2 + 1e-8)) * phi_right
            
            phi = term1 + term2
            
        return phi

    def forward(self, h, regime_probs):
        # h: (batch, hidden_dim)
        # regime_probs: (batch, n_regimes)
        
        # Apply batch norm (Fix 7)
        if self.training and h.size(0) > 1:
            h = self.bn(h)
        
        # 1. Compute basis functions for ALL regimes
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
