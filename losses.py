import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(embeddings, regime_probs, margin=1.0, lambda_gp=0.1):
    """
    Equation 18: Contrastive Loss for Regime Separation
    
    L_contrastive = E[||z_i - z_j||^2 * y_ij]
    
    where:
    - z_i, z_j: latent embeddings from spline-based regime detection layer
    - y_ij: indicator function (1 if same regime, 0 if different regimes)
    - E: expectation over training mini-batch B
    
    Purpose: Enforces orthogonality in the latent space, improving separability between regimes.
    
    Args:
        embeddings: (batch, hidden_dim) - Latent embeddings z from Φ_detect(x)
        regime_probs: (batch, n_regimes) - Soft regime assignments from Gumbel Softmax
        margin: float - Margin for dissimilar samples (not used in Eq 18, kept for compatibility)
        lambda_gp: float - Gradient penalty weight to prevent embedding collapse
    
    Returns:
        loss: scalar - Contrastive loss value
    """
    batch_size = embeddings.size(0)
    
    # ============================================================
    # STEP 1: Compute pairwise distances ||z_i - z_j||^2
    # ============================================================
    # dist_matrix[i, j] = ||z_i - z_j||^2
    dist_matrix = torch.cdist(embeddings, embeddings, p=2).pow(2)  # (batch, batch)
    
    # ============================================================
    # STEP 2: Compute regime similarity indicator y_ij
    # ============================================================
    # Paper: y_ij indicates if samples i and j belong to same regime
    # Using soft assignment: y_ij = sum_k p_{ik} * p_{jk}
    # This is differentiable approximation to hard indicator
    
    # regime_probs: (batch, n_regimes)
    # similarity[i, j] = sum_k prob[i,k] * prob[j,k]
    y_ij = torch.matmul(regime_probs, regime_probs.t())  # (batch, batch)
    
    # ============================================================
    # STEP 3: Equation 18 - Contrastive Loss
    # L_contrastive = E[||z_i - z_j||^2 * y_ij]
    # ============================================================
    # Interpretation:
    # - If y_ij = 1 (same regime): penalize large distance (pull together)
    # - If y_ij = 0 (different regime): no penalty (allow separation)
    
    contrastive_term = dist_matrix * y_ij
    
    # Expectation over mini-batch (exclude diagonal i=j)
    mask = (1 - torch.eye(batch_size, device=embeddings.device))
    loss = (contrastive_term * mask).sum() / (mask.sum() + 1e-8)
    
    # ============================================================
    # ADDITIONAL: Gradient Penalty (not in Eq 18, for stability)
    # ============================================================
    # Prevent embedding collapse by encouraging unit norm
    embedding_norm = torch.norm(embeddings, p=2, dim=1).mean()
    gradient_penalty = torch.abs(embedding_norm - 1.0)
    
    return loss + lambda_gp * gradient_penalty

def contrastive_loss_hard(embeddings, regime_assignments):
    """
    Alternative: Hard contrastive loss using discrete regime assignments.
    Useful for evaluation or when using hard Gumbel Softmax.
    
    Args:
        embeddings: (batch, hidden_dim)
        regime_assignments: (batch,) - Hard regime labels (integers)
    
    Returns:
        loss: scalar
    """
    batch_size = embeddings.size(0)
    
    # Pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2).pow(2)
    
    # Hard indicator: y_ij = 1 if regime[i] == regime[j], else 0
    regime_i = regime_assignments.unsqueeze(1)  # (batch, 1)
    regime_j = regime_assignments.unsqueeze(0)  # (1, batch)
    y_ij = (regime_i == regime_j).float()  # (batch, batch)
    
    # Contrastive loss
    contrastive_term = dist_matrix * y_ij
    
    # Average over pairs (exclude diagonal)
    mask = (1 - torch.eye(batch_size, device=embeddings.device))
    loss = (contrastive_term * mask).sum() / (mask.sum() + 1e-8)
    
    return loss

def contrastive_regime_loss(regime_probs, y_true, margin=0.5):
    """
    Simplified contrastive loss for ablation studies.
    Pushes samples with opposite return signs into different regimes.
    
    Args:
        regime_probs: (batch, n_regimes) - soft regime assignments
        y_true: (batch,) - true returns
        margin: float - margin for contrastive separation
    
    Returns:
        loss: scalar - contrastive regime loss
    """
    batch_size = y_true.size(0)
    
    # Determine if returns are positive or negative
    signs = torch.sign(y_true)  # +1, 0, or -1
    
    # Pairwise sign agreement: 1 if same sign, 0 otherwise
    # sign_agreement[i,j] = 1 if sign[i] == sign[j] and both != 0
    sign_matrix = signs.unsqueeze(1) * signs.unsqueeze(0)  # (batch, batch)
    same_sign = (sign_matrix > 0).float()  # 1 if same sign, 0 otherwise
    
    # Regime similarity: dot product of regime probabilities
    # regime_sim[i,j] = sum_k prob[i,k] * prob[j,k]
    regime_sim = torch.matmul(regime_probs, regime_probs.t())  # (batch, batch)
    
    # Contrastive loss:
    # - If same sign: encourage high regime similarity (minimize (1 - sim)^2)
    # - If different sign: encourage low regime similarity (minimize sim^2)
    loss = same_sign * torch.pow(1.0 - regime_sim, 2) + \
           (1 - same_sign) * torch.pow(regime_sim, 2)
    
    # Average over all pairs (excluding diagonal)
    mask = (1 - torch.eye(batch_size, device=y_true.device))  # Exclude self-pairs
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    
    return loss

def orthogonality_loss(weights):
    """
    Enforces orthogonality on weights: ||W W^T - I||^2_F
    weights: (n_regimes, out_dim, in_dim) or similar
    """
    # Assuming weights is (n_regimes, ...)
    # Let's flatten to (n_regimes, -1) if needed, but the spec says W_r
    # If W_r is a matrix for each regime
    total_loss = 0
    k = weights.size(0)
    for i in range(k):
        w = weights[i]
        if w.dim() > 2:
            w = w.view(w.size(0), -1)
        
        # normalized weights
        w_norm = F.normalize(w, p=2, dim=1)
        m = torch.matmul(w_norm, w_norm.t())
        identity = torch.eye(m.size(0), device=m.device)
        total_loss += torch.norm(m - identity, p='fro')
        
    return total_loss / k

def l1_regularization(model, lambda_l1=0.001):
    """
    Sparsity enforcement on model weights.
    """
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def regime_balance_loss(regime_probs):
    """
    Prevents the model from collapsing into a single regime.
    Enforces uniform distribution of regime assignments on average.
    
    Args:
        regime_probs: (batch, n_regimes) - Soft regime probabilities from Gumbel Softmax
    
    Returns:
        kl_div: scalar - KL divergence from uniform distribution
    """
    # Average regime probabilities over batch
    avg_probs = torch.mean(regime_probs, dim=0)  # (n_regimes,)
    
    # Target: uniform distribution
    uniform_target = torch.full_like(avg_probs, 1.0 / regime_probs.size(1))
    
    # KL Divergence: KL(Uniform || Observed) = sum(p * log(p/q))
    # We want to minimize this to match uniform distribution
    kl_div = torch.sum(uniform_target * torch.log(uniform_target / (avg_probs + 1e-8)))
    return kl_div

def composite_loss(y_pred, y_true, embeddings, regime_probs, model, 
                   delta=1.0, 
                   lambda_contr=0.05,      # ⬆️ Increased from 0.01 (stronger embedding separation)
                   lambda_l1=0.0001,       # ⬇️ Decreased from 0.001 (allow more coefficients)
                   lambda_orth=0.15,       # ⬆️ Increased from 0.05 (force regime separation)
                   lambda_balance=0.30):   # ⬆️ Increased from 0.15 (heavily penalize imbalance)
    """
    Full training objective combining all loss components.
    
    L_total = L_Huber + λ_contr * L_contrastive + λ_L1 * L_sparse + 
              λ_orth * L_orthogonality + λ_balance * L_balance
    
    Components:
    - L_Huber: Prediction error (robust to outliers)
    - L_contrastive: Equation 18 - Regime separation in embedding space
    - L_sparse: L1 regularization on forecasting coefficients
    - L_orthogonality: Forces regime-specific weights to be orthogonal
    - L_balance: Prevents regime collapse (KL divergence from uniform)
    
    Updated with aggressive parameters to prevent regime collapse and encourage bolder predictions.
    """
    # Huber Loss for robustness to outliers
    l_huber = F.huber_loss(y_pred.squeeze(), y_true.squeeze(), delta=delta)
    
    # Contrastive Loss (Equation 18)
    l_contr = contrastive_loss(embeddings, regime_probs)
    
    # L1 Regularization (Sparsity on forecasting coefficients)
    # Handle Ensembles
    if hasattr(model, 'models'):
        l_sparse = sum(torch.sum(torch.abs(m.forecasting.coeffs)) for m in model.models) / len(model.models)
    else:
        l_sparse = torch.sum(torch.abs(model.forecasting.coeffs))
    
    # Orthogonality Constraint
    l_orth = model.get_orthogonality_loss()
    
    # Balance Loss
    l_balance = regime_balance_loss(regime_probs)
    
    # ============================================================
    # DETECT REGIME COLLAPSE (Reviewer-driven Fix #3)
    # ============================================================
    # If any regime is ignored (max prob never points to it), heavily penalize
    regime_assignments = regime_probs.argmax(dim=1)
    unique_regimes = torch.unique(regime_assignments)
    
    collapse_penalty_multiplier = 1.0
    if len(unique_regimes) < model.n_regimes:
        # Regime collapse detected - HEAVILY penalize
        if model.training: # Only print during training to avoid terminal spam
            print(f"  ⚠️ WARNING: Regime collapse detected ({len(unique_regimes)}/{model.n_regimes} active). Applying 5x penalty.")
        collapse_penalty_multiplier = 5.0
    
    # Dynamic Entropy Check (additional safeguard)
    regime_entropy = -torch.sum(
        regime_probs.mean(dim=0) * torch.log(regime_probs.mean(dim=0) + 1e-8)
    )
    max_entropy = torch.log(torch.tensor(regime_probs.size(1), dtype=torch.float, device=regime_probs.device))
    
    entropy_penalty_multiplier = 1.0
    if regime_entropy / max_entropy < 0.5:
        entropy_penalty_multiplier = 2.0
    
    # Apply dynamic penalties
    effective_lambda_orth = lambda_orth * collapse_penalty_multiplier
    effective_lambda_balance = lambda_balance * collapse_penalty_multiplier * entropy_penalty_multiplier
        
    total_loss = l_huber + \
                 lambda_contr * l_contr + \
                 lambda_l1 * l_sparse + \
                 effective_lambda_orth * l_orth + \
                 effective_lambda_balance * l_balance
                 
    return total_loss, {
        'huber': l_huber.item(),
        'contrastive': l_contr.item(),
        'l1': l_sparse.item(),
        'orthogonality': l_orth.item(),
        'balance': l_balance.item(),
        'regime_entropy': regime_entropy.item()
    }
