import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(embeddings, regime_probs, margin=1.0):
    """
    Enforces separation between different regimes in the latent space.
    Uses soft regime probabilities to weight the distances.
    """
    # pairwise distances between embeddings
    # embeddings: (batch, hidden_dim)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # pairwise similarity based on regime probs
    # regime_probs: (batch, n_regimes)
    # similarity[i, j] = sum_k p_{ik} * p_{jk}
    similarity = torch.matmul(regime_probs, regime_probs.t())
    
    # Loss: pull similar ones together, push dissimilar ones apart
    # L = similarity * dist^2 + (1 - similarity) * max(0, margin - dist)^2
    loss = similarity * torch.pow(dist_matrix, 2) + \
           (1 - similarity) * torch.pow(F.relu(margin - dist_matrix), 2)
           
    return loss.mean()

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
    """
    avg_probs = torch.mean(regime_probs, dim=0)
    uniform_target = torch.full_like(avg_probs, 1.0 / regime_probs.size(1))
    
    # KL Divergence from uniform: sum(target * log(target/avg))
    # We want to minimize this
    kl_div = torch.sum(uniform_target * torch.log(uniform_target / (avg_probs + 1e-8)))
    return kl_div

def composite_loss(y_pred, y_true, embeddings, regime_probs, model, 
                   delta=1.0, 
                   lambda_contr=0.01, lambda_l1=0.001, 
                   lambda_orth=0.01, lambda_balance=0.05):
    """
    Full training objective: L = Huber + L1 + Contrastive + Orthogonal + Balance
    Based on paper Table I weights.
    """
    # Huber Loss for robustness
    l_huber = F.huber_loss(y_pred.squeeze(), y_true.squeeze(), delta=delta)
    
    # Contrastive Loss
    l_contr = contrastive_loss(embeddings, regime_probs)
    
    # L1 Regularization (Sparsity on forecasting coefficients)
    l_sparse = torch.sum(torch.abs(model.forecasting.coeffs))
    
    # Orthogonality Constraint
    l_orth = model.get_orthogonality_loss()
    
    # Balance Loss (KL Divergence from uniform)
    l_balance = regime_balance_loss(regime_probs)
    
    total_loss = l_huber + \
                 lambda_contr * l_contr + \
                 lambda_l1 * l_sparse + \
                 lambda_orth * l_orth + \
                 lambda_balance * l_balance
                 
    return total_loss, {
        'huber': l_huber.item(),
        'contrastive': l_contr.item(),
        'l1': l_sparse.item(),
        'orthogonality': l_orth.item(),
        'balance': l_balance.item()
    }
