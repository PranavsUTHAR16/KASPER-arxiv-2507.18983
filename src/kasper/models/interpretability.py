import torch
import numpy as np

def compute_monte_carlo_shapley_batched(model, sample, regime_idx, n_iterations=100):
    """
    Approximates Shapley values for features using batched Monte Carlo.
    sample: (1, n_features)
    """
    model.eval()
    n_features = sample.shape[1]
    shapley_values = np.zeros(n_features)
    
    batch_size = n_iterations
    
    with torch.no_grad():
        for i in range(n_features):
            # For each iteration, a random permutation and a position of feature i
            # We can vectorize this but let's stick to simple batching for now
            
            # Construct a batch of samples with and without feature i
            batch_with = torch.zeros(batch_size, n_features, device=sample.device)
            batch_without = torch.zeros(batch_size, n_features, device=sample.device)
            
            for b in range(batch_size):
                perm = np.random.permutation(n_features)
                pos = np.where(perm == i)[0][0]
                S = perm[:pos]
                
                batch_with[b, S] = sample[0, S]
                batch_with[b, i] = sample[0, i]
                batch_without[b, S] = sample[0, S]
                
            # Get predictions for the whole batch
            if hasattr(model, 'models'):
                # For ensembles, we use the forward pass directly
                _, regime_probs, embeddings = model(batch_with)
                phi_with = model.models[0].forecasting.b_spline_basis(embeddings.unsqueeze(-1), model.models[0].forecasting.spline_order)
                masked_coeff = model.models[0].forecasting.coeffs[regime_idx] * model.models[0].forecasting.sparsity_mask[regime_idx]
                preds_with = torch.sum(masked_coeff * phi_with, dim=(-1, -2))
                
                _, _, embeddings_without = model(batch_without)
                phi_without = model.models[0].forecasting.b_spline_basis(embeddings_without.unsqueeze(-1), model.models[0].forecasting.spline_order)
                preds_without = torch.sum(masked_coeff * phi_without, dim=(-1, -2))
            else:
                _, embeddings_with = model.regime_detection(batch_with)
                _, embeddings_without = model.regime_detection(batch_without)
                
                # Get regime predictions
                phi_with = model.forecasting.b_spline_basis(embeddings_with.unsqueeze(-1), model.forecasting.spline_order)
                masked_coeff = model.forecasting.coeffs[regime_idx] * model.forecasting.sparsity_mask[regime_idx]
                preds_with = torch.sum(masked_coeff * phi_with, dim=(-1, -2))
                
                phi_without = model.forecasting.b_spline_basis(embeddings_without.unsqueeze(-1), model.forecasting.spline_order)
                preds_without = torch.sum(masked_coeff * phi_without, dim=(-1, -2))
            
            # Mean difference
            shapley_values[i] = torch.mean(preds_with - preds_without).item()
            
    return shapley_values

def extract_regime_rules(model, X_test, feature_names, n_samples=50, gamma=0.9):
    """
    Extracts rules by calculating Shapley values for each regime.
    Implements Equation 24-28 including Temporal Weighting (Eq 26).
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        # Use direct call instead of .regime_detection to support ensembles
        _, regime_probs, _ = model(X_tensor)
        regime_assignments = torch.argmax(regime_probs, dim=1)
        
    n_regimes = model.n_regimes
    
    # Store per-sample Shapley values for each regime
    # shapley_values_per_sample[r_idx][t] will be a vector of shapley values for sample t in regime r_idx
    shapley_values_per_sample = {r: [] for r in range(n_regimes)}
    sample_indices_per_regime = {r: [] for r in range(n_regimes)}

    # Calculate Shapley values for each sample assigned to a regime
    for i in range(len(X_tensor)):
        r_idx = regime_assignments[i].item()
        sample = X_tensor[i:i+1] # (1, n_features)
        
        # Compute Shapley for this single sample and its assigned regime
        sample_shap_vals = compute_monte_carlo_shapley_batched(model, sample, r_idx, n_iterations=10) 
        shapley_values_per_sample[r_idx].append(sample_shap_vals)
        sample_indices_per_regime[r_idx].append(i) # Keep track of original indices for temporal weighting

    regime_rules = {}
    for r_idx in range(n_regimes):
        samples_in_regime = shapley_values_per_sample[r_idx]
        original_indices = sample_indices_per_regime[r_idx]

        if samples_in_regime:
            # Aggregate across selected points with Temporal Weighting (Eq 26)
            # Since we use test set samples, they are ordered daily.
            # weights: wt = gamma^(T-t) / sum(gamma^(T-t))
            
            # Sort samples by their original time index to apply temporal weighting correctly
            sorted_samples_info = sorted(zip(original_indices, samples_in_regime), key=lambda x: x[0])
            sorted_shapley_vals = [info[1] for info in sorted_samples_info]

            T_regime = len(sorted_shapley_vals)
            
            # If only one sample, no temporal weighting needed, just use its shapley
            if T_regime == 1:
                aggregated_shapley = sorted_shapley_vals[0]
            else:
                weights = np.array([gamma**(T_regime - t - 1) for t in range(T_regime)])
                weights /= weights.sum()
                
                aggregated_shapley = np.zeros(len(feature_names))
                for t in range(T_regime):
                    aggregated_shapley += weights[t] * sorted_shapley_vals[t]
            
            # Select top 3 features (Eq 27)
            top_indices = np.argsort(np.abs(aggregated_shapley))[-3:][::-1]
            rules = [feature_names[i] for i in top_indices]
            regime_rules[r_idx] = rules
        else:
            regime_rules[r_idx] = ["No samples assigned to this regime"]
            
    return regime_rules
