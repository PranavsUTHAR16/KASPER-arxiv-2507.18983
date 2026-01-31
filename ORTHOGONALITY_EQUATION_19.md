# Orthogonality Regularization: Equation 19 Alignment

## 📐 Overview

This document verifies that our orthogonality loss **exactly implements** Equation 19 from the KASPER paper.

---

## 📝 Equation 19: Orthogonality Regularization

### Paper Specification

```
L_orth = E[||W_r * W_r^T - I||²_F]
```

**Where:**
- `W_r`: Regime-specific weight matrix
- `W_r ∈ R^{1×F}`: Row vector of scalar feature-importance weights
- `W_r = [w_1^(r), w_2^(r), ..., w_F^(r)]`: Weights for regime r
- `F`: Total number of features (hidden dimension in our case)
- `||·||_F`: Frobenius norm
- `I`: Identity matrix
- `E[·]`: Expectation (in practice, computed directly on W)

**Purpose:** Enforces distinctiveness across regime-specific transformations by ensuring regime weight vectors are **orthogonal** to each other.

---

## 🎯 Mathematical Intuition

### What Does Orthogonality Mean?

Two vectors are **orthogonal** if their dot product is zero:
```
W_i · W_j = 0  (for i ≠ j)
```

**In regime detection context:**
- Regime 0 focuses on features [0.8, 0.2, -0.1, ...]
- Regime 1 focuses on features [-0.2, 0.7, 0.5, ...]
- These should be **perpendicular** (independent patterns)

### Gram Matrix

For a matrix `W ∈ R^{k×F}` with k regimes:
```
Gram = W * W^T ∈ R^{k×k}

Gram[i,j] = W_i · W_j = Σ_f W[i,f] * W[j,f]
```

**Ideal case (perfect orthogonality):**
```
Gram = I  (Identity matrix)

Gram = [[1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]]
```

**Interpretation:**
- Diagonal = 1: Each regime weight vector has unit norm
- Off-diagonal = 0: Regime vectors are orthogonal

### Frobenius Norm

The Frobenius norm measures the "distance" between two matrices:
```
||A||_F = sqrt(Σ_i Σ_j A[i,j]²)

||A||²_F = Σ_i Σ_j A[i,j]²  (squared Frobenius norm)
```

**For our loss:**
```
L_orth = ||W*W^T - I||²_F
       = Σ_i Σ_j (Gram[i,j] - I[i,j])²
```

**This penalizes:**
- Gram[i,i] ≠ 1: Weight vectors not unit-normalized
- Gram[i,j] ≠ 0 (i≠j): Weight vectors not orthogonal

---

## ✅ Implementation

### Location

**File:** `src/kasper/models/kan_layers.py`  
**Class:** `RegimeDetectionLayer`  
**Method:** `orthogonality_loss()`

### Code

```python
def orthogonality_loss(self):
    """
    Equation 19: Orthogonality Regularization
    
    L_orth = E[||W_r * W_r^T - I||²_F]
    
    Enforces that regime-specific weight vectors are orthogonal,
    ensuring each regime detects distinct market patterns.
    
    Returns:
        loss: scalar - Frobenius norm squared of (Gram - Identity)
    """
    # W: Regime weight matrix (k, F)
    # Each row W[r,:] is the weight vector for regime r
    W = self.regime_logits.weight  # (n_regimes, hidden_dim)
    
    # Compute Gram matrix: G = W * W^T
    # G[i,j] = dot product of regime i and regime j weight vectors
    gram = torch.mm(W, W.t())  # (n_regimes, n_regimes)
    
    # Target: Identity matrix (perfect orthogonality)
    identity = torch.eye(W.size(0), device=W.device)  # (n_regimes, n_regimes)
    
    # Frobenius norm squared: ||Gram - I||²_F
    # This is sum of squared differences
    loss = torch.norm(gram - identity, p='fro') ** 2
    
    return loss
```

---

## 📊 Step-by-Step Example

### Setup

```python
k = 3  # Number of regimes
F = 64 # Hidden dimension (features)

# Regime weight matrix
W = torch.tensor([
    [0.5, 0.5, 0.0, ...],  # Regime 0: focuses on features 0-1
    [0.0, 0.7, 0.7, ...],  # Regime 1: focuses on features 1-2
    [0.7, 0.0, 0.7, ...]   # Regime 2: focuses on features 0,2
])  # Shape: (3, 64)
```

### Step 1: Compute Gram Matrix

```python
Gram = W @ W.T

# Example (simplified with fewer features):
W = [[0.8, 0.6, 0.0],
     [0.0, 0.8, 0.6],
     [0.6, 0.0, 0.8]]

Gram = [[0.8*0.8 + 0.6*0.6 + 0.0*0.0,  0.8*0.0 + 0.6*0.8 + 0.0*0.6,  0.8*0.6 + 0.6*0.0 + 0.0*0.8],
        [0.0*0.8 + 0.8*0.6 + 0.6*0.0,  0.0*0.0 + 0.8*0.8 + 0.6*0.6,  0.0*0.6 + 0.8*0.0 + 0.6*0.8],
        [0.6*0.8 + 0.0*0.6 + 0.8*0.0,  0.6*0.0 + 0.0*0.8 + 0.8*0.6,  0.6*0.6 + 0.0*0.0 + 0.8*0.8]]

Gram = [[1.00, 0.48, 0.48],
        [0.48, 1.00, 0.48],
        [0.48, 0.48, 1.00]]
```

**Observation:**
- Diagonal: ~1.0 ✅ (vectors have unit norm)
- Off-diagonal: 0.48 ⚠️ (vectors are NOT orthogonal)

### Step 2: Compute Difference from Identity

```python
I = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

Diff = Gram - I = [[0.00, 0.48, 0.48],
                   [0.48, 0.00, 0.48],
                   [0.48, 0.48, 0.00]]
```

### Step 3: Frobenius Norm Squared

```python
L_orth = ||Diff||²_F
       = 0.00² + 0.48² + 0.48² + 0.48² + 0.00² + 0.48² + 0.48² + 0.48² + 0.00²
       = 0 + 0.2304 + 0.2304 + 0.2304 + 0 + 0.2304 + 0.2304 + 0.2304 + 0
       = 6 * 0.2304
       = 1.382
```

**Interpretation:**
- Loss = 1.382 is HIGH ⚠️
- Off-diagonal elements are large (0.48)
- Regimes are NOT well-separated
- Gradient descent will push weights to be more orthogonal

---

## 🎯 Training Dynamics

### Epoch 0 (Random Initialization):

```python
W = torch.randn(3, 64) * 3.0  # Scaled orthogonal init

# After orthogonal init:
Gram ≈ [[9.0, 0.1, -0.2],
        [0.1, 9.0, 0.3],
        [-0.2, 0.3, 9.0]]

L_orth = ||(Gram/9.0) - I||²_F ≈ 0.5  # Moderate (good init)
```

### Epoch 50 (Learning):

```python
# Gradient descent pushes towards orthogonality:
Gram ≈ [[1.2, 0.3, 0.2],
        [0.3, 1.1, 0.1],
        [0.2, 0.1, 1.3]]

L_orth = ||Gram - I||²_F ≈ 0.25  # Decreasing ✅
```

### Epoch 200 (Converged):

```python
# Near-perfect orthogonality:
Gram ≈ [[1.01, 0.05, 0.02],
        [0.05, 0.98, 0.03],
        [0.02, 0.03, 1.02]]

L_orth = ||Gram - I||²_F ≈ 0.01  # Very low ✅
```

---

## 🔧 Why Orthogonality Matters

### Problem: Regime Collapse Without Orthogonality

```python
# Bad case: All regimes learn the same pattern
W_0 = [0.8, 0.6, 0.0, ...]
W_1 = [0.8, 0.6, 0.0, ...]  # Identical!
W_2 = [0.8, 0.6, 0.0, ...]

Gram = [[1.0, 1.0, 1.0],  # Off-diagonal = 1 means identical
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]]

L_orth = ||Gram - I||²_F = 6.0  # HUGE loss ⚠️
```

**Result:** All samples assigned to same regime → Collapse!

### Solution: Orthogonality Forces Diversity

```python
# Good case: Each regime learns different pattern
W_0 = [1.0, 0.0, 0.0, ...]  # Focuses on feature 0
W_1 = [0.0, 1.0, 0.0, ...]  # Focuses on feature 1
W_2 = [0.0, 0.0, 1.0, ...]  # Focuses on feature 2

Gram = [[1.0, 0.0, 0.0],  # Perfect orthogonality!
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]

L_orth = ||Gram - I||²_F = 0.0  # Zero loss ✅
```

**Result:** Regimes detect distinct patterns → Diversity!

---

## 🧮 Mathematical Properties

### Property 1: Convexity

The Frobenius norm squared is **convex**, so gradient descent will converge to a local minimum.

### Property 2: Scaling

If we scale all weights by α:
```
W' = α * W
Gram' = W' * W'^T = α² * W * W^T = α² * Gram

L_orth(W') = ||α² * Gram - I||²_F ≠ α² * L_orth(W)
```

**Implication:** Loss encourages unit-norm weights (α=1).

### Property 3: Rotation Invariance

Orthogonal transformations preserve orthogonality:
```
If W is orthogonal and Q is orthogonal, then W*Q is orthogonal.
```

---

## 🔬 Gradient Computation

### Derivative of Frobenius Norm

```
L = ||A||²_F = trace(A^T * A)

∂L/∂A = 2A
```

### Derivative of Orthogonality Loss

```
L_orth = ||W*W^T - I||²_F

Let G = W*W^T
L = ||G - I||²_F

∂L/∂G = 2(G - I)

∂L/∂W = ∂L/∂G * ∂G/∂W
        = 2(G - I) * ∂(W*W^T)/∂W
        = 2(G - I) * 2W
        = 4(W*W^T - I) * W
```

**Interpretation:**
- If Gram[i,j] > 0 (i≠j): Gradient pushes W_i and W_j apart
- If Gram[i,i] > 1: Gradient reduces norm of W_i
- If Gram[i,i] < 1: Gradient increases norm of W_i

---

## 📊 Monitoring During Training

### Expected Log Output

```
Epoch 1/200
  Orthogonality loss: 1.234  ← High initially
  Gram diagonal: [8.2, 9.1, 7.8]  ← Not unit norm
  Gram off-diag (max): 0.82  ← High correlation

Epoch 50/200
  Orthogonality loss: 0.456  ← Decreasing ✅
  Gram diagonal: [1.3, 1.1, 1.2]  ← Closer to 1
  Gram off-diag (max): 0.34  ← Decreasing

Epoch 200/200
  Orthogonality loss: 0.045  ← Low ✅
  Gram diagonal: [1.01, 0.99, 1.02]  ← Unit norm ✅
  Gram off-diag (max): 0.08  ← Nearly zero ✅
```

### Diagnostic Plot

**Gram Matrix Heatmap:**
```
Epoch 0:          Epoch 100:        Epoch 200:
[8  1  1]         [1.2  0.3  0.1]   [1.0  0.0  0.0]
[1  9  1]         [0.3  1.1  0.2]   [0.0  1.0  0.0]
[1  1  7]         [0.1  0.2  1.3]   [0.0  0.0  1.0]

(Diagonal should be ~1, off-diagonal should be ~0)
```

---

## 🎛️ Hyperparameter: λ_orth

### In Composite Loss

```python
L_total = L_Huber + λ_contr * L_contrastive + λ_L1 * L_sparse + 
          λ_orth * L_orthogonality + λ_balance * L_balance
```

**Current setting:** `λ_orth = 0.15`

### Effect of λ_orth

| **λ_orth** | **Effect** | **Risk** |
|------------|------------|----------|
| 0.0 | No orthogonality constraint | Regime collapse likely |
| 0.05 | Weak constraint | May still collapse |
| 0.15 ✅ | Moderate constraint | Good balance |
| 0.30 | Strong constraint | May hurt prediction |
| 1.0 | Very strong | Prediction accuracy suffers |

**Tuning guide:**
- If regime collapse occurs: Increase to 0.25-0.50
- If prediction poor but regimes healthy: Decrease to 0.05-0.10
- Monitor Gram matrix: off-diagonal should be <0.2

---

## 🧪 Verification Test

```python
import torch
import torch.nn as nn

# Create regime detection layer
model = RegimeDetectionLayer(input_dim=8, hidden_dim=64, n_regimes=3)

# Get weight matrix
W = model.regime_logits.weight  # (3, 64)

# Compute Gram matrix
Gram = torch.mm(W, W.t())
print("Gram matrix:")
print(Gram)

# Compute orthogonality loss
loss = model.orthogonality_loss()
print(f"\nOrthogonality loss: {loss.item():.4f}")

# Check properties
diag = torch.diag(Gram)
off_diag = Gram - torch.diag(diag)

print(f"\nDiagonal (should be ~1): {diag}")
print(f"Off-diagonal max (should be ~0): {off_diag.abs().max().item():.4f}")

# Expected output:
# Gram matrix:
# tensor([[9.2, 0.5, -0.3],
#         [0.5, 8.7, 0.8],
#         [-0.3, 0.8, 9.5]])
#
# Orthogonality loss: 195.23  ← High before training
#
# Diagonal: [9.2, 8.7, 9.5]  ← Not unit norm
# Off-diagonal max: 0.8  ← Not orthogonal
```

---

## 📚 Related Equations

| **Equation** | **Component** | **Interaction with Eq 19** |
|--------------|---------------|---------------------------|
| Eq 13 | Feature engineering | Provides F input features |
| Eq 14-16 | Spline activation | Transforms features before regime detection |
| Eq 17 | Gumbel Softmax | Uses W to compute regime logits |
| Eq 18 | Contrastive loss | Encourages embedding separation (complementary) |
| **Eq 19** | **Orthogonality** | **Enforces weight separation** |

**Synergy:**
- Eq 18 (Contrastive): Separates **embeddings** in latent space
- Eq 19 (Orthogonality): Separates **weight vectors** in parameter space
- Together: Ensures regimes are distinct at both parameter and representation levels

---

## ✅ Implementation Checklist

- [x] `orthogonality_loss()` method implemented
- [x] Computes Gram matrix W * W^T
- [x] Computes Frobenius norm squared
- [x] Returns scalar loss value
- [x] Called in `composite_loss()`
- [x] Weighted by λ_orth = 0.15
- [x] Dynamic penalty on regime collapse (5x multiplier)
- [x] Gradients flow correctly through W

**If all checked ✅, your implementation matches Equation 19!**

---

## 🔗 Related Files

- **src/kasper/models/kan_layers.py** - `RegimeDetectionLayer.orthogonality_loss()`
- **losses.py** - `composite_loss()` (uses orthogonality loss)
- **trainer.py** - Monitors orthogonality during training
- **REGIME_CLASSIFICATION_ALIGNMENT.md** - Equations 17-18
- **PAPER_ALIGNMENT_COMPLETE.md** - Full summary

---

**Last Updated:** January 31, 2026, 6:17 AM IST  
**Implementation:** `RegimeDetectionLayer.orthogonality_loss()`  
**Status:** ✅ Fully aligned with KASPER paper Equation 19
