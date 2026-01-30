# KASPER: Kernel-based Adaptive Spline-activated Probabilistic Extreme Regime

[![ArXiv](https://img.shields.io/badge/ArXiv-2507.18983-B31B1B.svg)](https://arxiv.org/abs/2507.18983)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![OpenReview](https://img.shields.io/badge/OpenReview-PD4jGJQtL8-blue)](https://openreview.net/forum?id=PD4jGJQtL8)

KASPER is a high-fidelity financial forecasting framework that leverages **Kolmogorov-Arnold Networks (KANs)** for dual-stage regime detection and adaptive forecasting. This implementation strictly adheres to the mathematical formulations and architectural specifications outlined in the research paper.

## 🚀 Key Features

- **Regime-Specific KANs**: Implements dual KAN layers (Regime Detection & Adaptive Forecasting)
- **Localized Spline Activations**: Piecewise linear and cubic hybrid activations for high-precision non-linear mapping
- **Cox-de Boor B-Splines**: Full recursive basis function implementation for forecasting
- **Quantile-Initialized Knots**: Automatically adapts knots to the distribution of latent regime embeddings
- **Temporally-Weighted Shapley Values**: Interpretable rule extraction that prioritizes recent market dynamics
- **Variance-Stabilized Preprocessing**: Log-transformed OHL returns and one-hot temporal encoding

## 📐 Architecture Overview

The model follows the core KASPER workflow:
1. **Regime Detection**: A KAN layer identifying market states (e.g., Crash, Bull, Neutral) via Gumbel-Softmax classification and Orthogonality Regularization
2. **Adaptive Forecasting**: A regime-specific KAN layer that generates return forecasts tailored to the current market state
3. **Interpretability**: Monte Carlo Shapley estimation weighted by temporal decay (γ) to extract the top-3 influential features per regime

## 📦 Installation

```bash
git clone https://github.com/PranavsUTHAR16/KASPER_arxiv_2507.18983.git
cd KASPER_arxiv_2507.18983
pip install -r requirements.txt
```

## 📊 Quick Start

### Basic Training

Run the full pipeline on the Dow Jones (US30) benchmark:

```bash
python main.py
```

### Baseline Comparisons (Section 4.7)

Compare KASPER against PatchTST and DLinear baselines:

```bash
python compare_models.py
```

**Expected Results** (from peer review):
- **PatchTST**: R²=0.10, Sharpe=2.24
- **DLinear**: R²=0.03, Sharpe=3.40  
- **KASPER**: R²=0.89, Sharpe=12.02 ✨

### Ablation Studies (Section 4.6)

Run ablation experiments to verify component importance:

```bash
python ablation_studies.py
```

**Validates Critical Findings**:
- ❌ Removing orthogonality regularization → **Regime collapse**
- ❌ Removing contrastive loss → **Regime entanglement**  
- ✅ Full KASPER → Stable, well-separated regimes

### Interpretability Analysis

Extract regime-specific feature importance:

```bash
python interpretability.py
```

## 🔬 Implementation Details

### Project Structure

```
KASPER_arxiv_2507.18983/
├── kasper_model.py          # Core KASPER architecture
├── kan_layers.py            # KAN layer implementations
├── losses.py                # Loss functions (MSE, orthogonality, contrastive)
├── trainer.py               # Training loop with early stopping
├── data_utils.py            # Data loading & feature engineering
├── metrics.py               # Evaluation metrics (R², Sharpe, etc.)
├── baselines.py             # PatchTST & DLinear implementations (Section 4.7)
├── compare_models.py        # Baseline comparison script (Table 4)
├── ablation_studies.py      # Ablation experiments (Section 4.6, Fig. 8)
├── interpretability.py      # Shapley value analysis
├── diagnostic.py            # Regime distribution diagnostics
├── walk_forward.py          # Walk-forward validation
└── main.py                  # Main entry point
```

### Key Components

#### 1. **Regime Detection KAN** (`kasper_model.py`)
- Gumbel-Softmax sampling for differentiable regime assignment
- Orthogonality regularization (λ_orth = 0.01) prevents regime collapse
- Contrastive loss (λ_contrast = 0.1) ensures regime separation

#### 2. **Forecasting KAN** (`kan_layers.py`)
- Cox-de Boor recursive B-spline basis functions
- Regime-specific weights learned via gradient descent
- Adaptive knot placement using quantile-based initialization

#### 3. **Loss Functions** (`losses.py`)
```python
L_total = L_MSE + λ_orth * L_orth + λ_contrast * L_contrast
```
- **L_MSE**: Mean squared error on return predictions
- **L_orth**: Orthogonality regularization (Frobenius norm)
- **L_contrast**: Contrastive regime loss (pushes positive/negative returns apart)

## 📑 Peer Review Updates (Jan 30, 2026)

### Authors' Response to Reviewer iu4b

**Status**: Paper under review at [TMLR (OpenReview: PD4jGJQtL8)](https://openreview.net/forum?id=PD4jGJQtL8)

The authors have addressed all reviewer concerns and revised the manuscript accordingly:

#### ✅ **1. Expanded Related Work** (Section 2.3, pages 5-6)
- Added dedicated subsection on **Explainability and Interpretability in Finance**
- Better positioned KASPER's contributions relative to existing methods
- Clarified how KASPER differs from earlier interpretability-focused approaches

#### ✅ **2. Added Baseline Comparisons** (Section 4.7, Table 4, pages 17-18)

**New Baselines Evaluated**:
- **PatchTST** (Nie et al., 2023): Patch-based Transformer
  - Results: R²=0.10, Sharpe=2.24
  - Limitation: Regime-agnostic (cannot model regime-dependent relationships)
  
- **DLinear** (Zeng et al., 2023): Decomposition Linear Model  
  - Results: R²=0.03, Sharpe=3.40
  - Limitation: Regime-agnostic (no regime detection capability)

**KASPER Performance**:
- R²=0.89, Sharpe=12.02
- **8.9x better R² than PatchTST**
- Regime detection + regime-specific transformations yield higher explanatory power

**Implementation**: See [`baselines.py`](baselines.py) and [`compare_models.py`](compare_models.py)

#### ✅ **3. Added Ablation Studies** (Section 4.6, Fig. 8, pages 16-17)

**Experiments**:
1. **Full KASPER** (all components enabled)
   - ✅ Stable regime assignments
   - ✅ High confidence (avg > 0.8)
   - ✅ Clear market state separation

2. **Ablation A: Remove Orthogonality Regularization**
   - ❌ **Regime Collapse**: 95%+ samples assigned to single regime
   - ❌ Regime-specific weights converge to near-identical representations
   - ❌ Performance degradation (regime diversity lost)

3. **Ablation B: Remove Contrastive Loss**
   - ❌ **Regime Entanglement**: Weak separation between market states
   - ❌ Low confidence assignments (avg < 0.6)
   - ❌ Bullish/bearish proportions nearly identical across regimes

**Key Insight**: Both regularization components are **critical** for stable regime learning

**Implementation**: See [`ablation_studies.py`](ablation_studies.py)

#### ✅ **4. Enhanced Methodology Explanation** (Kasper Framework)
- Revised Section 3 with clearer component descriptions
- Added motivation behind design choices
- Clarified module interactions within the overall framework
- Fixed equation definitions (e.g., Eq. 13 now has explicit expressions)

#### ✅ **5. Clarified Knot Initialization**

**Question**: "What is percentile-based initialization?"

**Answer**: Knots are scaled using robust percentile bounds (1st and 99th) of each feature's training distribution:

```python
k_i = p_low + (i / N_linear) * (p_high - p_low)
```

where `p_low = 1st percentile`, `p_high = 99th percentile`

This prevents outliers from distorting the spline grid.

### 🔄 Follow Progress
- ⭐ **Star this repo** for implementation updates
- 📖 Monitor the [OpenReview page](https://openreview.net/forum?id=PD4jGJQtL8) for public comments  
- 📄 Check [arXiv](https://arxiv.org/abs/2507.18983) for revised version (v2)

**Last Updated**: January 30, 2026

## 🎯 Reproducibility

### Hardware Requirements
- CPU: Any modern multi-core processor
- RAM: 8GB minimum
- GPU: Optional (speeds up training 3-5x)

### Expected Runtime
- Training KASPER: ~10-15 minutes (CPU), ~3-5 minutes (GPU)
- Baseline training: ~5-8 minutes each
- Ablation studies: ~30-40 minutes total
- Full comparison suite: ~1 hour

### Random Seeds
All experiments use fixed random seeds for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
```

## 📈 Results

### Dow Jones (US30) Benchmark

| Metric | KASPER | PatchTST | DLinear |
|--------|---------|----------|----------|
| **R² Score** | **0.89** | 0.10 | 0.03 |
| **Sharpe Ratio** | **12.02** | 2.24 | 3.40 |
| **Win Rate (%)** | **68.4** | 52.1 | 54.3 |
| **Max Drawdown** | **-8.2%** | -18.5% | -16.9% |
| **Annualized Return** | **42.1%** | 14.3% | 18.2% |

### Regime Analysis

**Market States Identified**:
- **Regime 0** (Crash): High volatility, negative returns, elevated VIX
- **Regime 1** (Bull): Positive momentum, low volatility, strong trends  
- **Regime 2** (Neutral): Range-bound, mixed signals, mean reversion

**Regime-Specific Performance**:
- Crash regime: 72% correct directional predictions
- Bull regime: 81% correct directional predictions
- Neutral regime: 59% correct directional predictions

## 🧪 Advanced Usage

### Custom Data

```python
from data_utils import engineer_features, prepare_data
import pandas as pd

# Load your own OHLCV data
df = pd.read_csv('your_data.csv')  # Requires: Date, Open, High, Low, Close, Volume
X, y = engineer_features(df)
bundles = prepare_data(X, y)

# Train KASPER
from kasper_model import KASPEREnsemble
model = KASPEREnsemble(input_dim=X.shape[1])
model.fit_knots(torch.FloatTensor(bundles['X_train']))
# ... continue training
```

### Hyperparameter Tuning

```python
model = KASPEREnsemble(
    input_dim=15,
    hidden_dim=128,      # Embedding dimension
    n_regimes=3,         # Number of market states
    n_models=3,          # Ensemble size
    n_basis=20,          # Number of B-spline basis functions
    lambda_orth=0.01,    # Orthogonality regularization
    lambda_contrast=0.1  # Contrastive loss weight
)
```

### Walk-Forward Validation

```python
python walk_forward.py
```

Performs rolling window validation with monthly retraining.

## 📊 Visualizations

The implementation generates several diagnostic plots:

1. **`comparison_results.png`**: Model performance comparison with equity curves
2. **`ablation_results.png`**: Fig. 8 from paper (ablation study results)
3. **`regime_diagnostic.png`**: Regime distribution across train/val/test
4. **`performance_plot.png`**: KASPER predictions vs. ground truth

## 🔍 Known Limitations

1. **Data Requirements**: Needs at least 1000 samples for stable regime learning
2. **Regime Collapse Risk**: Can occur with insufficient regularization (see ablation studies)
3. **Computational Cost**: KAN layers are slower than standard MLPs (~2-3x)
4. **Market Regime Assumptions**: Assumes finite number of discrete market states

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- [ ] Multi-asset portfolio optimization
- [ ] Online learning / streaming updates
- [ ] Additional baseline comparisons (LSTM, Temporal Fusion Transformer)
- [ ] Hyperparameter sensitivity analysis
- [ ] Extended interpretability visualizations

## 📄 Citation

If you use this code or find the paper helpful, please cite:

```bibtex
@article{suthar2025kasper,
  title={KASPER: Kernel-based Adaptive Spline-activated Probabilistic Extreme Regime for Financial Forecasting},
  author={Suthar, Pranav and others},
  journal={arXiv preprint arXiv:2507.18983},
  year={2025},
  note={Under review at TMLR (OpenReview: PD4jGJQtL8)}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Reviewers**: Special thanks to Reviewer iu4b and others for constructive feedback
- **TMLR**: For providing an open, transparent peer review process
- **KAN Framework**: Liu et al. (2024) for the Kolmogorov-Arnold Network foundation
- **Baselines**: Nie et al. (2023) for PatchTST, Zeng et al. (2023) for DLinear

---

**Maintained by**: [@PranavsUTHAR16](https://github.com/PranavsUTHAR16)  
**Last Updated**: January 30, 2026  
**Paper Status**: Under review at TMLR ([OpenReview](https://openreview.net/forum?id=PD4jGJQtL8))
