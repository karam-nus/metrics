---
title: "Correlation (Pearson & Spearman)"
---

# Correlation (Pearson & Spearman)

## Overview

Correlation metrics measure the statistical relationship between two tensors. **Pearson correlation** captures the strength of the *linear* relationship — whether corresponding elements co-vary proportionally. **Spearman correlation** captures the strength of the *monotonic* relationship — whether the rank ordering of elements is preserved, regardless of the functional form. In model debugging, Pearson correlation checks if a transformation preserved linear relationships in activations, while Spearman correlation checks if relative ordering (which element is largest, second-largest, etc.) is maintained. Spearman is particularly valuable for attention scores and activation rankings where ordinal structure matters more than exact values.

## Formula

**Pearson Correlation Coefficient:**

$$
r(\mathbf{x}, \mathbf{y}) = \frac{\text{Cov}(\mathbf{x}, \mathbf{y})}{\sigma_x \sigma_y}
= \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

**Spearman Rank Correlation:**

$$
\rho = r(\text{rank}(\mathbf{x}), \text{rank}(\mathbf{y}))
$$

where $\text{rank}(\mathbf{x})$ replaces each value with its ordinal rank. For no tied ranks:

$$
\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}, \quad d_i = \text{rank}(x_i) - \text{rank}(y_i)
$$

**Relationship to Cosine Similarity:** Pearson correlation is cosine similarity applied to mean-centered data:

$$
r(\mathbf{x}, \mathbf{y}) = \cos\_sim(\mathbf{x} - \bar{x}, \mathbf{y} - \bar{y})
$$

## Visual Diagram

```
Pearson r = 1.0            Pearson r = 0.95          Pearson r = 0.0
(perfect linear)           (strong linear)           (no linear relationship)

 y │    •••               y │   •  •               y │  •    •
   │   •••                  │  •• •  •                │    •  •  •
   │  •••                   │  • •••                  │  •  •  •
   │ •••                    │ •• •• •                  │   •• •
   │•••                     │ • •  •                   │  •  •  •
   └──────── x              └──────── x               └──────── x

Spearman ρ = 1.0           Spearman ρ = 1.0          Spearman ρ = 0.0
(perfect monotonic)        (still perfect monotonic!) (no monotonic relationship)

 y │        ••             y │       •••              y │  •    •
   │      ••                 │    •••                   │    •  •  •
   │    ••                   │  •••                     │  •  •  •
   │  ••                     │ ••                       │   •• •
   │••                       │•                         │  •  •  •
   └──────── x               └──────── x               └──────── x
   (linear)                  (monotonic, non-linear)
```

<!-- IMAGE: Scatter plots showing perfect Pearson, high Pearson, zero Pearson, and monotonic-but-nonlinear cases -->

## Range & Interpretation

| Value         | Pearson Interpretation                     | Spearman Interpretation                      |
|---------------|-------------------------------------------|----------------------------------------------|
| 1.0           | Perfect positive linear relationship       | Perfect positive monotonic relationship       |
| 0.99 – 1.0   | Excellent linear preservation              | Excellent rank preservation                   |
| 0.95 – 0.99  | Strong — minor deviations                  | Strong — few rank swaps                       |
| 0.80 – 0.95  | Moderate — noticeable degradation          | Moderate — some rank reordering              |
| 0.0           | No linear relationship                     | No monotonic relationship                     |
| −1.0          | Perfect negative linear relationship       | Perfect rank inversion                        |

**When Pearson ≠ Spearman:** A large gap indicates a non-linear transformation. Pearson ≈ 1.0 but Spearman < 0.95 means the linear relationship holds globally but local rank orderings are scrambled (common with aggressive quantization of similar-valued elements).

## When to Use

- **Rank preservation checking:** After quantization, do the largest activations remain the largest? Spearman answers this directly — critical for attention mechanisms and top-k selection.
- **Linear relationship verification:** After a transformation, is the relationship between old and new values linear? Pearson answers this — relevant for linear layers and scaling operations.
- **Centered comparison:** Pearson is mean-invariant, making it appropriate when comparing tensors that may have different means (e.g., before/after batch normalization).
- **Activation pattern analysis:** Whether the "shape" of an activation pattern (which neurons fire strongly) is preserved.
- **Cross-layer feature tracking:** Checking if feature importance rankings are preserved after distillation.

## When NOT to Use

- **Absolute error needed:** Correlation says nothing about error magnitude. `y = 1000x + 500` has Pearson = 1.0 with x, despite huge absolute differences. Use RMSE/MAE.
- **Constant tensors:** If either tensor is constant (zero variance), correlation is undefined (0/0). Guard with a variance check.
- **Very small tensors:** With < 10 elements, correlation is statistically unstable and carries wide confidence intervals.
- **Probability distributions:** Use KL divergence for distributional comparison.
- **Binary/boolean tensors:** Use agreement metrics (accuracy, F1) instead.
- **Outlier-dominated:** A single outlier pair can inflate Pearson correlation. Use Spearman for robustness, or examine the scatter plot.

## What It Can Tell You

- Whether a transformation preserved the linear structure of a tensor (Pearson).
- Whether the rank ordering of elements is maintained (Spearman) — crucial for attention and top-k.
- The strength and direction (positive/negative) of the relationship.
- Pearson vs. Spearman gap reveals whether distortion is linear or non-linear.

## What It Cannot Tell You

- Absolute error magnitude — correlation is invariant to scale and shift.
- Per-element correctness (use allclose).
- Whether the tensor means or variances match.
- Distributional differences (use KL divergence).
- Worst-case element deviation (use max absolute error).

## Sensitivity

- **Outliers:** Pearson is sensitive — a single extreme value pair can dominate the covariance. Spearman is robust — it operates on ranks, so outlier magnitudes are irrelevant.
- **Scale:** Both are invariant to positive linear scaling. `corr(x, ax+b) = 1.0` for a > 0.
- **Distribution shape:** Pearson assumes an approximately linear relationship. Non-linear monotonic relationships (e.g., log-transform) will have Pearson < Spearman.
- **Ties:** Spearman handles ties via midrank assignment, but many ties (e.g., quantized to few levels) reduce its resolution.
- **Dimensionality:** Both are computed on flattened tensors. Very high-dimensional tensors tend to have stable correlations.

## Alternatives & When to Prefer Them

| Metric              | Prefer When                                                   |
|---------------------|---------------------------------------------------------------|
| Cosine Similarity   | Don't need mean-centering; faster; non-centered comparison    |
| RMSE / MAE          | Need absolute error magnitude                                 |
| Kendall's Tau       | Need a rank correlation robust to small samples               |
| Mutual Information   | Need to capture arbitrary (non-monotonic) dependencies        |
| KL Divergence       | Comparing probability distributions                           |
| SQNR                | Quantization quality on dB scale                              |

## Code Example

```python
import torch
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

# Simulate FP32 and quantized attention scores
# Shape: (batch=1, heads=12, seq_len=64, seq_len=64) — attention matrix
torch.manual_seed(0)
attn_fp32 = torch.randn(1, 12, 64, 64).softmax(dim=-1)  # (1, 12, 64, 64)
# Add noise that scrambles some rankings (simulated INT8 quantization)
noise = 0.02 * torch.randn_like(attn_fp32)
attn_quant = (attn_fp32 + noise).clamp(min=0)  # (1, 12, 64, 64)
attn_quant = attn_quant / attn_quant.sum(dim=-1, keepdim=True)  # renormalize

# Flatten for correlation computation
fp32_flat = attn_fp32.flatten()   # (49152,)
quant_flat = attn_quant.flatten()  # (49152,)

# --- Pearson Correlation ---
pearson_metric = PearsonCorrCoef()
pearson_metric.update(quant_flat, fp32_flat)
pearson_val = pearson_metric.compute()
print(f"Pearson correlation:  {pearson_val.item():.6f}")

# --- Spearman Correlation ---
spearman_metric = SpearmanCorrCoef()
spearman_metric.update(quant_flat, fp32_flat)
spearman_val = spearman_metric.compute()
print(f"Spearman correlation: {spearman_val.item():.6f}")

# --- Manual Pearson ---
x_centered = fp32_flat - fp32_flat.mean()
y_centered = quant_flat - quant_flat.mean()
pearson_manual = (
    torch.dot(x_centered, y_centered)
    / (x_centered.norm() * y_centered.norm())
)
print(f"Pearson (manual):     {pearson_manual.item():.6f}")

# --- Per-head Spearman (which attention heads are most affected?) ---
print("\nPer-head Spearman correlation:")
for head in range(12):
    fp32_head = attn_fp32[0, head].flatten()    # (4096,)
    quant_head = attn_quant[0, head].flatten()  # (4096,)
    sp = SpearmanCorrCoef()
    sp.update(quant_head, fp32_head)
    print(f"  Head {head:2d}: {sp.compute().item():.4f}")
```

## Debugging Use Case

**Scenario: Checking if quantized activations preserve relative ordering in a recommendation model**

A recommendation model uses inner-product attention over user-item embeddings. The top-k items by attention score are recommended. After INT8 quantization, recommendations change for 5% of users:

1. For 1000 users, capture FP32 and INT8 attention score vectors (shape: 1000 items per user).
2. Compute per-user Spearman correlation between FP32 and INT8 attention scores.
3. Results: Mean Spearman = 0.992 (excellent). But for 3% of users, Spearman < 0.95.
4. Investigate the low-Spearman users: their FP32 attention scores have many near-identical values in the top-20 (within 0.001 of each other). INT8 quantization noise reorders these tied values, changing the top-k set.
5. Also compute Pearson: Mean Pearson = 0.998 for all users, even the problematic ones. This confirms the issue is rank-ordering (Spearman) not linear relationship (Pearson) — the overall correlation is fine, but local ties are scrambled.
6. Fix: Use FP16 for the final attention score computation (the top-k selection layer) while keeping the rest in INT8. Spearman improves to > 0.999 for all users, and recommendation changes drop to < 0.1%.

Spearman correlation directly measures what matters for ranking tasks — ordinal preservation — which Pearson and RMSE cannot capture.

## Related Metrics

- [Cosine Similarity](cosine_similarity.md) — Non-centered version of Pearson; faster but includes mean offset.
- [RMSE](rmse.md) — Absolute error magnitude that correlation ignores.
- [MAE](mae.md) — Average absolute error, complementary to correlation.
- [KL Divergence](kl_divergence.md) — Distributional comparison for probability-valued tensors.
- [SQNR](sqnr.md) — Quantization quality on dB scale.
- [Allclose](allclose.md) — Element-wise tolerance check when correlation is not strict enough.
