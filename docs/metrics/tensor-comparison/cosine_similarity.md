---
title: "Cosine Similarity"
---

# Cosine Similarity

## Overview

Cosine similarity measures the directional alignment between two tensors by computing the cosine of the angle between them when treated as high-dimensional vectors. It ignores magnitude and focuses purely on orientation in the vector space. A cosine similarity of 1.0 means the tensors point in exactly the same direction (identical up to a positive scalar), 0.0 means they are orthogonal (completely uncorrelated directions), and -1.0 means they point in exactly opposite directions. This makes it the go-to metric for detecting whether a transformation (quantization, pruning, distillation) has altered the *direction* of a tensor's representation, independent of any rescaling.

## Formula

$$
\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \; \|\mathbf{B}\|}
= \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \; \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

**Aliases:** cosine distance ($1 - \cos(\theta)$), angular distance ($\arccos(\cos(\theta)) / \pi$).

When applied element-wise across batches, the reduction is typically a mean over the batch dimension.

## Visual Diagram

```
        B (quantized)
       /
      / θ  ← cosine_similarity = cos(θ)
     /
    /_________ A (baseline)
   O

θ ≈ 0°  → cos(θ) ≈ 1.0   (directions aligned, quantization preserved orientation)
θ = 90° → cos(θ) = 0.0    (orthogonal — severe drift)
θ ≈ 180°→ cos(θ) ≈ -1.0   (inverted — catastrophic failure)
```

<!-- IMAGE: 2D unit-circle diagram showing vectors A and B with angle θ between them -->

## Range & Interpretation

| Value         | Interpretation                                              |
|---------------|-------------------------------------------------------------|
| 1.0           | Perfect directional alignment (identical up to positive scale) |
| 0.99 – 1.0   | Excellent — typical of well-calibrated INT8 quantization     |
| 0.95 – 0.99  | Good — minor directional drift, check downstream accuracy    |
| 0.80 – 0.95  | Concerning — noticeable drift, likely accuracy degradation   |
| 0.0           | Orthogonal — no directional relationship preserved           |
| < 0.0         | Anti-correlated — representation is inverted                 |

## When to Use

- **Quantization validation:** Comparing activations or weights of a quantized model (INT8, FP16, INT4) against the FP32 baseline to verify directional fidelity.
- **Layer-by-layer drift analysis:** Profiling each layer's activation cosine similarity to locate where quantization error accumulates.
- **Distillation monitoring:** Checking that a student network's intermediate representations align with the teacher's.
- **Embedding comparison:** Verifying that embedding vectors from a modified model preserve semantic direction.
- **Scale-invariant comparison needed:** When tensors may differ in magnitude (e.g., after batch normalization rescaling) but direction matters.

## When NOT to Use

- **Magnitude matters:** Cosine similarity is blind to scale. If tensor A = [1, 2, 3] and tensor B = [1000, 2000, 3000], cosine similarity is 1.0. Use RMSE or MAE if absolute values matter.
- **Sparse tensors with many zeros:** When both tensors have large zero regions, the non-zero elements dominate and the metric can be misleadingly high. Filter to non-zero regions first.
- **Zero-norm tensors:** If either tensor is all zeros, the metric is undefined (division by zero). Guard against this in code.
- **Scalar or very low-dimensional tensors:** With 1-2 elements, cosine similarity collapses to sign agreement and carries little information.
- **Distribution comparison:** Cosine similarity compares paired elements, not distributions. Use KL divergence for distributional analysis.

## What It Can Tell You

- Whether a transformation has preserved the *direction* of a representation.
- Which layers in a network have the most directional drift after quantization.
- Whether two models produce semantically equivalent embeddings (up to scaling).
- Whether accumulated quantization error is directional or purely magnitude-based (combine with RMSE to disambiguate).

## What It Cannot Tell You

- The absolute magnitude of error between tensors.
- Whether individual elements are within tolerance (use allclose for that).
- Whether the error is concentrated in a few elements or spread uniformly (use max absolute error for worst-case).
- Whether downstream task accuracy is affected — high cosine similarity is necessary but not sufficient.
- Nothing about distributional shifts in outputs (use KL divergence).

## Sensitivity

- **Outliers:** Moderately sensitive. A single large-magnitude outlier element can dominate both the dot product and norms, pulling cosine similarity toward 1.0 even if many small elements diverge.
- **Scale:** Completely invariant to positive uniform scaling. `cos(αA, βB) = cos(A, B)` for α, β > 0. Sensitive to sign flips.
- **Distribution shift:** Insensitive to shifts that preserve direction. A constant additive offset will change cosine similarity.
- **Sparsity:** When both tensors are sparse, cosine similarity is computed only over the non-zero support, which may overstate agreement. Consider masking.
- **Dimensionality:** More stable in high dimensions. In very low dimensions (< 10), the metric is noisy.

## Alternatives & When to Prefer Them

| Metric               | Prefer When                                                   |
|----------------------|---------------------------------------------------------------|
| RMSE                 | Absolute magnitude of error matters                           |
| RRMSE                | Scale-normalized magnitude error needed                       |
| SQNR                 | Quantization-specific quality assessment in dB                |
| Pearson Correlation  | Need a centered (mean-subtracted) directional comparison      |
| Spearman Correlation | Care about rank preservation, not linear relationship         |
| KL Divergence        | Comparing probability distributions, not raw tensors          |
| Max Absolute Error   | Need worst-case element-wise deviation                        |

## Code Example

```python
import torch
import torch.nn.functional as F

# Simulate FP32 baseline activations and INT8-dequantized activations
# Shape: (batch=8, channels=512) — e.g., layer 12 of a transformer
baseline = torch.randn(8, 512)  # (8, 512) FP32 reference
quantized = baseline + 0.05 * torch.randn(8, 512)  # (8, 512) simulated quant noise

# --- Method 1: torch.nn.functional (per-sample, returns shape (8,)) ---
cos_sim = F.cosine_similarity(baseline, quantized, dim=1)  # (8,)
print(f"Per-sample cosine similarity: {cos_sim}")
print(f"Mean cosine similarity: {cos_sim.mean().item():.6f}")

# --- Method 2: Manual computation (flattened, single scalar) ---
a_flat = baseline.flatten()  # (4096,)
b_flat = quantized.flatten()  # (4096,)
manual_cos = torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())
print(f"Flattened cosine similarity: {manual_cos.item():.6f}")

# --- Method 3: torchmetrics ---
from torchmetrics.regression import CosineSimilarity

metric = CosineSimilarity(reduction="mean")
metric.update(baseline, quantized)
result = metric.compute()
print(f"torchmetrics cosine similarity: {result.item():.6f}")
```

## Debugging Use Case

**Scenario: Comparing layer 12 activations of INT8 vs FP32 ResNet-50**

You have quantized a ResNet-50 to INT8 using post-training quantization (PTQ) and observe a 2.1% top-1 accuracy drop on ImageNet. To diagnose where quantization error is most severe:

1. Hook both models to capture activations at every residual block output.
2. For each layer, compute `cosine_similarity(fp32_act, int8_dequant_act)` across a calibration batch.
3. Plot per-layer cosine similarity. You observe layers 1–10 are > 0.998, but layers 11–14 drop to 0.92–0.96.
4. This identifies layers 11–14 as the primary sources of directional drift.
5. Apply mixed-precision: keep layers 11–14 in FP16 while the rest stays INT8. Re-measure cosine similarity (now > 0.998 everywhere) and verify accuracy recovers to within 0.3% of baseline.

Cosine similarity per layer acts as a heat map for quantization damage, allowing targeted precision allocation.

## Related Metrics

- [RMSE](rmse.md) — Captures magnitude error that cosine similarity ignores.
- [RRMSE](rrmse.md) — Scale-normalized version of RMSE.
- [SQNR](sqnr.md) — Quantization-specific signal quality metric.
- [Correlation](correlation.md) — Pearson is the centered version of cosine similarity.
- [Allclose](allclose.md) — Element-wise tolerance check for strict numerical equivalence.
- [KL Divergence](kl_divergence.md) — For distributional comparison of output logits.
