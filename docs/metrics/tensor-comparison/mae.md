---
title: "Mean Absolute Error (MAE)"
---

# Mean Absolute Error (MAE)

## Overview

Mean Absolute Error computes the average of the absolute differences between corresponding elements of two tensors. Unlike RMSE, MAE does not square the errors before averaging, making it linearly sensitive to all deviations and significantly more robust to outliers. In tensor comparison for model debugging, MAE provides an intuitive "average per-element deviation" that directly answers: "On average, how far off is each element?" It is especially useful when a few large outlier errors (common in quantization) would inflate RMSE but should not dominate the quality assessment.

## Formula

$$
\text{MAE}(\mathbf{x}, \mathbf{y}) = \frac{1}{n} \sum_{i=1}^{n} |x_i - y_i|
$$

Equivalently:

$$
\text{MAE} = \text{mean}(|\mathbf{x} - \mathbf{y}|) = \frac{\|\mathbf{x} - \mathbf{y}\|_1}{n}
$$

**Aliases:** L1 loss (when used as a loss function), Mean Absolute Deviation (MAD, though MAD also refers to median absolute deviation).

## Visual Diagram

```
Tensor A (FP32):     [ 1.00,  2.00,  3.00,  4.00,  5.00 ]
Tensor B (quantized): [ 1.03,  1.95,  3.08,  3.85,  5.50 ]
                        ↓       ↓       ↓       ↓       ↓
|Differences|:        [ 0.03,  0.05,  0.08,  0.15,  0.50 ]

MAE = mean([0.03, 0.05, 0.08, 0.15, 0.50]) = 0.162

Compare with:
RMSE = sqrt(mean([0.0009, 0.0025, 0.0064, 0.0225, 0.2500])) = 0.237
       ↑ RMSE is inflated by the 0.50 outlier; MAE is not
```

<!-- IMAGE: Error distribution histogram with MAE and RMSE marked, showing RMSE pulled toward outlier tail -->

## Range & Interpretation

| MAE Value          | Interpretation                                              |
|--------------------|-------------------------------------------------------------|
| 0.0                | Perfect match — tensors are identical                       |
| Small (relative)   | Typical per-element deviation is small — acceptable error   |
| Large (relative)   | Average element is far off — significant perturbation       |
| ∞                  | Inf/NaN present in at least one element                     |

Like RMSE, MAE is scale-dependent and must be interpreted relative to the tensor's value range. For cross-layer comparison, normalize by `mean(|x|)` to get relative MAE.

## When to Use

- **Outlier-robust comparison:** When quantization produces a few extreme errors but is otherwise accurate, MAE gives a truer picture of typical error than RMSE.
- **Average per-element deviation:** When you need an intuitive "how wrong is each element on average" answer.
- **Weight perturbation analysis:** Measuring the average magnitude of weight changes after quantization, pruning, or fine-tuning.
- **L1-based optimization:** When the downstream loss uses L1 (e.g., image generation), MAE for intermediate tensor comparison aligns with the optimization objective.
- **Comparing error profiles:** MAE vs. RMSE gap reveals whether error is uniformly distributed (MAE ≈ RMSE/√2) or outlier-dominated (MAE << RMSE).

## When NOT to Use

- **Outlier detection desired:** If you specifically want to detect large errors, RMSE or max absolute error amplifies them. MAE suppresses outlier signal.
- **Penalizing large errors:** In safety-critical applications where large deviations are disproportionately bad, RMSE or max absolute error is more appropriate.
- **Scale-free comparison:** MAE is scale-dependent. Use RRMSE or relative MAE for cross-layer comparison.
- **Directional analysis:** MAE measures magnitude only. Use cosine similarity for directional drift.
- **Distribution comparison:** Use KL divergence for probability distribution comparisons.

## What It Can Tell You

- The average per-element absolute deviation between tensors.
- A robust summary of tensor difference that resists outlier inflation.
- When compared with RMSE, the *shape* of the error distribution: if RMSE >> MAE, errors are outlier-dominated.
- Which layers have the highest typical element-wise quantization error.

## What It Cannot Tell You

- Where the worst-case error is (use max absolute error).
- Whether the error is directional or isotropic (use cosine similarity).
- The scale-normalized quality (use RRMSE or relative MAE).
- Whether errors are systematic (bias) or random — MAE of both is the same.
- Impact on downstream task accuracy.

## Sensitivity

- **Outliers:** Low sensitivity compared to RMSE. A single outlier with error E increases MAE by E/n, whereas it increases MSE by E²/n. This is MAE's primary advantage.
- **Scale:** Directly proportional. Scaling both tensors by α scales MAE by |α|.
- **Distribution shift:** A constant bias δ contributes exactly δ to MAE (when all elements shift by δ).
- **Sparsity:** Same as RMSE — mismatched sparsity patterns contribute their full absolute values.
- **Dimensionality:** Stable across dimensions due to 1/n normalization.

## Alternatives & When to Prefer Them

| Metric              | Prefer When                                                  |
|---------------------|--------------------------------------------------------------|
| RMSE                | Want to penalize large deviations more heavily               |
| RRMSE               | Need scale-invariant comparison across layers                |
| Max Absolute Error  | Need worst-case single-element deviation                     |
| Median AE           | Want even more robustness to outliers than MAE               |
| Cosine Similarity   | Care about directional preservation                          |
| SQNR                | Want dB-scale quantization quality metric                    |

## Code Example

```python
import torch
from torchmetrics.regression import MeanAbsoluteError

# Simulate FP32 and quantized weights for a convolutional layer
# Shape: (out_channels=64, in_channels=128, kernel=3, kernel=3)
fp32_weights = torch.randn(64, 128, 3, 3)  # (64, 128, 3, 3)
quant_weights = fp32_weights + 0.01 * torch.randn_like(fp32_weights)  # simulated quant noise

# --- Method 1: torchmetrics ---
mae_metric = MeanAbsoluteError()
mae_metric.update(quant_weights.flatten(), fp32_weights.flatten())
mae_val = mae_metric.compute()
print(f"MAE (torchmetrics): {mae_val.item():.6f}")

# --- Method 2: Manual PyTorch ---
mae_manual = torch.mean(torch.abs(fp32_weights - quant_weights))
print(f"MAE (manual):       {mae_manual.item():.6f}")

# --- Compare MAE vs RMSE to understand error distribution ---
rmse_val = torch.sqrt(torch.mean((fp32_weights - quant_weights) ** 2))
print(f"RMSE:               {rmse_val.item():.6f}")
print(f"RMSE/MAE ratio:     {rmse_val.item() / mae_val.item():.3f}")
# Ratio ≈ 1.25 (≈ sqrt(π/2)) for Gaussian errors; >> 1.25 indicates outliers

# --- Per-channel MAE ---
per_channel_mae = torch.mean(
    torch.abs(fp32_weights - quant_weights).view(64, -1), dim=1
)  # (64,)
print(f"Per-channel MAE — max: {per_channel_mae.max().item():.6f}, "
      f"min: {per_channel_mae.min().item():.6f}")
```

## Debugging Use Case

**Scenario: Average per-element deviation in quantized weights of MobileNetV2**

MobileNetV2 uses depthwise separable convolutions with small kernels. After INT8 quantization, you want to assess per-element accuracy:

1. For each of the 53 convolutional layers, compute MAE between FP32 and dequantized INT8 weights.
2. Also compute RMSE for each layer. Examine the RMSE/MAE ratio.
3. Results: Most layers have RMSE/MAE ≈ 1.25 (Gaussian-like error), but the final pointwise convolution has RMSE/MAE = 3.2 — indicating heavy outlier errors.
4. Investigate: The final layer has a few channels with very large weights (> 5σ from mean), and INT8's limited range clips them. The clipping creates large outlier errors visible in RMSE but hidden if you only looked at MAE.
5. Action: For this layer, use per-channel quantization to give each channel its own scale factor. MAE stays similar (was already low for most elements), but RMSE drops to RMSE/MAE ≈ 1.3, confirming outliers are resolved.

The MAE vs. RMSE comparison acts as a diagnostic tool for understanding error distribution shape.

## Related Metrics

- [RMSE](rmse.md) — More sensitive to outliers; complements MAE for error distribution analysis.
- [RRMSE](rrmse.md) — Scale-normalized RMSE for cross-layer comparison.
- [Max Absolute Error](max_absolute_error.md) — Worst-case single element; the extreme complement to MAE.
- [Norm Difference](norm_difference.md) — L1 norm difference is MAE × n (total vs. average).
- [Allclose](allclose.md) — Per-element tolerance check instead of aggregate statistics.
- [Cosine Similarity](cosine_similarity.md) — Directional fidelity, orthogonal to magnitude-based MAE.
