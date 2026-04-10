---
title: "Root Mean Squared Error (RMSE)"
---

# Root Mean Squared Error (RMSE)

## Overview

Root Mean Squared Error measures the average magnitude of element-wise differences between two tensors, with heavier penalization of large deviations due to the squaring step. It is the most commonly used absolute-error metric for tensor comparison because it is in the same units as the original tensor values and has well-understood statistical properties. In the context of model debugging, RMSE quantifies *how much* a transformation (quantization, pruning, fine-tuning) has perturbed tensor values. Unlike cosine similarity, RMSE is sensitive to both directional and magnitude changes.

## Formula

$$
\text{RMSE}(\mathbf{x}, \mathbf{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2}
$$

Equivalently:

$$
\text{RMSE} = \sqrt{\text{MSE}} = \| \mathbf{x} - \mathbf{y} \|_2 \; / \; \sqrt{n}
$$

**Aliases:** Root Mean Square Deviation (RMSD), L2 error (when unnormalized by $\sqrt{n}$).

## Visual Diagram

```
Tensor A (FP32):     [ 1.00,  2.00,  3.00,  4.00 ]
Tensor B (INT8 deq): [ 1.02,  1.95,  3.10,  3.80 ]
                       ↓       ↓       ↓       ↓
Differences:         [ 0.02, -0.05,  0.10, -0.20 ]
Squared:             [ 0.0004, 0.0025, 0.0100, 0.0400 ]
Mean of squared:       0.013225
RMSE = sqrt(0.013225) = 0.1150
```

<!-- IMAGE: Bar chart showing per-element squared errors and the RMSE as a horizontal reference line -->

## Range & Interpretation

| Value              | Interpretation                                               |
|--------------------|--------------------------------------------------------------|
| 0.0                | Perfect match — tensors are identical                        |
| Small (relative)   | Minor perturbation — likely acceptable quantization error    |
| Large (relative)   | Significant deviation — investigate layer-by-layer           |
| ∞                  | At least one element has diverged to inf/NaN                 |

Interpretation is **scale-dependent**: an RMSE of 0.01 is excellent for weights in [−1, 1] but meaningless for activations in [−1000, 1000]. Always contextualize with tensor magnitude or use RRMSE.

## When to Use

- **Weight drift monitoring:** Tracking how much model weights change after quantization, pruning, or fine-tuning.
- **Activation comparison:** Measuring the absolute deviation of intermediate activations between a baseline and modified model.
- **Regression tasks:** RMSE is the standard loss/metric when comparing predicted vs. ground-truth continuous values.
- **Gradient checking:** Comparing analytical vs. numerical gradients to verify backprop correctness.
- **Per-layer error profiling:** Computing RMSE for each layer to build an error profile of a quantized model.

## When NOT to Use

- **Different tensor scales:** If comparing layers with wildly different magnitude ranges, raw RMSE is misleading. Use RRMSE instead.
- **Outlier-dominated analysis:** A single catastrophic outlier will inflate RMSE. Use MAE for a robust alternative or Max Absolute Error for explicit outlier detection.
- **Directional analysis:** RMSE cannot distinguish between errors that change direction vs. errors that only change magnitude. Use cosine similarity for directional fidelity.
- **Probability distributions:** RMSE on logits is fine, but for comparing probability distributions use KL divergence.
- **Pass/fail gating:** Use allclose for binary pass/fail checks with explicit tolerances.

## What It Can Tell You

- The average magnitude of per-element error between two tensors, in original units.
- Which layers have the largest absolute quantization error.
- Whether quantization error is growing across layers (error accumulation).
- Relative quality of different quantization schemes applied to the same model.

## What It Cannot Tell You

- Whether errors are directional or merely scalar (use cosine similarity).
- Where the worst-case error is (use max absolute error).
- Whether the error is uniformly distributed or concentrated (inspect the error histogram).
- Whether the error affects downstream accuracy — low RMSE is necessary but not sufficient.
- Scale-independent quality — RMSE of 0.1 means nothing without knowing the tensor's range.

## Sensitivity

- **Outliers:** High sensitivity. Squaring amplifies large deviations. A single element with error 100× the median will dominate the RMSE. This is a feature for detecting catastrophic failures, but a liability for robust comparison.
- **Scale:** Directly proportional. Scaling both tensors by α scales RMSE by |α|. This means RMSE cannot be compared across layers with different scales.
- **Distribution shift:** Captures both mean shift and variance change. A constant bias of δ adds δ to RMSE.
- **Sparsity:** Zeros that match contribute nothing; zeros in one tensor vs. non-zeros in the other contribute their full squared value. High sparsity mismatch inflates RMSE.
- **Dimensionality:** Stable across dimensions due to the 1/n normalization.

## Alternatives & When to Prefer Them

| Metric              | Prefer When                                                  |
|---------------------|--------------------------------------------------------------|
| RRMSE               | Need scale-invariant comparison across layers                |
| MAE                 | Want outlier-robust average error                            |
| Max Absolute Error  | Need worst-case element-wise error                           |
| SQNR                | Quantization-specific quality in dB scale                    |
| Cosine Similarity   | Care about directional preservation, not magnitude           |
| Norm Difference     | Want total (unnormalized) error magnitude                    |

## Code Example

```python
import torch
from torchmetrics.regression import MeanSquaredError

# Simulate FP32 weights and quantized-dequantized weights
# Shape: (out_features=256, in_features=512) — e.g., a linear layer
fp32_weights = torch.randn(256, 512)  # (256, 512)
quant_weights = fp32_weights + 0.02 * torch.randn(256, 512)  # (256, 512) simulated noise

# --- Method 1: torchmetrics ---
mse_metric = MeanSquaredError(squared=False)  # squared=False → returns RMSE
mse_metric.update(quant_weights.flatten(), fp32_weights.flatten())
rmse_val = mse_metric.compute()
print(f"RMSE (torchmetrics): {rmse_val.item():.6f}")

# --- Method 2: Manual PyTorch ---
diff = fp32_weights - quant_weights  # (256, 512)
rmse_manual = torch.sqrt(torch.mean(diff ** 2))
print(f"RMSE (manual):       {rmse_manual.item():.6f}")

# --- Per-row RMSE (per output neuron) ---
per_row_rmse = torch.sqrt(torch.mean(diff ** 2, dim=1))  # (256,)
print(f"Per-row RMSE — max: {per_row_rmse.max().item():.6f}, "
      f"min: {per_row_rmse.min().item():.6f}, "
      f"mean: {per_row_rmse.mean().item():.6f}")
```

## Debugging Use Case

**Scenario: Weight drift after INT8 post-training quantization of BERT**

After quantizing BERT-base to INT8, you notice a 1.5% F1 drop on SQuAD. To identify the problematic layers:

1. Extract the FP32 and dequantized INT8 weight tensors for all 12 transformer layers.
2. Compute per-layer RMSE for each weight matrix (query, key, value, FFN).
3. Results show layers 0–8 have RMSE < 0.005, but layers 9–11 FFN weights have RMSE of 0.03–0.06.
4. Inspect the FFN weight distributions: layers 9–11 have heavier tails (larger dynamic range), causing more quantization error with uniform INT8 quantization.
5. Solution: Apply per-channel quantization to FFN layers 9–11, reducing their RMSE to < 0.008. F1 recovers to within 0.2% of baseline.

RMSE per layer provides a direct error magnitude that correlates with the severity of quantization damage.

## Related Metrics

- [RRMSE](rrmse.md) — RMSE normalized by reference magnitude; use for cross-layer comparison.
- [MAE](mae.md) — Mean absolute error; more robust to outliers.
- [Max Absolute Error](max_absolute_error.md) — Worst-case single-element error.
- [SQNR](sqnr.md) — Signal-to-quantization-noise ratio; dB-scale quality metric.
- [Cosine Similarity](cosine_similarity.md) — Directional fidelity, ignoring magnitude.
- [Norm Difference](norm_difference.md) — Total (unnormalized) L2 error.
