---
title: "Max Absolute Error (L∞ Norm)"
---

# Max Absolute Error (L∞ Norm)

## Overview

Max Absolute Error is the largest element-wise absolute difference between two tensors — the L∞ (infinity) norm of the difference tensor. While aggregate metrics like RMSE and MAE summarize the average behavior, max absolute error captures the **worst-case** deviation. A single catastrophic outlier that would be diluted in RMSE (averaged over millions of elements) is fully exposed by this metric. In quantization debugging, max absolute error is critical for detecting clipping artifacts, overflow errors, and numerical edge cases that affect only a few elements but can cause catastrophic model failures (e.g., NaN propagation, attention score explosion).

## Formula

$$
\text{MaxAE}(\mathbf{x}, \mathbf{y}) = \max_{i} |x_i - y_i| = \|\mathbf{x} - \mathbf{y}\|_\infty
$$

This is the L∞ norm of the difference tensor, also known as the Chebyshev distance or supremum norm.

**Relationship to Lp norms:**

$$
\lim_{p \to \infty} \|\mathbf{d}\|_p = \|\mathbf{d}\|_\infty = \max_i |d_i|
$$

## Visual Diagram

```
Element-wise absolute errors:

  0.50 │                                    ██
       │                                    ██ ← MAX ABSOLUTE ERROR
  0.40 │                                    ██
       │                                    ██
  0.30 │                                    ██
       │                                    ██
  0.20 │          ██                        ██
       │    ██    ██    ██                  ██
  0.10 │    ██    ██    ██    ██            ██
       │    ██    ██    ██    ██    ██      ██
  0.00 └────────────────────────────────────────
       e1   e2   e3   e4   e5   e6   ...  eN

  MAE  = 0.08   (average — looks fine)
  RMSE = 0.12   (slightly elevated — mildly concerning)
  MaxAE = 0.50  (catastrophic outlier — MUST investigate)
```

<!-- IMAGE: Scatter plot of per-element errors with MaxAE highlighted as the top outlier -->

## Range & Interpretation

| MaxAE Value          | Interpretation                                              |
|----------------------|-------------------------------------------------------------|
| 0.0                  | Perfect match — all elements identical                      |
| ≤ quantization step  | Within single-step quantization error — expected            |
| 2–5× quant step      | Rounding boundary effect — usually acceptable               |
| >> quant step         | Clipping, overflow, or numerical error — investigate        |
| ∞ or NaN             | Catastrophic numerical failure                               |

**For INT8 quantization:** If the quantization scale is `s`, the maximum expected error is `s/2` (half-step rounding). MaxAE >> `s/2` indicates clipping or implementation bugs.

## When to Use

- **Clipping detection:** INT8/INT4 quantization clips values outside the representable range. MaxAE reveals the worst clipping error.
- **Overflow/underflow detection:** FP16 has limited range (max ~65504). MaxAE catches overflow to inf.
- **Numerical correctness verification:** Custom CUDA kernels, fused operations, or compiler optimizations may introduce worst-case errors visible only in MaxAE.
- **Safety-critical validation:** When even a single large error is unacceptable (e.g., autonomous driving, medical).
- **Complementing RMSE/MAE:** If MaxAE >> RMSE, the error is concentrated in outliers. If MaxAE ≈ RMSE, error is uniform.

## When NOT to Use

- **Average behavior assessment:** MaxAE reflects a single worst element, not typical behavior. Use MAE or RMSE.
- **Scale-free comparison:** MaxAE is in original units and not normalized. Use RRMSE for cross-layer comparison.
- **Distributional comparison:** Use KL divergence for probability distributions.
- **Directional analysis:** Use cosine similarity.
- **Noisy tensors with known outliers:** If you expect a few large errors (e.g., from stochastic rounding), MaxAE will always alarm. Use a high-percentile error (99th percentile) instead.

## What It Can Tell You

- The worst-case single-element error in the entire tensor.
- Whether quantization clipping has occurred and how severe it is.
- Whether there are numerical edge cases (inf, NaN, overflow) in the approximation.
- The location (index) of the worst error, enabling targeted debugging.
- Whether the error distribution has heavy tails (compare MaxAE to RMSE ratio).

## What It Cannot Tell You

- How many elements have large errors (it only reports the maximum). Pair with a histogram or percentile analysis.
- The average error (use MAE/RMSE).
- Whether the overall tensor is a good approximation (one bad element inflates MaxAE even if everything else is perfect).
- Directional or distributional properties.

## Sensitivity

- **Outliers:** Maximum sensitivity by definition — this *is* the outlier detection metric.
- **Scale:** Directly proportional. Scaling tensors by α scales MaxAE by |α|.
- **Tensor size:** Weakly increases with tensor size because the maximum of n random variables grows as O(√(log n)) for Gaussian errors.
- **Sparsity:** If one tensor is sparse and the other is not, MaxAE equals the largest non-zero element in the denser tensor at a zero position.
- **Distribution shape:** Heavy-tailed error distributions (common in quantization with clipping) produce much higher MaxAE relative to RMSE.

## Alternatives & When to Prefer Them

| Metric              | Prefer When                                                  |
|---------------------|--------------------------------------------------------------|
| MAE                 | Want average per-element error, robust to outliers           |
| RMSE                | Want average error with moderate outlier sensitivity         |
| Percentile Error    | Want top-k% error rather than absolute worst case            |
| Allclose            | Need pass/fail with explicit tolerance (uses both atol + rtol)|
| SQNR                | Want aggregate quantization quality on dB scale              |
| Norm Difference     | Want total (not max) error magnitude                         |

## Code Example

```python
import torch

# Simulate FP32 activations and INT8-quantized activations with clipping
# Shape: (batch=1, channels=64, height=32, width=32) — e.g., conv layer output
torch.manual_seed(42)
fp32_act = torch.randn(1, 64, 32, 32) * 3.0  # (1, 64, 32, 32) some values > 6σ

# Simulate INT8 quantization with clipping at ±6.0
clip_min, clip_max = -6.0, 6.0
scale = (clip_max - clip_min) / 255.0
int8_quant = torch.clamp(fp32_act, clip_min, clip_max)
int8_quant = torch.round(int8_quant / scale) * scale  # (1, 64, 32, 32)

# --- Max Absolute Error ---
abs_diff = torch.abs(fp32_act - int8_quant)  # (1, 64, 32, 32)
max_ae = torch.max(abs_diff)
print(f"Max Absolute Error: {max_ae.item():.6f}")

# --- Location of worst error ---
flat_idx = torch.argmax(abs_diff)
idx = torch.unravel_index(flat_idx, abs_diff.shape)
print(f"Worst error at index: batch={idx[0]}, ch={idx[1]}, h={idx[2]}, w={idx[3]}")
print(f"FP32 value:  {fp32_act[idx].item():.4f}")
print(f"INT8 value:  {int8_quant[idx].item():.4f}")
print(f"Error:       {abs_diff[idx].item():.4f}")

# --- Compare with RMSE and MAE ---
rmse = torch.sqrt(torch.mean(abs_diff ** 2))
mae = torch.mean(abs_diff)
print(f"\nMAE:              {mae.item():.6f}")
print(f"RMSE:             {rmse.item():.6f}")
print(f"Max Absolute:     {max_ae.item():.6f}")
print(f"MaxAE / RMSE:     {max_ae.item() / rmse.item():.1f}x")

# --- Percentile analysis ---
sorted_errors = torch.sort(abs_diff.flatten()).values
n = sorted_errors.numel()
for pct in [50, 90, 99, 99.9, 100]:
    idx_pct = min(int(pct / 100 * n), n - 1)
    print(f"  P{pct:5.1f}: {sorted_errors[idx_pct].item():.6f}")

# --- Expected max error for uniform quantization ---
expected_max = scale / 2  # half-step rounding error
print(f"\nExpected max (half-step): {expected_max:.6f}")
print(f"Actual max / expected:    {max_ae.item() / expected_max:.1f}x")
clipped_count = ((fp32_act < clip_min) | (fp32_act > clip_max)).sum().item()
print(f"Clipped elements:         {clipped_count} ({100*clipped_count/fp32_act.numel():.3f}%)")
```

## Debugging Use Case

**Scenario: Finding catastrophic outlier errors in quantized tensors of a YOLOv5 detector**

After INT8 quantization of YOLOv5, you observe rare but severe false positives — the model occasionally predicts bounding boxes with extreme confidence (> 0.99) for non-existent objects:

1. Hook the detection head and capture FP32 vs. INT8 activations for 500 images.
2. Compute per-image MaxAE for the detection head output tensor.
3. Results: 95% of images have MaxAE < 0.05 (within normal quantization error). But 2% of images have MaxAE > 2.0.
4. Inspect the high-MaxAE cases: the worst errors occur at spatial positions where the FP32 activation has values > 8.0, which exceeds the INT8 calibration range of [-6, 6]. The clipping error of 2+ units in the objectness score logit translates to near-1.0 sigmoid confidence.
5. Fix: Expand the calibration range for the detection head from [-6, 6] to [-10, 10], or use per-channel calibration. MaxAE drops to < 0.1 for all images. False positives disappear.

MaxAE is the only metric that would have caught this — RMSE was 0.03 (excellent) because 99.99% of elements were fine, masking the catastrophic 0.01%.

## Related Metrics

- [MAE](mae.md) — Average absolute error; MaxAE is the extreme case.
- [RMSE](rmse.md) — RMS average; moderate sensitivity between MAE and MaxAE.
- [Norm Difference](norm_difference.md) — L1/L2 total norms; MaxAE is L∞ norm.
- [Allclose](allclose.md) — Uses a tolerance band; MaxAE tells you if it would pass.
- [SQNR](sqnr.md) — Aggregate quality metric; use MaxAE to find the outliers that degrade SQNR.
- [Cosine Similarity](cosine_similarity.md) — Directional metric; insensitive to the outliers MaxAE catches.
