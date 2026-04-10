---
title: "Signal-to-Quantization-Noise Ratio (SQNR)"
---

# Signal-to-Quantization-Noise Ratio (SQNR)

## Overview

Signal-to-Quantization-Noise Ratio is the definitive metric for assessing the fidelity of a quantized tensor relative to its original full-precision representation. It measures the ratio of the original signal's power to the power of the quantization error (noise), expressed in decibels (dB). SQNR is the single most important metric when evaluating quantization schemes because it directly quantifies how much of the original signal is preserved versus how much is corrupted by the quantization process. It is mathematically identical to SNR but is specifically applied to quantization noise and carries domain-specific quality thresholds.

## Formula

$$
\text{SQNR}(\mathbf{x}, \mathbf{x}_q) = 10 \cdot \log_{10}\left(\frac{\|\mathbf{x}\|^2}{\|\mathbf{x} - \mathbf{x}_q\|^2}\right)
= 10 \cdot \log_{10}\left(\frac{\sum_{i=1}^{n} x_i^2}{\sum_{i=1}^{n} (x_i - x_{q,i})^2}\right)
$$

where $\mathbf{x}$ is the original FP32 tensor and $\mathbf{x}_q$ is the quantized-then-dequantized tensor.

**Theoretical maximum for uniform $b$-bit quantization:**

$$
\text{SQNR}_{\max} \approx 6.02 \cdot b + 1.76 \;\text{dB}
$$

| Bits | Theoretical Max SQNR |
|------|----------------------|
| 8    | ~49.9 dB             |
| 4    | ~25.8 dB             |
| 2    | ~13.8 dB             |

Actual SQNR is always lower due to non-uniform distributions, clipping, and rounding.

## Visual Diagram

```
Original signal x:     ████████████████████████████████  (||x||² = signal power)
Quant noise (x - xq):  ███                               (||x - xq||² = noise power)
                       └────────────────────────────────┘
                       SQNR = 10·log10(signal / noise) dB

Per-layer SQNR profile of a quantized model:
Layer  0: ████████████████████████████████████████  45 dB ✓
Layer  1: ███████████████████████████████████████   43 dB ✓
Layer  2: ██████████████████████████████████████    41 dB ✓
  ...
Layer 10: ██████████████████████████                31 dB ✓
Layer 11: ████████████████                          23 dB ⚠
Layer 12: █████████                                 15 dB ✗ ← PROBLEM
```

<!-- IMAGE: Heatmap of per-layer SQNR across a transformer model, with red highlighting layers below threshold -->

## Range & Interpretation

| SQNR (dB) | Quality Level | Action                                                  |
|------------|---------------|---------------------------------------------------------|
| > 45 dB   | Excellent     | Near-lossless quantization. No action needed.           |
| 35 – 45   | Good          | Minimal accuracy impact. Standard operating range.      |
| 25 – 35   | Acceptable    | Some accuracy loss likely. Monitor downstream metrics.  |
| 20 – 25   | Degraded      | Noticeable accuracy impact. Consider mixed precision.   |
| 15 – 20   | Poor          | Significant degradation. Layer needs higher precision.  |
| < 15 dB   | Broken        | Output is noise-dominated. Quantization is unacceptable.|

**Rule of thumb:** For INT8 quantization, most layers should achieve 35–50 dB. For INT4, expect 20–30 dB. Below 20 dB, accuracy almost certainly suffers.

## When to Use

- **Per-layer quantization quality assessment:** The primary use case. Compute SQNR for each layer to identify weak points in a quantized model.
- **Comparing quantization schemes:** PTQ vs. QAT, symmetric vs. asymmetric, per-tensor vs. per-channel — compare by SQNR.
- **Mixed-precision decisions:** Use SQNR to decide which layers can be INT4, which need INT8, and which must stay FP16.
- **Calibration evaluation:** After changing calibration data or strategy, compare per-layer SQNR to see if quality improved.
- **Quantization-aware training monitoring:** Track per-layer SQNR during QAT epochs to verify convergence.

## When NOT to Use

- **Non-quantization noise:** Use SNR for general noise analysis (compression, approximate computing).
- **Distributional comparison:** SQNR compares paired tensors, not distributions. Use KL divergence.
- **Zero-signal tensors:** SQNR is undefined when the reference signal power is zero (bias terms near zero, sparse activations). Floor the denominator.
- **Already in linear scale:** If you prefer linear ratios over dB, use RRMSE (they carry the same information).

## What It Can Tell You

- Which layers are most damaged by quantization, with precise quality numbers.
- Whether a quantization configuration meets a minimum quality bar (e.g., all layers > 30 dB).
- How much quality headroom exists for further compression (e.g., a layer at 45 dB can tolerate more aggressive quantization).
- The theoretical vs. actual efficiency of the quantization scheme (actual SQNR vs. 6.02b + 1.76).

## What It Cannot Tell You

- Which specific elements have the largest error (use max absolute error).
- Whether the error is directional (use cosine similarity).
- The downstream accuracy impact — SQNR is a necessary-but-not-sufficient proxy. A model can have good per-layer SQNR but accumulate errors across layers.
- Element-wise tolerance compliance (use allclose).
- How errors interact across layers (SQNR is per-layer, not end-to-end).

## Sensitivity

- **Outliers:** Highly sensitive. Outlier signal values inflate signal power; outlier noise values inflate noise power. Both affect SQNR. In practice, quantization noise outliers (from clipping) are the main concern.
- **Scale:** Invariant — scaling both tensors by α does not change SQNR.
- **Distribution shape:** Non-uniform distributions (heavy tails, multi-modal) achieve lower SQNR than Gaussian for the same bit-width due to suboptimal range utilization.
- **Clipping:** The dominant source of low SQNR. When the quantization range is too narrow, clipped values generate large noise. SQNR is an excellent detector of clipping damage.
- **Sparsity:** Many zeros in the signal deflate signal power, lowering SQNR. For sparse activations, consider SQNR only over the non-zero support.

## Alternatives & When to Prefer Them

| Metric            | Prefer When                                                   |
|-------------------|---------------------------------------------------------------|
| SNR               | Noise source is not quantization                              |
| RRMSE             | Want linear-scale equivalent (RRMSE = 10^(−SQNR/20))         |
| RMSE              | Need error in absolute units                                  |
| Cosine Similarity | Directional fidelity matters more than magnitude              |
| Max Absolute Error| Need worst-case element error (clipping detection)            |
| KL Divergence     | Comparing output probability distributions                    |

## Code Example

```python
import torch

def sqnr_db(original: torch.Tensor, quantized: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute Signal-to-Quantization-Noise Ratio in dB.

    Args:
        original: FP32 reference tensor.
        quantized: Quantized-then-dequantized tensor (same shape).
        eps: Floor to prevent log(0).

    Returns:
        SQNR in decibels. Higher is better.
    """
    signal_power = torch.sum(original ** 2)
    noise_power = torch.sum((original - quantized) ** 2)
    return (10 * torch.log10(signal_power / (noise_power + eps))).item()

# --- Simulate per-layer INT8 quantization of a 6-layer transformer ---
torch.manual_seed(42)
num_layers = 6
print("Layer | SQNR (dB) | Verdict")
print("------+-----------+--------")

for layer in range(num_layers):
    # Simulate weights with varying dynamic range
    # Shape: (hidden=512, hidden=512) — e.g., attention projection
    scale_factor = 1.0 + 0.5 * layer  # Later layers have wider range
    weights_fp32 = scale_factor * torch.randn(512, 512)  # (512, 512)

    # Simulate INT8 quantization: scale → round → clamp → dequantize
    qmin, qmax = -128, 127
    w_min, w_max = weights_fp32.min(), weights_fp32.max()
    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = qmin - torch.round(w_min / scale)
    quantized_int = torch.clamp(torch.round(weights_fp32 / scale + zero_point), qmin, qmax)
    dequantized = (quantized_int - zero_point) * scale  # (512, 512)

    sqnr = sqnr_db(weights_fp32, dequantized)
    verdict = "✓ OK" if sqnr > 30 else ("⚠ WARN" if sqnr > 20 else "✗ BAD")
    print(f"  {layer:3d}  | {sqnr:8.1f}  | {verdict}")

# --- Targeted analysis of worst layer ---
print(f"\nTheoretical max SQNR for 8-bit: {6.02 * 8 + 1.76:.1f} dB")
```

## Debugging Use Case

**Scenario: Per-layer quantization quality assessment — finding problem layers in INT8 ResNet-50**

After INT8 PTQ of ResNet-50, top-1 accuracy drops by 3.2%. SQNR profiling reveals the cause:

1. Hook both FP32 and INT8 models. For each of the 53 conv layers, compute `sqnr_db(fp32_weights, int8_dequant_weights)`.
2. Results: 48 layers have SQNR > 38 dB (good). Layers `conv4_3`, `conv5_1`, `conv5_2`, `conv5_3`, and `conv5_4` have SQNR 18–24 dB.
3. Root cause: These layers have outlier weights (> 6σ) that force the INT8 range to be very wide, causing most weights to be quantized with excessive step size.
4. Apply per-channel quantization to the 5 problematic layers. SQNR improves from 18–24 dB to 38–42 dB.
5. With mixed per-channel/per-tensor quantization, top-1 accuracy recovers to within 0.4% of FP32 baseline.

SQNR is the standard diagnostic for identifying which layers need higher precision or better calibration.

## Related Metrics

- [SNR](snr.md) — Generalized version for any noise source, same formula.
- [RRMSE](rrmse.md) — Linear-scale equivalent: RRMSE = 10^(−SQNR/20).
- [RMSE](rmse.md) — Absolute error in original units; the numerically un-normalized form.
- [Max Absolute Error](max_absolute_error.md) — Catches clipping errors that degrade SQNR.
- [Cosine Similarity](cosine_similarity.md) — Directional quality, complementary to power-based SQNR.
- [Allclose](allclose.md) — Per-element tolerance check, useful after SQNR-guided fixes.
