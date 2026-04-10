---
title: "Signal-to-Noise Ratio (SNR)"
---

# Signal-to-Noise Ratio (SNR)

## Overview

Signal-to-Noise Ratio measures the ratio of useful signal power to noise power, expressed in decibels (dB). In tensor comparison, the "signal" is the reference tensor and the "noise" is the difference between the reference and the approximation. SNR provides a logarithmic quality scale where every 10 dB increase represents a 10× improvement in power ratio (or ~3.16× improvement in amplitude ratio). It is the generalized form of SQNR and applies to any noise source (compression artifacts, numerical precision loss, communication errors), not just quantization.

## Formula

$$
\text{SNR} = 10 \cdot \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)
= 10 \cdot \log_{10}\left(\frac{\sum_{i=1}^{n} x_i^2}{\sum_{i=1}^{n} (x_i - \hat{x}_i)^2}\right)
$$

where $\mathbf{x}$ is the reference (signal) and $\hat{\mathbf{x}}$ is the approximation (signal + noise).

Equivalently, using RMS values:

$$
\text{SNR} = 20 \cdot \log_{10}\left(\frac{\text{RMS}(\mathbf{x})}{\text{RMS}(\mathbf{x} - \hat{\mathbf{x}})}\right)
= -20 \cdot \log_{10}(\text{RRMSE})
$$

**Aliases:** Signal-to-Distortion Ratio (SDR) in audio processing.

## Visual Diagram

```
Signal x:         ████████████████████████████  (power = Σxi²)
Noise (x - x̂):   ██                            (power = Σ(xi-x̂i)²)
                  └──────────────────────────┘
                  SNR = 10·log10(signal_power / noise_power)

           ┌──────────────────────────────────────┐
 60 dB     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │  Lossless-grade
 40 dB     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓          │  Excellent
 20 dB     │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                  │  Acceptable
 10 dB     │  ▓▓▓▓▓▓▓▓                          │  Poor
  0 dB     │  ▓▓▓▓                              │  Noise = Signal
           └──────────────────────────────────────┘
```

<!-- IMAGE: Power spectral density plot showing signal and noise floors with SNR annotation -->

## Range & Interpretation

| SNR (dB)   | Power Ratio        | Interpretation                                       |
|------------|--------------------|----------------------------------------------------- |
| > 60 dB    | > 1,000,000:1      | Near-lossless, FP32 round-trip precision level       |
| 40 – 60 dB| 10,000 – 1M:1      | Excellent — imperceptible error in most applications |
| 30 – 40 dB| 1,000 – 10,000:1   | Good — minor quality degradation                     |
| 20 – 30 dB| 100 – 1,000:1      | Acceptable — noticeable in sensitive applications    |
| 10 – 20 dB| 10 – 100:1         | Poor — significant noise, accuracy likely impacted   |
| 0 – 10 dB | 1 – 10:1           | Bad — noise comparable to signal                     |
| < 0 dB    | < 1:1              | Noise exceeds signal — output is dominated by error  |

## When to Use

- **General-purpose quality assessment:** When the noise source is not specifically quantization (e.g., lossy compression, approximate computation, communication noise).
- **Audio/signal processing models:** SNR is the native quality metric in audio ML (speech enhancement, codec evaluation).
- **Comparing compression schemes:** Evaluating different compression methods (pruning, low-rank, quantization) on a common dB scale.
- **Cross-domain communication:** Engineers from signal processing, communications, and ML all understand dB-scale SNR.
- **Monitoring over time:** Tracking model quality degradation across multiple rounds of compression.

## When NOT to Use

- **Quantization-specific analysis:** Use SQNR, which has the same formula but quantization-specific naming and thresholds.
- **Zero-signal tensors:** If the reference tensor is all zeros, SNR is undefined (−∞ dB). Guard with a power floor.
- **Scale-sensitive comparison:** SNR is scale-invariant (like RRMSE). If absolute error magnitude matters, use RMSE.
- **Per-element analysis:** SNR is a single aggregate number. Use allclose or max absolute error for element-level checks.
- **Distribution comparison:** Use KL divergence for distributional analysis.

## What It Can Tell You

- The overall quality of an approximation on an intuitive logarithmic scale.
- Whether noise is negligible relative to the signal (> 40 dB) or comparable (< 10 dB).
- A common baseline for comparing different approximation methods.
- How quality degrades as compression becomes more aggressive.

## What It Cannot Tell You

- Where in the tensor the error is concentrated (it averages over all elements).
- Whether error is directional or isotropic (use cosine similarity).
- Whether individual elements are within tolerance (use allclose).
- The perceptual impact of the noise — 20 dB SNR may be fine for classification but unacceptable for generation.

## Sensitivity

- **Outliers:** Moderately sensitive. Squaring amplifies large errors in the noise term. A single catastrophic outlier can drop SNR significantly.
- **Scale:** Completely invariant. `SNR(αx, αx̂) = SNR(x, x̂)` for any α ≠ 0.
- **Distribution shift:** A constant bias δ adds `n·δ²` to the noise power, reducing SNR. Very sensitive to systematic bias.
- **Sparsity:** Zeros in the reference reduce signal power without affecting noise from non-zero approximation values, deflating SNR. Compute over non-zero mask.
- **Dimensionality:** Independent of n due to the ratio formulation.

## Alternatives & When to Prefer Them

| Metric            | Prefer When                                                   |
|-------------------|---------------------------------------------------------------|
| SQNR              | Specifically analyzing quantization noise (same formula, domain-specific thresholds) |
| RRMSE             | Want linear (not logarithmic) scale; RRMSE = 10^(−SNR/20)    |
| RMSE              | Need absolute error in original units                         |
| PESQ / POLQA      | Audio quality with perceptual weighting                       |
| SSIM              | Image quality with structural similarity                      |
| Cosine Similarity | Directional fidelity, not power ratio                         |

## Code Example

```python
import torch

def snr_db(reference: torch.Tensor, approximation: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute Signal-to-Noise Ratio in dB.

    Args:
        reference: Clean/baseline tensor (any shape).
        approximation: Noisy/modified tensor (same shape).
        eps: Floor to prevent log(0).

    Returns:
        SNR in decibels.
    """
    signal_power = torch.sum(reference ** 2)
    noise_power = torch.sum((reference - approximation) ** 2)
    return (10 * torch.log10(signal_power / (noise_power + eps))).item()

# Simulate clean activations and compressed version
# Shape: (batch=16, channels=256, height=7, width=7) — e.g., ResNet layer4 output
clean = torch.randn(16, 256, 7, 7)  # (16, 256, 7, 7)

# Low noise (good compression)
low_noise = clean + 0.01 * torch.randn_like(clean)
print(f"Low noise  SNR: {snr_db(clean, low_noise):.1f} dB")  # ~40 dB

# Medium noise (aggressive compression)
med_noise = clean + 0.1 * torch.randn_like(clean)
print(f"Med noise  SNR: {snr_db(clean, med_noise):.1f} dB")  # ~20 dB

# High noise (broken compression)
high_noise = clean + 1.0 * torch.randn_like(clean)
print(f"High noise SNR: {snr_db(clean, high_noise):.1f} dB")  # ~0 dB

# --- Per-channel SNR ---
per_channel_snr = []
for c in range(clean.shape[1]):
    ch_snr = snr_db(clean[:, c], low_noise[:, c])
    per_channel_snr.append(ch_snr)
per_channel_snr = torch.tensor(per_channel_snr)  # (256,)
print(f"Per-channel SNR — min: {per_channel_snr.min():.1f} dB, "
      f"mean: {per_channel_snr.mean():.1f} dB, "
      f"max: {per_channel_snr.max():.1f} dB")
```

## Debugging Use Case

**Scenario: Measuring noise introduced by model weight compression**

You are using low-rank approximation (SVD) to compress a language model's weight matrices. For each layer, you truncate to rank $k$ and want to find the minimum $k$ that maintains quality:

1. For rank $k$ in {16, 32, 64, 128, 256}, reconstruct each weight matrix as $W_k = U_k \Sigma_k V_k^T$.
2. Compute SNR(W, W_k) for each layer and rank.
3. Plot SNR vs. rank. You observe that attention projection matrices reach 40 dB at rank 64, while FFN matrices need rank 128 for the same SNR.
4. Set quality target: SNR ≥ 35 dB for all layers.
5. Use the per-layer SNR curves to select the minimum rank per layer, achieving 2.1× compression with all layers > 35 dB.
6. Cross-validate: The model's perplexity increases by only 0.3 with the SNR-guided rank selection, confirming the threshold was appropriate.

SNR provides a universal quality yardstick for comparing different compression strategies on a common scale.

## Related Metrics

- [SQNR](sqnr.md) — Same formula, quantization-specific naming and thresholds.
- [RRMSE](rrmse.md) — Linear-scale equivalent: RRMSE = 10^(−SNR/20).
- [RMSE](rmse.md) — Absolute error in original units (not dB).
- [Cosine Similarity](cosine_similarity.md) — Directional fidelity, complementary to power-based SNR.
- [Max Absolute Error](max_absolute_error.md) — Worst-case element, not captured by aggregate SNR.
