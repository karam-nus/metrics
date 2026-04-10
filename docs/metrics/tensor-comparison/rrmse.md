---
title: "Relative Root Mean Squared Error (RRMSE)"
---

# Relative Root Mean Squared Error (RRMSE)

## Overview

Relative RMSE normalizes the root mean squared error by the RMS (root mean square) of the reference tensor, producing a dimensionless, scale-invariant error ratio. This solves RMSE's fundamental limitation: an RMSE of 0.01 is excellent for weights with magnitude ~0.1 but negligible for activations with magnitude ~1000. By dividing by the reference signal's RMS, RRMSE enables direct comparison of quantization error across layers, tensors, and models with different dynamic ranges. An RRMSE of 0.01 means the error is 1% of the reference signal magnitude, regardless of scale.

## Formula

$$
\text{RRMSE}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{\text{RMSE}(\mathbf{x}, \hat{\mathbf{x}})}{\text{RMS}(\mathbf{x})}
= \frac{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2}}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}}
= \frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_2}{\|\mathbf{x}\|_2}
$$

where $\mathbf{x}$ is the reference (baseline) tensor and $\hat{\mathbf{x}}$ is the approximation (quantized/modified).

**Aliases:** Normalized RMSE (NRMSE) when normalized by range; Relative L2 Error.

> **Note:** Some sources normalize by `mean(x)`, `max(x) - min(x)`, or `std(x)` instead. The RMS normalization above is standard for tensor comparison because it handles zero-mean data correctly.

## Visual Diagram

```
Layer    │ RMSE   │ RMS(ref) │ RRMSE  │ Verdict
─────────┼────────┼──────────┼────────┼──────────────
embed    │ 0.0005 │ 0.05     │ 0.0100 │ 1.0% — OK
attn_q_3 │ 0.0080 │ 0.80     │ 0.0100 │ 1.0% — OK
ffn_7    │ 0.0500 │ 1.20     │ 0.0417 │ 4.2% — Warning
ffn_11   │ 0.1200 │ 0.90     │ 0.1333 │ 13.3% — BROKEN
─────────┴────────┴──────────┴────────┴──────────────
         RMSE varies 240×, but RRMSE reveals ffn_11 is the real problem.
```

<!-- IMAGE: Side-by-side bar chart comparing raw RMSE vs RRMSE across layers, showing RRMSE correctly identifies the problem layer -->

## Range & Interpretation

| RRMSE Value   | Interpretation                                             |
|---------------|------------------------------------------------------------|
| 0.0           | Perfect match                                              |
| < 0.01        | Excellent — < 1% relative error, negligible impact         |
| 0.01 – 0.05  | Good — 1–5% relative error, monitor downstream accuracy    |
| 0.05 – 0.10  | Concerning — 5–10% relative error, likely accuracy impact  |
| 0.10 – 0.20  | Poor — significant degradation, investigate or revert      |
| > 0.20        | Broken — > 20% relative error, quantization unacceptable   |

These thresholds are empirical heuristics. The acceptable RRMSE depends on the sensitivity of the downstream task and the layer's position in the network.

## When to Use

- **Cross-layer comparison:** Comparing quantization error across layers with different weight/activation scales — RRMSE's primary purpose.
- **Cross-model comparison:** Comparing quantization quality between different architectures (e.g., ResNet vs. ViT) where absolute scales differ.
- **Automated quality gates:** Setting a single RRMSE threshold (e.g., < 5%) that applies uniformly across all layers.
- **Tracking over calibration iterations:** Monitoring how quantization error converges as you refine calibration data.

## When NOT to Use

- **Zero or near-zero reference tensors:** If `RMS(x) ≈ 0`, RRMSE explodes to infinity. This occurs with bias tensors that are near zero, or sparse activation maps. Guard with a floor: `max(RMS(x), epsilon)`.
- **Single-layer absolute error:** When you need the error in original units (e.g., "the error is 0.003 in weight units"), use RMSE.
- **Directional analysis:** RRMSE measures magnitude error, not directional drift. Use cosine similarity.
- **Worst-case analysis:** RRMSE averages over elements. Use max absolute error for outlier detection.

## What It Can Tell You

- The *proportion* of error relative to the signal, enabling apples-to-apples comparison across layers.
- Which layers have disproportionately high quantization error relative to their signal magnitude.
- Whether a quantization scheme uniformly distributes error or concentrates it in specific layers.
- A single threshold that works across all layers regardless of scale.

## What It Cannot Tell You

- The absolute magnitude of error in original units.
- Whether errors are directional or isotropic.
- The distribution of per-element errors (all-elements-slightly-off vs. few-elements-very-off yield the same RRMSE).
- Whether the error impacts downstream accuracy — RRMSE is a proxy, not a guarantee.

## Sensitivity

- **Outliers:** Same as RMSE (high sensitivity) — outliers inflate the numerator. However, if the reference also has high-magnitude outliers, they inflate the denominator, partially canceling out.
- **Scale:** Invariant by construction. Scaling both tensors by α leaves RRMSE unchanged: `RRMSE(αx, αx̂) = RRMSE(x, x̂)`.
- **Distribution shift:** A constant additive bias δ contributes `δ/RMS(x)` to RRMSE. Sensitive to mean shift when signal is small.
- **Sparsity:** Reference sparsity deflates `RMS(x)`, inflating RRMSE. Consider computing RRMSE only over non-zero elements.
- **Dimensionality:** Stable due to element-count normalization in both numerator and denominator.

## Alternatives & When to Prefer Them

| Metric            | Prefer When                                                    |
|-------------------|----------------------------------------------------------------|
| RMSE              | Need error in original units, not relative                     |
| MAE               | Need outlier-robust average error                              |
| SQNR              | Want quantization quality on a log (dB) scale                  |
| Cosine Similarity | Care about directional alignment, not magnitude error          |
| SNR               | Comparing signal vs. generic noise, not specifically quant     |

**Relationship to SQNR:** SQNR = $-20 \log_{10}(\text{RRMSE})$ dB. They carry the same information on different scales:

| RRMSE  | SQNR (dB) |
|--------|-----------|
| 0.001  | 60 dB     |
| 0.01   | 40 dB     |
| 0.10   | 20 dB     |
| 0.316  | 10 dB     |

## Code Example

```python
import torch

# Simulate activations from two layers with very different scales
# Layer A: small-scale embeddings, shape (batch=32, dim=128)
ref_a = 0.05 * torch.randn(32, 128)      # (32, 128), values ≈ [-0.15, 0.15]
quant_a = ref_a + 0.0005 * torch.randn(32, 128)  # small absolute noise

# Layer B: large-scale FFN activations, shape (batch=32, dim=512)
ref_b = 5.0 * torch.randn(32, 512)       # (32, 512), values ≈ [-15, 15]
quant_b = ref_b + 0.05 * torch.randn(32, 512)    # larger absolute noise

def rrmse(reference: torch.Tensor, approximation: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Relative RMSE (RRMSE) = RMSE / RMS(reference)."""
    diff = reference - approximation
    rmse_val = torch.sqrt(torch.mean(diff ** 2))
    rms_ref = torch.sqrt(torch.mean(reference ** 2))
    return (rmse_val / torch.clamp(rms_ref, min=eps)).item()

# Compare RMSE vs RRMSE
rmse_a = torch.sqrt(torch.mean((ref_a - quant_a) ** 2)).item()
rmse_b = torch.sqrt(torch.mean((ref_b - quant_b) ** 2)).item()

print(f"Layer A — RMSE: {rmse_a:.6f}, RRMSE: {rrmse(ref_a, quant_a):.6f}")
print(f"Layer B — RMSE: {rmse_b:.6f}, RRMSE: {rrmse(ref_b, quant_b):.6f}")
# RMSE(B) >> RMSE(A), but RRMSE(A) ≈ RRMSE(B) — both ~1% relative error
```

## Debugging Use Case

**Scenario: Comparing quantization error across layers with vastly different scales**

You are quantizing a GPT-2 model and want to find the weakest layers. The embedding layer has weight magnitudes ~0.02, while later FFN layers have magnitudes ~2.0. Raw RMSE shows the FFN layers have 100× higher error, but that is expected given their scale.

1. Compute RRMSE for each of the 12 transformer blocks (attention weights + FFN weights).
2. Set a uniform threshold of RRMSE < 0.05 (5% relative error).
3. Results: All attention weights pass. FFN layers 0–9 pass. But FFN layers 10 and 11 have RRMSE of 0.08 and 0.12.
4. Diagnosis: Layers 10–11 have outlier weights that push the quantization range, causing higher relative error for the majority of weights.
5. Solution: Apply per-channel quantization or clip outliers for layers 10–11. RRMSE drops to < 0.03 for all layers.

Without RRMSE, you would either miss the problem (if only looking at small-scale layers) or over-investigate non-issues (if alarmed by large raw RMSE in legitimately large-scale layers).

## Related Metrics

- [RMSE](rmse.md) — Absolute (non-normalized) version; use when you need error in original units.
- [SQNR](sqnr.md) — Logarithmic (dB) transformation of the same information: SQNR = −20·log₁₀(RRMSE).
- [SNR](snr.md) — Generalized signal-to-noise ratio.
- [MAE](mae.md) — Outlier-robust alternative to RMSE.
- [Cosine Similarity](cosine_similarity.md) — Directional alignment, complementary to magnitude-based RRMSE.
