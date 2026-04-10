---
title: "Allclose (Numerical Equivalence Testing)"
---

# Allclose (Numerical Equivalence Testing)

## Overview

`torch.allclose` and `torch.isclose` provide element-wise and aggregate boolean checks for whether two tensors are numerically equivalent within specified absolute and relative tolerances. Unlike statistical metrics (RMSE, MAE), allclose gives a hard **pass/fail** verdict: either every element is within tolerance, or the check fails. This is the standard tool for gating model exports, verifying ONNX conversions, confirming that quantization-dequantization round-trips are lossless to within precision limits, and validating numerical correctness of custom kernels.

## Formula

For each element pair $(a_i, b_i)$, `isclose` evaluates:

$$
|a_i - b_i| \leq \texttt{atol} + \texttt{rtol} \times |b_i|
$$

- **atol** (absolute tolerance): Constant tolerance floor. Dominates when $|b_i|$ is small.
- **rtol** (relative tolerance): Scales with the magnitude of the reference. Dominates when $|b_i|$ is large.

`allclose` returns `True` if and only if **all** elements pass `isclose`.

**Defaults:** `atol=1e-8`, `rtol=1e-5` — appropriate for FP32 vs. FP32 comparison.

## Visual Diagram

```
                     Tolerance Band
                 ┌─────────────────────┐
                 │  atol + rtol × |b|  │
                 │         ↕           │
     ────────────┤    PASS region      ├────────────
     b_i value   │                     │  a_i value
     ────────────┤                     ├────────────
                 │         ↕           │
                 │  atol + rtol × |b|  │
                 └─────────────────────┘

For b = 0.0:    tolerance = atol (relative term vanishes)
For b = 1000.0: tolerance = atol + rtol × 1000 (relative term dominates)
```

<!-- IMAGE: Plot showing tolerance band widening with reference magnitude |b| -->

## Range & Interpretation

| Result   | Interpretation                                                |
|----------|---------------------------------------------------------------|
| `True`   | All elements within tolerance — numerical equivalence holds   |
| `False`  | At least one element exceeds tolerance — investigate          |

For `isclose`, the per-element boolean mask tells you *which* elements fail.

**Recommended tolerances by dtype:**

| Comparison               | atol      | rtol      | Rationale                                  |
|--------------------------|-----------|-----------|---------------------------------------------|
| FP32 vs. FP32            | 1e-6      | 1e-5      | Accumulated FP32 rounding (default-ish)     |
| FP32 vs. FP16            | 1e-3      | 1e-3      | FP16 has ~3 decimal digits of precision     |
| FP32 vs. BF16            | 1e-2      | 1.6e-2    | BF16 has ~2 decimal digits of precision     |
| FP32 vs. INT8 (dequant)  | 0.5/scale | 0.05      | Quantization step size is 1/scale           |
| FP32 vs. FP32 (reorder)  | 1e-5      | 1e-4      | Floating-point non-associativity            |
| FP32 vs. ONNX RT FP32    | 1e-5      | 1e-4      | Different operator implementations          |

## When to Use

- **Model export validation:** Verifying that ONNX, TorchScript, or TensorRT exported models produce identical outputs to the PyTorch baseline.
- **Quantization round-trip:** Confirming that quantize → dequantize produces values within the expected quantization step size.
- **Custom kernel validation:** Checking that a CUDA/Triton kernel matches the PyTorch reference implementation.
- **Numerical gradient checking:** `torch.autograd.gradcheck` uses allclose internally.
- **CI/CD gates:** Automated pass/fail checks in continuous integration pipelines.
- **Cross-platform reproducibility:** Ensuring model outputs match across CPU/GPU/different hardware.

## When NOT to Use

- **Quantitative error assessment:** Allclose is binary — it tells you pass/fail but not *how much* error exists. Use RMSE or MAE for quantitative analysis.
- **Statistical comparison:** Allclose checks element-wise, not distributional properties. Use KL divergence for distribution comparison.
- **Tolerance is unknown:** If you do not have a principled basis for choosing atol/rtol, allclose can give false confidence (too loose) or false alarms (too tight). Compute RMSE first to understand the error magnitude, then set tolerances.
- **Stochastic outputs:** Models with dropout, sampling, or non-deterministic operations will always fail allclose. Disable stochasticity first or use statistical metrics.

## What It Can Tell You

- Whether two tensors are identical within a specified precision envelope.
- *Which* elements fail (via `torch.isclose` mask) — useful for diagnosing systematic patterns.
- Whether a model export or conversion preserved numerical fidelity.
- Whether tolerance settings are appropriate for the dtype being used.

## What It Cannot Tell You

- The magnitude of error (only whether it exceeds tolerance).
- Whether the errors that pass are uniformly distributed or concentrated.
- Whether marginal pass/fail matters for downstream accuracy.
- Distributional differences between tensors.

## Sensitivity

- **Outliers:** A single outlier element beyond tolerance causes allclose to return `False`, regardless of how well all other elements match. This is by design — allclose is conservative.
- **Scale:** The `rtol` term adapts to scale — large-magnitude elements get wider tolerance. Set `rtol=0` for purely absolute comparison.
- **Distribution shift:** A constant bias of `δ` causes failure whenever `δ > atol + rtol × |b_i|` for any element. Very sensitive to systematic bias.
- **Sparsity:** Elements near zero are checked with tolerance ≈ `atol` (rtol term vanishes). This is where failures most commonly occur.
- **NaN handling:** By default, NaN ≠ NaN. Use `equal_nan=True` if both tensors may contain NaN in the same positions.

## Alternatives & When to Prefer Them

| Metric / Tool          | Prefer When                                                 |
|------------------------|-------------------------------------------------------------|
| RMSE / MAE             | Need quantitative error magnitude, not pass/fail            |
| Max Absolute Error     | Need the worst-case element error as a number               |
| `torch.equal`          | Need bitwise-exact equality (same dtype, no tolerance)      |
| `numpy.testing.assert_allclose` | Working in NumPy with informative error messages  |
| SQNR                   | Quantization-specific quality on dB scale                   |
| Cosine Similarity      | Care about directional agreement, not element-wise match    |

## Code Example

```python
import torch

# Simulate FP32 baseline and FP16-round-tripped tensor
# Shape: (batch=4, seq_len=128, hidden=768) — transformer output
baseline_fp32 = torch.randn(4, 128, 768)  # (4, 128, 768) FP32
fp16_roundtrip = baseline_fp32.half().float()  # (4, 128, 768) FP32 after FP16 cast

# --- Check 1: Default tolerances (atol=1e-8, rtol=1e-5) — will FAIL for FP16 ---
result_default = torch.allclose(baseline_fp32, fp16_roundtrip)
print(f"Default tolerances: {result_default}")  # False

# --- Check 2: FP16-appropriate tolerances ---
result_fp16 = torch.allclose(baseline_fp32, fp16_roundtrip, atol=1e-3, rtol=1e-3)
print(f"FP16 tolerances (atol=1e-3, rtol=1e-3): {result_fp16}")  # True

# --- Diagnose failures with isclose ---
close_mask = torch.isclose(baseline_fp32, fp16_roundtrip, atol=1e-4, rtol=1e-4)
num_failures = (~close_mask).sum().item()
total_elements = close_mask.numel()
print(f"Elements failing (atol=1e-4, rtol=1e-4): {num_failures}/{total_elements} "
      f"({100 * num_failures / total_elements:.2f}%)")

# --- Find worst offenders ---
abs_diff = torch.abs(baseline_fp32 - fp16_roundtrip)  # (4, 128, 768)
worst_idx = torch.argmax(abs_diff)
print(f"Max absolute error: {abs_diff.max().item():.6f}")
print(f"Worst element index (flat): {worst_idx.item()}")

# --- BF16 example ---
bf16_roundtrip = baseline_fp32.bfloat16().float()
result_bf16 = torch.allclose(baseline_fp32, bf16_roundtrip, atol=1e-2, rtol=1.6e-2)
print(f"BF16 tolerances (atol=1e-2, rtol=1.6e-2): {result_bf16}")  # True
```

## Debugging Use Case

**Scenario: Verifying numerical equivalence of an ONNX-exported transformer**

You export a BERT model to ONNX and want to verify that the ONNX Runtime output matches PyTorch:

1. Run the same input through both PyTorch and ONNX Runtime, collecting output logits.
2. First check: `torch.allclose(pytorch_out, onnx_out, atol=1e-5, rtol=1e-4)` → **False**.
3. Diagnose with `torch.isclose`: 0.3% of elements fail. Inspect the failure pattern.
4. The failures concentrate in the final LayerNorm output — LayerNorm implementations differ slightly between PyTorch and ONNX Runtime due to floating-point operation ordering.
5. Loosen tolerance to `atol=1e-4, rtol=1e-3` → **True**. Verify that this tolerance is acceptable by checking that the argmax of logits matches for all samples (100% agreement).
6. Encode the tolerance in your CI/CD pipeline: `assert torch.allclose(pytorch_out, onnx_out, atol=1e-4, rtol=1e-3)`.

Allclose acts as a **gate**: it enforces that numerical drift stays within a budget you explicitly define. The `isclose` mask helps you understand *where* drift occurs.

## Related Metrics

- [RMSE](rmse.md) — Quantitative magnitude of error when allclose fails.
- [MAE](mae.md) — Average element-wise error magnitude.
- [Max Absolute Error](max_absolute_error.md) — The element that causes allclose to fail.
- [SQNR](sqnr.md) — dB-scale quantization quality, useful for setting informed tolerances.
- [Cosine Similarity](cosine_similarity.md) — Directional check when element-wise allclose is too strict.
- [Norm Difference](norm_difference.md) — Aggregate error magnitude as an alternative to per-element checks.
