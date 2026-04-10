---
title: "Norm Difference (L1 and L2)"
---

# Norm Difference (L1 and L2)

## Overview

Norm difference measures the total magnitude of the difference between two tensors using vector norms. The L1 norm (sum of absolute differences) gives the total absolute deviation, while the L2 norm (Euclidean distance) gives the total Euclidean deviation. Unlike MAE and RMSE which normalize by element count, norm differences are **unnormalized** — they scale with tensor size, making them useful for measuring the *total* perturbation magnitude. This is critical when you care about the aggregate impact of changes (e.g., total weight perturbation budget after pruning) rather than the per-element average.

## Formula

**L1 Norm Difference (Manhattan Distance):**

$$
\|\mathbf{x} - \mathbf{y}\|_1 = \sum_{i=1}^{n} |x_i - y_i|
$$

**L2 Norm Difference (Euclidean Distance):**

$$
\|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

**General Lp Norm Difference:**

$$
\|\mathbf{x} - \mathbf{y}\|_p = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}
$$

**Relationships:**
- $\text{MAE} = \|\mathbf{x} - \mathbf{y}\|_1 / n$
- $\text{RMSE} = \|\mathbf{x} - \mathbf{y}\|_2 / \sqrt{n}$
- $\text{Max Absolute Error} = \|\mathbf{x} - \mathbf{y}\|_\infty$

## Visual Diagram

```
Difference tensor (x - y):  [0.01, -0.02, 0.05, 0.00, -0.03, 0.01, 0.04, -0.02]

L1 norm = |0.01| + |0.02| + |0.05| + |0.00| + |0.03| + |0.01| + |0.04| + |0.02|
        = 0.18  (total absolute perturbation)

L2 norm = sqrt(0.01² + 0.02² + 0.05² + 0.00² + 0.03² + 0.01² + 0.04² + 0.02²)
        = sqrt(0.0060)
        = 0.0775  (Euclidean distance in 8D space)

L1 is the sum of bar heights:    |█|██|█████| |███|█|████|██|
L2 is the diagonal distance:      ─────────────────▶ (one vector in n-D)
```

<!-- IMAGE: Geometric diagram in 2D showing L1 (Manhattan) and L2 (Euclidean) distances between two points -->

## Range & Interpretation

| Value            | Interpretation                                               |
|------------------|--------------------------------------------------------------|
| 0.0              | Identical tensors                                            |
| Small            | Small total perturbation                                     |
| Large            | Large total perturbation — but check tensor size             |
| ∞                | Inf/NaN present                                              |

**Critical:** Norm differences grow with tensor size. A 1M-element tensor with per-element error 0.01 has L1 norm = 10,000 and L2 norm ≈ 10. Compare norms between tensors of the **same size** or normalize (→ MAE/RMSE).

**Relationship between L1 and L2:**
$$
\|\mathbf{d}\|_2 \leq \|\mathbf{d}\|_1 \leq \sqrt{n} \cdot \|\mathbf{d}\|_2
$$
When L1 ≈ √n · L2, error is uniformly distributed. When L1 ≈ L2, error is concentrated in one element.

## When to Use

- **Total perturbation budget:** When you have a constraint on the total amount of weight change allowed (e.g., fine-tuning within an L2 ball, adversarial perturbation bounds).
- **Pruning analysis:** Measuring the total magnitude of weights set to zero (L1 norm of pruned weights = total removed magnitude).
- **Gradient norms:** Monitoring gradient norms during training for stability (gradient clipping is L2-norm based).
- **Comparing same-size tensors:** When you want a single number for the total displacement between two tensors.
- **Regularization-aligned analysis:** L1 norm difference aligns with L1 regularization; L2 with L2 regularization / weight decay.

## When NOT to Use

- **Different-sized tensors:** Norms are not comparable across tensors of different sizes. Use MAE or RMSE instead.
- **Per-element interpretation:** Norms aggregate. For per-element analysis, use the raw difference tensor or allclose.
- **Scale-invariant comparison:** Use RRMSE or cosine similarity.
- **Probability distributions:** Use KL divergence.
- **Directional analysis:** Use cosine similarity. Note that L2 norm difference and cosine similarity are related for unit-normalized vectors.

## What It Can Tell You

- The total magnitude of perturbation introduced by a transformation.
- Whether a modification stays within a specified perturbation budget.
- The L1/L2 ratio reveals error distribution shape: concentrated (ratio near 1) vs. spread (ratio near √n).
- Gradient health during training (exploding = large norm, vanishing = tiny norm).

## What It Cannot Tell You

- Per-element error magnitude (it's aggregated).
- Whether error is concentrated in a few elements or spread uniformly (unless you compare L1 and L2).
- Directional change.
- Scale-relative quality.
- Impact on downstream task performance.

## Sensitivity

- **Outliers:** L2 is more sensitive than L1 (squaring amplifies large deviations). L1 treats all deviations linearly.
- **Scale:** Directly proportional. Scaling tensors by α scales norms by |α|.
- **Tensor size:** Directly proportional (L1 grows as O(n), L2 grows as O(√n) for fixed per-element error).
- **Sparsity:** Sparse errors (few non-zero differences) yield L1 ≈ L2. Dense errors yield L1 >> L2.
- **Dimensionality:** L2 distance in high dimensions concentrates — random vectors in high-D space have nearly constant L2 distance (curse of dimensionality).

## Alternatives & When to Prefer Them

| Metric              | Prefer When                                                  |
|---------------------|--------------------------------------------------------------|
| MAE                 | Need per-element average (= L1 / n)                         |
| RMSE                | Need per-element RMS average (= L2 / √n)                    |
| Max Absolute Error  | Need worst-case single element (= L∞ norm)                   |
| RRMSE               | Need scale-invariant comparison                              |
| Cosine Similarity   | Care about direction, not distance                           |
| Frobenius Norm      | Working with matrices; equivalent to L2 on flattened tensor  |

## Code Example

```python
import torch

# Simulate FP32 weights and pruned weights
# Shape: (out=1024, in=1024) — e.g., large linear layer
weights_original = torch.randn(1024, 1024)  # (1024, 1024)

# Simulate magnitude pruning: zero out smallest 50% of weights
threshold = torch.quantile(weights_original.abs(), 0.5)
weights_pruned = weights_original.clone()
weights_pruned[weights_original.abs() < threshold] = 0.0  # (1024, 1024)

diff = weights_original - weights_pruned  # (1024, 1024)

# --- L1 Norm Difference ---
l1_diff = torch.norm(diff, p=1)
print(f"L1 norm difference: {l1_diff.item():.2f}")
print(f"  → MAE = L1/n:    {l1_diff.item() / diff.numel():.6f}")

# --- L2 Norm Difference (Frobenius for matrices) ---
l2_diff = torch.norm(diff, p=2)
print(f"L2 norm difference: {l2_diff.item():.4f}")
print(f"  → RMSE = L2/√n:  {l2_diff.item() / (diff.numel() ** 0.5):.6f}")

# --- Frobenius norm (equivalent to L2 for matrices) ---
frob_diff = torch.linalg.norm(diff, ord="fro")
print(f"Frobenius norm:     {frob_diff.item():.4f}")
assert torch.allclose(l2_diff, frob_diff)  # identical

# --- L1/L2 ratio reveals error distribution shape ---
ratio = l1_diff.item() / l2_diff.item()
max_ratio = diff.numel() ** 0.5  # √n = maximum ratio (uniform error)
print(f"L1/L2 ratio:        {ratio:.1f}  (max possible: {max_ratio:.1f})")
print(f"  → Error spread:   {ratio / max_ratio:.2%} of maximum uniformity")

# --- Relative norm difference ---
rel_l2 = l2_diff / torch.norm(weights_original, p=2)
print(f"Relative L2 norm:   {rel_l2.item():.4f}  (= RRMSE)")
```

## Debugging Use Case

**Scenario: Total magnitude of weight perturbation after structured pruning**

You apply structured pruning (removing entire filters) to a VGG-16 model and need to verify the total perturbation stays within a budget:

1. For each conv layer, compute L2 norm of the difference between original and pruned weight tensors.
2. Results:
   - Conv1_1: L2 = 0.82 (1.2% of filters removed)
   - Conv3_3: L2 = 15.4 (25% of filters removed)
   - Conv5_3: L2 = 48.2 (50% of filters removed)
3. Compute relative L2 norm (= RRMSE): Conv5_3 has relative L2 = 0.45, meaning 45% of the weight energy was removed.
4. Compare L1 and L2: Conv5_3 has L1/L2 ratio = 280 with √n = 332. Ratio/√n = 0.84, indicating the error (pruned weights) is spread across many elements — consistent with structured pruning removing entire filters.
5. For fine-tuning, constrain the weight update to stay within a L2 ball of radius = pruning L2 norm, preventing the model from drifting too far from the pruned initialization.

Norm differences provide the natural perturbation measure that aligns with regularization-based fine-tuning and adversarial robustness analysis.

## Related Metrics

- [MAE](mae.md) — L1 norm normalized by element count: MAE = L1 / n.
- [RMSE](rmse.md) — L2 norm normalized by √n: RMSE = L2 / √n.
- [Max Absolute Error](max_absolute_error.md) — L∞ norm: the extreme case of Lp norms.
- [RRMSE](rrmse.md) — Relative L2 norm difference.
- [Cosine Similarity](cosine_similarity.md) — Directional metric related to normalized L2 distance.
- [SQNR](sqnr.md) — dB-scale version of relative L2 difference squared.
