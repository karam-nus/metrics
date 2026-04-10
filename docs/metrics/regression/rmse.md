---
title: "Root Mean Squared Error (RMSE)"
---

# Root Mean Squared Error (RMSE)

## Overview

Root Mean Squared Error is the square root of the Mean Squared Error. By undoing the squaring, RMSE restores the error to the same units as the target variable, making it directly interpretable (e.g., "the model is off by ~4.2 kg on average"). Like MSE, RMSE disproportionately penalizes large errors because the squaring happens before the root, but the final number is on a human-readable scale. RMSE is one of the most commonly reported evaluation metrics in regression tasks across domains — from weather forecasting to recommender systems. It is not a proper loss function for direct optimization (due to the square root introducing a non-linearity in the gradient), so models typically optimize MSE and report RMSE.

## Formula

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } = \sqrt{\text{MSE}}
$$

Equivalently: $\text{RMSE} = \| \mathbf{y} - \hat{\mathbf{y}} \|_2 / \sqrt{n}$, the L2 norm of the residual vector scaled by $1/\sqrt{n}$.

## Visual Diagram

```
  RMSE
   ▲
   │        ╱
   │      ╱
   │    ╱          ← grows as sqrt of MSE;
   │  ╱               sublinear in squared error
   │╱
   ┼──────────────► MSE
   0
```

<!-- IMAGE: Concave square-root curve mapping MSE (x-axis) to RMSE (y-axis) -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | 0 (perfect predictions) |
| Maximum  | ∞ (unbounded) |
| Units    | Same as target variable |
| Optimal  | Lower is better |

RMSE ≈ standard deviation of the residuals when predictions are unbiased. A useful rule of thumb: compare RMSE to the standard deviation of the target — if RMSE ≪ σ(y), the model captures most of the variance.

## When to Use

- **Reporting model accuracy to stakeholders**: RMSE is in the same units as the prediction target, making it directly interpretable ("average error of ±3.2°C").
- **When large errors are costly** but you still need a human-readable number — RMSE preserves the outlier sensitivity of MSE.
- **Comparing models on the same dataset**: RMSE provides a single scalar summary that reflects both bias and variance.
- **Leaderboard / competition metrics**: Kaggle and many benchmarks default to RMSE.

## When NOT to Use

- **Outlier-heavy data**: like MSE, RMSE amplifies outlier influence. Prefer [MAE](../regression/mae.md) or [Huber Loss](../regression/huber_loss.md).
- **As a direct optimization objective**: the square root introduces a $1/(2\sqrt{\text{MSE}})$ factor in the gradient, which can cause numerical instability near zero. Optimize MSE instead.
- **Cross-scale comparison**: RMSE is scale-dependent. Use [MAPE](../regression/mape.md) or [SMAPE](../regression/smape.md) for relative comparisons.
- **When median error is more representative**: for skewed error distributions, MAE (which targets the conditional median) is more appropriate.

## What It Can Tell You

- The typical magnitude of prediction error in the target's original units.
- An upper bound on MAE (by Jensen's inequality, RMSE ≥ MAE), so RMSE contextualizes MAE.
- Whether the model's error is practically acceptable given domain tolerances.
- Relative model ranking under the same evaluation set.

## What It Cannot Tell You

- Error direction (over- vs. under-prediction) — RMSE is symmetric.
- How errors are distributed across samples — two models can share the same RMSE but have very different error distributions.
- Relative error — an RMSE of 5 means different things for targets of magnitude 10 vs. 10,000.
- How much variance the model explains — use [R²](../regression/r_squared.md) for that.

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | High. A single outlier with error $e$ inflates RMSE by roughly $e / \sqrt{n}$ (more if $e$ is large relative to other errors). |
| **Scale** | Directly proportional to target scale; always compare RMSE values on the same target/normalization. |
| **Distribution shift** | RMSE increases under drift but cannot distinguish bias shift from increased variance without supplementary analysis. |
| **Sparsity** | For sparse targets, RMSE may be dominated by a few large-magnitude entries; consider separate evaluation on zero vs. non-zero subsets. |

## Alternatives & When to Prefer Them

| Metric | When to Prefer Over RMSE |
|--------|-------------------------|
| [MSE](../regression/mse.md) | When using as an optimization loss (avoids sqrt gradient issues). |
| [MAE](../regression/mae.md) | When outlier robustness matters more than penalizing large errors. |
| [Huber Loss](../regression/huber_loss.md) | When training with noisy labels; quadratic near zero, linear in the tails. |
| [MAPE](../regression/mape.md) | When relative (percentage) error is the evaluation criterion. |
| [SMAPE](../regression/smape.md) | When relative error is needed and actuals can approach zero. |
| [R²](../regression/r_squared.md) | When a unitless, proportion-of-variance metric is required. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])   # (4,)
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])     # (4,)

# --- torchmetrics (squared=False returns RMSE directly) ---
rmse_metric = torchmetrics.MeanSquaredError(squared=False)
rmse_metric.update(y_pred, y_true)
rmse_value = rmse_metric.compute()
print(f"RMSE (torchmetrics): {rmse_value.item():.6f}")
# Expected: sqrt(0.2875) ≈ 0.536262

# --- Manual computation via PyTorch ---
rmse_manual = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
print(f"RMSE (manual):       {rmse_manual.item():.6f}")

# --- From MSE loss object ---
criterion = torch.nn.MSELoss(reduction="mean")
mse_val = criterion(y_pred, y_true)
rmse_from_mse = torch.sqrt(mse_val)
print(f"RMSE (from MSELoss): {rmse_from_mse.item():.6f}")
```

## Debugging Use Case

**Scenario — Evaluating regression model accuracy in original units:**

You trained a temperature-forecasting model. Validation MSE is 6.25, but the project manager asks: "How many degrees off are we on average?"

1. **Convert to RMSE**: $\sqrt{6.25} = 2.5°C$. This is the typical error magnitude in the units stakeholders understand.
2. **Compare to baseline**: if climatological mean prediction yields RMSE = 4.1°C, your model's 2.5°C represents a 39% improvement.
3. **Segment by regime**: compute RMSE separately for summer vs. winter. If winter RMSE = 4.0°C while summer = 1.2°C, the model struggles with cold-season dynamics — consider domain-specific features.
4. **Sanity-check with MAE**: if MAE = 2.4°C and RMSE = 2.5°C, errors are relatively uniform. If MAE = 1.5°C and RMSE = 2.5°C, a few large errors are inflating RMSE — investigate those samples for data quality issues or regime-specific failure.
5. **Set acceptance threshold**: domain experts specify ±3°C is acceptable for 95% of predictions. Check the 95th percentile of absolute errors, not just RMSE, since RMSE is a summary statistic.

## Related Metrics

- [MSE](../regression/mse.md) — RMSE², used as the optimization objective.
- [MAE](../regression/mae.md) — linear error penalty; lower bound of RMSE.
- [Huber Loss](../regression/huber_loss.md) — hybrid MSE/MAE with tunable transition.
- [R²](../regression/r_squared.md) — proportion of variance explained; unitless.
- [MAPE](../regression/mape.md) — relative percentage error.
- [SMAPE](../regression/smape.md) — symmetric relative error.
