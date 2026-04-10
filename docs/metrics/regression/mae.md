---
title: "Mean Absolute Error (MAE)"
---

# Mean Absolute Error (MAE)

## Overview

Mean Absolute Error is the arithmetic mean of the absolute differences between predicted and actual values. It provides a linear, symmetric penalty for errors of any magnitude, making it significantly more robust to outliers than MSE or RMSE. MAE corresponds to the L1 loss and, under maximum-likelihood estimation, is optimal when the residuals follow a Laplace distribution. In optimization, MAE is the objective that produces the conditional median of the target distribution — contrasted with MSE, which targets the conditional mean. The non-differentiability of MAE at zero error can cause gradient issues during training, but subgradient methods and smooth approximations handle this well in practice.

## Formula

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Equivalently: $\text{MAE} = \frac{1}{n} \| \mathbf{y} - \hat{\mathbf{y}} \|_1$, the L1 norm of the residual vector divided by $n$.

## Visual Diagram

```
  Loss
   ▲
   │        ╱
   │      ╱
   │    ╱          ← linear growth: doubling the
   │  ╱               error doubles the loss
   │╱
   ┼──────────────► |Error| = |y - ŷ|
   0
```

<!-- IMAGE: V-shaped absolute value function centered at zero error -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | 0 (perfect predictions) |
| Maximum  | ∞ (unbounded) |
| Units    | Same as target variable |
| Optimal  | Lower is better |

MAE directly tells you the average magnitude of errors in the target's native units. An MAE of 2.3 kg means predictions are, on average, 2.3 kg away from the true value — no exponentiation or root extraction needed.

## When to Use

- **Outlier-contaminated data**: MAE treats all errors linearly, so a single extreme outlier contributes proportionally rather than quadratically. Preferred when the dataset has heavy-tailed noise or mislabeled samples.
- **Median estimation**: when the conditional median is a better central tendency than the conditional mean (e.g., skewed income distributions).
- **Interpretability**: MAE is the most intuitive error metric — average absolute deviation in original units.
- **Robust model selection**: for comparing models when you want a metric that is not dominated by a few worst-case predictions.

## When NOT to Use

- **When large errors must be penalized more**: if a 10-unit error is far worse than ten 1-unit errors, use [MSE](../regression/mse.md) or [RMSE](../regression/rmse.md).
- **Gradient-based optimization with strict smoothness requirements**: MAE has a discontinuous gradient at zero. Use [Huber Loss](../regression/huber_loss.md) as a smooth alternative.
- **When relative error matters**: MAE is scale-dependent and does not convey percentage deviation. Use [MAPE](../regression/mape.md) or [SMAPE](../regression/smape.md).
- **Bias–variance decomposition**: MSE decomposes cleanly into bias² + variance; MAE does not admit a similarly clean decomposition.

## What It Can Tell You

- The average magnitude of prediction errors in target units.
- A robust central-tendency measure of model error that is less influenced by outliers than RMSE.
- Whether model performance is acceptable relative to domain-specific tolerances.
- The relationship to RMSE: by Jensen's inequality, MAE ≤ RMSE. The gap between them indicates error dispersion — a large gap means a few samples have disproportionately large errors.

## What It Cannot Tell You

- Error direction (sign) — MAE is symmetric by design.
- Whether errors are concentrated in specific regions of the input space.
- Relative error — an MAE of 5 is excellent if the target range is [0, 10000] but poor if it is [0, 10].
- The variance of individual prediction errors; MAE summarizes only the first moment of the absolute error distribution.

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | Moderate. Each outlier contributes $|e_i| / n$, linear in error magnitude — much less impactful than $e_i^2 / n$ in MSE. |
| **Scale** | Directly proportional to target scale; normalize before cross-target comparison. |
| **Distribution shift** | MAE increases under drift but, like MSE, cannot distinguish bias from variance increase without decomposition. |
| **Sparsity** | For zero-inflated targets, MAE on the non-zero subset is often more informative than global MAE. |

## Alternatives & When to Prefer Them

| Metric | When to Prefer Over MAE |
|--------|------------------------|
| [MSE](../regression/mse.md) | When large errors need quadratic penalization; smoother optimization landscape. |
| [RMSE](../regression/rmse.md) | When you want outlier sensitivity but still in original units. |
| [Huber Loss](../regression/huber_loss.md) | When you need smooth gradients near zero error with linear tails for robustness. |
| [MAPE](../regression/mape.md) | When percentage error is more meaningful than absolute deviation. |
| [SMAPE](../regression/smape.md) | When relative error is needed and targets include values near zero. |
| [R²](../regression/r_squared.md) | When a unitless proportion-of-variance metric is needed. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])   # (4,)
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])     # (4,)

# --- torchmetrics (stateful, accumulates across batches) ---
mae_metric = torchmetrics.MeanAbsoluteError()
mae_metric.update(y_pred, y_true)
mae_value = mae_metric.compute()
print(f"MAE (torchmetrics): {mae_value.item():.6f}")
# Expected: mean([0.5, 0.5, 0.1, 0.8]) = 0.475

# --- Manual PyTorch computation ---
mae_manual = torch.mean(torch.abs(y_true - y_pred))
print(f"MAE (manual):       {mae_manual.item():.6f}")

# --- torch.nn L1Loss (usable as training objective) ---
criterion = torch.nn.L1Loss(reduction="mean")
mae_loss = criterion(y_pred, y_true)
print(f"MAE (nn.L1Loss):    {mae_loss.item():.6f}")
```

## Debugging Use Case

**Scenario — Median-like error assessment for skewed targets:**

You are predicting household income. The target distribution is right-skewed (median $55K, mean $78K). Validation RMSE = $42K, MAE = $18K.

1. **Interpret the gap**: RMSE is 2.3× MAE, indicating a heavy right tail in the error distribution — a few high-income households have very large prediction errors.
2. **Examine the outliers**: sort by absolute error descending. The top 2% of samples (incomes > $500K) account for 35% of total squared error but only 8% of total absolute error. These are luxury outliers.
3. **Switch evaluation metric**: MAE of $18K is more representative of model quality for the median household. Report MAE as the primary metric to stakeholders.
4. **Consider median regression**: if the goal is predicting the "typical" income, retrain with L1 loss (MAE) to target the conditional median, which is more robust to the skewed tail.
5. **Per-quantile analysis**: compute MAE for each income decile. If MAE is low for deciles 1–8 and spikes for 9–10, the model is excellent for most of the population and only struggles at the extreme.

## Related Metrics

- [MSE](../regression/mse.md) — quadratic penalty; targets conditional mean.
- [RMSE](../regression/rmse.md) — square root of MSE; same units, more outlier-sensitive.
- [Huber Loss](../regression/huber_loss.md) — smooth blend of MAE and MSE.
- [MAPE](../regression/mape.md) — percentage-based relative error.
- [SMAPE](../regression/smape.md) — symmetric percentage error.
- [R²](../regression/r_squared.md) — proportion of variance explained.
