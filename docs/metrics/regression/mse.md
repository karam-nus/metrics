---
title: "Mean Squared Error (MSE)"
---

# Mean Squared Error (MSE)

## Overview

Mean Squared Error is the arithmetic mean of the squared differences between predicted and actual values. It is the most widely used regression loss function and serves as the default optimization objective for least-squares regression. MSE is differentiable everywhere, convex, and has a unique global minimum, making it well-suited for gradient-based optimization. Because errors are squared before averaging, MSE disproportionately penalizes large deviations — a property that is desirable when large errors are particularly costly but problematic when the data contains outliers. MSE is expressed in squared units of the target variable, which can make direct interpretation less intuitive compared to metrics like RMSE or MAE.

## Formula

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where $y_i$ is the ground-truth value, $\hat{y}_i$ is the predicted value, and $n$ is the number of samples. In matrix form: $\text{MSE} = \frac{1}{n} \| \mathbf{y} - \hat{\mathbf{y}} \|_2^2$.

## Visual Diagram

```
  Loss
   ▲
   │          ╱
   │        ╱
   │      ╱        ← quadratic growth: doubling the
   │    ╱             error quadruples the loss
   │  ╱
   │╱
   ┼──────────────► Error (y - ŷ)
   0
```

<!-- IMAGE: Parabolic curve centered at zero error, illustrating quadratic penalty growth -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | 0 (perfect predictions) |
| Maximum  | ∞ (unbounded) |
| Units    | (target units)² |
| Optimal  | Lower is better |

An MSE of 0 indicates every prediction exactly matches the ground truth. Values grow quadratically with the magnitude of individual errors. Because the units are squared, MSE values are not directly comparable across targets measured in different units without normalization.

## When to Use

- **Default training loss** for regression models optimized via gradient descent; the smooth, convex surface guarantees stable convergence.
- **When large errors are disproportionately costly** — e.g., structural load prediction, financial risk estimation — because the quadratic penalty naturally up-weights big misses.
- **Variance-sensitive evaluation**: MSE decomposes as $\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$, making it the canonical metric for bias–variance analysis.
- **Comparing models under the same target scale** during hyperparameter sweeps or architecture searches.

## When NOT to Use

- **Outlier-contaminated data**: a single extreme error can dominate the aggregate MSE, masking otherwise good performance. Prefer [MAE](../regression/mae.md) or [Huber Loss](../regression/huber_loss.md).
- **When interpretable units matter**: squared units are unintuitive for stakeholders. Use [RMSE](../regression/rmse.md) instead.
- **Cross-scale comparison**: MSE is scale-dependent. Use [MAPE](../regression/mape.md) or [SMAPE](../regression/smape.md) for relative error.
- **Targets near zero with percentage semantics**: MSE gives no sense of relative magnitude.

## What It Can Tell You

- The average squared deviation of predictions from ground truth.
- Which model has lower overall error magnitude (under the same data and target scale).
- Whether training loss is converging (monotonic decrease in MSE across epochs).
- The bias–variance decomposition of your model's generalization error.

## What It Cannot Tell You

- The direction of errors (over- vs. under-prediction); MSE is symmetric.
- How large a "typical" error is in the original units — take the square root to get [RMSE](../regression/rmse.md).
- Whether errors are uniformly distributed or concentrated in a few samples.
- Relative error magnitude — a 10-unit MSE means very different things for targets in [0, 10] vs. [0, 10000].

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | Extremely sensitive. A single outlier with error $e$ contributes $e^2 / n$, which can dominate the mean. |
| **Scale** | Directly proportional to the square of the target scale. Always normalize or standardize before comparing MSE across different targets. |
| **Distribution shift** | MSE will increase under covariate or concept drift, but does not distinguish between bias shift and variance increase without further decomposition. |
| **Sparsity** | In sparse regression (many zero targets), MSE is dominated by the non-zero entries if their magnitudes are large; zero-inflated metrics may be more appropriate. |

## Alternatives & When to Prefer Them

| Metric | When to Prefer Over MSE |
|--------|------------------------|
| [RMSE](../regression/rmse.md) | When you need error in the original target units for interpretability. |
| [MAE](../regression/mae.md) | When outlier robustness is required; MAE penalizes all errors linearly. |
| [Huber Loss](../regression/huber_loss.md) | When you want quadratic sensitivity for small errors but linear for large (outlier-robust, still differentiable). |
| [MAPE](../regression/mape.md) | When relative (percentage) error matters more than absolute magnitude. |
| [SMAPE](../regression/smape.md) | When relative error is needed and targets can be near zero. |
| [R²](../regression/r_squared.md) | When you want a unitless proportion-of-variance-explained measure. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])   # (4,)
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])     # (4,)

# --- torchmetrics (stateful, supports accumulation across batches) ---
mse_metric = torchmetrics.MeanSquaredError()
mse_metric.update(y_pred, y_true)
mse_value = mse_metric.compute()
print(f"MSE (torchmetrics): {mse_value.item():.6f}")
# Expected: mean([0.25, 0.25, 0.01, 0.64]) = 0.2875

# --- PyTorch functional (stateless, single-batch) ---
mse_manual = torch.mean((y_true - y_pred) ** 2)
print(f"MSE (manual):       {mse_manual.item():.6f}")

# --- torch.nn loss (useful as training objective) ---
criterion = torch.nn.MSELoss(reduction="mean")
mse_loss = criterion(y_pred, y_true)
print(f"MSE (nn.MSELoss):   {mse_loss.item():.6f}")
```

## Debugging Use Case

**Scenario — Monitoring training loss convergence:**

You are training a feed-forward network for house-price prediction. After 50 epochs the training MSE plateaus at 1.2 × 10⁹ while validation MSE oscillates.

1. **Check the loss surface**: plot per-epoch MSE. A plateau with oscillation suggests the learning rate is too high for fine-grained convergence — reduce by 0.3–0.5×.
2. **Inspect per-sample squared errors**: sort samples by $(y_i - \hat{y}_i)^2$. If the top 1% of errors contribute >50% of total MSE, a handful of outlier houses (e.g., luxury estates) are dominating. Consider switching to [Huber Loss](../regression/huber_loss.md) or clipping targets.
3. **Bias–variance check**: compute MSE on bootstrap resamples. If variance across resamples is high relative to the mean, the model is overfitting; add regularization. If bias dominates, increase model capacity.
4. **Unit sanity**: MSE of 1.2 × 10⁹ in dollars² corresponds to an RMSE ≈ $34,641 — compare that against the target range to judge if the error is acceptable.

## Related Metrics

- [RMSE](../regression/rmse.md) — square root of MSE; same units as target.
- [MAE](../regression/mae.md) — linear penalty; more robust to outliers.
- [Huber Loss](../regression/huber_loss.md) — piecewise combination of MSE and MAE.
- [R²](../regression/r_squared.md) — normalized version: $R^2 = 1 - \text{MSE} / \text{Var}(y)$.
- [MAPE](../regression/mape.md) — percentage-based relative error.
- [SMAPE](../regression/smape.md) — symmetric percentage error.
