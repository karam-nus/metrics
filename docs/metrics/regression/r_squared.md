---
title: "Coefficient of Determination (R²)"
---

# Coefficient of Determination (R²)

## Overview

R² (R-squared) measures the proportion of variance in the target variable that is explained by the model's predictions. It is defined as one minus the ratio of residual sum of squares to total sum of squares. An R² of 1 indicates perfect prediction, 0 indicates performance equal to predicting the mean, and negative values indicate worse-than-mean performance. R² is unitless and scale-invariant, making it ideal for comparing models across different targets or datasets. However, R² can be misleading in non-linear settings, is sensitive to the variance of the target, and can increase spuriously with more features (adjusted R² corrects for this in linear regression). R² is best understood as a relative measure: it tells you how much better the model is than the trivial mean-baseline, not how good the predictions are in absolute terms.

## Formula

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

where $\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i$ is the target mean, $SS_{\text{res}}$ is the residual sum of squares, and $SS_{\text{tot}}$ is the total sum of squares. Equivalently: $R^2 = 1 - \text{MSE} / \text{Var}(y)$ (using population variance with $1/n$).

## Visual Diagram

```
  R²
   ▲
 1 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─  Perfect model (SS_res = 0)
   │
   │     Useful models
   │
 0 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─  Mean-baseline (ŷ = ȳ)
   │
   │  Worse than mean
   │
   ┼──────────────────► Model complexity / fit quality
```

<!-- IMAGE: Scatter plots showing R²=0.95, R²=0.50, R²=0.0 (horizontal line), R²<0 -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | −∞ (arbitrarily bad predictions) |
| Maximum  | 1 (perfect predictions) |
| Units    | Dimensionless |
| Optimal  | Higher is better; 1 = perfect |

| R² Value | Interpretation |
|----------|---------------|
| 1.0      | Perfect fit — all variance explained |
| 0.9–1.0  | Excellent — model captures most variability |
| 0.7–0.9  | Good — substantial explanatory power |
| 0.5–0.7  | Moderate — useful but significant unexplained variance |
| 0.0–0.5  | Poor — barely better than the mean baseline |
| < 0      | Model is worse than simply predicting the target mean |

## When to Use

- **Unitless model comparison**: R² is independent of target scale, making it valid for cross-dataset and cross-target comparisons.
- **Baseline-relative evaluation**: R² directly answers "how much better is this model than predicting the mean?"
- **Feature importance assessment**: in linear models, R² quantifies the total variance explained; incremental R² from adding a feature measures its marginal contribution.
- **Quick model sanity check**: negative R² immediately flags a broken model or severe overfitting on train with no generalization.

## When NOT to Use

- **Non-linear models evaluated out-of-distribution**: R² can be negative even for a "reasonable" model if the test set has a different target distribution than training.
- **When absolute error magnitude matters**: R² does not tell you the size of errors. A model with R² = 0.99 can still have unacceptable absolute errors if target variance is large. Always pair R² with [RMSE](../regression/rmse.md) or [MAE](../regression/mae.md).
- **Comparing models with different numbers of features** (in linear regression): use adjusted R² to penalize model complexity.
- **Multi-output regression without careful aggregation**: R² per output can be misleading if outputs have very different variances.
- **Low-variance targets**: if the target has near-zero variance, R² becomes numerically unstable (division by near-zero $SS_{\text{tot}}$).

## What It Can Tell You

- The fraction of target variance captured by the model.
- Whether the model outperforms the trivial mean-prediction baseline (R² > 0).
- Relative ranking of models on the same evaluation set.
- A dimensionless summary that is comparable across targets measured in different units.

## What It Cannot Tell You

- The absolute magnitude of prediction errors — use [RMSE](../regression/rmse.md) or [MAE](../regression/mae.md).
- Whether the model is biased (R² can be high even if predictions are systematically shifted).
- Whether the model is overfitting (R² on training data is always optimistic; evaluate on held-out data).
- Distributional properties of errors — residual analysis is needed for that.

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | Highly sensitive. A single outlier in the target inflates $SS_{\text{tot}}$, which can artificially increase R². Conversely, an outlier in predictions inflates $SS_{\text{res}}$, decreasing R². |
| **Scale** | Scale-invariant — R² is unchanged by linear transformations of the target. |
| **Distribution shift** | If test-set target variance differs from training, R² can be misleadingly high or low. A high-variance test set inflates $SS_{\text{tot}}$, making R² appear higher. |
| **Sparsity** | For sparse targets with many zeros, $SS_{\text{tot}}$ is dominated by non-zero entries; R² may not reflect performance on the zero class. |

## Alternatives & When to Prefer Them

| Metric | When to Prefer Over R² |
|--------|------------------------|
| [MSE](../regression/mse.md) | When you need a differentiable training objective. |
| [RMSE](../regression/rmse.md) | When absolute error magnitude in original units is needed. |
| [MAE](../regression/mae.md) | When a robust, outlier-resistant absolute error measure is preferred. |
| [MAPE](../regression/mape.md) | When percentage error is more relevant than variance explained. |
| [SMAPE](../regression/smape.md) | When relative error with near-zero handling is required. |
| Adjusted R² | When comparing linear models with different numbers of predictors. |
| [Huber Loss](../regression/huber_loss.md) | When training under noisy labels with a robust loss. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])   # (4,)
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])     # (4,)

# --- torchmetrics R2Score ---
r2_metric = torchmetrics.R2Score()
r2_metric.update(y_pred, y_true)
r2_value = r2_metric.compute()
print(f"R² (torchmetrics): {r2_value.item():.6f}")
# SS_res = 0.25 + 0.25 + 0.01 + 0.64 = 1.15
# mean_y = 2.875
# SS_tot = (3-2.875)² + (-0.5-2.875)² + (2-2.875)² + (7-2.875)² = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
# R² = 1 - 1.15/29.1875 ≈ 0.960600

# --- Manual computation ---
ss_res = torch.sum((y_true - y_pred) ** 2)
ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
r2_manual = 1 - ss_res / ss_tot
print(f"R² (manual):       {r2_manual.item():.6f}")

# --- Demonstrating negative R² (worse than mean) ---
y_bad = torch.tensor([10.0, 10.0, 10.0, 10.0])  # (4,) — constant bad prediction
r2_bad = torchmetrics.R2Score()
r2_bad.update(y_bad, y_true)
print(f"R² (bad model):    {r2_bad.compute().item():.6f}")  # Negative
```

## Debugging Use Case

**Scenario — Checking if the model explains variance:**

You built a neural network for stock-return prediction. Training R² = 0.85, but test R² = −0.12.

1. **Interpret negative R²**: the model is worse than predicting the mean return on the test set. This is a strong signal of overfitting or distribution shift.
2. **Check for data leakage**: if training R² is suspiciously high for a noisy target like stock returns, verify that future information is not leaking into features (look-ahead bias).
3. **Compare target distributions**: compute $\text{Var}(y_{\text{train}})$ vs. $\text{Var}(y_{\text{test}})$. If the test set covers a different market regime (e.g., 2008 crisis), the target distribution has shifted.
4. **Compute supplementary metrics**: test RMSE = 0.034 vs. std(y_test) = 0.031. Since RMSE > std(y), predictions add noise rather than signal — confirming the negative R².
5. **Regularize and simplify**: reduce model complexity, add dropout, or switch to a simpler baseline. For stock returns, an R² of 0.01–0.05 out-of-sample is realistic; R² = 0.85 in-sample was almost certainly memorization.
6. **Use adjusted R²**: if using a linear model with many features, adjusted R² penalizes overfitting from feature proliferation.

## Related Metrics

- [MSE](../regression/mse.md) — $R^2 = 1 - \text{MSE}/\text{Var}(y)$; the numerator of R²'s ratio.
- [RMSE](../regression/rmse.md) — absolute error in target units; complements R²'s unitless summary.
- [MAE](../regression/mae.md) — robust absolute error metric.
- [MAPE](../regression/mape.md) — relative percentage error.
- [SMAPE](../regression/smape.md) — symmetric relative error.
- [Huber Loss](../regression/huber_loss.md) — robust training loss.
