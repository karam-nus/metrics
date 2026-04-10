---
title: "Mean Absolute Percentage Error (MAPE)"
---

# Mean Absolute Percentage Error (MAPE)

## Overview

Mean Absolute Percentage Error expresses prediction error as a percentage of the actual value, providing a scale-independent measure of accuracy. MAPE is ubiquitous in demand forecasting, supply-chain planning, and financial modeling because stakeholders intuitively understand percentage errors. However, MAPE has well-known pathologies: it is undefined when any actual value is zero, it is asymmetric (over-predictions are penalized less than under-predictions of the same absolute magnitude), and it is unbounded above. These limitations have driven the development of alternatives like [SMAPE](../regression/smape.md), but MAPE remains the industry-standard percentage-error metric in many domains.

## Formula

$$
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

where $y_i \neq 0$ for all $i$. Each term $|y_i - \hat{y}_i| / |y_i|$ represents the relative error for sample $i$.

## Visual Diagram

```
  Actual y=100         Actual y=10
  ┌────────────┐       ┌────────────┐
  │ Pred: 110  │       │ Pred: 11   │
  │ |Error|=10 │       │ |Error|=1  │
  │ MAPE =10%  │       │ MAPE =10%  │
  └────────────┘       └────────────┘
  Same absolute error → same MAPE when relative magnitude is equal.
  Different absolute errors can yield equal MAPE values.
```

<!-- IMAGE: Bar chart comparing absolute vs. percentage errors across different target scales -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | 0% (perfect predictions) |
| Maximum  | ∞ (unbounded; exceeds 100% when prediction deviates by more than the actual) |
| Units    | Percentage (%) |
| Optimal  | Lower is better |

Common benchmarks in forecasting: MAPE < 10% is considered highly accurate, 10–20% is good, 20–50% is reasonable, and > 50% is inaccurate — though these thresholds are domain-dependent.

## When to Use

- **Cross-scale evaluation**: comparing accuracy across products, regions, or time series with different magnitudes (e.g., SKU-level demand forecasting).
- **Stakeholder communication**: percentage errors are universally understood by non-technical audiences.
- **When all actual values are safely non-zero** and the target distribution does not approach zero.
- **Industry-standard reporting**: MAPE is the de facto metric in supply chain, energy forecasting, and retail demand planning.

## When NOT to Use

- **Actual values at or near zero**: MAPE is undefined at $y_i = 0$ and explodes as $y_i \to 0$. Use [SMAPE](../regression/smape.md) or [MAE](../regression/mae.md).
- **Intermittent demand / sparse data**: zero-demand periods make MAPE unusable. Consider scaled errors or coverage metrics.
- **When symmetric treatment of over/under-prediction is needed**: MAPE penalizes under-prediction more heavily. For $y=100$, pred=50 gives 50% error, but pred=150 also gives 50% — yet for $y=50$, pred=100 gives 100% while pred=0 gives 100%. The asymmetry arises because the denominator is fixed at $|y|$.
- **Optimization objective**: MAPE's gradient is $\text{sign}(\hat{y} - y) / |y|$, which is constant in $\hat{y}$ and can cause slow convergence in gradient-based methods.

## What It Can Tell You

- The average relative magnitude of prediction errors as a percentage.
- Whether accuracy is consistent across items of different scales.
- How the model's percentage accuracy changes over time (trend in MAPE across forecast horizons).

## What It Cannot Tell You

- Absolute error magnitude — 10% MAPE on a $1M forecast and a $100 forecast have very different dollar impacts.
- Error direction (over- vs. under-prediction); MAPE is absolute.
- Performance on zero or near-zero actuals — these samples are excluded or cause division-by-zero.
- Whether errors are driven by bias or variance.

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | Moderate for large actuals (error is divided by a large denominator), but extreme for small actuals (division by near-zero amplifies even tiny absolute errors). |
| **Scale** | Scale-invariant by design — that is its primary advantage. |
| **Distribution shift** | MAPE increases under drift, but the percentage framing can mask large absolute errors on high-value items. |
| **Sparsity** | Catastrophic. Any zero actual value makes MAPE undefined. Near-zero actuals produce arbitrarily large percentage errors that dominate the average. |

## Alternatives & When to Prefer Them

| Metric | When to Prefer Over MAPE |
|--------|-------------------------|
| [SMAPE](../regression/smape.md) | When actuals can be zero or near-zero; provides bounded [0, 200%] range. |
| [MAE](../regression/mae.md) | When absolute (not relative) error is the appropriate measure. |
| [RMSE](../regression/rmse.md) | When large absolute errors need quadratic penalization. |
| [MSE](../regression/mse.md) | When used as a differentiable training loss. |
| [R²](../regression/r_squared.md) | When a unitless variance-explained metric is needed without percentage-error semantics. |
| [Huber Loss](../regression/huber_loss.md) | When training with noisy labels and you need a robust, differentiable objective. |
| Weighted MAPE (WMAPE) | When you want to weight each sample's contribution by its actual value, reducing the influence of small-actual outliers. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([100.0, 50.0, 30.0, 80.0])   # (4,) — no zeros!
y_pred = torch.tensor([110.0, 45.0, 33.0, 78.0])    # (4,)

# --- torchmetrics ---
mape_metric = torchmetrics.MeanAbsolutePercentageError()
mape_metric.update(y_pred, y_true)
mape_value = mape_metric.compute()
print(f"MAPE (torchmetrics): {mape_value.item():.6f}")
# Per-sample: [10/100, 5/50, 3/30, 2/80] = [0.1, 0.1, 0.1, 0.025]
# Mean = 0.08125 (i.e., 8.125%)

# --- Manual PyTorch computation ---
mape_manual = torch.mean(torch.abs((y_true - y_pred) / y_true))
print(f"MAPE (manual):       {mape_manual.item():.6f}")

# Convert to percentage
print(f"MAPE (%):            {mape_manual.item() * 100:.4f}%")
```

## Debugging Use Case

**Scenario — Relative error across different scales in demand forecasting:**

You forecast daily demand for 10,000 SKUs. Global MAE = 12 units, but stakeholders complain that "small SKUs have terrible forecasts."

1. **Compute per-SKU MAPE**: SKUs with demand 1–5 units/day have MAPE > 80%, while SKUs with demand > 100 units/day have MAPE < 8%. A flat MAE hides this disparity.
2. **Identify the problem**: for a SKU with actual demand = 2, a prediction of 4 yields MAPE = 100% but absolute error = 2 — perfectly reasonable in absolute terms but catastrophic in percentage terms.
3. **Stratify reporting**: report MAPE by demand tier (low / medium / high volume). This gives stakeholders a more nuanced view.
4. **Handle zeros**: 1,200 SKUs had zero demand on certain days. Exclude these from MAPE computation and report them separately (e.g., as a classification accuracy for zero vs. non-zero demand).
5. **Consider alternatives**: switch to [SMAPE](../regression/smape.md) for a bounded metric, or use Weighted MAPE (WMAPE = total |error| / total |actual|) to down-weight low-volume SKUs proportionally.

## Related Metrics

- [SMAPE](../regression/smape.md) — symmetric, bounded alternative for near-zero actuals.
- [MAE](../regression/mae.md) — absolute error without percentage normalization.
- [RMSE](../regression/rmse.md) — quadratic penalty in original units.
- [MSE](../regression/mse.md) — squared error, standard training loss.
- [R²](../regression/r_squared.md) — unitless variance-explained measure.
- [Huber Loss](../regression/huber_loss.md) — robust loss for training under noise.
