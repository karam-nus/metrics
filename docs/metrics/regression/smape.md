---
title: "Symmetric Mean Absolute Percentage Error (SMAPE)"
---

# Symmetric Mean Absolute Percentage Error (SMAPE)

## Overview

Symmetric Mean Absolute Percentage Error addresses the two major weaknesses of MAPE: asymmetry between over- and under-predictions, and instability when actual values are zero. SMAPE achieves this by normalizing the absolute error by the average of the absolute actual and absolute predicted values rather than by the actual alone. The result is a bounded metric in [0, 200%] (or [0, 100%] depending on convention) that treats positive and negative errors more symmetrically. SMAPE is widely used in forecasting competitions (e.g., the M-competitions) and time-series evaluation. However, SMAPE is not perfectly symmetric — it still exhibits mild asymmetry — and its behavior when both actual and predicted are zero requires careful handling (conventionally defined as 0% error in that case).

## Formula

$$
\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{2 \, |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}
$$

When $y_i = \hat{y}_i = 0$, the term is defined as 0 by convention. The factor of 2 in the numerator ensures the range is [0, 200%]. Some implementations omit the factor of 2, giving a range of [0, 100%] — always check the convention being used.

## Visual Diagram

```
  SMAPE per sample (%)
   ▲
200├─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Upper bound
   │         ╱ ╲
   │       ╱     ╲       ← Bounded: unlike MAPE,
   │     ╱         ╲       cannot exceed 200%
100├── ╱─ ─ ─ ─ ─ ─ ╲──
   │ ╱                 ╲
   │╱                    ╲
  0┼──────────────────────► ŷ (for fixed y > 0)
   0            y         2y

  At ŷ=0: SMAPE = 200% | At ŷ=y: SMAPE = 0% | At ŷ=2y: SMAPE = 66.7%
```

<!-- IMAGE: SMAPE value as a function of predicted value for a fixed positive actual, showing bounded behavior -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | 0% (perfect predictions) |
| Maximum  | 200% (maximum possible deviation) |
| Units    | Percentage (%) |
| Optimal  | Lower is better |

The 200% upper bound occurs when one of $y_i$ or $\hat{y}_i$ is zero while the other is non-zero: $2|a| / |a| = 2 = 200\%$. In practice, SMAPE values above 100% are rare and indicate severe prediction failures.

## When to Use

- **Forecasting with near-zero actuals**: SMAPE remains finite when $y_i = 0$ (as long as $\hat{y}_i \neq 0$, giving 200%, or both are zero, giving 0%), unlike [MAPE](../regression/mape.md) which is undefined.
- **Bounded relative error**: the [0, 200%] range makes SMAPE easier to aggregate and compare than unbounded MAPE.
- **Competition and benchmarking**: SMAPE is the primary metric in the M3, M4, and M5 forecasting competitions.
- **Cross-scale comparison**: like MAPE, SMAPE is scale-independent, enabling comparison across series of different magnitudes.

## When NOT to Use

- **When perfect symmetry is required**: despite its name, SMAPE is not fully symmetric — $\text{SMAPE}(y=100, \hat{y}=150) \neq \text{SMAPE}(y=150, \hat{y}=100)$ because the denominator changes. The asymmetry is mild but present.
- **When absolute error magnitude matters**: SMAPE obscures the actual size of errors. Use [MAE](../regression/mae.md) or [RMSE](../regression/rmse.md).
- **When both $y_i$ and $\hat{y}_i$ can be zero frequently**: the 0/0 convention (defined as 0%) is arbitrary and inflates apparent accuracy.
- **As a training loss**: the gradient of SMAPE is complex and non-convex; it is an evaluation metric, not an optimization objective. Train with [MSE](../regression/mse.md) or [Huber Loss](../regression/huber_loss.md) and evaluate with SMAPE.

## What It Can Tell You

- The average relative deviation of predictions from actuals, bounded and more stable than MAPE.
- Comparative accuracy across time series or products with different magnitude scales.
- Whether forecasts degrade at specific horizons (plot SMAPE vs. forecast step).

## What It Cannot Tell You

- The absolute magnitude of errors — 10% SMAPE on a $1B forecast vs. a $100 forecast are vastly different in impact.
- Error direction — SMAPE uses absolute values.
- Distributional properties of errors — it is a single-point summary.
- Optimal model parameters — SMAPE is not designed for gradient-based optimization.

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | More controlled than MAPE because the denominator grows with large predictions, dampening the percentage. However, near-zero denominators still cause inflated values. |
| **Scale** | Scale-invariant by design — its primary advantage. |
| **Distribution shift** | SMAPE increases under drift, but the bounded range limits extreme values compared to MAPE. |
| **Sparsity** | Better than MAPE for zero actuals (defined as 200% or 0% depending on the prediction), but systematic near-zero values still inflate the average. |
| **Convention** | Critical: the factor-of-2 convention (range [0, 200%]) vs. no factor (range [0, 100%]) changes absolute values. Always report which convention is used. |

## Alternatives & When to Prefer Them

| Metric | When to Prefer Over SMAPE |
|--------|--------------------------|
| [MAPE](../regression/mape.md) | When actuals are guaranteed non-zero and the industry standard requires MAPE (e.g., regulatory reporting). |
| [MAE](../regression/mae.md) | When absolute error in original units is needed, not relative error. |
| [RMSE](../regression/rmse.md) | When large absolute errors need to be highlighted. |
| [MSE](../regression/mse.md) | When a differentiable training loss is needed. |
| [R²](../regression/r_squared.md) | When a unitless variance-explained metric is preferred over percentage error. |
| [Huber Loss](../regression/huber_loss.md) | For robust training; evaluate with SMAPE afterward. |
| MASE (Mean Absolute Scaled Error) | When comparing across series with different scales and you want a metric normalized by the naïve forecast error. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([100.0, 50.0, 0.0, 80.0])    # (4,) — includes a zero!
y_pred = torch.tensor([110.0, 45.0, 5.0, 78.0])     # (4,)

# --- torchmetrics SymmetricMeanAbsolutePercentageError ---
smape_metric = torchmetrics.SymmetricMeanAbsolutePercentageError()
smape_metric.update(y_pred, y_true)
smape_value = smape_metric.compute()
print(f"SMAPE (torchmetrics): {smape_value.item():.6f}")
# Per-sample: 2*|y-ŷ|/(|y|+|ŷ|)
# [2*10/210, 2*5/95, 2*5/5, 2*2/158]
# = [0.09524, 0.10526, 2.0, 0.02532]
# Mean = 0.55645 (i.e., ~55.6%)
# Note: torchmetrics returns as a fraction, not percentage

# --- Manual PyTorch computation ---
numerator = 2.0 * torch.abs(y_true - y_pred)
denominator = torch.abs(y_true) + torch.abs(y_pred)
# Handle 0/0 case: where both y and ŷ are zero, SMAPE = 0
smape_per_sample = torch.where(
    denominator == 0,
    torch.zeros_like(numerator),
    numerator / denominator
)
smape_manual = torch.mean(smape_per_sample)
print(f"SMAPE (manual):       {smape_manual.item():.6f}")

# Convert to percentage (0-200% scale)
print(f"SMAPE (%):            {smape_manual.item() * 100:.4f}%")

# --- Demonstrate boundedness ---
y_extreme = torch.tensor([100.0])
y_pred_zero = torch.tensor([0.0])
smape_max = 2 * torch.abs(y_extreme - y_pred_zero) / (torch.abs(y_extreme) + torch.abs(y_pred_zero))
print(f"SMAPE max (y=100, ŷ=0): {smape_max.item() * 100:.1f}%")  # 200.0%
```

## Debugging Use Case

**Scenario — Forecasting evaluation with intermittent demand:**

You are evaluating a demand-forecasting model for a retailer. Many SKUs have intermittent demand (zero sales on most days, occasional spikes).

1. **MAPE fails**: 3,000 out of 10,000 daily observations have $y_i = 0$. MAPE is undefined for these, and excluding them biases the metric toward active-demand days.
2. **Switch to SMAPE**: SMAPE handles $y_i = 0$ — when $\hat{y}_i > 0$, SMAPE = 200% (maximum penalty for predicting demand when there is none). When $y_i = \hat{y}_i = 0$, SMAPE = 0% (correct no-demand prediction).
3. **Diagnose by segment**: active-demand days have SMAPE = 18% (good), but zero-demand days where the model predicts > 0 contribute SMAPE = 200% each, pulling the average to 65%.
4. **Hybrid approach**: decompose the problem — use a binary classifier for "demand vs. no demand" and SMAPE only on predicted-demand days. Overall accuracy combines classification accuracy and regression SMAPE.
5. **Compare conventions**: the M5 competition uses the 2× convention (0–200%). Your internal dashboard uses the 1× convention (0–100%). Report: "SMAPE = 32.7% (using 2× convention, range 0–200%)."
6. **Cross-series comparison**: SMAPE for high-volume SKUs averages 12%, while low-volume SKUs average 85%. This confirms that the model needs a different architecture or training strategy for sparse/intermittent demand.

## Related Metrics

- [MAPE](../regression/mape.md) — asymmetric, unbounded predecessor; undefined at zero.
- [MAE](../regression/mae.md) — absolute error in original units.
- [RMSE](../regression/rmse.md) — quadratic penalty in original units.
- [MSE](../regression/mse.md) — standard differentiable training loss.
- [R²](../regression/r_squared.md) — unitless proportion of variance explained.
- [Huber Loss](../regression/huber_loss.md) — robust training objective for noisy data.
