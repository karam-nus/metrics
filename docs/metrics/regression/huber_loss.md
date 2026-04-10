---
title: "Huber Loss"
---

# Huber Loss

## Overview

Huber Loss is a piecewise-defined loss function that behaves quadratically (like MSE) for small errors and linearly (like MAE) for large errors, with a transition controlled by the hyperparameter δ (delta). This hybrid design makes Huber Loss differentiable everywhere (unlike MAE) while being robust to outliers (unlike MSE). It was introduced by Peter Huber in 1964 for robust estimation and has become a standard choice for training regression models on noisy data. The delta parameter is the critical knob: small δ makes Huber behave like MAE (more robust, less efficient under Gaussian noise), while large δ makes it behave like MSE (more efficient under Gaussian, less robust). Selecting δ typically requires cross-validation or domain knowledge about the expected noise level.

## Formula

$$
L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{if } |y - \hat{y}| > \delta
\end{cases}
$$

The mean Huber Loss over $n$ samples:

$$
\text{Huber} = \frac{1}{n} \sum_{i=1}^{n} L_\delta(y_i, \hat{y}_i)
$$

The function is continuous and differentiable everywhere, including at $|y - \hat{y}| = \delta$.

## Visual Diagram

```
  Loss
   ▲
   │          ╱
   │        ╱       ← linear region (slope = δ)
   │      ╱
   │    ·╱
   │  ·· ·         ← quadratic region (smooth curve)
   │··     ··
   ┼───────────────► |Error| = |y - ŷ|
   0       δ
           ↑
     transition point
```

<!-- IMAGE: Piecewise function showing parabolic center and linear tails, with delta marked at the transition -->

## Range & Interpretation

| Property | Value |
|----------|-------|
| Minimum  | 0 (perfect predictions) |
| Maximum  | ∞ (unbounded, but grows linearly, not quadratically) |
| Units    | (target units)² for small errors, target units × δ for large errors |
| Optimal  | Lower is better |
| Key parameter | δ (delta) — controls the quadratic-to-linear transition |

For errors within δ, Huber Loss equals $\frac{1}{2}e^2$ (identical to half-MSE). For errors beyond δ, it equals $\delta|e| - \frac{1}{2}\delta^2$, growing linearly with slope δ.

## When to Use

- **Training with noisy labels or outliers**: Huber Loss limits the influence of large errors while retaining the gradient properties of MSE for small errors. Ideal when you suspect label noise but cannot clean the data.
- **When MAE's non-differentiability at zero is problematic**: Huber is smooth everywhere, producing more stable gradients near zero error.
- **Reinforcement learning**: Huber Loss (often called "smooth L1") is standard in DQN and actor-critic value estimation to stabilize training.
- **Object detection**: Smooth L1 loss (a variant of Huber with δ=1) is the standard bounding-box regression loss in Faster R-CNN and SSD.

## When NOT to Use

- **Clean data with Gaussian noise**: plain MSE is statistically optimal (minimum variance unbiased estimator) when errors are truly Gaussian. Huber Loss sacrifices efficiency for robustness you don't need.
- **When you need strict L1 behavior**: if all errors should be penalized linearly regardless of magnitude, use [MAE](../regression/mae.md) directly.
- **When δ tuning is impractical**: Huber Loss introduces an additional hyperparameter. If you cannot afford the tuning cost, MSE or MAE are simpler defaults.
- **When percentage error is the evaluation criterion**: Huber Loss is scale-dependent. Use [MAPE](../regression/mape.md) or [SMAPE](../regression/smape.md).

## What It Can Tell You

- A robust measure of average prediction error that balances sensitivity to small errors with resilience to large ones.
- Whether training converges more stably compared to MSE on noisy data (monitor Huber loss curves vs. MSE loss curves).
- The effective "noise threshold" of the data, as reflected in the optimal δ found via cross-validation.

## What It Cannot Tell You

- The absolute error in interpretable units — Huber Loss is a mixed-unit quantity (squared for small, linear for large).
- Whether errors are over- or under-predictions — Huber Loss is symmetric.
- Relative (percentage) error — it is scale-dependent.
- The proportion of variance explained — use [R²](../regression/r_squared.md) for that.

## Sensitivity

| Factor | Impact |
|--------|--------|
| **Outliers** | Controlled. Errors beyond δ contribute linearly (slope = δ), not quadratically. The larger the outlier, the more savings relative to MSE. |
| **Scale** | Dependent on δ and target scale. δ should be set relative to the expected error distribution (e.g., δ ≈ 1.345σ for 95% asymptotic efficiency under Gaussian noise). |
| **Distribution shift** | More resilient than MSE because the linear tail limits the influence of novel large errors. |
| **Sparsity** | Same considerations as MSE/MAE; Huber Loss does not inherently handle zero-inflated targets. |
| **Delta (δ)** | Critical hyperparameter. Too small → undertrained (all errors in linear regime, weak gradients). Too large → effectively MSE, losing robustness. |

## Alternatives & When to Prefer Them

| Metric / Loss | When to Prefer Over Huber Loss |
|---------------|-------------------------------|
| [MSE](../regression/mse.md) | Clean Gaussian data; no outliers; maximum statistical efficiency. |
| [MAE](../regression/mae.md) | Maximum outlier robustness; when strict L1 penalty is desired. |
| [RMSE](../regression/rmse.md) | For evaluation/reporting in original units (not as a training loss). |
| Log-Cosh Loss | Smooth approximation of MAE without the δ hyperparameter; twice differentiable everywhere. |
| Quantile Loss | When predicting specific quantiles (e.g., median, 90th percentile) rather than conditional mean. |
| [MAPE](../regression/mape.md) | When relative error is the primary concern. |
| [R²](../regression/r_squared.md) | When a unitless evaluation metric is needed. |

## Code Example

```python
import torch
import torchmetrics

# Ground truth and predictions — shape: (batch_size,)
y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])   # (4,)
y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8])     # (4,)

# --- torch.nn.HuberLoss (default delta=1.0) ---
criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)
huber_value = criterion(y_pred, y_true)
print(f"Huber Loss (delta=1.0): {huber_value.item():.6f}")
# Errors: [0.5, 0.5, 0.1, 0.8]
# All |e| <= 1.0, so all in quadratic regime: 0.5 * e^2
# Losses: [0.125, 0.125, 0.005, 0.320]
# Mean = 0.14375

# --- Custom delta ---
criterion_small = torch.nn.HuberLoss(reduction="mean", delta=0.3)
huber_small = criterion_small(y_pred, y_true)
print(f"Huber Loss (delta=0.3): {huber_small.item():.6f}")
# Errors: [0.5, 0.5, 0.1, 0.8] — only 0.1 is in quadratic regime
# 0.5: 0.3*0.5 - 0.5*0.09 = 0.105
# 0.1: 0.5*0.01 = 0.005
# 0.8: 0.3*0.8 - 0.5*0.09 = 0.195
# Mean = (0.105 + 0.105 + 0.005 + 0.195) / 4 = 0.1025

# --- Manual implementation ---
def huber_loss_manual(y, y_hat, delta=1.0):
    error = torch.abs(y - y_hat)
    quadratic = 0.5 * error ** 2
    linear = delta * error - 0.5 * delta ** 2
    return torch.mean(torch.where(error <= delta, quadratic, linear))

huber_manual = huber_loss_manual(y_true, y_pred, delta=1.0)
print(f"Huber Loss (manual):    {huber_manual.item():.6f}")

# --- Using in training loop (sketch) ---
# model = MyRegressionModel()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = torch.nn.HuberLoss(delta=1.0)
# for batch in dataloader:
#     pred = model(batch.x)                    # (B,)
#     loss = criterion(pred, batch.y)           # scalar
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
```

## Debugging Use Case

**Scenario — Training with noisy labels:**

You are training a regression model to predict housing prices. The dataset contains ~5% mislabeled samples (e.g., typographic errors where $500K becomes $5M). Training with MSE diverges or converges to a poor local minimum.

1. **Diagnose the problem**: plot per-sample MSE. The top 5% of samples contribute 60% of total loss. These are the mislabeled outliers driving gradients.
2. **Switch to Huber Loss**: set δ = 1.0 (after standardizing targets to zero mean, unit variance). Errors beyond 1σ now contribute linearly instead of quadratically.
3. **Tune δ via cross-validation**: try δ ∈ {0.5, 1.0, 1.345, 2.0}. δ = 1.345 provides 95% asymptotic efficiency relative to MSE under Gaussian noise. If validation loss is best at δ = 0.5, the data has heavier-than-Gaussian tails.
4. **Compare training dynamics**: Huber Loss shows monotonic decrease and stable convergence, while MSE showed erratic spikes when outlier-heavy batches were sampled.
5. **Post-training evaluation**: report [RMSE](../regression/rmse.md) and [MAE](../regression/mae.md) on a clean validation set (with verified labels). The Huber-trained model achieves RMSE = $28K vs. $45K for the MSE-trained model.
6. **Consider data cleaning**: use the Huber-trained model to identify outliers (samples with loss > 3δ), manually inspect them, and retrain on cleaned data with MSE for maximum efficiency.

## Related Metrics

- [MSE](../regression/mse.md) — Huber's quadratic regime; optimal under Gaussian noise.
- [MAE](../regression/mae.md) — Huber's linear regime; maximum outlier robustness.
- [RMSE](../regression/rmse.md) — evaluation metric in original units.
- [R²](../regression/r_squared.md) — unitless variance-explained measure.
- [MAPE](../regression/mape.md) — relative percentage error.
- [SMAPE](../regression/smape.md) — symmetric relative error.
