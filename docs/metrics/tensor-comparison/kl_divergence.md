---
title: "Kullback-Leibler Divergence (KL Divergence)"
---

# Kullback-Leibler Divergence (KL Divergence)

## Overview

Kullback-Leibler Divergence measures how one probability distribution diverges from a reference distribution. In tensor comparison for model debugging, it quantifies how much the output distribution of a modified model (quantized, pruned, distilled) differs from the baseline. Unlike element-wise metrics (RMSE, MAE), KL divergence operates on *distributions* — it captures shifts in probability mass, not individual element errors. This makes it the right metric for comparing softmax outputs, attention distributions, or any tensor that represents a probability distribution. KL divergence is non-symmetric: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.

## Formula

**Discrete case:**

$$
D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

**Continuous case:**

$$
D_{KL}(P \| Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
$$

**For tensors representing log-probabilities (PyTorch convention):**

$$
D_{KL}(P \| Q) = \sum_{i} P_i \cdot (\log P_i - \log Q_i)
$$

where $P$ is the target (reference) distribution and $Q$ is the approximation.

**Aliases:** KL distance (informal, technically not a distance), relative entropy, information gain.

**Symmetric variant (Jensen-Shannon Divergence):**

$$
D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M), \quad M = \frac{P + Q}{2}
$$

## Visual Diagram

```
Baseline softmax P:   [0.70, 0.20, 0.05, 0.03, 0.02]  ← FP32 model
Quantized softmax Q:  [0.65, 0.22, 0.06, 0.04, 0.03]  ← INT8 model

KL(P||Q) = 0.70·log(0.70/0.65) + 0.20·log(0.20/0.22) + ...
         = 0.70·0.074 + 0.20·(-0.095) + 0.05·(-0.182) + 0.03·(-0.288) + 0.02·(-0.405)
         = 0.0518 - 0.0190 - 0.0091 - 0.0086 - 0.0081
         = 0.007 nats

         P: ████████████████████████████████████   70%
         Q: ██████████████████████████████████     65%  ← mass shifted away
              ↑ KL measures the "cost" of this shift in information-theoretic terms
```

<!-- IMAGE: Side-by-side bar charts of P and Q distributions with KL divergence annotated as area between them -->

## Range & Interpretation

| KL Divergence    | Interpretation                                             |
|------------------|------------------------------------------------------------|
| 0.0              | Identical distributions                                    |
| < 0.001          | Negligible divergence — distributions are effectively equal |
| 0.001 – 0.01    | Very small — minor distribution shift                      |
| 0.01 – 0.1      | Small — noticeable difference, check accuracy impact       |
| 0.1 – 1.0       | Moderate — significant distributional change               |
| > 1.0            | Large — distributions are substantially different          |
| ∞                | P has support where Q is zero (Q assigns zero probability) |

**Note:** KL divergence is in **nats** when using natural log, **bits** when using log₂. PyTorch's `kl_div` uses natural log by default.

## When to Use

- **Softmax output comparison:** Comparing the output probability distributions of a quantized/pruned model against the baseline.
- **Attention distribution analysis:** Checking whether quantized attention patterns match the original.
- **Knowledge distillation:** KL divergence between teacher and student softmax outputs is the standard distillation loss.
- **Output distribution monitoring:** Detecting distributional drift in model outputs over time or across configurations.
- **Calibration analysis:** Measuring whether predicted probabilities match empirical frequencies.

## When NOT to Use

- **Non-probability tensors:** KL divergence requires proper probability distributions (non-negative, sum to 1). For raw activations or weights, use RMSE, cosine similarity, or SQNR.
- **Q has zeros where P is non-zero:** KL divergence is infinite. This happens with clipped or heavily quantized distributions. Use label smoothing, epsilon-smoothing, or Jensen-Shannon divergence.
- **Symmetric comparison needed:** $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$. If you need symmetry, use Jensen-Shannon divergence or compute both directions.
- **Element-wise comparison:** KL divergence is a distribution-level metric, not element-level. Use allclose for element-wise tolerance checks.
- **The tensors don't represent distributions:** Forcing non-distribution tensors through softmax to compute KL divergence loses information.

## What It Can Tell You

- Whether a transformation has shifted the output distribution of a model.
- The information-theoretic cost of using the approximate distribution instead of the true one.
- Which output classes have gained or lost probability mass.
- Whether attention patterns are preserved after quantization.

## What It Cannot Tell You

- Element-wise error magnitudes (it operates on distributions, not elements).
- Whether the argmax (predicted class) has changed — small KL can still flip predictions near decision boundaries.
- Directional alignment of internal representations (use cosine similarity).
- Absolute numerical deviation (use RMSE/MAE).
- Whether the error is acceptable for a specific downstream task.

## Sensitivity

- **Outliers in probability:** Highly sensitive to near-zero probabilities. If $Q(x) \to 0$ where $P(x) > 0$, KL divergence diverges to infinity. Always apply epsilon-smoothing.
- **Scale:** Not applicable — inputs must be valid probability distributions.
- **Temperature:** Softmax temperature dramatically affects KL divergence. Higher temperature spreads mass, reducing KL divergence between similar distributions. Lower temperature concentrates mass, amplifying small differences.
- **Vocabulary/class size:** Larger output spaces tend to have smaller per-token KL divergence because probability mass is spread more thinly.
- **Asymmetry:** $D_{KL}(P\|Q)$ penalizes Q for missing P's mass (mode-dropping). $D_{KL}(Q\|P)$ penalizes Q for placing mass where P does not (hallucination). Choose direction based on what matters.

## Alternatives & When to Prefer Them

| Metric                  | Prefer When                                                |
|-------------------------|------------------------------------------------------------|
| Jensen-Shannon Div.     | Need a symmetric, bounded [0, ln2] divergence              |
| Cross-Entropy           | Want the loss directly (CE = H(P) + KL(P\|\|Q))           |
| Wasserstein Distance    | Distributions are over an ordered/metric space             |
| Total Variation Distance| Want a simple max-difference between distributions         |
| Cosine Similarity       | Comparing raw tensors, not distributions                   |
| RMSE on logits          | Comparing pre-softmax logits element-wise                  |

## Code Example

```python
import torch
import torch.nn.functional as F

# Simulate softmax outputs from FP32 and INT8 models
# Shape: (batch=16, num_classes=1000) — e.g., ImageNet classifier
torch.manual_seed(0)
logits_fp32 = torch.randn(16, 1000)   # (16, 1000) FP32 logits
logits_int8 = logits_fp32 + 0.3 * torch.randn(16, 1000)  # (16, 1000) with quant noise

# Convert to probabilities
p = F.softmax(logits_fp32, dim=1)   # (16, 1000) reference distribution
q = F.softmax(logits_int8, dim=1)   # (16, 1000) quantized distribution

# --- Method 1: torch.nn.functional.kl_div ---
# PyTorch expects log-probabilities as INPUT and probabilities as TARGET
log_q = F.log_softmax(logits_int8, dim=1)  # (16, 1000) log-probabilities
kl_per_sample = F.kl_div(log_q, p, reduction="none").sum(dim=1)  # (16,)
kl_mean = kl_per_sample.mean()
print(f"KL(P || Q) per-sample mean: {kl_mean.item():.6f} nats")

# --- Method 2: Manual computation ---
eps = 1e-10  # smoothing to avoid log(0)
kl_manual = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=1).mean()
print(f"KL(P || Q) manual:          {kl_manual.item():.6f} nats")

# --- Method 3: torchmetrics ---
from torchmetrics.regression import KLDivergence

kl_metric = KLDivergence(log_prob=False)
kl_metric.update(p, q)
kl_tm = kl_metric.compute()
print(f"KL(P || Q) torchmetrics:    {kl_tm.item():.6f} nats")

# --- Show asymmetry ---
kl_reverse = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=1).mean()
print(f"KL(Q || P) reverse:         {kl_reverse.item():.6f} nats")
print(f"Asymmetry ratio:            {kl_mean.item() / (kl_reverse.item() + eps):.3f}")
```

## Debugging Use Case

**Scenario: Comparing output distributions of quantized vs. original softmax layers**

You have quantized a DistilBERT model to INT8 for deployment and observe that while top-1 accuracy drops only 0.5%, the confidence calibration has deteriorated (ECE increases from 0.03 to 0.08). KL divergence helps diagnose this:

1. For 1000 validation samples, collect softmax output distributions from both FP32 and INT8 models.
2. Compute per-sample KL(FP32 || INT8). Mean KL = 0.015 nats — small but non-zero.
3. Sort samples by KL divergence. The top-10% (high-KL samples) have mean confidence drop from 0.92 to 0.78 — the quantized model is *under-confident* on these.
4. Inspect these samples: they are ambiguous examples near decision boundaries where small logit perturbations cause large probability shifts after softmax.
5. Apply temperature scaling (T=1.1) to the INT8 model's logits before softmax. Per-sample KL drops to 0.004 nats, and ECE returns to 0.035.

KL divergence revealed that quantization noise, while small in logit space (low RMSE), caused significant distributional shifts in probability space for borderline samples.

## Related Metrics

- [Cosine Similarity](cosine_similarity.md) — Directional comparison of raw tensors, not distributions.
- [RMSE](rmse.md) — Element-wise magnitude error on logits (pre-softmax).
- [SQNR](sqnr.md) — Quantization quality metric for raw tensors.
- [Correlation](correlation.md) — Linear/rank correlation between tensors.
- [Allclose](allclose.md) — Element-wise tolerance check for pre-softmax logits.
- [SNR](snr.md) — Signal quality metric for non-distributional tensors.
