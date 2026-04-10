---
title: "PR-AUC"
---
# PR-AUC

## Overview

PR-AUC (Area Under the Precision-Recall Curve) summarizes the trade-off between precision and recall across all possible thresholds into a single scalar. Unlike ROC-AUC, PR-AUC focuses exclusively on the positive class and is not inflated by a large number of true negatives. This makes PR-AUC the preferred threshold-invariant metric for imbalanced datasets where the positive class is rare—fraud detection, rare disease screening, anomaly detection. The baseline for PR-AUC is the positive class prevalence ($\pi = P/N$), not 0.5 as in ROC-AUC.

## Formula

The Precision-Recall curve is parameterized by threshold $t$:

$$
\text{Precision}(t) = \frac{TP(t)}{TP(t) + FP(t)}, \quad \text{Recall}(t) = \frac{TP(t)}{TP(t) + FN(t)}
$$

$$
\text{PR-AUC} = \int_0^1 \text{Precision}(\text{Recall}^{-1}(r)) \, dr
$$

**Average Precision (AP)** — the standard discrete approximation:

$$
\text{AP} = \sum_{k=1}^{n} (\text{Recall}_k - \text{Recall}_{k-1}) \cdot \text{Precision}_k
$$

where $k$ indexes thresholds sorted by decreasing score, and $\text{Recall}_0 = 0$.

**Baseline (no-skill):**

$$
\text{PR-AUC}_{\text{random}} = \frac{|\text{positives}|}{|\text{total}|} = \pi
$$

## Visual Diagram

```
ASCII Precision-Recall Curve:

Precision
1.0 ┤●
    │ ·●
    │   ·●
0.8 ┤     ·●
    │       ·●        Good model
    │         ·●      (AP ≈ 0.80)
0.6 ┤           ·●
    │             ·●
    │               ·●
0.4 ┤                 ·●
    │                   ·●
    │                     ·●
0.2 ┤                       ·●
    │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Random baseline
    │                               (π = prevalence)
0.0 ┤───────────────────────────────────────────
    0.0   0.2   0.4   0.6   0.8   1.0
                    Recall

    AUC = area under the curve
    No-skill baseline = horizontal line at prevalence (π)
```

```
Key relationship:

    High threshold → High precision, Low recall (top-left)
    Low threshold  → Low precision, High recall (bottom-right)
    
    The curve starts near (0, 1) and ends near (1, π).
    A perfect model maintains Precision=1.0 across all Recall.
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Perfect — precision stays 1.0 at all recall levels |
| = $\pi$ (prevalence) | No-skill classifier; random guessing |
| < $\pi$ | Worse than random |
| > 0.5 | Only meaningful relative to baseline $\pi$ |

- **Range:** $[0, 1]$
- **Critical:** A PR-AUC of 0.3 on a dataset with 1% positives ($\pi = 0.01$) is excellent. A PR-AUC of 0.3 on a balanced dataset ($\pi = 0.5$) is poor.
- Always compare against the prevalence baseline.

## When to Use

- **Imbalanced datasets** where the positive class is rare (fraud, rare disease, anomaly detection).
- When **false positives among predicted positives** are more informative than false positive rate.
- **Information retrieval** — Average Precision is the standard metric for ranked retrieval.
- When ROC-AUC is misleadingly high due to the abundance of true negatives.
- **Object detection** — mAP (mean Average Precision) is the primary metric.

## When NOT to Use

- **Balanced datasets** — ROC-AUC is equally informative and more widely understood.
- **When TN performance matters** — PR-AUC ignores true negatives entirely.
- **Comparing across datasets** with different prevalences — the baseline changes.
- **When you need a fixed-threshold metric** — use [F1](../classification/f1_score.md) or [precision](../classification/precision.md)/[recall](../classification/recall.md) at the operating point.

## What It Can Tell You

- How well the model separates positives from negatives, focusing on positive-class retrieval.
- The full precision-recall trade-off without committing to a threshold.
- Which model is better at **retrieving positives** while maintaining precision.
- The practical performance ceiling on imbalanced tasks where ROC-AUC overstates quality.

## What It Cannot Tell You

- Negative class performance (TN is excluded).
- Calibration quality — only rank ordering matters.
- Performance at a specific operating threshold.
- Whether the model generalizes — PR-AUC can overfit to the ranking of the test set.

## Sensitivity

- **Prevalence:** The random baseline shifts with class balance. Always report prevalence alongside PR-AUC.
- **Interpolation method:** Linear interpolation vs. step-function interpolation can give different AUC values. `torchmetrics` uses trapezoidal; some libraries use the step method.
- **Number of positive instances:** With few positives, the PR curve is noisy and the AUC estimate has high variance.
- **Score ties:** Ties in predicted scores create flat segments; tie-breaking strategy affects AUC.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [ROC-AUC](../classification/roc_auc.md) | Balanced data; need FPR-TPR trade-off | Well-understood, 0.5 baseline |
| [F1 Score](../classification/f1_score.md) | Fixed threshold needed | Single-threshold balance |
| [Precision](../classification/precision.md) | FP cost is primary concern | Direct false alarm control |
| [Recall](../classification/recall.md) | FN cost is primary concern | Direct miss-rate control |
| [MCC](../classification/mcc.md) | Need balanced metric with TN | All-quadrant evaluation |
| Precision@K | Ranked retrieval, top-K matters | Focuses on top results |

## Code Example

```python
import torch
import torchmetrics

# --- Binary PR-AUC (Average Precision) ---
preds = torch.tensor([0.95, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05, 0.02])
target = torch.tensor([1,    1,    0,    1,    0,    0,    0,    0,    0,    0])
# shape: (10,) — 3 positives out of 10 (π = 0.3)

ap = torchmetrics.AveragePrecision(task="binary")
result = ap(preds, target)
print(f"PR-AUC (Average Precision): {result.item():.4f}")

# --- Multiclass PR-AUC ---
preds_mc = torch.softmax(torch.randn(200, 4), dim=1)  # shape: (200, 4)
target_mc = torch.randint(0, 4, (200,))                # shape: (200,)

ap_mc = torchmetrics.AveragePrecision(
    task="multiclass", num_classes=4, average="macro"
)
result_mc = ap_mc(preds_mc, target_mc)
print(f"Multiclass PR-AUC (macro): {result_mc.item():.4f}")

# --- PR curve data for plotting ---
from torchmetrics.classification import BinaryPrecisionRecallCurve

pr_curve = BinaryPrecisionRecallCurve()
precision_vals, recall_vals, thresholds = pr_curve(preds, target)
print(f"Precision: {precision_vals}")
print(f"Recall: {recall_vals}")
# Plot: plt.plot(recall_vals, precision_vals)
```

## Debugging Use Case

**Scenario:** Rare event detection — credit card fraud (positive rate = 0.1%).

A fraud detection model reports ROC-AUC = 0.98 but PR-AUC = 0.32. The high ROC-AUC is misleading: with 99.9% negatives, FPR remains low even with hundreds of false positives.

**Debugging steps:**
1. Compute the **baseline PR-AUC** = 0.001 (prevalence). PR-AUC of 0.32 is actually 320× above random — not bad.
2. Plot the **PR curve** — check if precision collapses at high recall (common pattern).
3. Identify the **recall level** required for the business case (e.g., catch 90% of fraud).
4. Read the precision at that recall level from the PR curve — if precision = 0.05, then 95% of alerts are false alarms.
5. Evaluate whether the false alarm rate is operationally acceptable (human review capacity).
6. If not, try:
   - **Oversampling** positives (SMOTE) during training.
   - **Focal loss** to emphasize hard positives.
   - **Feature engineering** targeting fraud patterns.
7. Compare PR curves of different models — a model with lower ROC-AUC but higher PR-AUC may be practically superior.

## Related Metrics

- [ROC-AUC](../classification/roc_auc.md) — complementary threshold-invariant metric
- [Precision](../classification/precision.md) — the y-axis of the PR curve
- [Recall](../classification/recall.md) — the x-axis of the PR curve
- [F1 Score](../classification/f1_score.md) — iso-F1 lines on the PR plot
- [Accuracy](../classification/accuracy.md) — often misleading when PR-AUC is needed
- [MCC](../classification/mcc.md) — balanced threshold-dependent alternative
