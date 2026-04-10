---
title: "ROC-AUC"
---
# ROC-AUC

## Overview

ROC-AUC (Area Under the Receiver Operating Characteristic Curve) measures a classifier's ability to discriminate between classes across all possible decision thresholds. The ROC curve plots True Positive Rate (Recall) against False Positive Rate at every threshold. AUC summarizes this curve into a single scalar: the probability that a randomly chosen positive instance is scored higher than a randomly chosen negative instance. ROC-AUC is threshold-invariant and scale-invariant, making it ideal for comparing models that output uncalibrated scores. However, it can be overly optimistic on highly imbalanced datasets.

## Formula

The ROC curve is parameterized by threshold $t \in (-\infty, +\infty)$:

$$
\text{TPR}(t) = \frac{TP(t)}{TP(t) + FN(t)}, \quad \text{FPR}(t) = \frac{FP(t)}{FP(t) + TN(t)}
$$

$$
\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx
$$

Equivalently (Mann-Whitney U statistic interpretation):

$$
\text{AUC} = P(\hat{y}_{+} > \hat{y}_{-}) = \frac{\sum_{i \in \text{pos}} \sum_{j \in \text{neg}} \mathbb{1}(s_i > s_j)}{|\text{pos}| \cdot |\text{neg}|}
$$

where $s_i, s_j$ are the predicted scores for positive instance $i$ and negative instance $j$.

For multiclass (One-vs-Rest):

$$
\text{AUC}_{\text{macro}} = \frac{1}{C}\sum_{c=1}^{C} \text{AUC}_c
$$

## Visual Diagram

```
ASCII ROC Curve:

TPR (Recall)
1.0 ┤                          ●───────────────
    │                       ●·´
    │                    ●·´
0.8 ┤                 ●·´
    │              ●·´         Perfect classifier
    │           ●·´            AUC = 1.0
0.6 ┤        ●·´
    │      ●·´     ← Good model (AUC ≈ 0.85)
    │    ●·´
0.4 ┤  ●·´
    │ ●·
    │●·     ╱ Random classifier
0.2 ┤·    ╱   (diagonal, AUC = 0.5)
    │   ╱
    │ ╱
0.0 ┤╱─────────────────────────────────────────
    0.0   0.2   0.4   0.6   0.8   1.0
                    FPR (1 - Specificity)

    AUC = area between the curve and the x-axis
```

<!-- IMAGE: ROC curve comparing good model, poor model, and random baseline -->

```
Confusion Matrix at threshold t:

                 Predicted
              +       -
         +---------+---------+
Actual + |  TP(t)  |  FN(t)  |  → TPR = TP/(TP+FN)
         +---------+---------+
Actual - |  FP(t)  |  TN(t)  |  → FPR = FP/(FP+TN)
         +---------+---------+
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Perfect discrimination — positive and negative score distributions are fully separated |
| 0.5 | Random — no discriminative power (diagonal line) |
| < 0.5 | Worse than random — model predictions are inverted (flip the sign) |
| 0.7–0.8 | Acceptable discrimination |
| 0.8–0.9 | Good discrimination |
| > 0.9 | Excellent discrimination |

- **Range:** $[0, 1]$
- AUC = 0.5 is the no-skill baseline regardless of class imbalance.
- AUC is invariant to monotonic transformations of the predicted score.

## When to Use

- **Comparing models** that produce probability or confidence scores, without committing to a threshold.
- **Balanced to moderately imbalanced datasets** where both TPR and FPR matter.
- **Medical diagnostics** — ROC analysis is the standard in clinical decision studies.
- When you need a **threshold-independent, scale-independent** evaluation metric.
- **Model selection** during hyperparameter tuning before choosing a deployment threshold.

## When NOT to Use

- **Highly imbalanced data:** ROC-AUC can be overly optimistic because FPR remains small even with many false positives when negatives vastly outnumber positives. Use [PR-AUC](../classification/pr_auc.md) instead.
- **When the operating threshold is fixed:** Use threshold-dependent metrics (F1, precision, recall).
- **Cost-sensitive settings with known cost ratios:** Use cost curves or directly optimize the metric of interest.
- **Multiclass with many classes:** One-vs-Rest AUC is expensive and may not capture inter-class confusion well.

## What It Can Tell You

- The probability that the model ranks a random positive above a random negative.
- Whether one model has uniformly better discrimination than another.
- The full spectrum of TPR-FPR trade-offs available by adjusting the threshold.
- Whether the model has learned a meaningful separation between classes.

## What It Cannot Tell You

- How well the model performs at a **specific threshold** — a high AUC does not guarantee good performance at any particular operating point.
- Whether predicted probabilities are **calibrated** (AUC is scale-invariant).
- Performance under **severe class imbalance** — small FPR changes map to large absolute FP counts.
- Per-class performance in multiclass — macro AUC averages over one-vs-rest curves.

## Sensitivity

- **Class imbalance:** FPR = FP/(FP+TN); when TN is huge, FPR stays low even with many FP. This makes AUC insensitive to false positives in imbalanced settings.
- **Score distribution:** AUC depends only on the rank ordering of scores, not their magnitude. Two models with different calibration but same ranking have the same AUC.
- **Ties:** Tied scores reduce AUC; proper tie-breaking (midpoint) is used by standard implementations.
- **Sample size:** AUC confidence intervals depend on both N+ and N−. Use DeLong's test for comparing AUCs.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [PR-AUC](../classification/pr_auc.md) | Highly imbalanced data | Focuses on positive class; ignores TN |
| [F1 Score](../classification/f1_score.md) | Fixed threshold evaluation | Direct P-R balance at operating point |
| [MCC](../classification/mcc.md) | Need all-quadrant binary evaluation | Balanced, threshold-dependent |
| [Accuracy](../classification/accuracy.md) | Balanced data, fixed threshold | Simple, interpretable |
| Log Loss | Calibration matters | Penalizes confident wrong predictions |
| [Specificity](../classification/specificity.md) | Need FPR control at a specific threshold | Direct negative class measure |

## Code Example

```python
import torch
import torchmetrics

# --- Binary ROC-AUC ---
# Predicted probabilities (continuous scores required, not hard labels)
preds = torch.tensor([0.9, 0.7, 0.4, 0.3, 0.8, 0.1, 0.6, 0.55, 0.2, 0.85])  # shape: (10,)
target = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])                          # shape: (10,)

auroc = torchmetrics.AUROC(task="binary")
result = auroc(preds, target)
print(f"Binary ROC-AUC: {result.item():.4f}")

# --- Multiclass ROC-AUC (One-vs-Rest) ---
preds_mc = torch.softmax(torch.randn(50, 5), dim=1)   # shape: (50, 5) — probabilities
target_mc = torch.randint(0, 5, (50,))                 # shape: (50,)

auroc_mc = torchmetrics.AUROC(task="multiclass", num_classes=5, average="macro")
result_mc = auroc_mc(preds_mc, target_mc)
print(f"Multiclass ROC-AUC (macro): {result_mc.item():.4f}")

# --- Plotting ROC curve data (manual) ---
from torchmetrics.classification import BinaryROC

roc = BinaryROC()
fpr, tpr, thresholds = roc(preds, target)
print(f"FPR: {fpr}")
print(f"TPR: {tpr}")
print(f"Thresholds: {thresholds}")
# Plot with: plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
```

## Debugging Use Case

**Scenario:** Comparing two models' discrimination ability across thresholds.

Model A has higher accuracy (0.88 vs. 0.84) but lower ROC-AUC (0.79 vs. 0.86) than Model B. This paradox arises because Model A was tuned for a specific threshold, while Model B produces better-separated score distributions.

**Debugging steps:**
1. Plot both ROC curves — Model B's curve dominates Model A's for most of the FPR range, even though at one operating point Model A is better.
2. Check if Model A's threshold was optimized on the test set (data leakage).
3. Compute AUC with **DeLong's confidence intervals** — is the difference statistically significant?
4. Examine the **score distributions** (histogram of P(positive) for true positives vs. true negatives) — Model B likely has better separation.
5. If deploying at a specific threshold, compute threshold-specific metrics ([F1](../classification/f1_score.md), [precision](../classification/precision.md), [recall](../classification/recall.md)) at that point.
6. If the dataset is imbalanced, recheck with [PR-AUC](../classification/pr_auc.md) — ROC-AUC may overstate Model B's practical utility.

## Related Metrics

- [PR-AUC](../classification/pr_auc.md) — better for imbalanced datasets
- [Recall](../classification/recall.md) — TPR, the y-axis of the ROC curve
- [Specificity](../classification/specificity.md) — 1 - FPR, the x-axis complement
- [F1 Score](../classification/f1_score.md) — threshold-dependent balance
- [Accuracy](../classification/accuracy.md) — threshold-dependent overall measure
- [MCC](../classification/mcc.md) — balanced binary metric at a fixed threshold
