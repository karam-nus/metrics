---
title: "Precision"
---
# Precision

## Overview

Precision (Positive Predictive Value) measures the fraction of positive predictions that are actually correct. It directly answers: "When the model says positive, how often is it right?" Precision is the primary metric when the cost of false positives is high—spam filters, content moderation, fraud alerts—where acting on a false positive wastes resources or harms user experience. Precision is independent of the number of true negatives, making it well-suited for imbalanced datasets where negatives dominate.

## Formula

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

For multiclass (per-class precision for class $c$):

$$
\text{Precision}_c = \frac{M_{cc}}{\sum_{c'=1}^{C} M_{c'c}}
$$

Aggregation variants:
- **Macro:** $\frac{1}{C}\sum_{c=1}^{C} \text{Precision}_c$
- **Micro:** $\frac{\sum_c TP_c}{\sum_c (TP_c + FP_c)}$
- **Weighted:** $\sum_{c=1}^{C} w_c \cdot \text{Precision}_c$, where $w_c = \frac{n_c}{N}$

## Visual Diagram

```
Confusion Matrix (Binary):

                 Predicted
              +       -
         +---------+---------+
Actual + |   TP    |   FN    |
         +---------+---------+
Actual - |   FP    |   TN    |
         +---------+---------+

Precision = TP / (TP + FP)
                   ▲
            predicted positive
            (column sum of +)
```

```
All Predictions
┌────────────────────────────────┐
│  Predicted Positive            │
│  ┌──────────┬────────────┐     │
│  │    TP    │     FP     │     │  ← Precision = TP / (TP+FP)
│  └──────────┴────────────┘     │
│  Predicted Negative            │
│  ┌──────────┬────────────┐     │
│  │    FN    │     TN     │     │
│  └──────────┴────────────┘     │
└────────────────────────────────┘
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Every positive prediction is correct (zero false positives) |
| 0.5 | Half of positive predictions are wrong |
| 0.0 | Every positive prediction is wrong |
| Undefined | No positive predictions made (TP + FP = 0); typically set to 0.0 |

- **Range:** $[0, 1]$
- High precision ≠ good model — a model predicting positive only once (correctly) has precision 1.0 but useless recall.
- Precision can be artificially inflated by raising the decision threshold (fewer but more confident positives).

## When to Use

- **False positives are costly:** spam filters (legitimate email marked as spam), content moderation (wrongful removal), fraud detection (blocking legitimate transactions).
- **Precision-oriented systems** where users lose trust upon encountering false alarms.
- **Information retrieval:** precision@K measures relevance of top-K returned documents.
- Paired with [recall](../classification/recall.md) or as part of [F1](../classification/f1_score.md) for a complete picture.

## When NOT to Use

- **False negatives are costly:** medical diagnosis (missing a cancer case), security screening (missing a threat). Use [recall](../classification/recall.md) instead.
- **As a sole metric:** precision alone ignores how many true positives were missed.
- **Highly imbalanced data without context:** high precision can mask poor recall.
- **Ranking tasks without a threshold:** use [PR-AUC](../classification/pr_auc.md) or [ROC-AUC](../classification/roc_auc.md).

## What It Can Tell You

- The false discovery rate: $FDR = 1 - \text{Precision}$.
- Whether the model's positive predictions are trustworthy.
- Directional impact of threshold adjustment on false positive rate.
- Per-class reliability in multiclass settings (which classes get confused as false positives).

## What It Cannot Tell You

- How many actual positives are being missed (that's [recall](../classification/recall.md)).
- Overall classification quality across both classes.
- Threshold-independent model quality (use [PR-AUC](../classification/pr_auc.md)).
- Whether the model generalizes—precision on train vs. test must be compared.

## Sensitivity

- **Threshold:** Raising the decision threshold increases precision (fewer, more confident positives) at the cost of recall.
- **Class imbalance:** Precision is less affected by class imbalance than accuracy because it ignores TN. However, with many negatives, even a small FP rate produces many false positives.
- **Base rate (prevalence):** Low prevalence means FP can easily outnumber TP, driving precision down even with a good model (Bayesian base-rate fallacy).
- **Label noise:** Mislabeled negatives counted as FP directly degrade precision.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [Recall](../classification/recall.md) | FN cost > FP cost | Captures missed positives |
| [F1 Score](../classification/f1_score.md) | Need balance between precision and recall | Single number trade-off |
| [PR-AUC](../classification/pr_auc.md) | Threshold-free evaluation on imbalanced data | Summarizes P-R curve |
| [Specificity](../classification/specificity.md) | Need to measure TN performance | Complementary to recall |
| [MCC](../classification/mcc.md) | Need a balanced single metric | Uses all 4 quadrants |

## Code Example

```python
import torch
import torchmetrics

# --- Binary precision ---
preds = torch.tensor([1, 1, 1, 0, 1, 0, 1, 0, 0, 1])    # shape: (10,)
target = torch.tensor([1, 0, 1, 0, 1, 1, 0, 0, 0, 1])    # shape: (10,)

precision = torchmetrics.Precision(task="binary")
result = precision(preds, target)
print(f"Binary Precision: {result.item():.4f}")  # TP=4, FP=2 → 0.6667

# --- With probability scores and threshold ---
preds_prob = torch.tensor([0.9, 0.7, 0.8, 0.2, 0.85, 0.4, 0.6, 0.1, 0.3, 0.95])
precision_thresh = torchmetrics.Precision(task="binary", threshold=0.75)
result_thresh = precision_thresh(preds_prob, target)
print(f"Precision (threshold=0.75): {result_thresh.item():.4f}")

# --- Multiclass macro precision ---
preds_mc = torch.tensor([0, 2, 1, 0, 2, 1, 0, 2])   # shape: (8,)
target_mc = torch.tensor([0, 2, 1, 1, 2, 0, 0, 1])   # shape: (8,)

precision_macro = torchmetrics.Precision(
    task="multiclass", num_classes=3, average="macro"
)
result_macro = precision_macro(preds_mc, target_mc)
print(f"Macro Precision: {result_macro.item():.4f}")
```

## Debugging Use Case

**Scenario:** Spam filter evaluation — too many legitimate emails flagged.

Users report that important emails end up in the spam folder. The model has recall of 0.95 (catches most spam) but precision of 0.60 (40% of flagged emails are legitimate).

**Debugging steps:**
1. Compute precision per class — confirm that the "spam" class has low precision.
2. Examine the **false positives**: what features cause legitimate emails to look like spam? (e.g., marketing language, links, short body).
3. Raise the classification threshold from 0.5 to 0.7 and recompute precision — expect improvement at the cost of recall.
4. Plot the [Precision-Recall curve](../classification/pr_auc.md) to find the optimal threshold for the desired precision level.
5. Consider retraining with **cost-sensitive loss** (higher weight on FP) or **hard negative mining** to reduce false positives.
6. Monitor precision over time — distribution shift (new email patterns) can degrade precision.

## Related Metrics

- [Recall](../classification/recall.md) — the complement focus: measures missed positives
- [F1 Score](../classification/f1_score.md) — harmonic mean of precision and recall
- [PR-AUC](../classification/pr_auc.md) — area under the precision-recall curve
- [Specificity](../classification/specificity.md) — true negative rate, analogous for negatives
- [Accuracy](../classification/accuracy.md) — overall correctness (includes TN)
- [MCC](../classification/mcc.md) — balanced metric using all quadrants
