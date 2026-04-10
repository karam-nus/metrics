---
title: "Recall"
---
# Recall

## Overview

Recall (Sensitivity, True Positive Rate, Hit Rate) measures the fraction of actual positives that the model correctly identifies. It answers: "Of all the real positives, how many did we catch?" Recall is the critical metric when missing a positive instance carries severe consequences—medical diagnosis, security threat detection, fault detection in safety-critical systems. Unlike precision, recall is indifferent to false positives: a model that predicts everything as positive achieves perfect recall (but zero precision).

## Formula

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{TP}{P}
$$

where $P = TP + FN$ is the total number of actual positives.

Also known as:
- **Sensitivity** (medical/statistical literature)
- **True Positive Rate (TPR)** (ROC analysis)
- **Hit Rate** (information retrieval)

For multiclass (per-class recall for class $c$):

$$
\text{Recall}_c = \frac{M_{cc}}{\sum_{c'=1}^{C} M_{cc'}}
$$

**Balanced Accuracy** is the macro-averaged recall:

$$
\text{Balanced Accuracy} = \frac{1}{C}\sum_{c=1}^{C} \text{Recall}_c
$$

## Visual Diagram

```
Confusion Matrix (Binary):

                 Predicted
              +       -
         +---------+---------+
Actual + |   TP    |   FN    |  ← Recall = TP / (TP + FN)
         +---------+---------+     (this row)
Actual - |   FP    |   TN    |
         +---------+---------+
```

```
All Actual Positives
┌──────────────────────────────┐
│  ┌───────────┬──────────┐    │
│  │  Caught   │  Missed  │    │
│  │   (TP)    │   (FN)   │    │
│  └───────────┴──────────┘    │
│   Recall = TP / (TP + FN)   │
└──────────────────────────────┘
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Every actual positive is detected (zero false negatives) |
| 0.5 | Half the actual positives are missed |
| 0.0 | No actual positives detected |
| Undefined | No actual positives exist (TP + FN = 0); typically set to 0.0 |

- **Range:** $[0, 1]$
- Perfect recall is trivially achievable by predicting all instances as positive — recall must always be evaluated alongside precision.
- On imbalanced data, recall for the minority class is often the most informative single number.

## When to Use

- **False negatives are catastrophic:** cancer screening (missing malignant tumors), fraud detection (missing fraudulent transactions), intrusion detection (missing attacks).
- **Safety-critical systems** where the cost of a miss far exceeds the cost of a false alarm.
- As the **y-axis (TPR) of the ROC curve** — [ROC-AUC](../classification/roc_auc.md) integrates recall across thresholds.
- **Paired with precision** to understand the full trade-off via [F1](../classification/f1_score.md) or the [PR curve](../classification/pr_auc.md).

## When NOT to Use

- **False positives are the primary concern:** use [precision](../classification/precision.md) instead.
- **As a sole metric:** a trivial all-positive classifier has recall = 1.0.
- **When class distribution is balanced and symmetric costs apply:** [accuracy](../classification/accuracy.md) or [MCC](../classification/mcc.md) may suffice.
- **Ranking without a fixed threshold:** use [ROC-AUC](../classification/roc_auc.md) or [PR-AUC](../classification/pr_auc.md).

## What It Can Tell You

- The miss rate: $FNR = 1 - \text{Recall}$.
- Whether the model is capturing the minority/positive class.
- Per-class detection rate in multiclass problems.
- The TPR at a given threshold — essential for ROC analysis.

## What It Cannot Tell You

- How many of the positive predictions are wrong (that's [precision](../classification/precision.md)).
- Model performance on the negative class (that's [specificity](../classification/specificity.md)).
- Threshold-independent quality — recall is threshold-dependent.
- Whether the model is clinically/operationally useful without knowing the false alarm rate.

## Sensitivity

- **Threshold:** Lowering the decision threshold increases recall (more predictions become positive) but increases false positives.
- **Class imbalance:** Recall is independent of TN count, so it remains meaningful on imbalanced data. However, low prevalence makes each FN more impactful.
- **Label noise:** Mislabeled positives as negatives (FN noise) directly degrades recall.
- **Sample size:** For rare positives, recall estimates have high variance — use confidence intervals.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [Precision](../classification/precision.md) | FP cost > FN cost | Controls false alarm rate |
| [F1 Score](../classification/f1_score.md) | Need to balance precision and recall | Single trade-off number |
| [Specificity](../classification/specificity.md) | Need to measure negative class detection | TNR is the recall of the negative class |
| [ROC-AUC](../classification/roc_auc.md) | Threshold-independent evaluation | Integrates TPR across all FPR |
| [PR-AUC](../classification/pr_auc.md) | Imbalanced + threshold-free | Better than ROC-AUC under imbalance |
| [MCC](../classification/mcc.md) | Need a single balanced metric | All-quadrant correlation |

## Code Example

```python
import torch
import torchmetrics

# --- Binary recall ---
preds = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])    # shape: (10,)
target = torch.tensor([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])    # shape: (10,)

recall = torchmetrics.Recall(task="binary")
result = recall(preds, target)
print(f"Binary Recall: {result.item():.4f}")  # TP=4, FN=2 → 0.6667

# --- With probability scores ---
preds_prob = torch.tensor([0.9, 0.3, 0.8, 0.7, 0.2, 0.85, 0.1, 0.4, 0.75, 0.6])
recall_low_thresh = torchmetrics.Recall(task="binary", threshold=0.3)
result_low = recall_low_thresh(preds_prob, target)
print(f"Recall (threshold=0.3): {result_low.item():.4f}")  # Lower threshold → higher recall

# --- Multiclass macro recall (= Balanced Accuracy) ---
preds_mc = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])   # shape: (8,)
target_mc = torch.tensor([0, 1, 2, 1, 1, 0, 0, 2])   # shape: (8,)

recall_macro = torchmetrics.Recall(
    task="multiclass", num_classes=3, average="macro"
)
result_macro = recall_macro(preds_mc, target_mc)
print(f"Macro Recall (Balanced Accuracy): {result_macro.item():.4f}")
```

## Debugging Use Case

**Scenario:** Medical diagnosis model — missing positive cases (cancer detection).

A mammography screening model achieves 98% accuracy, but recall for malignant cases is only 0.62 — missing 38% of cancers. Precision for malignant is 0.91.

**Debugging steps:**
1. Confirm the class distribution: if only 2% of cases are malignant, 98% accuracy is trivially achievable.
2. Compute the **confusion matrix** — identify how many malignant cases are misclassified as benign (FN).
3. Lower the decision threshold (e.g., from 0.5 to 0.2) and recompute: recall should increase while precision drops.
4. Plot the **ROC curve** — find the threshold where TPR (recall) ≥ 0.95 and note the corresponding FPR.
5. Plot the **PR curve** to understand the precision cost of achieving target recall.
6. Consider **oversampling** the minority class (SMOTE), **focal loss**, or **class-weighted cross-entropy** to push the model toward higher recall.
7. In medical settings, a high-recall, moderate-precision model is preferred — false alarms lead to further testing, but missed cancers are fatal.

## Related Metrics

- [Precision](../classification/precision.md) — complementary: measures FP control
- [F1 Score](../classification/f1_score.md) — harmonic mean of precision and recall
- [Specificity](../classification/specificity.md) — recall of the negative class (TNR)
- [ROC-AUC](../classification/roc_auc.md) — integrates recall (TPR) across FPR thresholds
- [PR-AUC](../classification/pr_auc.md) — integrates precision across recall thresholds
- [Accuracy](../classification/accuracy.md) — overall correctness
- [MCC](../classification/mcc.md) — balanced single metric
