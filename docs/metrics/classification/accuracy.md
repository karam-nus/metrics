---
title: "Accuracy"
---
# Accuracy

## Overview

Accuracy is the most intuitive classification metric: the fraction of all predictions that are correct. It counts both true positives and true negatives against the total population. While universally understood, accuracy is a deceptive metric on imbalanced datasets—a classifier that always predicts the majority class achieves accuracy equal to the prevalence of that class, providing zero discriminative value. Use accuracy as a first-pass sanity check, never as the sole evaluation metric.

## Formula

For binary classification with confusion matrix entries TP, TN, FP, FN:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Equivalently, for $N$ samples with indicator $\mathbb{1}$:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i)
$$

For multiclass with $C$ classes and confusion matrix $\mathbf{M}$:

$$
\text{Accuracy} = \frac{\sum_{c=1}^{C} M_{cc}}{\sum_{c=1}^{C}\sum_{c'=1}^{C} M_{cc'}} = \frac{\text{trace}(\mathbf{M})}{\|\mathbf{M}\|_1}
$$

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

Accuracy = (TP + TN) / (TP + TN + FP + FN)
              ▲                  ▲
         correct            everything
```

<!-- IMAGE: Bar chart showing correct vs incorrect predictions, with accuracy as the ratio -->

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Perfect classification — every prediction is correct |
| 0.5 | No better than random guessing (balanced binary) |
| 0.0 | Every prediction is wrong (invert predictions for 1.0) |
| = class prevalence | Model predicts majority class only — no discrimination |

- **Range:** $[0, 1]$
- On a dataset with 95% negatives, a trivial all-negative classifier achieves 0.95 accuracy.
- Accuracy is symmetric: it weights TP and TN equally, which is rarely appropriate in practice.

## When to Use

- **Balanced datasets** where positive and negative classes are roughly equal in size.
- **Quick sanity checks** early in development to verify the model is learning.
- **Multiclass problems** with roughly uniform class distribution.
- When stakeholders require a single, easily communicable number.
- As a **baseline reference** alongside more informative metrics (F1, MCC, AUC).

## When NOT to Use

- **Imbalanced datasets** — accuracy inflates performance of majority-class predictors.
- **Cost-sensitive settings** — when false positives and false negatives carry different costs (e.g., medical diagnosis, fraud detection).
- **Ranking or scoring tasks** — accuracy requires a hard threshold; use AUC-based metrics instead.
- When you need to understand **per-class performance** — use macro/weighted F1 or per-class recall.
- **Information retrieval** — precision and recall are standard.

## What It Can Tell You

- Whether the model has learned anything beyond the trivial baseline.
- The overall error rate ($1 - \text{Accuracy}$).
- Relative comparison between models **on the same balanced dataset**.
- A coarse measure of model quality for presentation to non-technical stakeholders.

## What It Cannot Tell You

- Which classes the model is confusing.
- The ratio of false positives to false negatives.
- Whether performance is acceptable under asymmetric cost assumptions.
- How the model ranks predictions (no threshold sensitivity information).
- Whether the model generalizes — accuracy on training data can be inflated by overfitting.

## Sensitivity

- **Class imbalance:** Highly sensitive. Accuracy degrades in informativeness as imbalance increases.
- **Threshold:** Accuracy depends on the decision threshold (default 0.5 for binary). Shifting the threshold trades TP for TN.
- **Number of classes:** In $C$-class problems, random baseline is $\approx 1/C$ (uniform) or prevalence of the most common class.
- **Dataset size:** Confidence intervals narrow with $\sim 1/\sqrt{N}$; use Wilson or Clopper-Pearson intervals for small $N$.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [F1 Score](../classification/f1_score.md) | Imbalanced data, care about both precision and recall | Harmonic mean penalizes extreme trade-offs |
| [MCC](../classification/mcc.md) | Imbalanced binary classification | Uses all 4 quadrants, balanced measure |
| [ROC-AUC](../classification/roc_auc.md) | Comparing models across thresholds | Threshold-invariant discrimination |
| [Balanced Accuracy](../classification/recall.md) | Imbalanced data, equal class importance | Average of per-class recall |
| [Cohen's Kappa](../classification/cohens_kappa.md) | Need chance-corrected agreement | Adjusts for expected agreement |
| [Precision](../classification/precision.md) / [Recall](../classification/recall.md) | Asymmetric costs | Independent control of FP vs FN |

## Code Example

```python
import torch
import torchmetrics

# --- Binary classification ---
preds = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])   # shape: (10,)
target = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])   # shape: (10,)

accuracy = torchmetrics.Accuracy(task="binary")
result = accuracy(preds, target)
print(f"Binary Accuracy: {result.item():.4f}")  # 0.8000

# --- Multiclass classification ---
# 4 samples, 3 classes, logits
preds_mc = torch.tensor([
    [0.1, 0.8, 0.1],
    [0.7, 0.2, 0.1],
    [0.2, 0.3, 0.5],
    [0.9, 0.05, 0.05],
])  # shape: (4, 3)
target_mc = torch.tensor([1, 0, 2, 0])  # shape: (4,)

accuracy_mc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
result_mc = accuracy_mc(preds_mc, target_mc)
print(f"Multiclass Accuracy: {result_mc.item():.4f}")  # 1.0000

# --- With running accumulation ---
accuracy_accum = torchmetrics.Accuracy(task="binary")
for batch_preds, batch_targets in zip(preds.chunk(2), target.chunk(2)):
    accuracy_accum.update(batch_preds, batch_targets)
print(f"Accumulated Accuracy: {accuracy_accum.compute().item():.4f}")
```

## Debugging Use Case

**Scenario:** Baseline sanity check for a new binary classifier.

A freshly initialized model produces ~50% accuracy on a balanced test set — confirming random-level predictions before training. After one epoch, accuracy jumps to 93% on a dataset with 90% negatives. This signals the model learned to predict the majority class, not meaningful patterns.

**Debugging steps:**
1. Compare accuracy against the **majority-class baseline** (prevalence of the most common class).
2. If accuracy ≈ prevalence → model is not discriminating. Check loss function, learning rate, data pipeline.
3. Compute the **confusion matrix** to see if one row/column dominates.
4. Switch to [F1](../classification/f1_score.md) or [MCC](../classification/mcc.md) for a clearer signal on imbalanced data.
5. Plot accuracy over epochs — a flat curve suggests the model is stuck; a spike suggests data leakage.

## Related Metrics

- [Precision](../classification/precision.md) — focuses on false positive control
- [Recall](../classification/recall.md) — focuses on false negative control
- [F1 Score](../classification/f1_score.md) — harmonic mean of precision and recall
- [MCC](../classification/mcc.md) — balanced binary classification metric
- [ROC-AUC](../classification/roc_auc.md) — threshold-invariant discrimination
- [Top-K Accuracy](../classification/top_k_accuracy.md) — relaxed accuracy for multiclass
- [Cohen's Kappa](../classification/cohens_kappa.md) — chance-corrected accuracy
