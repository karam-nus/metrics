---
title: "Specificity"
---
# Specificity

## Overview

Specificity (True Negative Rate, Selectivity) measures the fraction of actual negatives that are correctly identified. It answers: "Of all the real negatives, how many did we correctly classify as negative?" Specificity is the mirror image of [recall](../classification/recall.md)—recall measures detection of positives, specificity measures detection of negatives. In medical screening, specificity controls the false alarm rate: a highly specific test rarely produces false positives, avoiding unnecessary follow-up procedures and patient anxiety. Specificity is the complement of the False Positive Rate: $\text{FPR} = 1 - \text{Specificity}$.

## Formula

$$
\text{Specificity} = \frac{TN}{TN + FP} = 1 - \text{FPR}
$$

where $\text{FPR} = \frac{FP}{FP + TN}$ is the False Positive Rate.

For multiclass (per-class specificity for class $c$, treating class $c$ as positive):

$$
\text{Specificity}_c = \frac{\sum_{i \neq c}\sum_{j \neq c} M_{ij}}{\sum_{i \neq c}\sum_{j=1}^{C} M_{ij}}
$$

This is the true negative rate where "negative" means "not class $c$."

## Visual Diagram

```
Confusion Matrix (Binary):

                 Predicted
              +       -
         +---------+---------+
Actual + |   TP    |   FN    |
         +---------+---------+
Actual - |   FP    |   TN    |  ← Specificity = TN / (TN + FP)
         +---------+---------+     (this row)
```

```
Mirror Relationship:

    Recall      = TP / (TP + FN)   ← among actual positives
    Specificity = TN / (TN + FP)   ← among actual negatives

    Both measure correct classification WITHIN their respective class.
    
    ROC Curve:  y-axis = Recall (TPR)
                x-axis = 1 - Specificity (FPR)
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Every actual negative is correctly identified (zero false positives) |
| 0.5 | Half the negatives are misclassified as positive |
| 0.0 | Every negative is misclassified |

- **Range:** $[0, 1]$
- Perfect specificity is trivially achievable by never predicting positive — but recall becomes 0.
- In medical testing: high specificity = "rule in" (SpPin: Specific test, Positive result rules IN the condition).
- Specificity is the complement of FPR used as the x-axis in the ROC curve.

## When to Use

- **Medical screening / diagnostics:** High specificity ensures few false alarms, reducing unnecessary invasive follow-up tests.
- **Legal / regulatory contexts:** False positives have legal or financial consequences (e.g., wrongful flagging).
- **When negatives vastly outnumber positives:** Specificity ensures the large negative class is handled correctly.
- **Alongside recall** to understand the full TPR-FPR trade-off.
- **ROC analysis:** Specificity defines the x-axis (as $1 - \text{Specificity} = \text{FPR}$).

## When NOT to Use

- **When false negatives are the primary concern:** Use [recall](../classification/recall.md).
- **As a sole metric:** High specificity alone is trivially achievable (predict all negative).
- **When the positive class is the focus:** [Precision](../classification/precision.md) or [recall](../classification/recall.md) are more directly useful.
- **Imbalanced datasets with few negatives:** Specificity is unreliable with small negative counts.

## What It Can Tell You

- The false alarm rate: $\text{FPR} = 1 - \text{Specificity}$.
- Whether the model correctly handles the negative class.
- Combined with sensitivity (recall), the pair (sensitivity, specificity) fully characterizes binary classification at a given threshold.
- Likelihood ratios: $LR+ = \frac{\text{Sensitivity}}{1 - \text{Specificity}}$, $LR- = \frac{1 - \text{Sensitivity}}{\text{Specificity}}$.

## What It Cannot Tell You

- How well the model detects positives (that's [recall](../classification/recall.md)).
- Overall classification quality — need both sensitivity and specificity.
- Positive predictive value — that depends on prevalence (Bayes' theorem).
- Performance across thresholds — specificity is threshold-dependent.

## Sensitivity

- **Threshold:** Raising the threshold increases specificity (fewer positive predictions) but decreases recall.
- **Class imbalance:** When negatives dominate, specificity is high even for mediocre models (many TN). Interpret with caution.
- **Label noise:** Mislabeled negatives as positives (FP noise) directly degrade specificity.
- **Sample size:** For rare negative classes, specificity estimates are unstable.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [Recall](../classification/recall.md) | FN cost > FP cost | Measures positive class detection |
| [Precision](../classification/precision.md) | Focus on reliability of positive predictions | PPV rather than class-conditional |
| [F1 Score](../classification/f1_score.md) | Need balanced P/R evaluation | Ignores TN (unlike specificity) |
| [ROC-AUC](../classification/roc_auc.md) | Threshold-free TPR-FPR evaluation | Integrates across all specificities |
| [MCC](../classification/mcc.md) | Need single balanced metric | Uses all four quadrants |
| Balanced Accuracy | Equal weight to both classes | $= \frac{\text{Sensitivity} + \text{Specificity}}{2}$ |

## Code Example

```python
import torch
import torchmetrics

# --- Binary specificity ---
preds = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])    # shape: (10,)
target = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])    # shape: (10,)

specificity = torchmetrics.Specificity(task="binary")
result = specificity(preds, target)
print(f"Binary Specificity: {result.item():.4f}")

# --- Verify manually ---
# TN=4, FP=2, FN=2, TP=2
# Specificity = TN/(TN+FP) = 4/6 = 0.6667
# FPR = 1 - Specificity = 0.3333

# --- With probability inputs ---
preds_prob = torch.tensor([0.1, 0.7, 0.2, 0.3, 0.8, 0.15, 0.6, 0.05, 0.25, 0.9])
specificity_prob = torchmetrics.Specificity(task="binary", threshold=0.5)
result_prob = specificity_prob(preds_prob, target)
print(f"Specificity (prob, t=0.5): {result_prob.item():.4f}")

# --- Multiclass specificity ---
preds_mc = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])   # shape: (8,)
target_mc = torch.tensor([0, 1, 1, 0, 2, 2, 1, 1])   # shape: (8,)

specificity_mc = torchmetrics.Specificity(
    task="multiclass", num_classes=3, average="macro"
)
result_mc = specificity_mc(preds_mc, target_mc)
print(f"Macro Specificity: {result_mc.item():.4f}")

# --- Sensitivity + Specificity pair ---
recall = torchmetrics.Recall(task="binary")
recall_result = recall(preds, target)
print(f"Sensitivity (Recall): {recall_result.item():.4f}")
print(f"Specificity: {result.item():.4f}")
balanced_acc = (recall_result.item() + result.item()) / 2
print(f"Balanced Accuracy: {balanced_acc:.4f}")
```

## Debugging Use Case

**Scenario:** Medical screening — controlling false alarms in COVID-19 antigen tests.

A rapid antigen test has sensitivity (recall) = 0.85 and specificity = 0.92. With a prevalence of 2%, what fraction of positive results are truly positive?

**Debugging steps:**
1. Apply **Bayes' theorem** for the Positive Predictive Value:
   - $PPV = \frac{0.85 \times 0.02}{0.85 \times 0.02 + 0.08 \times 0.98} = \frac{0.017}{0.017 + 0.078} = 0.179$
   - Only 18% of positive test results are true positives, despite 92% specificity!
2. This illustrates the **base-rate fallacy**: even high specificity produces many false positives when prevalence is low.
3. To improve PPV, increase specificity (e.g., confirmatory PCR test with specificity > 0.999).
4. Plot **PPV vs. prevalence** at fixed specificity to show stakeholders how false alarm rate changes with population infection rate.
5. Consider **two-stage testing**: rapid screen (high sensitivity) → confirmatory test (high specificity).
6. Monitor specificity over time — new viral variants or test degradation can reduce specificity.

## Related Metrics

- [Recall](../classification/recall.md) — sensitivity, the mirror metric for positives (TPR)
- [Precision](../classification/precision.md) — positive predictive value (different from specificity)
- [ROC-AUC](../classification/roc_auc.md) — integrates sensitivity vs. 1-specificity
- [Accuracy](../classification/accuracy.md) — combines sensitivity and specificity (weighted by prevalence)
- [MCC](../classification/mcc.md) — balanced metric using all quadrants
- [F1 Score](../classification/f1_score.md) — ignores specificity (TN)
- [Cohen's Kappa](../classification/cohens_kappa.md) — chance-corrected agreement
