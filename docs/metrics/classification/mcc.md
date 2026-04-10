---
title: "Matthews Correlation Coefficient"
---
# Matthews Correlation Coefficient (MCC)

## Overview

The Matthews Correlation Coefficient (MCC) is widely regarded as the most informative single metric for binary classification. It produces a balanced measure even when classes are of very different sizes, because it uses all four quadrants of the confusion matrix (TP, TN, FP, FN). MCC is the Pearson correlation between the observed and predicted binary labels, ranging from −1 (perfect inverse) through 0 (random) to +1 (perfect). Unlike F1, which ignores TN, or accuracy, which is skewed by class imbalance, MCC provides a reliable single-number summary of classification quality.

## Formula

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

Equivalently, as Pearson's $\phi$ coefficient between binary vectors $\mathbf{y}$ and $\hat{\mathbf{y}}$:

$$
\text{MCC} = \frac{N \cdot TP - S \cdot P}{\sqrt{S \cdot (N - S) \cdot P \cdot (N - P)}}
$$

where $N$ = total samples, $S = TP + FP$ (predicted positives), $P = TP + FN$ (actual positives).

**Multiclass generalization** (R_k coefficient):

$$
\text{MCC} = \frac{c \cdot s - \sum_{k=1}^{C} p_k \cdot t_k}{\sqrt{(s^2 - \sum_k p_k^2)(s^2 - \sum_k t_k^2)}}
$$

where $c = \text{trace}(\mathbf{M})$, $s = \sum_{ij} M_{ij}$, $p_k = \sum_i M_{ik}$, $t_k = \sum_j M_{kj}$.

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

MCC uses ALL FOUR cells:

        Numerator:  (TP × TN) - (FP × FN)
                      ↗           ↗
              correct product  error product
        
        Denominator: geometric mean of marginal products
```

```
MCC Interpretation Scale:

-1.0         0.0         +1.0
 ├────────────┼────────────┤
 Perfect      Random       Perfect
 inverse      (no skill)   agreement
 prediction
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| +1.0 | Perfect classification |
| +0.7 to +1.0 | Strong positive correlation |
| +0.3 to +0.7 | Moderate positive correlation |
| 0.0 | No better than random (no correlation) |
| −0.3 to 0.0 | Weak inverse correlation |
| −1.0 | Perfect inverse (every prediction is wrong) |

- **Range:** $[-1, +1]$
- **Key property:** MCC = 0 for any trivial classifier (all-positive, all-negative, random), regardless of class imbalance.
- **Undefined** when any marginal sum is zero (e.g., no positive predictions); typically returns 0.

## When to Use

- **Imbalanced binary classification** — MCC is the single best summary metric.
- When you need a **balanced evaluation** that accounts for all four confusion matrix entries.
- **Comparing models** on datasets with different class distributions.
- **Biomedical / bioinformatics** — MCC is the recommended metric by many journals.
- When [accuracy](../classification/accuracy.md) is misleading and [F1](../classification/f1_score.md) ignores TN.
- As a **sanity check** alongside other metrics to catch degenerate classifiers.

## When NOT to Use

- **Multiclass with many classes** — MCC generalization exists but is harder to interpret. Use macro F1 or per-class analysis.
- **When threshold-invariance is needed** — MCC is threshold-dependent. Use [ROC-AUC](../classification/roc_auc.md) or [PR-AUC](../classification/pr_auc.md).
- **Ranking tasks** — MCC evaluates a single threshold, not the full ranking.
- **When stakeholders need intuitive interpretation** — accuracy or F1 may be more accessible.

## What It Can Tell You

- Whether the model has genuine predictive power (MCC >> 0) or is trivial (MCC ≈ 0).
- A single number that reflects performance on both positive and negative classes.
- Correlation strength between predictions and ground truth, corrected for class distribution.
- Whether a high-accuracy model is actually trivial (accuracy = 95% but MCC = 0 on 95:5 imbalanced data).

## What It Cannot Tell You

- The decomposition into precision vs. recall — MCC merges them.
- Performance across thresholds (it's a point estimate at one threshold).
- Per-class performance in multiclass settings.
- Whether the model is calibrated.

## Sensitivity

- **Class imbalance:** MCC is robust to imbalance — a trivial classifier always gets MCC = 0, unlike accuracy.
- **Threshold:** MCC changes with the threshold. The optimal threshold for MCC is not always 0.5.
- **Edge cases:** Undefined when a row or column of the confusion matrix sums to 0 (no positive predictions or no positive actuals). Implementations typically return 0.
- **Sample size:** On small datasets, MCC can fluctuate significantly. Use permutation tests or bootstrap CIs.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [F1 Score](../classification/f1_score.md) | TN is irrelevant or focus is on positive class | Simpler interpretation, widely adopted |
| [ROC-AUC](../classification/roc_auc.md) | Need threshold-invariant evaluation | Summarizes all thresholds |
| [PR-AUC](../classification/pr_auc.md) | Imbalanced + threshold-free | Positive-class focused |
| [Cohen's Kappa](../classification/cohens_kappa.md) | Need chance-corrected agreement | Interpretable agreement measure |
| [Accuracy](../classification/accuracy.md) | Balanced data, simple reporting | Universally understood |
| [Precision](../classification/precision.md)/[Recall](../classification/recall.md) | Asymmetric costs | Fine-grained control |

## Code Example

```python
import torch
import torchmetrics

# --- Binary MCC ---
preds = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])    # shape: (10,)
target = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])    # shape: (10,)

mcc = torchmetrics.MatthewsCorrCoef(task="binary")
result = mcc(preds, target)
print(f"Binary MCC: {result.item():.4f}")

# --- With probability inputs ---
preds_prob = torch.tensor([0.9, 0.6, 0.2, 0.85, 0.3, 0.7, 0.1, 0.15, 0.8, 0.25])
mcc_prob = torchmetrics.MatthewsCorrCoef(task="binary")
result_prob = mcc_prob(preds_prob, target)
print(f"MCC (prob inputs): {result_prob.item():.4f}")

# --- Demonstrate MCC = 0 for trivial classifier ---
trivial_preds = torch.ones(10, dtype=torch.long)  # predict all positive
mcc_trivial = torchmetrics.MatthewsCorrCoef(task="binary")
result_trivial = mcc_trivial(trivial_preds, target)
print(f"MCC (all-positive): {result_trivial.item():.4f}")  # ≈ 0.0

# --- Multiclass MCC ---
preds_mc = torch.tensor([0, 1, 2, 0, 1, 2, 0, 2])   # shape: (8,)
target_mc = torch.tensor([0, 1, 2, 1, 1, 0, 0, 2])   # shape: (8,)

mcc_mc = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=3)
result_mc = mcc_mc(preds_mc, target_mc)
print(f"Multiclass MCC: {result_mc.item():.4f}")
```

## Debugging Use Case

**Scenario:** Balanced evaluation on imbalanced datasets — verifying a model isn't trivial.

A model on a 95:5 imbalanced dataset reports 96% accuracy. Is it good or trivial?

**Debugging steps:**
1. Compute MCC:
   - All-negative classifier: Accuracy = 95%, **MCC = 0.0** → trivial.
   - Trained model: Accuracy = 96%, **MCC = 0.45** → moderate predictive power.
   - Improved model: Accuracy = 93%, **MCC = 0.62** → better despite lower accuracy.
2. The model with lower accuracy but higher MCC is genuinely better — it correctly identifies more positives (higher recall) at the cost of some false positives (lower precision on negatives).
3. Compare MCC with [F1](../classification/f1_score.md): if MCC = 0.62 but F1 = 0.40, the model may have decent TN performance (captured by MCC, ignored by F1) but poor positive-class performance.
4. Plot MCC vs. threshold to find the threshold that maximizes MCC — often different from the F1-optimal threshold.
5. Use MCC as the **primary selection criterion** for model comparison on imbalanced data.

## Related Metrics

- [Accuracy](../classification/accuracy.md) — simpler but misleading on imbalanced data
- [F1 Score](../classification/f1_score.md) — ignores TN, focuses on positive class
- [Cohen's Kappa](../classification/cohens_kappa.md) — related chance-corrected agreement metric
- [ROC-AUC](../classification/roc_auc.md) — threshold-invariant discrimination
- [PR-AUC](../classification/pr_auc.md) — threshold-invariant, positive-class focused
- [Precision](../classification/precision.md) / [Recall](../classification/recall.md) — decomposed positive-class metrics
- [Specificity](../classification/specificity.md) — TN performance captured by MCC
