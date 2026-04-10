---
title: "Cohen's Kappa"
---
# Cohen's Kappa

## Overview

Cohen's Kappa ($\kappa$) measures classification agreement adjusted for chance. While accuracy tells you how often the classifier agrees with the ground truth, $\kappa$ tells you how much better this agreement is compared to what would be expected by random chance alone. Originally designed for measuring inter-rater reliability between two human annotators, $\kappa$ is equally applicable to model-vs-ground-truth evaluation. It accounts for the class distribution, making it more informative than raw accuracy on imbalanced datasets. $\kappa$ is closely related to [MCC](../classification/mcc.md) but uses a different chance-correction formula.

## Formula

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where:
- $p_o$ = observed agreement (= accuracy):

$$
p_o = \frac{TP + TN}{N} = \text{Accuracy}
$$

- $p_e$ = expected agreement by chance:

$$
p_e = \frac{(TP + FP)(TP + FN) + (FN + TN)(FP + TN)}{N^2}
$$

For multiclass with $C$ classes and confusion matrix $\mathbf{M}$:

$$
p_o = \frac{\sum_{c=1}^{C} M_{cc}}{N}, \quad p_e = \frac{\sum_{c=1}^{C} \left(\sum_{i} M_{ic}\right)\left(\sum_{j} M_{cj}\right)}{N^2}
$$

**Weighted Kappa** (for ordinal classes, penalizes distant disagreements more):

$$
\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} \cdot M_{ij}}{\sum_{i,j} w_{ij} \cdot E_{ij}}
$$

where $w_{ij}$ are weights (linear or quadratic) and $E_{ij}$ is the expected frequency.

## Visual Diagram

```
Confusion Matrix (Binary):

                 Predicted
              +       -
         +---------+---------+
Actual + |   TP    |   FN    |  row sum: TP+FN
         +---------+---------+
Actual - |   FP    |   TN    |  row sum: FP+TN
         +---------+---------+
col sum:   TP+FP     FN+TN     N

p_o = (TP + TN) / N           ← observed agreement
p_e = (row₁·col₁ + row₂·col₂) / N²  ← expected by chance

κ = (p_o - p_e) / (1 - p_e)   ← excess agreement above chance
```

```
Kappa Interpretation Scale (Landis & Koch, 1977):

 <0.00    0.00-0.20  0.21-0.40  0.41-0.60  0.61-0.80  0.81-1.00
  Poor     Slight      Fair     Moderate  Substantial   Almost
                                                        Perfect
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Perfect agreement (beyond chance) |
| 0.81–1.00 | Almost perfect agreement |
| 0.61–0.80 | Substantial agreement |
| 0.41–0.60 | Moderate agreement |
| 0.21–0.40 | Fair agreement |
| 0.0 | Agreement equals chance expectation |
| < 0.0 | Agreement worse than chance (systematic disagreement) |

- **Range:** $[-1, 1]$ (theoretically; in practice, negative values are rare and indicate systematic errors).
- $\kappa = 0$ does not mean 0% accuracy — it means accuracy equals what random agreement would produce.
- $\kappa$ depends on class prevalence: with extreme imbalance, $p_e$ is high, making $\kappa$ harder to increase.

## When to Use

- **Inter-annotator agreement:** Measuring consistency between two human labelers.
- **Model vs. human comparison:** Does the model agree with humans better than chance?
- **Ordinal classification:** Weighted $\kappa$ penalizes distant disagreements (e.g., rating scales 1–5).
- **Chance-corrected evaluation** on imbalanced datasets where accuracy is misleading.
- **Regulatory / publication requirements:** Some fields (medical, psychological) require $\kappa$ for validation.

## When NOT to Use

- **When the two raters have different marginal distributions:** Use Scott's $\pi$ or Krippendorff's $\alpha$ instead.
- **Ranking/scoring tasks:** $\kappa$ requires hard class assignments.
- **When threshold-independence is needed:** Use [ROC-AUC](../classification/roc_auc.md).
- **Binary classification where all four quadrants matter equally:** [MCC](../classification/mcc.md) is often preferred (proportional to $\kappa$ for 2×2 but has clearer geometric meaning).

## What It Can Tell You

- Whether agreement exceeds chance expectation — the key question for annotation quality.
- Comparative reliability: Annotator A vs. B (κ=0.72) vs. Annotator A vs. C (κ=0.55).
- Model quality relative to chance, accounting for class distribution.
- For weighted $\kappa$: how far off the disagreements are (severity of errors).

## What It Cannot Tell You

- Which classes are being confused — examine the confusion matrix.
- The direction of disagreement (who over-predicts which class).
- Threshold-independent discrimination — $\kappa$ is at a single operating point.
- Whether the metric is reliable with small sample sizes (use bootstrap CIs).

## Sensitivity

- **Prevalence / class imbalance:** With extreme imbalance, $p_e$ approaches $p_o$, compressing $\kappa$ toward 0 even for decent classifiers. This is sometimes called the "kappa paradox."
- **Marginal homogeneity:** $\kappa$ assumes both raters (or the model and truth) have similar marginal distributions. Violations bias $\kappa$ downward.
- **Number of categories:** More categories generally lower $p_e$, making higher $\kappa$ easier to achieve.
- **Weighting scheme:** Linear vs. quadratic weights produce different $\kappa_w$ values; report the scheme used.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [MCC](../classification/mcc.md) | Binary classification | Correlation-based, uses all 4 quadrants directly |
| [Accuracy](../classification/accuracy.md) | Balanced data, simple reporting | No chance correction needed |
| [F1 Score](../classification/f1_score.md) | Focus on positive class | Ignores TN, standard in NLP |
| Krippendorff's α | Multiple annotators, missing data | Handles >2 raters |
| Scott's π | Raters with different marginals | Corrects for marginal asymmetry |
| Fleiss' κ | Multiple raters (>2) | Extends Cohen's κ to more raters |
| [ROC-AUC](../classification/roc_auc.md) | Threshold-independence | Summarizes all thresholds |

## Code Example

```python
import torch
import torchmetrics

# --- Binary Cohen's Kappa ---
preds = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])    # shape: (10,)
target = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 0, 0])    # shape: (10,)

kappa = torchmetrics.CohenKappa(task="binary")
result = kappa(preds, target)
print(f"Binary Cohen's Kappa: {result.item():.4f}")

# --- Compare with accuracy to see chance correction ---
acc = torchmetrics.Accuracy(task="binary")
acc_result = acc(preds, target)
print(f"Accuracy: {acc_result.item():.4f}")
print(f"Kappa adjusts for chance agreement in accuracy")

# --- Multiclass with weighted kappa (ordinal) ---
# Rating predictions on a 0-4 scale (ordinal)
preds_ord = torch.tensor([0, 1, 2, 3, 4, 2, 1, 3, 4, 0])  # shape: (10,)
target_ord = torch.tensor([0, 1, 3, 3, 4, 1, 1, 2, 4, 0])  # shape: (10,)

kappa_linear = torchmetrics.CohenKappa(
    task="multiclass", num_classes=5, weights="linear"
)
result_linear = kappa_linear(preds_ord, target_ord)
print(f"Weighted Kappa (linear): {result_linear.item():.4f}")

kappa_quadratic = torchmetrics.CohenKappa(
    task="multiclass", num_classes=5, weights="quadratic"
)
result_quadratic = kappa_quadratic(preds_ord, target_ord)
print(f"Weighted Kappa (quadratic): {result_quadratic.item():.4f}")
```

## Debugging Use Case

**Scenario:** Inter-annotator agreement — model vs. human comparison.

A text classification model is compared against human annotators. Human-human κ = 0.78, Model-human κ = 0.61.

**Debugging steps:**
1. The model's agreement exceeds chance (κ > 0) but is substantially below human-human level.
2. Compute **per-class κ** or examine the confusion matrix to find which classes drive disagreement.
3. Compute the **confusion matrix** between model and each annotator separately — does the model systematically disagree on certain classes?
4. Check the **marginal distributions**: if the model predicts class A 40% of the time but humans label it 25%, marginal asymmetry biases κ down.
5. Compute **accuracy alongside κ**: if accuracy = 0.82 and κ = 0.61, the gap reveals that much of the accuracy is due to chance agreement (high $p_e$).
6. For ordinal labels, compare **linear vs. quadratic weighted κ** — quadratic penalizes distant errors more heavily, revealing whether errors are near-misses or far-off.
7. Target: improve model κ to be within 0.05 of human-human κ for production readiness.

## Related Metrics

- [MCC](../classification/mcc.md) — closely related; Pearson correlation of binary predictions
- [Accuracy](../classification/accuracy.md) — the numerator component ($p_o$) of Kappa
- [F1 Score](../classification/f1_score.md) — imbalanced evaluation without chance correction
- [Specificity](../classification/specificity.md) — negative class agreement
- [Precision](../classification/precision.md) / [Recall](../classification/recall.md) — decomposed positive class metrics
