---
title: "F1 Score"
---
# F1 Score

## Overview

The F1 Score is the harmonic mean of precision and recall, providing a single number that balances both concerns. It penalizes extreme trade-offs: if either precision or recall is low, the F1 score is pulled down disproportionately. This makes F1 the default go-to metric for imbalanced classification problems where accuracy is misleading. F1 is a special case of the $F_\beta$ family with $\beta = 1$ (equal weight to precision and recall). For multiclass problems, micro, macro, and weighted averaging strategies produce meaningfully different results.

## Formula

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

General $F_\beta$ (weights recall $\beta$ times as much as precision):

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

**Multiclass aggregation variants:**

| Variant | Formula | Use Case |
|---------|---------|----------|
| **Micro** | $F_1 = \frac{2\sum_c TP_c}{2\sum_c TP_c + \sum_c FP_c + \sum_c FN_c}$ | Global performance; dominated by frequent classes |
| **Macro** | $F_1 = \frac{1}{C}\sum_{c=1}^{C} F_{1,c}$ | Equal importance per class; harsh on rare classes |
| **Weighted** | $F_1 = \sum_{c=1}^{C} \frac{n_c}{N} \cdot F_{1,c}$ | Accounts for class frequency |

## Visual Diagram

```
                    Precision
                   ┌─────────┐
                   │         │
                   │    P    │
              ┌────┤         ├────┐
              │    └────┬────┘    │
              │         │         │
     ┌────────▼─────────▼────────▼────────┐
     │     Harmonic Mean (F1)             │
     │     = 2·P·R / (P + R)             │
     │                                     │
     │     Unlike arithmetic mean,         │
     │     penalizes if either is low:     │
     │     P=1.0, R=0.1 → F1=0.18        │
     │     P=0.5, R=0.5 → F1=0.50        │
     └────────▲─────────▲────────▲────────┘
              │         │         │
              │    ┌────┴────┐    │
              └────┤         ├────┘
                   │    R    │
                   │         │
                   └─────────┘
                    Recall
```

```
Confusion Matrix:

                 Predicted
              +       -
         +---------+---------+
Actual + |   TP    |   FN    |
         +---------+---------+
Actual - |   FP    |   TN    |
         +---------+---------+

F1 = 2·TP / (2·TP + FP + FN)
     ▲ ignores TN entirely
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | Perfect precision and recall |
| 0.5 | Moderate — one or both of P, R are mediocre |
| 0.0 | Either precision or recall (or both) is zero |

- **Range:** $[0, 1]$
- F1 = 0 whenever TP = 0 (no correct positive predictions).
- F1 is **undefined** when there are no positive predictions and no actual positives; typically set to 0.0.
- The harmonic mean is always ≤ the arithmetic mean: $F_1 \leq \frac{P + R}{2}$.
- F1 is **symmetric in P and R** — it does not inherently favor one over the other.

## When to Use

- **Imbalanced binary classification** where accuracy is misleading (e.g., fraud detection, rare disease, anomaly detection).
- **NER, POS tagging, and structured prediction** — F1 is the standard metric in NLP entity-level evaluation.
- When you need a **single number** that captures both FP and FN performance.
- **Model selection** when you want a balanced precision-recall trade-off.
- **Macro F1** for multiclass with imbalanced classes where minority class performance matters.

## When NOT to Use

- **When TN matters:** F1 ignores true negatives. Use [MCC](../classification/mcc.md) or [accuracy](../classification/accuracy.md).
- **Asymmetric cost settings:** If FP and FN have very different costs, use $F_\beta$ with appropriate $\beta$, or directly optimize precision/recall.
- **Ranking/scoring tasks:** F1 requires a hard threshold; use [ROC-AUC](../classification/roc_auc.md) or [PR-AUC](../classification/pr_auc.md).
- **When you need threshold-independence:** F1 at default threshold ≠ model quality across thresholds.

## What It Can Tell You

- Whether the model balances precision and recall effectively.
- That a model with F1 = 0.4 has a severe weakness in at least one of P or R.
- Comparison between models under the same precision-recall trade-off assumption.
- Per-class F1 reveals which classes are well-handled vs. poorly-handled.

## What It Cannot Tell You

- Which of P or R is the bottleneck — decompose into precision and recall.
- How the model handles the negative class (TN is excluded from F1).
- Ranking quality or calibration of predicted probabilities.
- Whether the model is better overall when cost asymmetry exists.

## Sensitivity

- **Harmonic mean effect:** If P = 0.95 and R = 0.10, F1 = 0.18 — the low component dominates.
- **Threshold:** F1 changes with the decision threshold. The threshold that maximizes F1 is not necessarily 0.5.
- **Class imbalance:** Micro F1 is biased toward the majority class. Macro F1 weights all classes equally (can be harsh on rare classes with few TP). Weighted F1 balances by support.
- **$\beta$ parameter:** $F_2$ weights recall twice as much; $F_{0.5}$ weights precision twice as much.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [MCC](../classification/mcc.md) | Need all-quadrant evaluation including TN | Balanced, uses full confusion matrix |
| [Precision](../classification/precision.md) | FP cost dominates | Direct FP control |
| [Recall](../classification/recall.md) | FN cost dominates | Direct FN control |
| [PR-AUC](../classification/pr_auc.md) | Threshold-free + imbalanced data | Summarizes full P-R curve |
| [ROC-AUC](../classification/roc_auc.md) | Threshold-free + balanced data | Discrimination across all thresholds |
| $F_\beta$ ($\beta \neq 1$) | Asymmetric P-R importance | Tunable P-R weighting |
| [Cohen's Kappa](../classification/cohens_kappa.md) | Chance-corrected evaluation | Adjusts for expected agreement |

## Code Example

```python
import torch
import torchmetrics

# --- Binary F1 ---
preds = torch.tensor([1, 1, 0, 1, 0, 1, 1, 0, 0, 1])    # shape: (10,)
target = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0, 1, 1])    # shape: (10,)

f1 = torchmetrics.F1Score(task="binary")
result = f1(preds, target)
print(f"Binary F1: {result.item():.4f}")

# --- Multiclass micro/macro/weighted ---
preds_mc = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # shape: (10,)
target_mc = torch.tensor([0, 1, 2, 1, 0, 2, 0, 2, 1, 0])  # shape: (10,)

for avg in ["micro", "macro", "weighted"]:
    f1_mc = torchmetrics.F1Score(
        task="multiclass", num_classes=3, average=avg
    )
    r = f1_mc(preds_mc, target_mc)
    print(f"F1 ({avg}): {r.item():.4f}")

# --- F-beta (beta=2, recall-heavy) ---
f2 = torchmetrics.FBetaScore(task="binary", beta=2.0)
result_f2 = f2(preds, target)
print(f"F2 Score: {result_f2.item():.4f}")

# --- Probability inputs with threshold ---
preds_prob = torch.randn(100).sigmoid()  # shape: (100,)
target_rand = torch.randint(0, 2, (100,))  # shape: (100,)

f1_prob = torchmetrics.F1Score(task="binary", threshold=0.5)
result_prob = f1_prob(preds_prob, target_rand)
print(f"F1 (prob inputs): {result_prob.item():.4f}")
```

## Debugging Use Case

**Scenario:** Named Entity Recognition (NER) model evaluation — low entity-level F1.

A NER model reports token-level accuracy of 96% but entity-level F1 of 0.58. The gap indicates that while individual tokens are classified well, entity boundaries are frequently wrong.

**Debugging steps:**
1. Decompose F1 into **precision and recall** per entity type:
   - `PER`: P=0.82, R=0.90, F1=0.86 ✓
   - `ORG`: P=0.45, R=0.38, F1=0.41 ✗
   - `LOC`: P=0.70, R=0.62, F1=0.66 ~
2. `ORG` is the bottleneck — both P and R are low. Check if organization names are underrepresented in training data.
3. Examine **false positives** for ORG — are common nouns or product names being tagged as organizations?
4. Examine **false negatives** for ORG — are multi-token org names partially matched but counted as FN in strict matching?
5. Compare **micro vs. macro F1** — if micro >> macro, frequent entities inflate the score.
6. Switch from strict to **partial entity matching** (if applicable) to quantify boundary errors.
7. Augment training data for underperforming entity types or use a **CRF layer** to improve boundary detection.

## Related Metrics

- [Precision](../classification/precision.md) — the P component of F1
- [Recall](../classification/recall.md) — the R component of F1
- [PR-AUC](../classification/pr_auc.md) — area under the precision-recall curve
- [MCC](../classification/mcc.md) — balanced metric that includes TN
- [Accuracy](../classification/accuracy.md) — overall correctness (often misleading vs. F1)
- [ROC-AUC](../classification/roc_auc.md) — threshold-invariant discrimination
- [Cohen's Kappa](../classification/cohens_kappa.md) — chance-corrected agreement
