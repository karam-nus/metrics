---
title: "Top-K Accuracy"
---
# Top-K Accuracy

## Overview

Top-K Accuracy measures whether the true label is among the model's top K highest-scoring predictions. Standard (top-1) accuracy requires the single highest prediction to be correct; top-K relaxes this to accept any of the K highest. This metric is essential for multiclass problems with many classes—ImageNet (1000 classes), large-scale product categorization, species identification—where the correct class may be semantically close to other high-scoring classes. Top-5 accuracy is the standard reporting metric for ImageNet benchmarks. Top-K accuracy captures whether the model has "narrowed down" the correct answer, even if it isn't the single top prediction.

## Formula

For a single sample with true label $y$ and predicted score vector $\hat{\mathbf{y}} \in \mathbb{R}^C$:

$$
\text{Top-K Correct}(y, \hat{\mathbf{y}}) = \mathbb{1}\left(y \in \text{argsort}_{desc}(\hat{\mathbf{y}})_{1:K}\right)
$$

Over $N$ samples:

$$
\text{Top-K Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left(y_i \in \text{Top-K}(\hat{\mathbf{y}}_i)\right)
$$

Note: Top-1 accuracy is standard [accuracy](../classification/accuracy.md).

## Visual Diagram

```
Example: 5-class problem, K=3

Predicted scores:   [0.05, 0.35, 0.10, 0.40, 0.10]
                     cls0  cls1  cls2  cls3  cls4

Sorted (desc):       cls3(0.40)  cls1(0.35)  cls2(0.10)  cls4(0.10)  cls0(0.05)
                     ─────────── Top-3 ──────────────────
                     rank 1      rank 2      rank 3

True label: cls1

Top-1: cls3 ≠ cls1 → ✗ (incorrect)
Top-3: {cls3, cls1, cls2} ∋ cls1 → ✓ (correct)
Top-5: {cls3, cls1, cls2, cls4, cls0} ∋ cls1 → ✓ (correct, trivially)
```

```
Top-K Accuracy vs K:

Accuracy
1.0 ┤                              ●─────────────
    │                         ●───·
0.9 ┤                    ●───·
    │               ●───·
0.8 ┤          ●───·
    │     ●───·
0.7 ┤●───·
    │
0.6 ┤
    │
    └──┬──┬──┬──┬──┬──┬──┬──┬──┬──
       1  2  3  4  5  6  7  8  9  10
                    K

    Always monotonically non-decreasing with K.
    Top-C accuracy = 1.0 (trivially).
```

## Range & Interpretation

| Value | Interpretation |
|-------|----------------|
| 1.0 | True label always in top-K predictions |
| 0.0 | True label never in top-K |
| Top-1 = Top-K | Model is already confident; top-K adds nothing |
| Top-1 << Top-K | Model frequently has the right answer in top-K but not at rank 1 |

- **Range:** $[0, 1]$
- **Monotonic:** Top-K accuracy ≥ Top-(K-1) accuracy ≥ ... ≥ Top-1 accuracy.
- **Trivial at K=C:** Top-C accuracy = 1.0 (all classes included).
- **Baseline:** For $C$ classes with uniform random prediction, Top-K accuracy = $K/C$.

## When to Use

- **Large-scale multiclass:** ImageNet (top-5), fine-grained classification, product taxonomy.
- When **semantic class overlap** is high — the model may confuse visually similar classes (e.g., different dog breeds), and top-K captures this.
- **Search / recommendation:** Is the correct item in the top-K results?
- **Human-in-the-loop systems:** Show top-K predictions to a human for selection.
- **Benchmarking:** Many standard benchmarks report top-1 and top-5 accuracy.

## When NOT to Use

- **Binary classification:** Top-K is meaningless for K≥1 (always 1.0 for K=2).
- **Few classes (C ≤ 5):** Top-K quickly becomes trivial; standard accuracy suffices.
- **When ranking quality matters:** Use MRR (Mean Reciprocal Rank) or NDCG for richer rank-aware evaluation.
- **When you need class-specific performance:** Use per-class recall or confusion matrix analysis.

## What It Can Tell You

- Whether the model's predictions are "in the right neighborhood," even if not exactly correct.
- The gap between top-1 and top-K accuracy reveals how often the model is "close but not quite right."
- For ensemble methods: if top-5 accuracy is high but top-1 is low, a reranking stage may help.
- Difficulty of the classification task: a large top-1 to top-5 gap suggests many confusable classes.

## What It Cannot Tell You

- The **rank** of the correct class within the top-K (was it rank 2 or rank K?). Use MRR for this.
- **Confidence calibration** — a model may have the right class in top-K with poorly calibrated scores.
- Per-class breakdown — which classes benefit most from the top-K relaxation?
- Whether the model is suitable for deployment where only the top prediction matters.

## Sensitivity

- **K value:** Must be chosen relative to $C$. Top-5 on a 10-class problem is very lenient (50% of classes). Top-5 on a 1000-class problem is strict (0.5%).
- **Number of classes:** More classes make top-K more meaningful but also increase the random baseline.
- **Class similarity:** Datasets with many similar classes (fine-grained) show larger top-1 to top-K gaps.
- **Score distribution:** Flat score distributions (low confidence) inflate top-K relative to top-1.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Advantage |
|--------|-------------|---------------|
| [Accuracy](../classification/accuracy.md) (Top-1) | Exact prediction matters | Strictest evaluation |
| MRR (Mean Reciprocal Rank) | Rank of correct answer matters | Captures position information |
| NDCG | Graded relevance | Weighted rank-aware metric |
| [F1 Score](../classification/f1_score.md) (macro) | Per-class balance needed | Class-level evaluation |
| Precision@K | Information retrieval | Multiple relevant items |
| Hit Rate@K | Recommendation systems | At least one relevant in top-K |

## Code Example

```python
import torch
import torchmetrics

# --- Top-5 Accuracy (ImageNet-style) ---
num_samples = 8
num_classes = 20

# Simulated logits — shape: (8, 20)
preds = torch.randn(num_samples, num_classes)
target = torch.randint(0, num_classes, (num_samples,))  # shape: (8,)

top1 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)

result_top1 = top1(preds, target)
result_top5 = top5(preds, target)
print(f"Top-1 Accuracy: {result_top1.item():.4f}")
print(f"Top-5 Accuracy: {result_top5.item():.4f}")

# --- Concrete example with known predictions ---
preds_known = torch.tensor([
    [0.1, 0.2, 0.5, 0.1, 0.1],   # top-1: cls2, top-3: {cls2, cls1, cls0/cls3/cls4}
    [0.3, 0.3, 0.1, 0.2, 0.1],   # top-1: cls0 or cls1
    [0.05, 0.05, 0.05, 0.8, 0.05],  # top-1: cls3
    [0.1, 0.4, 0.3, 0.1, 0.1],   # top-1: cls1, top-2: {cls1, cls2}
])  # shape: (4, 5)

target_known = torch.tensor([2, 1, 3, 2])  # shape: (4,)

for k in [1, 2, 3]:
    metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=5, top_k=k
    )
    result = metric(preds_known, target_known)
    print(f"Top-{k} Accuracy: {result.item():.4f}")

# --- Accumulating over batches ---
top5_accum = torchmetrics.Accuracy(
    task="multiclass", num_classes=num_classes, top_k=5
)
for i in range(4):
    batch_preds = torch.randn(32, num_classes)   # shape: (32, 20)
    batch_target = torch.randint(0, num_classes, (32,))
    top5_accum.update(batch_preds, batch_target)

print(f"Top-5 Accuracy (accumulated): {top5_accum.compute().item():.4f}")
```

## Debugging Use Case

**Scenario:** ImageNet classification — top-1 accuracy plateaus but top-5 is high.

A ResNet model achieves top-1 accuracy = 0.72 but top-5 accuracy = 0.91 on ImageNet validation. The 19-point gap suggests the model frequently confuses semantically similar classes.

**Debugging steps:**
1. Compute **per-class top-1 vs top-5 gap**: identify classes with the largest gap (e.g., dog breeds, car models).
2. Examine the **confusion matrix** for high-gap classes — are confusions between visually similar classes (e.g., Siberian Husky vs. Alaskan Malamute)?
3. Check the **predicted score distribution**: are the top-2 or top-3 scores very close? If so, the model is uncertain, not wrong.
4. Try **label smoothing** or **knowledge distillation** to improve separation between similar classes.
5. For fine-grained classes, consider:
   - **Hierarchical classification** (coarse → fine).
   - **Part-based models** that focus on discriminative regions.
   - **Increased resolution** input to capture fine details.
6. If deploying in a **human-in-the-loop** system, showing top-5 predictions (with 91% chance of correct) is an effective UX pattern.
7. Compare with **MRR** to understand the average rank of the correct class (not just whether it's in top-K).

## Related Metrics

- [Accuracy](../classification/accuracy.md) — top-1 accuracy (standard accuracy)
- [F1 Score](../classification/f1_score.md) — per-class evaluation
- [Precision](../classification/precision.md) — positive predictive value per class
- [Recall](../classification/recall.md) — per-class detection rate
- [ROC-AUC](../classification/roc_auc.md) — threshold-invariant, but typically binary/one-vs-rest
- [PR-AUC](../classification/pr_auc.md) — threshold-invariant positive-class metric
