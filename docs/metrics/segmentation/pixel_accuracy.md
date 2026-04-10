---
title: "Pixel Accuracy"
---

# Pixel Accuracy

## Overview

Pixel Accuracy is the simplest segmentation metric: the fraction of pixels in the image that are assigned the correct class label. It is the direct analogue of classification accuracy applied to every pixel independently. While easy to compute and interpret, Pixel Accuracy is dominated by the majority class in imbalanced segmentation tasks—a model that predicts "background" everywhere on a dataset where 95% of pixels are background achieves 95% Pixel Accuracy despite being completely useless for foreground segmentation.

For this reason, Pixel Accuracy is rarely used as the primary evaluation metric in modern segmentation benchmarks. It serves as a **sanity check** and a complement to mIoU or Dice. PASCAL VOC and Cityscapes report it alongside mIoU for completeness. Mean class accuracy (macro-averaged per-class accuracy) partially addresses the class imbalance issue but remains less informative than IoU-based metrics because it still counts true negatives implicitly through the per-class formulation.

## Formula

**Overall Pixel Accuracy:**

$$
\text{PA} = \frac{\sum_{k=0}^{K-1} \text{TP}_k}{\text{Total Pixels}} = \frac{\text{Number of correctly classified pixels}}{\text{Total number of pixels}}
$$

Equivalently, from the $K \times K$ confusion matrix $C$ where $C_{ij}$ = number of pixels of class $i$ predicted as class $j$:

$$
\text{PA} = \frac{\sum_{k=0}^{K-1} C_{kk}}{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} C_{ij}} = \frac{\text{trace}(C)}{\text{sum}(C)}
$$

**Mean Class Accuracy (Mean Accuracy):**

$$
\text{MCA} = \frac{1}{K} \sum_{k=0}^{K-1} \frac{C_{kk}}{\sum_{j=0}^{K-1} C_{kj}} = \frac{1}{K} \sum_{k=0}^{K-1} \text{Recall}_k
$$

This is the macro-average of per-class recall (sensitivity).

## Visual Diagram

```
Ground Truth                Prediction                  Correct?
┌───┬───┬───┬───┬───┐      ┌───┬───┬───┬───┬───┐      ┌───┬───┬───┬───┬───┐
│ 0 │ 0 │ 1 │ 1 │ 0 │      │ 0 │ 0 │ 0 │ 1 │ 0 │      │ ✓ │ ✓ │ ✗ │ ✓ │ ✓ │
├───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │      │ 0 │ 1 │ 1 │ 1 │ 0 │      │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
├───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┤
│ 0 │ 0 │ 1 │ 0 │ 0 │      │ 0 │ 1 │ 1 │ 0 │ 0 │      │ ✓ │ ✗ │ ✓ │ ✓ │ ✓ │
├───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │      │ 0 │ 0 │ 0 │ 0 │ 0 │      │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
└───┴───┴───┴───┴───┘      └───┴───┴───┴───┴───┘      └───┴───┴───┴───┴───┘

Total pixels: 20       Correct: 18       Incorrect: 2

Pixel Accuracy = 18 / 20 = 0.90

But note: class 0 has 14 pixels (70% of image), class 1 has 6 pixels (30%).
A trivial "all-zero" predictor would get PA = 14/20 = 0.70.
The 0.90 PA looks impressive but only 4/6 = 67% of foreground pixels are correct.
```

## Range & Interpretation

| PA Value | Interpretation |
|----------|----------------|
| 0.0 | Every pixel misclassified (virtually impossible in practice). |
| class_freq_max | Achievable by trivially predicting the majority class everywhere. This is the baseline. |
| 0.90–0.95 | Common range for decent models on imbalanced datasets—may hide poor minority-class performance. |
| 0.95–0.99 | Strong overall, but verify per-class accuracy to ensure minorities are not sacrificed. |
| 1.0 | Perfect pixel-level agreement. |

**Critical baseline:** Always compare PA against the trivial baseline (majority-class frequency). On Cityscapes, "road" + "building" + "vegetation" cover ~75% of pixels, so PA ≥ 0.75 is achievable without any real segmentation.

## When to Use

- **Sanity check**: Verify that your model is learning something beyond majority-class prediction. If PA < majority-class frequency, something is fundamentally broken (data loading, label mapping, loss computation).
- **Alongside mIoU**: Report PA as a secondary metric to give readers a sense of overall per-pixel correctness.
- **Balanced datasets**: When class frequencies are roughly equal, PA is a reasonable primary metric (rare in segmentation).
- **Quick iteration**: During early development, PA is fast to compute and interpret—useful as a dashboard metric before investing in per-class analysis.

## When NOT to Use

- **As the primary metric on imbalanced data**: This is the main pitfall. A model with PA = 0.95 might completely ignore rare but important classes (pedestrians, cyclists in autonomous driving).
- **Comparing models**: Two models with the same PA can have drastically different per-class behavior. Always use mIoU or per-class metrics for model comparison.
- **Medical imaging**: Foreground (lesion, organ) typically occupies <5% of the image. PA is almost entirely determined by background classification accuracy.
- **Detecting regression on rare classes**: A model update that breaks a rare-class segmentation head might cause zero change in PA. Use per-class IoU or Dice to catch this.
- **Any scenario where false negatives on specific classes have high cost**: Autonomous driving (missing a pedestrian), medical imaging (missing a tumor).

## What It Can Tell You

- Whether your model has learned something beyond trivial majority-class prediction.
- The overall fraction of correctly classified pixels—useful when all classes are roughly equally important and equally frequent.
- A quick, interpretable number for non-technical stakeholders who need a single "accuracy" figure.
- A smoke test for data pipeline correctness: if PA suddenly drops to near-chance, your labels or preprocessing are likely broken.

## What It Cannot Tell You

- **Per-class performance**: PA can be 0.95 while a critical class has 0% recall.
- **Spatial quality**: PA says nothing about whether errors are at boundaries, in interiors, or scattered randomly.
- **Model ranking reliability**: Two models with PA 0.93 and 0.94 may differ by 10+ mIoU points on rare classes.
- **Detection of small objects**: A model that misses every small object (pedestrian, sign) can still achieve very high PA.
- **Segmentation quality beyond correct/incorrect**: No notion of distance-to-boundary, smoothness, or instance separation.

## Sensitivity

- **Class frequency**: PA is dominated by the most frequent class. On a dataset where 90% of pixels are class 0, a model predicting all-class-0 achieves PA = 0.90. Improving class-1 recall from 0% to 100% only increases PA by 0.10.
- **Number of classes**: With more classes, the majority-class baseline is lower, making PA slightly more informative—but imbalance still dominates.
- **Image resolution**: Higher resolution means more pixels, so PA variance across images decreases. But it also means more boundary pixels, which are harder to classify correctly.
- **Ignore labels**: Pixels labeled as "ignore" (e.g., 255 in Cityscapes) must be excluded from both numerator and denominator. Failing to exclude them corrupts PA.
- **Spatial distribution of errors**: PA is completely insensitive to where errors occur. Boundary errors and interior errors are treated identically.

## Alternatives & When to Prefer Them

| Metric | Relationship to PA | When to Prefer |
|--------|--------------------|----------------|
| Mean Class Accuracy (MCA) | Macro-average of per-class recall | When you want class-balanced accuracy without full IoU computation. |
| [IoU / Jaccard (mIoU)](iou.md) | Ignores TN; penalizes FP and FN per class | Standard for segmentation benchmarks; handles imbalance properly. |
| [Dice Coefficient](dice.md) | F1 at pixel level; ignores TN | Medical imaging standard; less punitive than IoU. |
| [Boundary F1](boundary_f1.md) | Accuracy at contour pixels only | When boundary quality matters more than bulk region accuracy. |
| Balanced Accuracy | Average of per-class recall | Directly addresses class imbalance; equivalent to MCA for segmentation. |
| Cohen's Kappa | PA corrected for chance agreement | When you need to account for the expected agreement under random prediction. |

## Code Example

```python
import torch
import torchmetrics

# --- Setup ---
# pred: model logits, shape (N, C, H, W)
# target: ground-truth class indices, shape (N, H, W)
N, C, H, W = 2, 19, 512, 1024  # Cityscapes dimensions

pred = torch.randn(N, C, H, W)          # raw logits, shape: (N, C, H, W)
target = torch.randint(0, C, (N, H, W)) # class indices, shape: (N, H, W)

# --- Overall Pixel Accuracy via torchmetrics ---
# Flatten spatial dims: treat each pixel as an independent classification
pixel_acc = torchmetrics.Accuracy(
    task="multiclass",
    num_classes=C,
    average="micro",       # micro = overall pixel accuracy
    ignore_index=255,      # exclude ignore-labeled pixels
)

pa = pixel_acc(pred, target)  # scalar tensor
print(f"Pixel Accuracy: {pa.item():.4f}")

# --- Mean Class Accuracy ---
mean_class_acc = torchmetrics.Accuracy(
    task="multiclass",
    num_classes=C,
    average="macro",       # macro = mean per-class accuracy (recall)
    ignore_index=255,
)

mca = mean_class_acc(pred, target)
print(f"Mean Class Accuracy: {mca.item():.4f}")

# --- Per-class Accuracy ---
per_class_acc = torchmetrics.Accuracy(
    task="multiclass",
    num_classes=C,
    average="none",  # returns shape (C,)
    ignore_index=255,
)

per_class = per_class_acc(pred, target)  # shape: (19,)
for c in range(C):
    print(f"  Class {c}: Accuracy = {per_class[c].item():.4f}")

# --- Manual computation for verification ---
pred_hard = pred.argmax(dim=1)           # shape: (N, H, W)
valid = target != 255                     # boolean mask, shape: (N, H, W)
correct = (pred_hard == target) & valid   # shape: (N, H, W)
pa_manual = correct.sum().float() / valid.sum().float()
print(f"PA (manual): {pa_manual.item():.4f}")

# --- Trivial baseline (majority class) ---
majority_class = target[valid].mode().values.item()
trivial_correct = (target == majority_class) & valid
trivial_pa = trivial_correct.sum().float() / valid.sum().float()
print(f"Trivial baseline PA (always predict class {majority_class}): {trivial_pa.item():.4f}")
```

## Debugging Use Case

**Scenario**: You train a new segmentation model and observe PA = 0.92, which seems reasonable. But mIoU is only 0.35.

**Diagnosis steps:**

1. **Compute trivial baseline PA**: If the majority class covers 90% of pixels, the trivial predictor achieves PA = 0.90. Your model's PA = 0.92 is barely above trivial—it is almost certainly predicting the majority class for most pixels.
2. **Examine per-class accuracy**: Compute per-class recall. If background recall = 0.99 but foreground classes have recall < 0.10, the model has collapsed to near-trivial prediction.
3. **Check prediction distribution**: Count how many pixels are assigned to each class. If the model predicts only 2–3 classes out of 19, it has mode-collapsed.
4. **Verify loss function**: Ensure your loss weights minority classes. Pixel-wise cross-entropy without class weighting or focal loss will drive the model toward majority-class prediction.
5. **Inspect learning curves**: If PA converges quickly but mIoU plateaus low, the model learned the background immediately but never learned foreground classes.
6. **Action**: Switch to Dice loss or weighted cross-entropy. Add OHEM. Verify data augmentation does not crop out foreground objects. Track mIoU (not PA) as the primary metric going forward.

**Key lesson:** PA and mIoU diverging is the strongest signal of class-imbalance-driven collapse. Always report both to catch this failure mode.

## Related Metrics

- [IoU / Jaccard Index (mIoU)](iou.md) — the standard metric that avoids PA's class-imbalance blindness.
- [Dice Coefficient](dice.md) — overlap metric ignoring TN; standard in medical imaging.
- [Boundary F1](boundary_f1.md) — contour-level accuracy.
- Mean Class Accuracy — macro-averaged per-class recall; a partial fix for PA's imbalance sensitivity.
- Cohen's Kappa — PA adjusted for chance agreement.
