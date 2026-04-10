---
title: "Dice Coefficient"
---

# Dice Coefficient

## Overview

The Dice Coefficient (also Sørensen–Dice index) measures the overlap between a predicted segmentation mask and a ground-truth mask. It is mathematically equivalent to the F1 score computed at the pixel level—the harmonic mean of pixel-wise precision and recall. Originally introduced in ecology (Dice 1945, Sørensen 1948), it became the dominant metric in medical image segmentation because it directly optimizes for overlap while being less punitive than IoU on small structures. Most segmentation challenges (BraTS, ISLES, Medical Segmentation Decathlon) report Dice as the primary metric.

Dice is intimately related to the Dice loss (1 − Dice), which is widely used as a training objective because it handles class imbalance better than pixel-wise cross-entropy. When you see "Dice" in a leaderboard, the evaluation is almost always macro-averaged per-class Dice over a dataset.

## Formula

$$
\text{Dice}(A, B) = \frac{2 \, |A \cap B|}{|A| + |B|}
$$

Where $A$ is the set of pixels predicted as positive and $B$ is the set of ground-truth positive pixels.

Equivalently, in terms of the confusion matrix entries TP (true positives), FP (false positives), FN (false negatives):

$$
\text{Dice} = \frac{2 \, \text{TP}}{2 \, \text{TP} + \text{FP} + \text{FN}}
$$

This is identical to the F1 score:

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \text{Dice}
$$

For soft (probabilistic) predictions, the soft Dice replaces set cardinalities with sums of probabilities:

$$
\text{Dice}_{\text{soft}} = \frac{2 \sum_i p_i \, g_i}{\sum_i p_i + \sum_i g_i}
$$

where $p_i \in [0,1]$ is the predicted probability and $g_i \in \{0,1\}$ is the ground-truth label at pixel $i$.

## Visual Diagram

```
Ground Truth (B)          Prediction (A)          Overlap (A ∩ B)
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
│   │ 1 │ 1 │ 1 │   │    │   │   │ 1 │ 1 │ 1 │    │   │   │ 1 │ 1 │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │ 1 │ 1 │ 1 │   │    │   │ 1 │ 1 │ 1 │   │    │   │ 1 │ 1 │ 1 │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │ 1 │ 1 │ 1 │   │    │   │ 1 │ 1 │   │   │    │   │ 1 │ 1 │   │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │   │   │   │   │    │   │   │   │   │   │    │   │   │   │   │   │
└───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘

|B| = 9 pixels             |A| = 8 pixels            |A ∩ B| = 6 pixels

Dice = 2 × 6 / (8 + 9) = 12 / 17 ≈ 0.706
```

## Range & Interpretation

| Dice Value | Interpretation |
|------------|----------------|
| 0.0 | No overlap at all between prediction and ground truth. |
| 0.0–0.3 | Poor segmentation; model is largely missing the target. |
| 0.3–0.6 | Moderate overlap; noticeable under- or over-segmentation. |
| 0.6–0.8 | Good segmentation for difficult tasks (e.g., small lesions). |
| 0.8–0.95 | Strong segmentation; typical target range for clinical use. |
| 1.0 | Perfect pixel-level agreement. |

**Edge case:** When both $A$ and $B$ are empty (no positive pixels in either prediction or ground truth), Dice is conventionally defined as 1.0 (perfect agreement on the absence of the structure). Some implementations return 0.0 or NaN—always verify the library behavior.

Dice is **not linear** in overlap. Moving from 0.80 to 0.90 requires proportionally more overlap improvement than moving from 0.50 to 0.60. This nonlinearity means Dice differences at high values represent larger absolute quality gains than the same numeric difference at low values.

## When to Use

- **Medical image segmentation**: Dice is the standard metric for organ, lesion, and tumor segmentation (BraTS, ACDC, AMOS, KiTS).
- **Binary or per-class evaluation**: When you need a single overlap number per class that balances precision and recall equally.
- **Training objective alignment**: If you train with Dice loss, evaluating with Dice ensures the metric and the loss are directly aligned.
- **Small structure segmentation**: Dice is more forgiving than IoU for small objects (a few misclassified pixels cause a smaller drop in Dice than in IoU).
- **Imbalanced foreground/background**: Dice ignores true negatives, so it is not inflated by large background regions.

## When NOT to Use

- **Multi-class ranking across datasets**: Dice is not directly comparable across tasks with different class frequencies or spatial scales.
- **Boundary-critical applications**: Dice treats all misclassified pixels equally—it does not distinguish interior errors from boundary errors. Use [Boundary F1](boundary_f1.md) or Hausdorff distance instead.
- **Instance segmentation**: Dice operates on semantic masks. For instance-level evaluation, use mask AP or panoptic quality.
- **When both masks may be empty frequently**: The Dice = 1.0 convention for empty masks can inflate macro-averaged scores. Filter or handle empty cases explicitly.
- **Ordinal or regression tasks**: Dice requires discrete (binary or one-hot) labels.

## What It Can Tell You

- Whether your model's predicted region substantially overlaps with the ground truth.
- The balance between over-segmentation (low precision → FP) and under-segmentation (low recall → FN). Decompose: Precision = TP/(TP+FP), Recall = TP/(TP+FN), and Dice is their harmonic mean.
- Relative performance across classes: per-class Dice reveals which structures your model handles well and which it struggles with.
- Training progress: Dice is a smooth, differentiable (in its soft form) metric that tracks learning dynamics well.

## What It Cannot Tell You

- **Where** the errors are located spatially. A Dice of 0.85 could mean uniform boundary erosion or a large hole in the interior—the scalar collapses all spatial information.
- **Distance** between predicted and true boundaries. Two predictions with identical Dice can have very different boundary quality (e.g., smooth vs. jagged contours). Use Hausdorff distance or Average Surface Distance for this.
- **Instance-level correctness**: If two adjacent objects merge into one predicted blob, per-class Dice may still be high.
- **Clinical significance**: A Dice of 0.90 on a large organ is less impressive than 0.90 on a 5-pixel lesion. Always contextualize with object size.

## Sensitivity

- **Object size**: Dice is highly sensitive to object size. For a 10-pixel object, misclassifying 2 pixels drops Dice by ~0.33. For a 1000-pixel object, the same 2 pixels change Dice by ~0.004.
- **FP vs. FN**: Dice penalizes FP and FN symmetrically—each extra FP or FN reduces Dice by the same amount.
- **Class imbalance**: Because Dice ignores TN, it is robust to large backgrounds. However, macro-averaging across classes with vastly different sizes can be dominated by noisy small-class Dice.
- **Threshold**: For probabilistic outputs, Dice depends on the binarization threshold. Report the threshold used, or compute soft Dice.
- **Spatial structure**: Dice is invariant to the spatial arrangement of errors. Clustered and scattered errors of the same count yield identical Dice.

## Alternatives & When to Prefer Them

| Metric | Relationship to Dice | When to Prefer |
|--------|---------------------|----------------|
| [IoU / Jaccard](iou.md) | Dice = 2·IoU/(1+IoU); IoU is always ≤ Dice | When you want a stricter overlap measure; standard in scene parsing (PASCAL VOC, COCO, Cityscapes). |
| [Pixel Accuracy](pixel_accuracy.md) | Counts all pixels including TN | Quick sanity check only; dominated by majority class. |
| Hausdorff Distance (HD95) | Measures worst-case boundary distance | When boundary localization accuracy matters (surgical planning). |
| Average Surface Distance (ASD) | Mean boundary distance | When you need average boundary quality rather than worst-case. |
| [Boundary F1](boundary_f1.md) | F1 on boundary pixels within tolerance | When contour quality matters more than region overlap. |
| Normalized Surface Dice (NSD) | Dice applied to boundary within tolerance τ | Combines boundary and overlap evaluation with a distance tolerance. |
| Panoptic Quality | Combines detection and segmentation quality | Instance + semantic segmentation (panoptic tasks). |

## Code Example

```python
import torch
import torchmetrics

# --- Setup ---
# pred: model output logits or probabilities, shape (N, C, H, W)
# target: ground-truth class indices, shape (N, H, W)
N, C, H, W = 4, 3, 256, 256  # batch=4, classes=3, 256x256 images

pred = torch.randn(N, C, H, W)        # raw logits, shape: (N, C, H, W)
target = torch.randint(0, C, (N, H, W))  # class indices, shape: (N, H, W)

# --- Dice via torchmetrics ---
dice_metric = torchmetrics.Dice(
    num_classes=C,
    average="macro",       # macro-average across classes
    ignore_index=None,     # set to 0 to ignore background
    threshold=0.5,         # binarization threshold for probabilities
)

score = dice_metric(pred, target)  # returns scalar tensor
print(f"Macro Dice: {score.item():.4f}")

# --- Per-class Dice ---
dice_per_class = torchmetrics.Dice(
    num_classes=C,
    average="none",  # returns tensor of shape (C,)
)

per_class = dice_per_class(pred, target)  # shape: (C,)
for cls_idx, d in enumerate(per_class):
    print(f"  Class {cls_idx}: Dice = {d.item():.4f}")

# --- Manual computation for verification ---
pred_hard = pred.argmax(dim=1)  # shape: (N, H, W)
for c in range(C):
    p = (pred_hard == c).float()
    g = (target == c).float()
    intersection = (p * g).sum()
    dice_manual = (2 * intersection) / (p.sum() + g.sum() + 1e-8)
    print(f"  Class {c} (manual): Dice = {dice_manual.item():.4f}")
```

## Debugging Use Case

**Scenario**: A brain tumor segmentation model (U-Net on BraTS) reports a macro Dice of 0.72, but clinicians complain that enhancing tumor boundaries are poor.

**Diagnosis steps:**

1. **Decompose per-class Dice**: Compute Dice for each sub-region (whole tumor, tumor core, enhancing tumor). If enhancing tumor Dice is 0.55 while whole tumor is 0.88, the problem is localized.
2. **Check precision vs. recall for the failing class**: If enhancing tumor precision is 0.80 but recall is 0.42, the model is under-segmenting—it misses enhancing voxels.
3. **Visualize false negatives on slices**: Overlay FN voxels on the MRI. If they cluster at boundaries, the issue is boundary localization. If they appear as missed interior regions, the model lacks sensitivity.
4. **Compare with Hausdorff distance**: If HD95 is high (>10 mm) but Dice is moderate, the model has outlier boundary errors—a few slices with gross misalignment.
5. **Action**: Increase weight of enhancing tumor in Dice loss, apply boundary-aware losses (e.g., HD loss, boundary loss), or use attention mechanisms for small structures. Re-evaluate with both Dice and [Boundary F1](boundary_f1.md) to confirm improvement.

## Related Metrics

- [IoU / Jaccard Index](iou.md) — monotonic transformation of Dice; IoU = Dice/(2−Dice).
- [Pixel Accuracy](pixel_accuracy.md) — simplest segmentation metric; includes TN.
- [Boundary F1](boundary_f1.md) — evaluates contour quality specifically.
- Hausdorff Distance — worst-case boundary error distance.
- Average Surface Distance — mean boundary error distance.
- Normalized Surface Dice — boundary-aware variant of Dice with distance tolerance.
