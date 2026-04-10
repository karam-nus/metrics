---
title: "IoU / Jaccard Index"
---

# IoU / Jaccard Index

## Overview

Intersection over Union (IoU), also called the Jaccard Index, is the ratio of the overlap between predicted and ground-truth segmentation masks to their union. It is the standard evaluation metric for semantic segmentation in computer vision benchmarks: PASCAL VOC, COCO, Cityscapes, ADE20K, and Mapillary Vistas all report mean IoU (mIoU) as the primary ranking metric.

IoU is stricter than Dice—it penalizes both false positives and false negatives more aggressively for the same amount of overlap. The two metrics are monotonically related (IoU = Dice/(2−Dice)), so they always rank models in the same order, but IoU scores are numerically lower, which can affect perception of quality. IoU generalizes naturally to multi-class evaluation via per-class computation followed by macro-averaging (mIoU), and to instance segmentation via per-instance matching (mask AP).

The Jaccard Index has its origin in set theory (Jaccard 1912) and is sometimes referred to as the Jaccard similarity coefficient in the information retrieval and clustering literature.

## Formula

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

In confusion-matrix terms:

$$
\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}
$$

**Mean IoU (mIoU)** across $K$ classes:

$$
\text{mIoU} = \frac{1}{K} \sum_{k=1}^{K} \text{IoU}_k
$$

**Frequency-weighted IoU (FWIoU):**

$$
\text{FWIoU} = \frac{1}{\sum_k t_k} \sum_{k=1}^{K} t_k \cdot \text{IoU}_k
$$

where $t_k$ is the total number of ground-truth pixels of class $k$.

**Relationship to Dice:**

$$
\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}, \qquad \text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}
$$

## Visual Diagram

```
Ground Truth (B)          Prediction (A)
┌───┬───┬───┬───┬───┐    ┌───┬───┬───┬───┬───┐
│   │ 1 │ 1 │ 1 │   │    │   │   │ 1 │ 1 │ 1 │
├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │ 1 │ 1 │ 1 │   │    │   │ 1 │ 1 │ 1 │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │ 1 │ 1 │ 1 │   │    │   │ 1 │ 1 │   │   │
├───┼───┼───┼───┼───┤    ├───┼───┼───┼───┼───┤
│   │   │   │   │   │    │   │   │   │   │   │
└───┴───┴───┴───┴───┘    └───┴───┴───┴───┴───┘

|B| = 9    |A| = 8    |A ∩ B| = 6    |A ∪ B| = 9 + 8 - 6 = 11

IoU  = 6 / 11 ≈ 0.545
Dice = 12 / 17 ≈ 0.706    (same masks, IoU is always ≤ Dice)

Pixel-level breakdown:
┌───┬───┬───┬───┬───┐
│ TN│ FN│ TP│ TP│ FP│   TP = 6  (both 1)
├───┼───┼───┼───┼───┤   FP = 2  (pred 1, gt 0)
│ TN│ TP│ TP│ TP│ TN│   FN = 3  (pred 0, gt 1)
├───┼───┼───┼───┼───┤   TN = 9  (both 0)
│ TN│ TP│ TP│ FN│ TN│
├───┼───┼───┼───┼───┤   IoU = 6 / (6+2+3) = 6/11
│ TN│ TN│ TN│ TN│ TN│
└───┴───┴───┴───┴───┘
```

## Range & Interpretation

| IoU Value | Interpretation |
|-----------|----------------|
| 0.0 | Zero overlap—prediction and ground truth are completely disjoint. |
| 0.0–0.2 | Very poor; model predictions barely intersect the target. |
| 0.2–0.5 | Weak segmentation; large spatial errors or missing regions. |
| 0.5 | Conventional detection threshold (PASCAL VOC uses IoU ≥ 0.5 for a "match"). |
| 0.5–0.7 | Moderate; acceptable for coarse tasks, insufficient for precise segmentation. |
| 0.7–0.85 | Good; typical mIoU range for competitive Cityscapes models. |
| 0.85–1.0 | Excellent; near-pixel-perfect segmentation. |
| 1.0 | Perfect agreement. |

**Edge case:** When both prediction and ground truth are empty for a class, IoU is conventionally defined as 1.0 (agree on absence). If only one is empty, IoU = 0.0. Some frameworks (e.g., PASCAL VOC evaluation scripts) skip classes not present in the ground truth for mIoU computation. Always verify how your evaluation handles this.

**Nonlinearity:** IoU is more nonlinear than Dice at the extremes. Improving from IoU 0.90 to 0.95 requires eliminating roughly half of all remaining FP+FN pixels—this compression makes high-IoU comparisons more meaningful but also harder to achieve.

## When to Use

- **Semantic segmentation benchmarks**: mIoU is the standard metric for PASCAL VOC, COCO-Stuff, Cityscapes, ADE20K. Use it for apples-to-apples comparison with published results.
- **Object detection overlap**: IoU defines the matching criterion in AP computation (IoU ≥ 0.5 for PASCAL, IoU ∈ {0.5:0.05:0.95} for COCO).
- **Strict overlap requirement**: IoU penalizes errors more than Dice for the same overlap, making it more discriminating in the high-quality regime.
- **Multi-class evaluation**: mIoU naturally balances across classes regardless of their pixel frequency.
- **Frequency-weighted evaluation**: FWIoU is appropriate when class importance scales with prevalence (e.g., autonomous driving where road matters more than poles).

## When NOT to Use

- **Medical imaging (by convention)**: Most medical segmentation challenges and papers report Dice, not IoU. Using IoU will confuse reviewers and break comparability.
- **Very small objects in isolation**: A single misclassified pixel on a 5-pixel object drops IoU from 1.0 to 0.67. Dice drops to 0.83. IoU's harshness can make small-object evaluation noisy.
- **Boundary quality assessment**: IoU does not distinguish boundary errors from interior errors. Use [Boundary F1](boundary_f1.md) or Hausdorff distance.
- **Instance segmentation ranking**: Use mask AP (which uses IoU internally for matching but reports AP curves). Raw per-instance IoU is not aggregated standardly.
- **When training with Dice loss**: If your loss is Dice-based, evaluating with IoU introduces a metric/loss mismatch. They rank identically, but the training signal and metric values diverge.

## What It Can Tell You

- The degree of spatial overlap between predicted and ground-truth regions, penalizing both FP and FN.
- Relative model ranking across different architectures/hyperparameters—IoU and Dice always agree on ordering.
- Per-class breakdown: which classes are well segmented vs. poorly segmented.
- Whether your model meets the IoU threshold required for downstream tasks (e.g., IoU ≥ 0.5 for detection, IoU ≥ 0.75 for "strict" matching in COCO).

## What It Cannot Tell You

- **Boundary precision**: Two masks with IoU = 0.80 may have wildly different boundary quality. One might have smooth contours 2 pixels off; the other might have jagged edges with localized 10-pixel errors.
- **Error location**: IoU is a global scalar. A prediction with a single large hole and one with scattered noise can yield the same IoU.
- **Instance-level performance**: mIoU treats all pixels of a class as one set. If your model merges two adjacent objects into one blob, per-class mIoU may remain high.
- **Clinical or operational significance**: IoU alone does not map to task-specific quality thresholds without domain calibration.

## Sensitivity

- **Object size**: IoU is highly sensitive to object size. For $n$ TP pixels and $e$ error pixels: IoU = $n / (n + e)$. At $n=10, e=3$: IoU = 0.77. At $n=1000, e=3$: IoU = 0.997. Small objects amplify the impact of each pixel error.
- **FP/FN symmetry**: Like Dice, IoU penalizes FP and FN identically—they both contribute to the union but not the intersection.
- **Relative to Dice**: For any overlap, IoU < Dice (except at 0 and 1). The gap is largest at Dice ≈ 0.5 (where IoU ≈ 0.33). At Dice = 0.90, IoU ≈ 0.82.
- **Number of classes**: mIoU treats all classes equally. A model excelling on 18 of 19 Cityscapes classes but failing on "truck" gets penalized proportionally. Consider frequency-weighted IoU if this is undesirable.
- **Threshold sensitivity**: Like Dice, soft-to-hard conversion affects IoU. Evaluate at the threshold used in deployment, or report soft IoU.

## Alternatives & When to Prefer Them

| Metric | Relationship to IoU | When to Prefer |
|--------|---------------------|----------------|
| [Dice Coefficient](dice.md) | Dice = 2·IoU/(1+IoU); monotonically related | Medical imaging conventions; less punitive on small objects. |
| [Pixel Accuracy](pixel_accuracy.md) | Includes TN in numerator and denominator | Quick sanity checks only; misleading under class imbalance. |
| Frequency-Weighted IoU (FWIoU) | Weighted mIoU by class pixel count | When class importance correlates with prevalence. |
| [Boundary F1](boundary_f1.md) | Evaluates contour rather than region | When boundary quality is the primary concern. |
| Hausdorff Distance (HD95) | Max boundary distance (95th percentile) | When worst-case boundary error matters (surgical planning). |
| Panoptic Quality (PQ) | IoU × detection quality | Combined instance detection + segmentation quality. |
| Mask AP | AP over IoU thresholds per instance | Instance segmentation ranking (COCO-style). |

## Code Example

```python
import torch
import torchmetrics

# --- Setup ---
# pred: model output logits, shape (N, C, H, W)
# target: ground-truth class indices, shape (N, H, W)
N, C, H, W = 4, 19, 512, 1024  # Cityscapes: 19 classes, 512x1024
                                 # (or 1024x2048 at full resolution)

pred = torch.randn(N, C, H, W)          # raw logits, shape: (N, C, H, W)
target = torch.randint(0, C, (N, H, W)) # class indices, shape: (N, H, W)

# --- mIoU via torchmetrics ---
iou_metric = torchmetrics.JaccardIndex(
    task="multiclass",
    num_classes=C,
    average="macro",       # macro = mIoU (unweighted mean across classes)
    ignore_index=255,      # Cityscapes uses 255 for "ignore" label
)

miou = iou_metric(pred, target)  # scalar tensor
print(f"mIoU: {miou.item():.4f}")

# --- Per-class IoU ---
iou_per_class = torchmetrics.JaccardIndex(
    task="multiclass",
    num_classes=C,
    average="none",  # returns shape (C,)
)

per_class = iou_per_class(pred, target)  # shape: (19,)
cityscapes_classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle",
]
for cls_idx, (name, iou_val) in enumerate(zip(cityscapes_classes, per_class)):
    print(f"  {name:>15s}: IoU = {iou_val.item():.4f}")

# --- Frequency-weighted IoU (manual) ---
pred_hard = pred.argmax(dim=1)  # shape: (N, H, W)
total_pixels = target.numel()
fwiou = 0.0
for c in range(C):
    p = (pred_hard == c).float()
    g = (target == c).float()
    intersection = (p * g).sum().item()
    union = p.sum().item() + g.sum().item() - intersection
    freq = g.sum().item() / total_pixels
    if union > 0:
        fwiou += freq * (intersection / union)
print(f"FWIoU: {fwiou:.4f}")
```

## Debugging Use Case

**Scenario**: A DeepLabV3+ model on Cityscapes achieves mIoU 0.74, but qualitative inspection shows poor results on thin structures (poles, traffic signs) and rare classes (train, truck).

**Diagnosis steps:**

1. **Per-class IoU breakdown**: Print per-class IoU. Expect: road ~0.97, building ~0.90, pole ~0.55, traffic sign ~0.60, train ~0.40. The low-IoU classes pull down mIoU.
2. **Confusion matrix analysis**: Build a C×C confusion matrix. If "pole" pixels are frequently predicted as "building" or "vegetation," the model lacks spatial resolution at thin structures.
3. **Resolution check**: Poles are 1–3 pixels wide at 512×1024. Downsampling in the encoder may erase them. Try evaluation at full resolution (1024×2048) or use a higher output stride.
4. **Class frequency vs. IoU scatter plot**: Plot each class's pixel frequency against its IoU. If there is a strong positive correlation, the model is underfitting rare classes. Apply class-balanced sampling or weighted loss.
5. **Compare mIoU vs. FWIoU**: If FWIoU is much higher than mIoU (e.g., 0.88 vs. 0.74), the model is strong on frequent classes but fails on rare ones—confirming the class-imbalance hypothesis.
6. **Action**: Increase crop size, use OHEM (Online Hard Example Mining) for rare-class pixels, add auxiliary losses at intermediate scales, or apply class-balanced Dice loss for low-IoU classes. Re-evaluate per-class IoU to confirm targeted improvement.

## Related Metrics

- [Dice Coefficient](dice.md) — monotonically related; Dice = 2·IoU/(1+IoU). Standard in medical imaging.
- [Pixel Accuracy](pixel_accuracy.md) — includes TN; dominated by majority class.
- [Boundary F1](boundary_f1.md) — contour-level evaluation.
- Panoptic Quality — for combined instance + semantic tasks.
- Mask AP — instance segmentation metric built on IoU matching.
