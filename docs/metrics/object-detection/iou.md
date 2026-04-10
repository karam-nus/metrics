---
title: "Intersection over Union (IoU)"
---

# Intersection over Union (IoU)

## Overview

Intersection over Union (IoU), also known as the Jaccard index for bounding boxes, is the foundational localization metric in object detection. It quantifies the spatial overlap between a predicted bounding box and a ground-truth bounding box as a ratio of their intersection area to their union area. IoU serves a dual role: (1) as a matching criterion in evaluation protocols (a detection is a true positive if IoU ≥ threshold), and (2) as a direct quality measure for individual predictions. Every major detection evaluation metric—mAP, AR, PASCAL VOC AP—relies on IoU thresholds for TP/FP assignment. IoU also appears in Non-Maximum Suppression (NMS), where overlapping predictions above an IoU threshold are suppressed to remove duplicates. Despite its simplicity, IoU has known limitations: it is zero for non-overlapping boxes (providing no gradient direction for learning), it does not capture how far apart non-overlapping boxes are, and it treats all overlapping configurations equally regardless of center alignment or aspect ratio agreement. These limitations motivated the GIoU, DIoU, and CIoU extensions.

## Formula

For two axis-aligned bounding boxes $A$ (predicted) and $B$ (ground truth):

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

In coordinate form, with boxes defined as $(x_1, y_1, x_2, y_2)$:

$$
x_I^{(1)} = \max(x_1^A, x_1^B), \quad y_I^{(1)} = \max(y_1^A, y_1^B)
$$

$$
x_I^{(2)} = \min(x_2^A, x_2^B), \quad y_I^{(2)} = \min(y_2^A, y_2^B)
$$

$$
\text{Intersection} = \max(0, x_I^{(2)} - x_I^{(1)}) \times \max(0, y_I^{(2)} - y_I^{(1)})
$$

$$
\text{Union} = (x_2^A - x_1^A)(y_2^A - y_1^A) + (x_2^B - x_1^B)(y_2^B - y_1^B) - \text{Intersection}
$$

$$
\text{IoU} = \frac{\text{Intersection}}{\text{Union}}
$$

## Visual Diagram

```
  Non-overlapping (IoU = 0.0):

  ┌──────────┐
  │    A      │                    ┌──────────┐
  │           │                    │    B      │
  └──────────┘                    │           │
                                   └──────────┘


  Partial overlap (0 < IoU < 1):

  ┌──────────────┐
  │       A       │
  │     ┌─────────┼──────────┐
  │     │/////////│          │
  │     │//INTER//│          │
  └─────┼─────────┘    B     │
        │                    │
        └────────────────────┘

  IoU = dark region / (A + B - dark region)


  Perfect overlap (IoU = 1.0):

  ┌──────────────┐
  │//////////////│
  │///A == B/////│
  │//////////////│
  └──────────────┘


  High IoU (~0.8):                    Low IoU (~0.2):

  ┌──────────────┐                    ┌──────────────┐
  │┌────────────┐│                    │ A             │
  ││////////////││                    │         ┌─────┼────┐
  ││////A∩B/////││                    │         │/////│    │
  │└────────────┘│                    └─────────┼─────┘    │
  └──────────────┘                              │    B     │
   A ≈ B                                        └──────────┘
```

## Range & Interpretation

| Value | Interpretation |
|-------|---------------|
| 0.0 | No overlap whatsoever between predicted and ground-truth boxes |
| 0.0–0.25 | Negligible overlap; prediction is largely mislocalized |
| 0.25–0.50 | Weak overlap; some spatial agreement but poor localization |
| 0.50 | Standard TP threshold in PASCAL VOC; coarse localization |
| 0.50–0.75 | Moderate to good overlap; acceptable for many applications |
| 0.75 | Strict COCO threshold (AP@0.75); good localization required |
| 0.75–0.90 | Strong overlap; high-quality localization |
| 0.90–1.0 | Near-perfect to perfect alignment |
| 1.0 | Exact match; predicted box equals ground-truth box |

IoU is **symmetric**: IoU(A, B) = IoU(B, A).

## When to Use

- **TP/FP assignment** in detection evaluation (the role IoU plays in mAP and AR).
- **NMS overlap criterion**: suppress duplicate detections above an IoU threshold.
- **Anchor box analysis**: measuring how well anchor boxes cover ground-truth distributions.
- **Localization quality reporting**: per-detection IoU as a diagnostic.
- **Simple pairwise box comparison** when you need a single overlap score.
- **Threshold selection**: sweeping IoU thresholds to understand localization sensitivity.

## When NOT to Use

- **As a training loss for non-overlapping boxes**: IoU is zero with zero gradient when boxes do not overlap. Use [GIoU](giou.md), [DIoU](diou.md), or [CIoU](ciou.md) instead.
- **When center-point distance matters**: two configurations can have the same IoU but very different center alignment. Use [DIoU](diou.md).
- **When aspect ratio consistency matters**: IoU does not penalize aspect ratio mismatch. Use [CIoU](ciou.md).
- **For rotated or oriented boxes**: standard IoU assumes axis-aligned boxes. Use rotated IoU implementations.
- **For comparing masks or polygons directly**: use mask IoU (pixel-level intersection/union).

## What It Can Tell You

- The spatial overlap quality of a single predicted box relative to ground truth.
- Whether a detection should be classified as TP or FP at a given threshold.
- How well anchor box priors match the ground-truth box distribution.
- The localization difficulty of a dataset (distribution of max achievable IoU per ground truth).

## What It Cannot Tell You

- **Direction of misalignment**: two boxes with IoU=0.3 could differ in position, scale, or aspect ratio—IoU conflates all these.
- **How to improve**: zero IoU gives no signal about which direction to move the box.
- **Classification quality**: IoU is purely geometric; a high-IoU box can have the wrong class label.
- **Global detection quality**: IoU is per-pair; you need mAP or AR for dataset-level assessment.

## Sensitivity

- **Scale**: IoU is scale-invariant (multiplying all coordinates by a constant does not change IoU). This is desirable.
- **Translation**: small translations cause larger IoU drops for small boxes than large boxes (same pixel shift is a larger fraction of a small box).
- **Aspect ratio**: IoU does not explicitly penalize aspect ratio mismatch, but extreme mismatches naturally reduce overlap.
- **Box size disparity**: if predicted box is much larger than ground truth, IoU can still be high because intersection ≈ ground truth area, but the large union prevents IoU from reaching 1.
- **Non-overlap cliff**: IoU drops to exactly 0 the instant boxes stop overlapping, regardless of proximity. This discontinuity is the primary motivation for GIoU/DIoU/CIoU.

## Alternatives & When to Prefer Them

| Metric | Relationship to IoU | When to Prefer |
|--------|-------------------|----------------|
| [GIoU](giou.md) | IoU minus enclosing-area penalty | Training loss; handles non-overlapping boxes |
| [DIoU](diou.md) | IoU plus center-distance penalty | Training loss; faster convergence than GIoU |
| [CIoU](ciou.md) | DIoU plus aspect-ratio penalty | Training loss; most comprehensive regression signal |
| Mask IoU | Pixel-level intersection/union | Instance segmentation evaluation |
| Rotated IoU | IoU for oriented bounding boxes | Aerial/satellite image detection |
| Corner Distance (L2) | Euclidean distance between corners | When absolute pixel error matters more than overlap |

## Code Example

```python
import torch
from torchvision.ops import box_iou

# Boxes in (x1, y1, x2, y2) format — top-left and bottom-right corners
# predictions: shape (N, 4)
pred_boxes = torch.tensor([
    [100.0, 100.0, 200.0, 200.0],  # prediction 0
    [150.0, 150.0, 250.0, 250.0],  # prediction 1
    [300.0, 300.0, 350.0, 350.0],  # prediction 2 (no overlap with GT)
])

# ground truth: shape (M, 4)
gt_boxes = torch.tensor([
    [110.0, 105.0, 210.0, 205.0],  # ground truth 0
    [145.0, 140.0, 245.0, 240.0],  # ground truth 1
])

# Pairwise IoU: returns (N, M) matrix
iou_matrix = box_iou(pred_boxes, gt_boxes)
print(f"IoU matrix (N={pred_boxes.shape[0]}, M={gt_boxes.shape[0]}):")
print(iou_matrix)
# Each element [i, j] is IoU between pred_boxes[i] and gt_boxes[j]

# Typical usage: find best matching GT for each prediction
best_iou, best_gt_idx = iou_matrix.max(dim=1)
print(f"\nBest IoU per prediction: {best_iou}")
print(f"Matched GT index:        {best_gt_idx}")

# Threshold at IoU ≥ 0.5 for TP/FP
tp_mask = best_iou >= 0.5
print(f"TP mask (IoU ≥ 0.5):     {tp_mask}")
```

## Debugging Use Case

**Scenario: Anchor box quality assessment for a custom dataset**

```
Symptom:  Single-stage detector (e.g., RetinaNet) has low recall despite
          sufficient training. mAP plateaus early.

Diagnosis using IoU:
  1. Compute IoU between all anchor boxes and all ground-truth boxes.
  2. For each GT box, find the maximum IoU across all anchors.
  3. Plot the distribution of max IoU values.

  max_iou_per_gt = box_iou(anchors, gt_boxes).max(dim=0).values
  # If many GT boxes have max_iou < 0.5, anchors cannot cover them

  Distribution analysis:
  ┌──────────────────────────────────────────────┐
  │  Count                                        │
  │  ████                                         │
  │  ████                                         │
  │  ████ ██                                      │
  │  ████ ████                                    │
  │  ████ ████ ████                    ████ ████  │
  │  ████ ████ ████ ████ ████ ████ ████ ████ ████ │
  └──┬────┬────┬────┬────┬────┬────┬────┬────┬──► │
    0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  │
                                    max IoU        │

  Problem: bimodal — many GT boxes below 0.4 → anchor mismatch.

Action:
  - Analyze GT aspect ratios and scales vs anchor configurations.
  - Add anchor ratios/scales to cover the underrepresented GT boxes.
  - Use k-means clustering on GT box dimensions (like YOLO) to derive
    optimal anchor priors.
  - After adjustment, re-plot: most GT boxes should have max_iou > 0.5.
```

## Related Metrics

- [Mean Average Precision (mAP)](map.md) — uses IoU thresholds for TP/FP matching
- [Average Recall (AR)](average_recall.md) — uses IoU thresholds to determine successful detections
- [Generalized IoU (GIoU)](giou.md) — extends IoU for non-overlapping boxes
- [Distance IoU (DIoU)](diou.md) — extends IoU with center-point distance
- [Complete IoU (CIoU)](ciou.md) — extends IoU with distance and aspect ratio
