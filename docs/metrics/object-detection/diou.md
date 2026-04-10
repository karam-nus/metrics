---
title: "Distance Intersection over Union (DIoU)"
---

# Distance Intersection over Union (DIoU)

## Overview

Distance IoU (DIoU), introduced by Zheng et al. (AAAI 2020), extends IoU by incorporating a normalized center-point distance penalty. While [GIoU](giou.md) addresses the zero-gradient problem of non-overlapping boxes through an enclosing-area term, it converges slowly because it indirectly encourages overlap via box expansion rather than directly minimizing the spatial gap. DIoU solves this by adding a penalty proportional to the squared Euclidean distance between box centers, normalized by the diagonal length of the smallest enclosing box. This provides a direct, continuous gradient signal that pulls predicted box centers toward ground-truth centers. DIoU converges significantly faster than GIoU in practice, particularly for boxes that are far apart or small. DIoU is also used as a replacement for IoU in NMS (DIoU-NMS), where it suppresses overlapping boxes while giving preference to centrally aligned detections, improving performance in crowded scenes.

## Formula

Given predicted box $A$ with center $(c_x^A, c_y^A)$ and ground-truth box $B$ with center $(c_x^B, c_y^B)$, let $C$ be the smallest enclosing box with diagonal length $d_C$:

$$
\rho^2(A, B) = (c_x^A - c_x^B)^2 + (c_y^A - c_y^B)^2
$$

$$
d_C^2 = \text{diagonal length}^2 \text{ of smallest enclosing box } C
$$

$$
\text{DIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(A, B)}{d_C^2}
$$

The DIoU loss:

$$
\mathcal{L}_{\text{DIoU}} = 1 - \text{DIoU}(A, B) = 1 - \text{IoU}(A, B) + \frac{\rho^2(A, B)}{d_C^2}
$$

The penalty term $\rho^2 / d_C^2$ is always in $[0, 1]$:
- 0 when centers coincide
- 1 when centers are at opposite corners of the enclosing box (theoretical maximum)

## Visual Diagram

```
  DIoU penalizes center-point distance:

  C (smallest enclosing box, diagonal = d_C)
  ┌──────────────────────────────────────┐
  │                                      │
  │   ┌──────────┐                       │
  │   │    A      │         ρ            │
  │   │     ●─ ─ ─ ─ ─ ─ ─ ─ ─●         │
  │   │   center  │        center        │
  │   └──────────┘    ┌──────────┐       │
  │                   │    B      │       │
  │                   │           │       │
  │                   └──────────┘       │
  │                                      │
  └──────────────────────────────────────┘

  DIoU = IoU - (ρ² / d_C²)

  ● = box center
  ρ = Euclidean distance between centers
  d_C = diagonal of enclosing box C


  Comparison: same IoU, different DIoU

  Configuration 1 (high DIoU):       Configuration 2 (low DIoU):

  ┌─────────────┐                     ┌──────────┐
  │   ┌─────────┼──┐                  │  A       │
  │   │/////////│  │                  │     ●    │
  │   │///●/////│  │                  └──────────┘
  │   │/////////│  │                       ρ (large)
  └───┼─────────┘  │                  ┌──────────┐
      │     ●      │                  │  B       │
      └────────────┘                  │     ●    │
       ρ (small)                      └──────────┘

  Same IoU ≈ 0.3, but Config 1 has higher DIoU
  because centers are closer together.
```

## Range & Interpretation

| Value | Interpretation |
|-------|---------------|
| −1.0 | Theoretical minimum: no overlap, centers at maximum distance in enclosing box |
| −1.0 to 0.0 | Non-overlapping or barely overlapping boxes with significant center distance |
| 0.0 | IoU exactly equals the center distance penalty |
| 0.0 to 0.5 | Moderate overlap with some center misalignment |
| 0.5 to 0.9 | Good overlap and center alignment |
| 0.9 to 1.0 | Excellent localization: high overlap and near-concentric centers |
| 1.0 | Perfect overlap with coincident centers (identical boxes) |

DIoU ≤ IoU always (the penalty term is non-negative). Equality holds only when centers are coincident.

## When to Use

- **Bounding-box regression loss** when faster convergence than GIoU is desired.
- **Small object detection**: center alignment is critical for small objects where minor center shifts cause large IoU drops.
- **DIoU-NMS**: replacing IoU-NMS to better handle occluded/crowded objects by considering center distance in suppression decisions.
- When the predicted box center tends to be far from the ground truth during early training (e.g., anchor-free detectors with poor initialization).
- **Complementary to GIoU**: if GIoU training shows the "box expansion" artifact (oversized predictions), DIoU avoids this by directly minimizing center distance.

## When NOT to Use

- **When aspect ratio consistency also matters**: DIoU does not penalize aspect ratio mismatch. A predicted box with correct center and area but wrong aspect ratio gets no additional penalty. Use [CIoU](ciou.md) for this.
- **As an evaluation metric**: standard benchmarks use IoU thresholds in mAP/AR computation. DIoU is a training loss, not an evaluation metric.
- **When boxes always heavily overlap**: if initialization guarantees good overlap (e.g., high-quality proposals in two-stage detectors), the center distance term adds complexity with diminishing returns. Standard IoU loss may suffice.

## What It Can Tell You

- How well the predicted box center aligns with the ground-truth center, normalized by the spatial extent of the detection context.
- Whether center alignment is the primary localization error (compare DIoU vs IoU: large gap implies center misalignment).
- Training convergence characteristics: DIoU loss should decrease faster than GIoU loss in early epochs.

## What It Cannot Tell You

- **Aspect ratio alignment**: a tall, narrow box centered on a short, wide ground-truth box can have high DIoU if the overlap and center coincide.
- **Classification correctness**: purely geometric, class-agnostic.
- **Absolute spatial error**: the normalization by enclosing box diagonal means the same center distance gets different penalties depending on how far apart the boxes are overall.

## Sensitivity

- **Center distance**: DIoU is highly sensitive to center misalignment—this is by design. Small center shifts produce proportional gradient signals.
- **Scale invariance**: DIoU is scale-invariant (normalizing by $d_C^2$). Scaling both boxes equally does not change DIoU.
- **Enclosing box size**: the penalty normalization by $d_C^2$ means the same center distance produces a smaller penalty when the enclosing box is large. This is intentional—it makes the penalty relative, not absolute.
- **Overlap regime**: when boxes overlap significantly, DIoU ≈ IoU because the center distance term becomes small relative to the IoU term.
- **Gradient behavior for non-overlapping boxes**: unlike GIoU, DIoU immediately provides a gradient that pulls centers together (not through box expansion). The gradient direction is always from predicted center toward GT center.
- **Collinear boxes**: the penalty works equally well for horizontal and vertical separation, unlike GIoU which can have anisotropic gradients due to the enclosing area term.

## Alternatives & When to Prefer Them

| Metric | Relationship to DIoU | When to Prefer |
|--------|---------------------|----------------|
| [IoU](iou.md) | DIoU without the center distance penalty | Evaluation; well-overlapping boxes |
| [GIoU](giou.md) | Uses enclosing area instead of center distance | When enclosing area context matters more than center alignment |
| [CIoU](ciou.md) | Adds aspect ratio penalty on top of DIoU | Most comprehensive loss; always unless simplicity is needed |
| [mAP](map.md) | Uses IoU thresholds for evaluation | Model benchmarking |
| Smooth-L1 | Direct coordinate regression | When scale invariance is not needed |
| SIoU | Adds angle cost to DIoU | When directional gradient decomposition helps |

## Code Example

```python
import torch
from torchvision.ops import distance_box_iou

# Boxes in (x1, y1, x2, y2) format
pred_boxes = torch.tensor([
    [100.0, 100.0, 200.0, 200.0],  # well-aligned with GT[0]
    [300.0, 300.0, 350.0, 350.0],  # center-shifted from GT[1]
    [10.0,  10.0,  60.0,  60.0],   # non-overlapping with GT[2]
], requires_grad=True)

gt_boxes = torch.tensor([
    [105.0, 102.0, 205.0, 202.0],  # slight offset
    [320.0, 340.0, 370.0, 390.0],  # center shift + size diff
    [200.0, 200.0, 250.0, 250.0],  # far away, no overlap
])

# Pairwise DIoU: returns (N, M) matrix
diou_matrix = distance_box_iou(pred_boxes, gt_boxes)
print(f"DIoU matrix:\n{diou_matrix}")

# Matched pairs DIoU loss (1-to-1 correspondence)
matched_diou = torch.diagonal(diou_matrix)
diou_loss = 1.0 - matched_diou
total_loss = diou_loss.mean()

print(f"\nMatched DIoU: {matched_diou}")
print(f"DIoU loss:    {diou_loss}")
print(f"Mean loss:    {total_loss.item():.4f}")

total_loss.backward()
print(f"\nGradients:\n{pred_boxes.grad}")

# Compare with IoU to see center-distance effect
from torchvision.ops import box_iou
iou_matrix = box_iou(pred_boxes.detach(), gt_boxes)
matched_iou = torch.diagonal(iou_matrix)
print(f"\nIoU:  {matched_iou}")
print(f"DIoU: {matched_diou.detach()}")
print(f"Gap (IoU - DIoU) = center penalty: {(matched_iou - matched_diou.detach())}")
```

## Debugging Use Case

**Scenario: Small object detection — poor localization despite reasonable mAP@0.50**

```
Symptom:  Model trained with GIoU loss on a dataset with many small objects
          (area < 32²). mAP@0.50 = 0.45 but mAP@0.75 = 0.12.
          Large objects: mAP@0.75 = 0.55. Problem is isolated to small objects.

Diagnosis:
  1. Small objects are highly sensitive to center misalignment.
     A 3-pixel center shift on a 20×20 object drops IoU much more than
     on a 200×200 object.
  2. GIoU's enclosing-area penalty provides weak gradients for small objects
     because the enclosing box area is dominated by the gap, not the boxes.
  3. DIoU's center distance penalty is normalized by enclosing diagonal,
     providing proportionally stronger center-alignment gradients for
     small boxes.

  Verification:
  - Compute mean center distance (in pixels) between predictions and GT
    for small vs large objects.
  - Small objects: mean center dist = 8.2 px (40% of box width)
  - Large objects: mean center dist = 6.5 px (3% of box width)
  - Absolute distances are similar, but relative impact is 13× worse
    for small objects.

Action:
  1. Switch from GIoU to DIoU loss (or CIoU).
  2. Re-train and compare:
     - GIoU: mAP_S@0.75 = 0.12
     - DIoU: mAP_S@0.75 = 0.21  (~75% relative improvement)
  3. If aspect ratio is also an issue, use CIoU instead.
  4. Additionally: increase feature resolution for small objects (FPN P2),
     use larger input resolution, or apply more aggressive anchor tiling
     at small scales.
```

## Related Metrics

- [Intersection over Union (IoU)](iou.md) — base overlap metric without distance penalty
- [Generalized IoU (GIoU)](giou.md) — uses enclosing area instead of center distance
- [Complete IoU (CIoU)](ciou.md) — extends DIoU with aspect ratio consistency penalty
- [Mean Average Precision (mAP)](map.md) — evaluation metric using IoU thresholds
- [Average Recall (AR)](average_recall.md) — detection coverage evaluation
