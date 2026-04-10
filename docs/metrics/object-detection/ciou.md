---
title: "Complete Intersection over Union (CIoU)"
---

# Complete Intersection over Union (CIoU)

## Overview

Complete IoU (CIoU), introduced alongside DIoU by Zheng et al. (AAAI 2020), is the most comprehensive IoU-based loss for bounding-box regression. It simultaneously optimizes three geometric properties: overlap area (via IoU), center-point distance (via the DIoU penalty), and aspect ratio consistency (via an additional penalty term). The key insight is that two boxes can have identical IoU and identical center distance but differ in aspect ratio, leading to visually different detections. CIoU addresses this with a penalty that measures the difference in aspect ratios between the predicted and ground-truth boxes, weighted by a balancing factor that increases as IoU improves (so aspect ratio refinement becomes more important in later training when overlap is already good). CIoU is the default box regression loss in YOLOv5, YOLOv7, YOLOv8, and many other modern single-stage detectors. It provides the richest gradient signal of any IoU variant, leading to the best convergence and final localization accuracy in most settings.

## Formula

Let $A$ be the predicted box and $B$ the ground-truth box:

$$
\text{CIoU}(A, B) = \text{IoU}(A, B) - \frac{\rho^2(\mathbf{c}_A, \mathbf{c}_B)}{d_C^2} - \alpha v
$$

where:

**Center distance penalty** (same as DIoU):
$$
\frac{\rho^2(\mathbf{c}_A, \mathbf{c}_B)}{d_C^2} = \frac{(c_x^A - c_x^B)^2 + (c_y^A - c_y^B)^2}{d_C^2}
$$

**Aspect ratio consistency term**:
$$
v = \frac{4}{\pi^2} \left( \arctan\frac{w^B}{h^B} - \arctan\frac{w^A}{h^A} \right)^2
$$

**Balancing weight**:
$$
\alpha = \frac{v}{(1 - \text{IoU}) + v}
$$

The $\alpha$ weighting ensures that aspect ratio refinement is prioritized when IoU is already high (i.e., overlap is good and the remaining error is in shape). When IoU is low, $\alpha$ is small and the loss focuses on overlap and center alignment.

The CIoU loss:

$$
\mathcal{L}_{\text{CIoU}} = 1 - \text{CIoU}(A, B)
$$

Note: during backpropagation, the gradient of $v$ with respect to box dimensions flows through the $\arctan$ terms, providing a smooth signal for aspect ratio adjustment. The $\alpha$ parameter is treated as a constant (stop-gradient) during optimization to avoid instability.

## Visual Diagram

```
  CIoU decomposes box regression into three objectives:

  ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
  │                                                             │
  │  1. OVERLAP (IoU term)                                      │
  │  ┌──────────┐                                               │
  │  │    A ┌───┼─────┐     Maximize intersection / union       │
  │  │      │///│     │                                         │
  │  └──────┼───┘     │                                         │
  │         │    B    │                                         │
  │         └─────────┘                                         │
  │                                                             │
  │  2. CENTER DISTANCE (ρ²/d_C² term)                          │
  │                                                             │
  │         ●A ─ ─ ─ ρ ─ ─ ─ ●B    Minimize center distance    │
  │                                                             │
  │  3. ASPECT RATIO (αv term)                                  │
  │                                                             │
  │  Pred:  ┌──────────┐   GT:  ┌────┐                         │
  │         │          │        │    │   Match w/h ratios       │
  │         └──────────┘        │    │                          │
  │         w_A/h_A = 2.5      │    │                          │
  │                             └────┘                          │
  │                             w_B/h_B = 0.5                   │
  │                                                             │
  │         v = (4/π²)(atan(w_B/h_B) - atan(w_A/h_A))²        │
  │                                                             │
  └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘


  Training progression with CIoU:

  Epoch 1:              Epoch 20:             Epoch 50:
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │ GT ┌───┐     │      │ GT ┌───┐     │      │ GT ┌───┐     │
  │    │   │     │      │  ┌─┼───┼─┐   │      │    │///│     │
  │    └───┘     │      │  │ │///│ │   │      │    │/A/│     │
  │              │      │  │ └───┘ │   │      │    │///│     │
  │     ┌────────┼┐     │  └───────┘   │      │    └───┘     │
  │     │   A    ││     │    A         │      │   A ≈ GT     │
  │     └────────┼┘     │              │      │              │
  └──────────────┘      └──────────────┘      └──────────────┘
  ↑ Center far,         ↑ Center close,       ↑ Overlap high,
    no overlap,           overlap growing,      center aligned,
    wrong aspect          aspect adjusting      aspect matched
```

## Range & Interpretation

| Value | Interpretation |
|-------|---------------|
| −1.0 | Theoretical minimum: no overlap, maximum center distance, maximum aspect ratio mismatch |
| −1.0 to 0.0 | Non-overlapping or poorly aligned boxes with significant distance and/or shape mismatch |
| 0.0 to 0.5 | Moderate localization: overlap exists but center and/or aspect ratio need improvement |
| 0.5 to 0.8 | Good localization: reasonable overlap, center, and aspect ratio alignment |
| 0.8 to 1.0 | Excellent localization across all three geometric properties |
| 1.0 | Identical boxes: perfect overlap, zero center distance, identical aspect ratio |

CIoU ≤ DIoU ≤ IoU always. Each successive penalty can only decrease the score.

## When to Use

- **Default box regression loss for modern detectors**: recommended for YOLOv5/v7/v8, custom single-stage detectors, and two-stage detectors where localization quality is critical.
- When **aspect ratio consistency** is important for the application (e.g., detecting vehicles, text, faces where shape matters).
- When you want the **best convergence and final localization** without manually balancing multiple loss terms.
- **Replacing GIoU or DIoU losses**: CIoU strictly subsumes both. If DIoU works, CIoU will work at least as well and often better.
- When training from **poor initializations** where predicted boxes may be far from ground truth with wrong aspect ratios.

## When NOT to Use

- **As an evaluation metric**: use [mAP](map.md) with IoU thresholds for benchmarking. CIoU is a training loss.
- **When simplicity and interpretability are paramount**: CIoU has more hyperparameter surface (the $\alpha$ balancing). For ablation studies, simpler IoU or DIoU losses may be easier to reason about.
- **When aspect ratio is intentionally variable**: if your detector must handle extreme aspect ratio variation within a class (e.g., detecting "objects" without shape prior), the aspect ratio penalty may fight the natural variation. In practice this is rare.
- **Rotated/oriented boxes**: standard CIoU is for axis-aligned boxes. Use oriented CIoU variants.

## What It Can Tell You

- The combined quality of a predicted box in terms of overlap, center alignment, and aspect ratio consistency.
- Which of the three components dominates the loss (by computing IoU, center distance, and $v$ terms separately).
- Whether the model's remaining localization error is in position, scale, or shape.

## What It Cannot Tell You

- **Classification correctness**: CIoU is purely geometric.
- **Which spatial direction to refine**: CIoU is a scalar loss; use per-coordinate analysis for directional debugging.
- **Absolute pixel-level accuracy**: CIoU is scale-invariant and relative.
- **Dataset-level performance**: CIoU is a per-pair measure; aggregate with mAP for dataset evaluation.

## Sensitivity

- **Overlap (IoU term)**: dominates when boxes barely overlap. In early training, this term drives most of the gradient.
- **Center distance (ρ²/d_C² term)**: provides strong gradients for non-overlapping or poorly aligned boxes. Normalized by enclosing diagonal for scale invariance.
- **Aspect ratio (αv term)**: the $\alpha$ balancing weight increases as IoU improves, making this term more influential in late training when overlap is already good. This is by design—early training should focus on overlap and positioning.
- **α stop-gradient**: $\alpha$ is treated as constant during backprop. Without this, gradients through $\alpha$ can cause oscillation because $\alpha$ itself depends on IoU.
- **Arctan sensitivity**: the $\arctan(w/h)$ parameterization is most sensitive to aspect ratio changes when $w/h$ is near 1 (square boxes) and less sensitive for extreme aspect ratios. This matches human perception—a change from 1:1 to 2:1 is more visually significant than from 10:1 to 11:1.
- **Scale invariance**: CIoU, like all IoU variants, is scale-invariant.

## Alternatives & When to Prefer Them

| Metric | Relationship to CIoU | When to Prefer |
|--------|---------------------|----------------|
| [IoU](iou.md) | Base overlap measure, no penalties | Evaluation; simplest loss when boxes overlap well |
| [GIoU](giou.md) | Enclosing area penalty only | When aspect ratio penalty is undesirable |
| [DIoU](diou.md) | CIoU without aspect ratio term | When aspect ratio variation is inherent to the problem |
| SIoU | Angle-aware penalty instead of center distance | Directional gradient decomposition |
| EIoU | Separate width/height penalties instead of arctan-based v | Explicit scale decomposition |
| Focal-CIoU | CIoU with focal weighting on IoU | Hard example mining for box regression |
| WIoU | Wise-IoU with attention-based weighting | Reducing impact of low-quality annotations |

## Code Example

```python
import torch
from torchvision.ops import complete_box_iou

# Boxes in (x1, y1, x2, y2) format
pred_boxes = torch.tensor([
    [100.0, 100.0, 200.0, 200.0],  # square, well-positioned
    [100.0, 100.0, 250.0, 150.0],  # wide box (aspect ratio mismatch)
    [300.0, 300.0, 340.0, 380.0],  # tall box, offset from GT
], requires_grad=True)

gt_boxes = torch.tensor([
    [105.0, 102.0, 205.0, 202.0],  # square, slight offset
    [100.0, 100.0, 150.0, 200.0],  # tall box (opposite aspect ratio!)
    [310.0, 310.0, 350.0, 390.0],  # tall box, nearby
])

# Pairwise CIoU: returns (N, M) matrix
ciou_matrix = complete_box_iou(pred_boxes, gt_boxes)
print(f"CIoU matrix:\n{ciou_matrix}")

# Matched pairs loss
matched_ciou = torch.diagonal(ciou_matrix)
ciou_loss = 1.0 - matched_ciou
total_loss = ciou_loss.mean()

print(f"\nMatched CIoU: {matched_ciou}")
print(f"CIoU loss:    {ciou_loss}")
print(f"Mean loss:    {total_loss.item():.4f}")

# Compare all IoU variants for the same box pairs
from torchvision.ops import box_iou, generalized_box_iou, distance_box_iou

iou_vals   = torch.diagonal(box_iou(pred_boxes.detach(), gt_boxes))
giou_vals  = torch.diagonal(generalized_box_iou(pred_boxes.detach(), gt_boxes))
diou_vals  = torch.diagonal(distance_box_iou(pred_boxes.detach(), gt_boxes))
ciou_vals  = matched_ciou.detach()

print("\n--- Comparison across IoU variants ---")
print(f"Pair |  IoU   |  GIoU  |  DIoU  |  CIoU")
print(f"─────┼────────┼────────┼────────┼────────")
for i in range(3):
    print(f"  {i}  | {iou_vals[i]:.4f} | {giou_vals[i]:.4f} | "
          f"{diou_vals[i]:.4f} | {ciou_vals[i]:.4f}")

# Note: pair 1 shows the largest gap between IoU and CIoU
# due to aspect ratio mismatch (wide pred vs tall GT).

total_loss.backward()
print(f"\nGradients on pred_boxes:\n{pred_boxes.grad}")
```

## Debugging Use Case

**Scenario: Aspect ratio sensitive detection — vehicle detection with mixed car/truck classes**

```
Symptom:  Detector trained with DIoU loss on vehicle dataset.
          Cars (aspect ratio ~2:1) detected well: AP = 0.72.
          Trucks (aspect ratio ~3:1) detected poorly: AP = 0.41.
          Visual inspection: truck predictions have correct position
          but consistently wrong aspect ratio (too short, too wide).

Diagnosis:
  1. DIoU optimizes center distance and overlap but does NOT penalize
     aspect ratio mismatch.
  2. For trucks, the model finds a local optimum where a shorter, wider
     box achieves reasonable IoU (since overlap area is decent) but the
     aspect ratio is wrong.
  3. At IoU threshold 0.75, these shape-mismatched boxes fail → low AP.

  Verification:
  - Compute predicted vs GT aspect ratios for truck detections:
      mean(w_pred/h_pred) = 2.3
      mean(w_gt/h_gt)     = 3.1
  - CIoU's v term for these predictions:
      v = (4/π²)(atan(3.1) - atan(2.3))² = 0.038
      α = v / (1 - IoU + v) ≈ 0.12
  - The αv penalty is non-trivial → CIoU would provide gradient to
    correct the aspect ratio.

Action:
  1. Switch from DIoU to CIoU loss.
  2. Re-train and compare per-class AP:
     - DIoU: car=0.72, truck=0.41
     - CIoU: car=0.73, truck=0.56  (truck AP ↑ 37% relative)
  3. The CIoU aspect ratio gradient encourages the model to predict
     longer, narrower boxes for trucks.
  4. Monitor: if some classes have legitimately variable aspect ratios,
     ensure the v penalty doesn't fight natural variation.
     Plot v distribution per class to verify.
```

## Related Metrics

- [Intersection over Union (IoU)](iou.md) — base overlap metric
- [Generalized IoU (GIoU)](giou.md) — enclosing area extension
- [Distance IoU (DIoU)](diou.md) — CIoU without the aspect ratio term
- [Mean Average Precision (mAP)](map.md) — evaluation metric using IoU thresholds
- [Average Recall (AR)](average_recall.md) — detection coverage evaluation
