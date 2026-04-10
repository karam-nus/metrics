---
title: "Mean Average Precision (mAP)"
---

# Mean Average Precision (mAP)

## Overview

Mean Average Precision is the de facto standard evaluation metric for object detection models. It aggregates per-class Average Precision (AP) scores—each derived from the area under a class-specific precision-recall curve—into a single scalar that captures both classification quality and localization accuracy. The COCO evaluation protocol computes AP at 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05, then averages them to produce `AP@[0.50:0.95]`. This multi-threshold averaging penalizes detections that are only coarsely localized, unlike the legacy PASCAL VOC metric (`AP@0.50`), which is lenient on bounding-box quality. mAP operates over ranked detection lists, where each prediction carries a confidence score and a class label. A detection is a true positive if it exceeds the IoU threshold with a ground-truth box of the same class and that ground-truth has not already been matched; otherwise it is a false positive. Precision-recall curves are interpolated using the all-point or 101-point interpolation method (COCO uses 101-point). The resulting AP per class is the mean interpolated precision at recall values `[0, 0.01, ..., 1.0]`. mAP is then the unweighted mean of all per-class AP values.

## Formula

$$
\text{AP}_c = \frac{1}{101} \sum_{r \in \{0, 0.01, \ldots, 1.0\}} p_{\text{interp}}(r)
$$

where the interpolated precision at recall $r$ is:

$$
p_{\text{interp}}(r) = \max_{\tilde{r} \geq r} p(\tilde{r})
$$

Single-threshold AP (e.g., AP@0.50) uses one IoU threshold $\tau$. COCO-style AP averages over thresholds:

$$
\text{AP}_c^{\text{COCO}} = \frac{1}{10} \sum_{\tau \in \{0.50, 0.55, \ldots, 0.95\}} \text{AP}_c(\tau)
$$

Mean Average Precision across $C$ classes:

$$
\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c
$$

## Visual Diagram

```
  Precision-Recall Curve for One Class at IoU=0.50
  1.0 |*****
      |     *****
  P   |          ****
  r   |              ***
  e   |                 ***
  c   |                    **
  i   |                      **
  s   |                        **
  i   |                          **
  o   |                            *****
  n   |                                 ******
  0.0 +-------------------------------------->
     0.0              Recall              1.0

  AP = shaded area under the interpolated curve

  COCO mAP: average this area across IoU ∈ {0.50, 0.55, ..., 0.95}
             then average across all classes
```

```
  Detection Matching at IoU threshold τ:

  Ground Truth        Prediction (conf=0.92)
  ┌───────────┐       ┌──────────────┐
  │           │       │              │
  │     ┌─────┼───────┼──┐          │
  │     │/////│///////│//│          │   IoU ≥ τ → TP
  │     │/////│///////│//│          │   IoU < τ → FP
  └─────┼─────┘       └──┼──────────┘
        └─────────────────┘
```

## Range & Interpretation

| Value | Interpretation |
|-------|---------------|
| 0.0 | No correct detections at any confidence or IoU threshold |
| 0.0–0.20 | Poor detector; most predictions are mislocalized or misclassified |
| 0.20–0.40 | Below average; acceptable only for very difficult tasks |
| 0.40–0.60 | Competitive; typical range for challenging benchmarks (COCO) |
| 0.60–0.80 | Strong; state-of-the-art on many benchmarks |
| 0.80–1.0 | Near-perfect; rare outside constrained domains |
| 1.0 | Every ground-truth is detected with perfect localization at every IoU threshold |

COCO leaderboard mAP@[0.50:0.95] for state-of-the-art models typically ranges from 0.50 to 0.65.

## When to Use

- **Primary evaluation** of any object detection model (YOLO, Faster R-CNN, DETR, SSD, etc.).
- **Benchmark comparisons** where a single number summarizes detection quality.
- **Model selection** and hyperparameter tuning when both classification and localization matter.
- **Multi-class evaluation** where per-class performance needs to be aggregated.
- When you need to evaluate across a **range of confidence thresholds** (the PR curve implicitly sweeps them).
- When you care about **localization quality** at multiple IoU thresholds (COCO-style).

## When NOT to Use

- **Real-time latency budgets**: mAP ignores inference speed entirely. Pair with FPS or latency metrics.
- **Class-imbalanced scenarios without inspection**: mAP weights all classes equally regardless of prevalence. A class with 5 instances has the same weight as one with 5,000. Inspect per-class AP before relying on mAP alone.
- **Instance segmentation ranking**: use mask AP (mAP computed on mask IoU) instead.
- **When operating point is fixed**: if your application uses a single confidence threshold, precision/recall/F1 at that threshold is more informative than mAP.
- **Counting or density estimation**: mAP does not measure count accuracy.
- **Tracking**: use MOTA/HOTA instead.

## What It Can Tell You

- Aggregate detection quality across classes and IoU thresholds.
- Which classes are well-detected vs. poorly-detected (via per-class AP breakdown).
- How localization quality affects score (compare AP@0.50 vs AP@0.75 vs AP@[0.50:0.95]).
- Whether raising confidence threshold improves precision without catastrophic recall loss (implicit in PR curve shape).
- Relative ranking between models on the same dataset.

## What It Cannot Tell You

- **Where** errors occur spatially in the image.
- **Why** a model fails (classification error vs. localization error vs. duplicate detection vs. missed detection). Use error analysis tools like TIDE for this.
- Inference cost or speed.
- Performance at a specific operating point (confidence threshold).
- How well the model handles class imbalance (equal class weighting hides this).
- Quality of predicted confidence calibration.

## Sensitivity

- **IoU threshold**: AP@0.50 is lenient (coarse boxes pass); AP@0.75 is strict. COCO's averaging smooths this but is still dominated by mid-range thresholds.
- **Number of classes**: adding easy classes inflates mAP; adding hard classes deflates it.
- **Class frequency**: mAP is insensitive to class frequency by design (unweighted mean). This is a feature, but can mask poor performance on rare classes in aggregate summaries if you only look at mAP.
- **Confidence score distribution**: poorly calibrated scores change the PR curve shape without affecting the best achievable AP (which depends only on ranking).
- **Duplicate detections**: multiple predictions on the same ground-truth box count as FPs (only the highest-confidence match is TP). NMS quality directly affects mAP.
- **Crowd annotations / `iscrowd` flag**: COCO evaluation ignores detections matched to crowd regions. Misconfigured crowd flags corrupt AP.
- **Small/medium/large split**: COCO reports AP_S, AP_M, AP_L. Aggregate mAP hides scale-specific failures.

## Alternatives & When to Prefer Them

| Metric | When to Prefer |
|--------|---------------|
| [AP@0.50 (PASCAL VOC)](map.md) | Legacy comparisons; when coarse localization suffices |
| [Average Recall (AR)](average_recall.md) | Evaluating proposal networks or detection coverage |
| [IoU](iou.md) | Evaluating localization quality of individual boxes |
| Precision / Recall / F1 at fixed threshold | Deployment operating-point evaluation |
| TIDE error analysis | Diagnosing error types (cls, loc, duplicate, background, missed) |
| LVIS AP | Long-tail / many-class evaluation with frequency-weighted reporting |
| Panoptic Quality (PQ) | Joint stuff + thing segmentation evaluation |
| MOTA / HOTA | Multi-object tracking evaluation |

## Code Example

```python
import torch
from torchmetrics.detection import MeanAveragePrecision

# Predictions: list of dicts, one per image
# Each dict: boxes (N,4) in xyxy, scores (N,), labels (N,)
preds = [
    {
        "boxes": torch.tensor([
            [100.0, 100.0, 200.0, 200.0],
            [50.0,  50.0, 150.0, 150.0],
            [300.0, 300.0, 400.0, 400.0],
        ]),
        "scores": torch.tensor([0.95, 0.80, 0.60]),
        "labels": torch.tensor([0, 1, 0]),
    }
]

# Targets: list of dicts, one per image
# Each dict: boxes (M,4) in xyxy, labels (M,)
targets = [
    {
        "boxes": torch.tensor([
            [105.0, 100.0, 205.0, 200.0],
            [55.0,  48.0, 155.0, 148.0],
        ]),
        "labels": torch.tensor([0, 1]),
    }
]

metric = MeanAveragePrecision(
    box_format="xyxy",           # also supports "xywh", "cxcywh"
    iou_type="bbox",             # "bbox" or "segm"
    iou_thresholds=None,         # None → COCO default [0.50:0.05:0.95]
    rec_thresholds=None,         # None → COCO default [0:0.01:1.00]
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True,          # per-class AP breakdown
)

metric.update(preds, targets)
result = metric.compute()

# Key outputs:
# result["map"]        → mAP@[0.50:0.95]  (scalar)
# result["map_50"]     → mAP@0.50          (scalar)
# result["map_75"]     → mAP@0.75          (scalar)
# result["map_small"]  → mAP for small objects (area < 32²)
# result["map_medium"] → mAP for medium objects (32² ≤ area < 96²)
# result["map_large"]  → mAP for large objects (area ≥ 96²)
# result["map_per_class"] → tensor of per-class AP values
# result["mar_1"]      → AR with max 1 detection per image
# result["mar_10"]     → AR with max 10 detections per image
# result["mar_100"]    → AR with max 100 detections per image

print(f"mAP@[0.50:0.95]: {result['map']:.4f}")
print(f"mAP@0.50:         {result['map_50']:.4f}")
print(f"mAP@0.75:         {result['map_75']:.4f}")
```

## Debugging Use Case

**Scenario: Comparing YOLOv8 vs Faster R-CNN on a custom dataset**

```
Symptom:  YOLOv8 has mAP@0.50 = 0.78, Faster R-CNN has mAP@0.50 = 0.72
          But mAP@[0.50:0.95]: YOLO=0.45, Faster R-CNN=0.52

Diagnosis:
  1. YOLO wins at coarse IoU (0.50) → good at detecting objects but boxes
     are less precise.
  2. Faster R-CNN wins at strict IoU → better localization from RoI pooling
     and two-stage refinement.
  3. Compare AP@0.75: if Faster R-CNN dominates here, the gap is in
     bounding-box regression quality, not classification.

Action:
  - Check AP_S / AP_M / AP_L for both: YOLO often struggles on small objects
    due to stride-induced resolution loss.
  - Inspect per-class AP: YOLO may dominate on large, common classes;
    Faster R-CNN on small, rare ones.
  - If deploying at IoU≥0.75 requirement: prefer Faster R-CNN.
  - If speed matters and IoU≥0.50 suffices: prefer YOLO.
  - Consider using TIDE to decompose errors into cls/loc/dup/bkg/miss.
```

## Related Metrics

- [Intersection over Union (IoU)](iou.md) — the localization criterion underlying mAP matching
- [Average Recall (AR)](average_recall.md) — complementary coverage-focused metric, also reported by COCO eval
- [Generalized IoU (GIoU)](giou.md) — used as a training loss but not directly in mAP computation
- [Distance IoU (DIoU)](diou.md) — training loss alternative with center-distance awareness
- [Complete IoU (CIoU)](ciou.md) — training loss combining overlap, distance, and aspect ratio
