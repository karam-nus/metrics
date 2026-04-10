---
title: "Average Recall (AR)"
---

# Average Recall (AR)

## Overview

Average Recall (AR) is a detection evaluation metric that measures how well a model covers all ground-truth objects, averaged across IoU thresholds. While [mAP](map.md) evaluates ranked detection lists by trading off precision and recall, AR focuses purely on recall—the fraction of ground-truth objects successfully detected—without penalizing false positives. COCO defines AR as the mean recall across IoU thresholds from 0.50 to 0.95 (step 0.05), computed at fixed maximum detection counts per image (maxDets = 1, 10, 100). AR is especially useful for evaluating proposal networks (e.g., Region Proposal Networks in Faster R-CNN, Selective Search), where the goal is to generate a set of candidate boxes that cover all objects with high recall, before a classifier refines them. AR also serves as a complementary diagnostic to mAP: a model with high mAP but low AR has good precision but misses many objects, while high AR with low mAP suggests the model detects most objects but with many false positives or poor ranking. The COCO evaluation API reports AR alongside mAP as standard output.

## Formula

For a given maximum detection count $k$ and IoU threshold $\tau$:

$$
\text{Recall}(\tau, k) = \frac{\text{TP}(\tau, k)}{\text{Total GT objects}}
$$

where $\text{TP}(\tau, k)$ counts ground-truth objects matched by at least one of the top-$k$ most confident predictions at IoU ≥ $\tau$.

Average Recall over IoU thresholds:

$$
\text{AR}_k = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{Recall}(\tau, k)
$$

where $\mathcal{T} = \{0.50, 0.55, 0.60, \ldots, 0.95\}$ (10 thresholds in COCO).

COCO reports three variants:

$$
\text{AR}_{1} = \text{AR with maxDets}=1
$$

$$
\text{AR}_{10} = \text{AR with maxDets}=10
$$

$$
\text{AR}_{100} = \text{AR with maxDets}=100
$$

Additionally, AR is broken down by object size:

$$
\text{AR}_k^{\text{small}} \quad (\text{area} < 32^2), \quad
\text{AR}_k^{\text{medium}} \quad (32^2 \leq \text{area} < 96^2), \quad
\text{AR}_k^{\text{large}} \quad (\text{area} \geq 96^2)
$$

An equivalent formulation: AR equals twice the area under the recall-IoU curve from IoU 0.5 to 1.0:

$$
\text{AR} = 2 \int_{0.5}^{1.0} \text{recall}(\tau) \, d\tau
$$

## Visual Diagram

```
  Recall vs IoU Threshold Curve (for fixed maxDets):

  1.0 ┌──────────────────────────────────────┐
      │ ████████████████                      │
  R   │ ████████████████████                  │
  e   │ ████████████████████████              │
  c   │ ████████████████████████████          │
  a   │ ████████████████████████████████      │
  l   │ ████████████████████████████████████  │
  l   │ ████████████████████████████████████  │
      │ ████████████████████████████████████  │
  0.0 └──┬────┬────┬────┬────┬────┬────┬────┬┘
        0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.95
                     IoU Threshold

  AR = mean height of bars = mean recall across thresholds
  (equivalently, 2× area under this curve from 0.5 to 1.0)


  AR at Different maxDets:

  maxDets=1:   Only 1 detection per image → low recall on multi-object images
  maxDets=10:  Reasonable for most images
  maxDets=100: Effectively uncapped for typical images

  AR₁ ≤ AR₁₀ ≤ AR₁₀₀  (more detections → higher or equal recall)


  Example: Proposal Network Coverage

  Image with 5 GT objects:
  ┌─────────────────────────────────────────┐
  │  ┌───┐   ┌──────┐                      │
  │  │GT1│   │ GT2  │   ┌────┐             │
  │  │ ✓ │   │  ✓   │   │GT3 │             │
  │  └───┘   └──────┘   │ ✗  │  ← missed   │
  │                      └────┘             │
  │     ┌──────────┐           ┌───┐        │
  │     │   GT4    │           │GT5│        │
  │     │    ✓     │           │ ✓ │        │
  │     └──────────┘           └───┘        │
  └─────────────────────────────────────────┘

  At IoU=0.50: 4/5 detected → Recall(0.50) = 0.80
  At IoU=0.75: 3/5 detected → Recall(0.75) = 0.60
  At IoU=0.90: 1/5 detected → Recall(0.90) = 0.20
  AR = mean across all thresholds
```

## Range & Interpretation

| Value | Interpretation |
|-------|---------------|
| 0.0 | No ground-truth objects detected at any IoU threshold |
| 0.0–0.20 | Very poor recall; most objects missed |
| 0.20–0.40 | Low recall; significant coverage gaps |
| 0.40–0.60 | Moderate recall; typical for challenging datasets with many small objects |
| 0.60–0.80 | Good recall; most objects detected at lenient IoU thresholds |
| 0.80–1.0 | Excellent recall; nearly all objects detected even at strict IoU |
| 1.0 | Every GT object matched at every IoU threshold by a top-k detection |

COCO state-of-the-art AR₁₀₀ typically ranges from 0.55–0.75. AR₁ is much lower (0.30–0.45) because single-detection-per-image constraint severely limits recall.

## When to Use

- **Proposal network evaluation**: RPN, Selective Search, EdgeBoxes—where the goal is high-recall candidate generation.
- **Detection coverage analysis**: understanding how many objects the model misses, independent of false positive count.
- **Complementary to mAP**: AR highlights recall failures that mAP may mask (e.g., high-precision model that ignores hard objects).
- **maxDets analysis**: AR₁ vs AR₁₀ vs AR₁₀₀ reveals whether the model is detection-count-limited.
- **Object size analysis**: AR_S, AR_M, AR_L reveal which scales are under-detected.
- **Proposal quality tuning**: when adjusting RPN hyperparameters (anchor scales, NMS threshold, top-k), AR is the primary tuning metric.

## When NOT to Use

- **When false positives matter**: AR ignores FPs entirely. A model predicting 10,000 boxes per image can achieve AR=1.0. Pair with mAP or precision.
- **When ranking quality matters**: AR does not evaluate the ordering of predictions by confidence. mAP captures ranking.
- **As a standalone deployment metric**: in production, precision-recall tradeoffs at a specific operating point matter more than AR.
- **Class-specific analysis**: COCO's standard AR is class-agnostic (aggregated across all classes). Per-class AR requires custom computation.

## What It Can Tell You

- The fraction of ground-truth objects your model can detect across IoU thresholds.
- Whether recall is limited by the number of detections (compare AR₁ vs AR₁₀₀).
- Which object scales are poorly covered (AR_S vs AR_M vs AR_L).
- Proposal network quality: whether the RPN generates enough high-quality candidates.
- The localization quality of detections (AR drops steeply at high IoU → poor localization).

## What It Cannot Tell You

- **False positive rate**: AR is blind to spurious detections.
- **Precision or confidence quality**: a model with AR=0.8 might achieve it with 100 or 10,000 detections per image.
- **Per-class recall distribution**: standard AR is class-agnostic.
- **Detection ranking**: two models with the same AR may differ in how well their top-ranked detections cover objects.

## Sensitivity

- **maxDets parameter**: AR₁ << AR₁₀₀ for images with many objects. For single-object-per-image datasets, AR₁ ≈ AR₁₀₀.
- **IoU threshold range**: averaging over [0.5:0.95] means strict thresholds (0.85, 0.90, 0.95) can drag AR down even if recall at 0.50 is high.
- **Object size distribution**: AR is sensitive to the size distribution of GT objects. Datasets with many small objects will have lower AR because small objects are harder to detect at high IoU.
- **Crowd annotations**: COCO ignores detections matched to `iscrowd` regions. Misconfigured crowd annotations inflate or deflate AR.
- **Number of GT objects per image**: images with many objects contribute more to AR than images with few objects (recall is per-object, not per-image).
- **NMS threshold**: aggressive NMS (low threshold) removes detections and can hurt recall on overlapping objects. AR is a good metric to tune NMS threshold.

## Alternatives & When to Prefer Them

| Metric | Relationship to AR | When to Prefer |
|--------|-------------------|----------------|
| [mAP](map.md) | Precision-recall tradeoff vs. recall-only | When false positives and ranking matter |
| Recall@k at fixed IoU | Single threshold, specific k | When you have a fixed operating point |
| [IoU](iou.md) | Per-box localization quality | When evaluating individual box quality |
| COCO AP_S / AP_M / AP_L | Size-specific precision-recall | When both precision and recall matter per size |
| Proposal Recall vs. # proposals curve | Recall as function of proposal count | Detailed proposal network analysis |
| Miss Rate vs. FPPI (log-log) | Pedestrian detection standard (Caltech) | Pedestrian/person detection benchmarks |

## Code Example

```python
import torch
from torchmetrics.detection import MeanAveragePrecision

# The MeanAveragePrecision metric computes AR alongside mAP
# Predictions: list of dicts per image
preds = [
    {
        "boxes": torch.tensor([
            [10.0, 10.0,  50.0,  50.0],
            [60.0, 60.0, 120.0, 120.0],
            [200.0, 200.0, 300.0, 300.0],
            [150.0, 150.0, 180.0, 180.0],
            [400.0, 400.0, 450.0, 420.0],
        ]),
        "scores": torch.tensor([0.95, 0.90, 0.85, 0.70, 0.50]),
        "labels": torch.tensor([0, 1, 0, 2, 1]),
    }
]

# Targets: list of dicts per image
targets = [
    {
        "boxes": torch.tensor([
            [12.0, 11.0,  52.0,  51.0],
            [58.0, 62.0, 118.0, 122.0],
            [205.0, 195.0, 305.0, 295.0],
            [350.0, 350.0, 400.0, 400.0],  # no matching prediction
        ]),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
]

metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_thresholds=None,  # COCO defaults
    max_detection_thresholds=[1, 10, 100],
    class_metrics=False,
)

metric.update(preds, targets)
result = metric.compute()

# Average Recall outputs:
print(f"AR @ maxDets=1:   {result['mar_1']:.4f}")
print(f"AR @ maxDets=10:  {result['mar_10']:.4f}")
print(f"AR @ maxDets=100: {result['mar_100']:.4f}")

# Size-specific AR (requires area annotations; COCO format)
# result['mar_small'], result['mar_medium'], result['mar_large']

# Pair with mAP for complete picture:
print(f"\nmAP@[0.50:0.95]:  {result['map']:.4f}")
print(f"mAP@0.50:          {result['map_50']:.4f}")
print(f"AR₁₀₀:             {result['mar_100']:.4f}")

# Diagnostic: if AR >> mAP, model has high recall but many FPs
# If mAP >> AR, model is selective but misses many objects
gap = result['mar_100'] - result['map']
if gap > 0.2:
    print("\n⚠ Large AR-mAP gap: model detects many objects but has "
          "high false positive rate or poor ranking.")
elif gap < -0.1:
    print("\n⚠ mAP > AR: unusual; check maxDets and evaluation config.")
```

## Debugging Use Case

**Scenario: Proposal network quality assessment — RPN in Faster R-CNN underperforming**

```
Symptom:  Faster R-CNN with ResNet-50 FPN backbone.
          mAP@[0.50:0.95] = 0.32, significantly below expected ~0.40.
          Per-class AP inspection shows ALL classes are uniformly low,
          suggesting the issue is upstream of the classifier head.

Diagnosis — evaluate the RPN proposals using AR:
  1. Extract RPN proposals before NMS and after NMS.
  2. Compute AR for raw proposals and post-NMS proposals.

  Results:
  ┌─────────────────────┬─────────┬─────────┐
  │ Stage               │ AR₁₀₀   │ AR₁₀₀₀  │
  ├─────────────────────┼─────────┼─────────┤
  │ Pre-NMS proposals   │ 0.68    │ 0.72    │
  │ Post-NMS (top 1000) │ 0.55    │ 0.55    │
  │ Post-NMS (top 300)  │ 0.42    │ —       │
  └─────────────────────┴─────────┴─────────┘

  Finding: NMS drops AR from 0.72 to 0.55 — significant proposal loss.

  Size breakdown:
  ┌─────────────────────┬─────────┬─────────┬─────────┐
  │ Stage               │ AR_S    │ AR_M    │ AR_L    │
  ├─────────────────────┼─────────┼─────────┼─────────┤
  │ Pre-NMS             │ 0.38    │ 0.72    │ 0.85    │
  │ Post-NMS (top 1000) │ 0.18    │ 0.62    │ 0.80    │
  └─────────────────────┴─────────┴─────────┴─────────┘

  Finding: Small-object AR drops by 53% through NMS.

Root Cause:
  - NMS IoU threshold = 0.7 (default) is too aggressive for small,
    densely packed objects.
  - Small object proposals overlap heavily due to dense anchor tiling
    at P2/P3 levels, causing excessive suppression.

Action:
  1. Increase NMS IoU threshold from 0.7 to 0.8 for RPN.
  2. Increase top-k post-NMS proposals from 1000 to 2000.
  3. Re-evaluate:
     - Post-NMS AR_S: 0.18 → 0.31 (+72% relative)
     - Post-NMS AR₁₀₀: 0.55 → 0.63
  4. Re-train Faster R-CNN with improved proposals:
     - mAP: 0.32 → 0.39 (+22% relative improvement)
  5. Further: add smaller anchors at P2 level, consider using
     FPN with P1 for extreme small objects.
```

## Related Metrics

- [Mean Average Precision (mAP)](map.md) — precision-recall evaluation that complements AR
- [Intersection over Union (IoU)](iou.md) — threshold criterion used in AR computation
- [Generalized IoU (GIoU)](giou.md) — training loss for box regression
- [Distance IoU (DIoU)](diou.md) — training loss with center distance penalty
- [Complete IoU (CIoU)](ciou.md) — most comprehensive box regression training loss
