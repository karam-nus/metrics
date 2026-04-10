---
title: "Boundary F1 Score"
---

# Boundary F1 Score

## Overview

The Boundary F1 Score (BF1, also called Boundary F-measure or Contour F1) evaluates the quality of segmentation boundaries rather than region overlap. While metrics like IoU and Dice measure how well predicted and ground-truth regions overlap in area, BF1 measures how well the predicted contour aligns with the ground-truth contour within a specified distance tolerance.

BF1 was introduced in the Berkeley segmentation benchmark (BSDS500) and adopted by Cityscapes and other benchmarks as a complement to mIoU. It computes precision and recall on boundary pixels: a predicted boundary pixel is a true positive if a ground-truth boundary pixel exists within distance $\theta$ (the tolerance), and vice versa. The F1 (harmonic mean) of boundary precision and recall yields BF1.

This metric is critical when downstream tasks depend on contour accuracy—instance segmentation, medical contouring for radiation therapy planning, autonomous driving lane boundary detection, and any application where the edge of a segmented region matters more than its interior. Two predictions with identical Dice/IoU can have vastly different BF1 if one has smooth, accurate boundaries and the other has jagged, shifted contours.

## Formula

Let $\partial A$ be the set of boundary pixels of prediction $A$ and $\partial B$ be the set of boundary pixels of ground truth $B$. Define a distance tolerance $\theta$ (typically 1–2 pixels, or a percentage of the image diagonal).

**Boundary Precision:**

$$
P_b = \frac{|\{p \in \partial A : \exists\, q \in \partial B,\; \|p - q\| \leq \theta\}|}{|\partial A|}
$$

The fraction of predicted boundary pixels that have a matching ground-truth boundary pixel within distance $\theta$.

**Boundary Recall:**

$$
R_b = \frac{|\{q \in \partial B : \exists\, p \in \partial A,\; \|p - q\| \leq \theta\}|}{|\partial B|}
$$

The fraction of ground-truth boundary pixels that have a matching predicted boundary pixel within distance $\theta$.

**Boundary F1:**

$$
\text{BF1} = \frac{2 \cdot P_b \cdot R_b}{P_b + R_b}
$$

Where $\|\cdot\|$ is typically the Euclidean distance. The boundary pixels $\partial A$, $\partial B$ are extracted via morphological erosion: a pixel is a boundary pixel if it belongs to the foreground but has at least one background neighbor (4- or 8-connectivity).

## Visual Diagram

```
Ground Truth Boundary (∂B)          Predicted Boundary (∂A)
┌───┬───┬───┬───┬───┬───┬───┐      ┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │      │   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┼───┼───┤
│   │   │ B │ B │ B │   │   │      │   │   │   │ A │ A │ A │   │
├───┼───┼───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┼───┼───┤
│   │ B │   │   │   │ B │   │      │   │   │ A │   │   │ A │   │
├───┼───┼───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┼───┼───┤
│   │ B │   │   │   │ B │   │      │   │   │ A │   │   │   │ A │
├───┼───┼───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┼───┼───┤
│   │   │ B │ B │ B │   │   │      │   │   │   │ A │ A │ A │   │
├───┼───┼───┼───┼───┼───┼───┤      ├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │      │   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘      └───┴───┴───┴───┴───┴───┴───┘

|∂B| = 12 pixels                    |∂A| = 12 pixels

With tolerance θ = 1 pixel (8-connectivity):
  Matched pred → gt: 9 of 12     →  Boundary Precision = 9/12 = 0.75
  Matched gt → pred: 9 of 12     →  Boundary Recall    = 9/12 = 0.75

  BF1 = 2 × 0.75 × 0.75 / (0.75 + 0.75) = 0.75

Note: The predicted boundary is shifted ~1 pixel right compared to GT.
With a larger tolerance (θ = 2), all boundary pixels would match → BF1 = 1.0.
```

**Boundary extraction detail:**

```
Mask (1 = foreground):              Boundary pixels (B):
┌───┬───┬───┬───┬───┐              ┌───┬───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │ 0 │              │   │   │   │   │   │
├───┼───┼───┼───┼───┤              ├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │              │   │ B │ B │ B │   │
├───┼───┼───┼───┼───┤              ├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │   erosion   │   │ B │   │ B │   │
├───┼───┼───┼───┼───┤   ───────►   ├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 1 │ 0 │              │   │ B │ B │ B │   │
├───┼───┼───┼───┼───┤              ├───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │              │   │   │   │   │   │
└───┴───┴───┴───┴───┘              └───┴───┴───┴───┴───┘

Boundary = Mask − Eroded(Mask)
Interior pixel (2,2) is removed; all edge pixels of the object remain.
```

## Range & Interpretation

| BF1 Value | Interpretation |
|-----------|----------------|
| 0.0 | No predicted boundary pixel is within tolerance of any ground-truth boundary pixel (and vice versa). Complete boundary failure. |
| 0.0–0.3 | Very poor boundary alignment; predicted contours are far from GT. |
| 0.3–0.6 | Moderate; boundaries are roughly in the right area but imprecise. |
| 0.6–0.8 | Good boundary quality; most contour pixels are within tolerance. |
| 0.8–0.95 | Excellent; contours closely follow GT. Typical target for high-quality models. |
| 1.0 | Every boundary pixel in both prediction and GT is matched within tolerance $\theta$. |

**Sensitivity to θ:** BF1 is heavily dependent on the tolerance parameter. A model achieving BF1 = 0.60 at θ = 1 might score BF1 = 0.90 at θ = 3. Always report the tolerance used. Cityscapes uses a tolerance proportional to the image diagonal (typically ~0.75% of the diagonal, which is roughly 2 pixels at 1024×2048).

## When to Use

- **Boundary-critical applications**: Radiation therapy contouring, where a 2-pixel boundary error on a tumor can irradiate healthy tissue. Autonomous driving lane detection, where boundary precision directly affects lane-keeping.
- **Complementing region metrics**: BF1 reveals boundary quality that IoU/Dice miss. A model with mIoU = 0.80 and BF1 = 0.50 has accurate regions but poor boundaries.
- **Instance segmentation evaluation**: Mask boundaries are important for downstream tasks (e.g., robotic grasping). BF1 per instance captures contour quality that mask IoU averages away.
- **Comparing boundary-aware architectures**: When evaluating boundary refinement modules (e.g., CascadePSP, SegFix, BPR), BF1 is the natural metric.
- **Post-processing evaluation**: CRF, morphological refinement, and active contour post-processing primarily affect boundaries. BF1 isolates their impact.

## When NOT to Use

- **When interior accuracy matters equally**: BF1 says nothing about whether interior pixels are correctly classified. A prediction with perfect boundaries but a hole in the interior scores high on BF1 but low on Dice/IoU.
- **Coarse segmentation tasks**: If the application only needs approximate region localization (e.g., image-level weakly supervised segmentation), boundary precision is irrelevant.
- **Varying tolerance standards**: If you cannot fix a single tolerance $\theta$ across experiments, BF1 comparisons are meaningless. Always use the same θ.
- **Very small objects**: For objects with only boundary pixels (thin structures, lines), BF1 degenerates to a pixel-matching metric similar to recall/precision, and Dice/IoU may be more interpretable.
- **When computation cost matters in training loops**: BF1 requires morphological operations, distance transforms, and boundary matching—it is significantly slower to compute than IoU or Dice.

## What It Can Tell You

- Whether your model's contours are spatially accurate within a defined tolerance.
- Whether boundary errors are due to over-segmentation (low boundary precision: too many predicted boundary pixels) or under-segmentation (low boundary recall: GT boundary pixels are missed).
- The marginal value of boundary refinement post-processing—compute BF1 before and after CRF/SegFix to quantify improvement.
- How robust boundaries are across classes: per-class BF1 can reveal that large smooth objects (road, sky) have high BF1 while small intricate objects (bicycle, person) have low BF1.

## What It Cannot Tell You

- **Interior correctness**: A hollow prediction (correct boundary, wrong interior) scores high on BF1 but low on Dice/IoU.
- **Absolute boundary distance**: BF1 is binary within tolerance—a boundary 0.5 pixels off and one 1.9 pixels off (with θ = 2) are equally "correct." Use Average Surface Distance for graded evaluation.
- **Global shape correctness**: BF1 evaluates point-wise boundary matching. It does not penalize topological errors (e.g., a predicted annulus when the GT is a disk—both have circular boundaries).
- **Instance detection**: BF1 assumes boundaries have been aligned to instances. It does not detect missing or spurious instances.

## Sensitivity

- **Tolerance θ**: The single most impactful parameter. Doubling θ can increase BF1 by 0.15–0.30 on typical datasets. Report θ prominently.
- **Image resolution**: At higher resolution, boundaries have more pixels, and a 2-pixel tolerance covers a smaller physical distance. BF1 at 1024×2048 with θ = 2 is stricter than at 512×1024 with θ = 2 in physical terms.
- **Object shape complexity**: Objects with smooth boundaries (circles, large convex shapes) naturally achieve higher BF1 than objects with intricate boundaries (trees, bicycles, hair).
- **Boundary extraction method**: 4-connectivity vs. 8-connectivity erosion produces slightly different boundary pixel sets. The choice affects BF1 by 1–3% typically.
- **Class size**: For very large objects, most boundary errors are localized to a small fraction of boundary pixels, so BF1 tends to be high. For small objects, each boundary pixel matters more.
- **Morphological noise**: A single-pixel noise spike at a boundary creates extra boundary pixels that count as FP in boundary precision, disproportionately affecting small objects.

## Alternatives & When to Prefer Them

| Metric | Relationship to BF1 | When to Prefer |
|--------|---------------------|----------------|
| [Dice Coefficient](dice.md) | Region overlap (area-based) | When overall region accuracy matters more than boundary precision. |
| [IoU / Jaccard](iou.md) | Region overlap (stricter than Dice) | Standard benchmark metric; does not isolate boundary quality. |
| Hausdorff Distance (HD / HD95) | Max (or 95th-percentile) distance between boundaries | When worst-case boundary error matters (e.g., surgical planning). |
| Average Surface Distance (ASD) | Mean distance between boundary points | When you want a graded (non-binary) measure of boundary quality. |
| Normalized Surface Dice (NSD) | Dice on boundary pixels within tolerance τ | Combines Dice's overlap philosophy with boundary distance tolerance. Similar spirit to BF1 but different formulation. |
| [Pixel Accuracy](pixel_accuracy.md) | All-pixel correct/total | Sanity check only; insensitive to boundaries. |
| Panoptic Quality | Detection × segmentation quality | Instance + semantic combined evaluation. |

## Code Example

```python
import torch
import numpy as np
from scipy import ndimage

def extract_boundary(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Extract boundary pixels from a binary mask using morphological erosion.

    Args:
        mask: binary array, shape (H, W), dtype bool or uint8
        connectivity: 4 or 8 (defines the erosion structuring element)

    Returns:
        boundary: binary array, shape (H, W), True at boundary pixels
    """
    if connectivity == 4:
        struct = ndimage.generate_binary_structure(2, 1)  # cross kernel
    else:
        struct = ndimage.generate_binary_structure(2, 2)  # 3x3 full kernel

    eroded = ndimage.binary_erosion(mask, structure=struct, border_value=0)
    boundary = mask.astype(bool) & ~eroded
    return boundary


def boundary_f1(
    pred: np.ndarray,
    target: np.ndarray,
    theta: float = 2.0,
    connectivity: int = 4,
) -> dict:
    """Compute Boundary F1 Score between predicted and ground-truth masks.

    Args:
        pred: binary prediction mask, shape (H, W)
        target: binary ground-truth mask, shape (H, W)
        theta: distance tolerance in pixels
        connectivity: 4 or 8 for boundary extraction

    Returns:
        dict with keys: 'precision', 'recall', 'f1'
    """
    # Extract boundaries
    bd_pred = extract_boundary(pred, connectivity)    # shape: (H, W)
    bd_target = extract_boundary(target, connectivity)  # shape: (H, W)

    # Handle edge cases
    if bd_pred.sum() == 0 and bd_target.sum() == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if bd_pred.sum() == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    if bd_target.sum() == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    # Compute distance transform from each boundary
    # distance_transform_edt gives distance from each False pixel to nearest True pixel
    dist_from_target = ndimage.distance_transform_edt(~bd_target)  # (H, W)
    dist_from_pred = ndimage.distance_transform_edt(~bd_pred)      # (H, W)

    # Boundary precision: fraction of pred boundary within theta of gt boundary
    precision = (dist_from_target[bd_pred] <= theta).sum() / bd_pred.sum()

    # Boundary recall: fraction of gt boundary within theta of pred boundary
    recall = (dist_from_pred[bd_target] <= theta).sum() / bd_target.sum()

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# --- Usage with PyTorch tensors ---
N, H, W = 4, 256, 256  # batch of 4, 256x256 images

pred_masks = torch.randint(0, 2, (N, H, W)).numpy()   # binary predictions
gt_masks = torch.randint(0, 2, (N, H, W)).numpy()      # binary ground truth

theta = 2.0  # pixel tolerance

# Per-image BF1
bf1_scores = []
for i in range(N):
    result = boundary_f1(pred_masks[i], gt_masks[i], theta=theta)
    bf1_scores.append(result["f1"])
    print(f"Image {i}: BF1={result['f1']:.4f}  "
          f"(P={result['precision']:.4f}, R={result['recall']:.4f})")

mean_bf1 = np.mean(bf1_scores)
print(f"\nMean BF1 (θ={theta}): {mean_bf1:.4f}")

# --- Multi-class BF1 ---
def multiclass_boundary_f1(
    pred: np.ndarray,       # shape (H, W), class indices
    target: np.ndarray,     # shape (H, W), class indices
    num_classes: int,
    theta: float = 2.0,
) -> dict:
    """Compute per-class and mean BF1 for multi-class segmentation."""
    per_class_bf1 = {}
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        result = boundary_f1(pred_c, target_c, theta=theta)
        per_class_bf1[c] = result["f1"]

    mean = np.mean(list(per_class_bf1.values()))
    return {"per_class": per_class_bf1, "mean_bf1": mean}
```

## Debugging Use Case

**Scenario**: An instance segmentation model (Mask R-CNN) on COCO achieves mask AP = 38.5, but qualitative inspection shows that predicted masks have jagged, imprecise boundaries compared to the ground truth.

**Diagnosis steps:**

1. **Compute per-instance BF1**: For each detected instance (IoU ≥ 0.5 with a GT instance), compute BF1 at θ = 2 pixels. Aggregate: if mean BF1 < 0.60 while mean per-instance IoU > 0.75, the model has good region overlap but poor boundary localization.
2. **Decompose BF1 into precision and recall**: If boundary precision is high but recall is low, the predicted contours are accurate where they exist but miss parts of the GT boundary (under-segmentation at edges). If precision is low but recall is high, the prediction has too many boundary pixels (jagged, noisy edges).
3. **Stratify by object size**: Compute mean BF1 for small (<32² pixels), medium (32²–96²), and large (>96²) objects. If BF1 drops sharply for small objects, the mask head lacks resolution.
4. **Check mask head output resolution**: Mask R-CNN's default 28×28 mask head output may be insufficient for large objects—boundaries are blocky after upsampling. Increasing to 56×56 or using PointRend can improve BF1 by 5–10%.
5. **Visualize boundary errors**: Overlay predicted and GT boundaries on the image. Color-code: green = matched within θ, red = unmatched predicted (FP), blue = unmatched GT (FN). This reveals whether errors are systematic (e.g., always on the left edge due to anchor misalignment) or random.
6. **Action**: Apply PointRend or boundary refinement heads (BPR, SegFix) to improve boundary quality. Alternatively, post-process with CRF or active contour fitting. Evaluate with BF1 before and after to quantify improvement. A BF1 improvement of 0.05–0.10 with negligible IoU change confirms that the intervention specifically improved boundaries.

## Related Metrics

- [Dice Coefficient](dice.md) — region overlap; does not distinguish boundary from interior errors.
- [IoU / Jaccard Index](iou.md) — region overlap; stricter than Dice but equally boundary-agnostic.
- [Pixel Accuracy](pixel_accuracy.md) — all-pixel accuracy; dominated by majority class and interior pixels.
- Hausdorff Distance (HD95) — worst-case boundary distance (95th percentile).
- Average Surface Distance (ASD) — mean boundary distance; graded alternative to BF1's binary matching.
- Normalized Surface Dice (NSD) — Dice on boundary pixels within tolerance τ; similar philosophy to BF1.
