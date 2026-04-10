---
title: "Structural Similarity Index Measure (SSIM)"
---
# Structural Similarity Index Measure (SSIM)

## Overview

SSIM evaluates the perceived quality of an image by comparing three components against a reference: **luminance** (mean intensity), **contrast** (variance of intensity), and **structure** (correlation of normalized signals). Unlike MSE/PSNR, SSIM is designed to model the human visual system's sensitivity to structural information. It operates on local patches (typically 11×11 Gaussian-weighted windows) and averages across the image. SSIM is symmetric, bounded, and widely used for image quality assessment in compression, restoration, and generation tasks.

## Formula

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Where:
- $\mu_x, \mu_y$: local means of images $x$ and $y$
- $\sigma_x^2, \sigma_y^2$: local variances
- $\sigma_{xy}$: local covariance
- $C_1 = (K_1 L)^2$, $C_2 = (K_2 L)^2$: stabilization constants ($L$ = dynamic range, $K_1=0.01$, $K_2=0.03$)

The three components factor as: $l(x,y) \cdot c(x,y) \cdot s(x,y)$ (luminance × contrast × structure).

## Visual Diagram

```
Image X ──► Local window (11×11 Gaussian) ──► μ_x, σ_x² ──┐
                                                             ├──► SSIM(x,y) per patch ──► Mean ──► MSSIM
Image Y ──► Local window (11×11 Gaussian) ──► μ_y, σ_y² ──┘
                                              σ_xy ─────────┘
```

<!-- IMAGE: Three-panel decomposition showing luminance comparison, contrast comparison, and structure comparison maps, combined into a final SSIM map. -->

## Range & Interpretation

| SSIM Value | Interpretation |
|------------|---------------|
| 1.0 | Identical images |
| 0.95–1.0 | Imperceptible differences |
| 0.8–0.95 | Minor visible artifacts |
| 0.5–0.8 | Clearly degraded quality |
| < 0.5 | Severe degradation |
| -1.0 | Perfectly inverted (theoretical) |

Range: **[-1, 1]**, where 1 = perfect structural similarity. In practice, values are typically in [0, 1] for natural images.

## When to Use

- Evaluating image compression algorithms (JPEG, WebP, AVIF).
- Comparing image restoration outputs (denoising, deblurring, super-resolution).
- Quality assessment where pixel alignment exists between reference and test images.
- As a loss function component for training image reconstruction models.

## When NOT to Use

- When images are not spatially aligned (SSIM assumes pixel correspondence).
- Comparing image distributions (use [FID](fid.md)).
- When subtle perceptual differences matter more than structure (use [LPIPS](lpips.md)).
- For images with large geometric transformations or different viewpoints.
- Non-image data without meaningful spatial structure.

## What It Can Tell You

- Whether structural information is preserved between reference and reconstructed images.
- Localized quality: SSIM maps reveal which regions are degraded.
- Relative ranking of compression levels or restoration methods.
- Whether luminance, contrast, or structural differences dominate.

## What It Cannot Tell You

- Whether an image looks "natural" or "realistic" in absolute terms.
- Perceptual quality for textures and high-frequency details (SSIM favors smoothness).
- Quality of generated images without a paired reference.
- Distribution-level quality or diversity.

## Sensitivity

- **Window size**: Default 11×11; smaller windows increase sensitivity to local artifacts but add noise.
- **Gaussian weighting**: σ=1.5 default; affects spatial weighting of local statistics.
- **Dynamic range**: Must match the actual image bit depth (255 for 8-bit, 1.0 for float [0,1]).
- **Color channels**: Often computed per-channel and averaged; RGB vs. luminance-only gives different results.
- **Blur/smoothing**: SSIM penalizes blur less than humans perceive it (known limitation).

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| MS-SSIM | Multi-scale quality assessment | Evaluates at multiple resolutions; better correlation with perception |
| [LPIPS](lpips.md) | Perceptual similarity matters most | Learned features; better human correlation |
| [PSNR](psnr.md) | Quick pixel-level check | Simpler but less perceptually meaningful |
| DISTS | Texture quality matters | Explicitly models texture similarity |
| VMAF | Video quality assessment | Combines multiple features; designed for video |

## Code Example

```python
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Initialize SSIM metric
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

# Input: paired images, (N, C, H, W), float [0, 1]
img_ref = torch.rand(4, 3, 256, 256)   # reference images
img_test = torch.rand(4, 3, 256, 256)  # test/reconstructed images

# Compute SSIM
score = ssim(img_test, img_ref)
print(f"SSIM: {score:.4f}")
# 1.0 = identical; higher is better
```

## Debugging Use Case

**Scenario**: Evaluating image compression quality at different levels.

```python
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def evaluate_compression_quality(original, compressed_versions, quality_levels):
    """Compare SSIM across compression quality levels to find the sweet spot."""
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    for compressed, q in zip(compressed_versions, quality_levels):
        # Both tensors: (1, C, H, W) float [0, 1]
        score = ssim(compressed.unsqueeze(0), original.unsqueeze(0))
        print(f"Quality {q}: SSIM = {score:.4f}")

        # Diagnostic thresholds:
        # SSIM > 0.98 → likely imperceptible; safe to use lower quality
        # SSIM 0.90-0.98 → minor artifacts; acceptable for most uses
        # SSIM < 0.90 → visible degradation; quality too low
        # Compare with LPIPS: if SSIM is high but LPIPS is also high,
        # structural info is preserved but textures are destroyed
```

## Related Metrics

- [PSNR](psnr.md) — pixel-level signal-to-noise ratio
- [LPIPS](lpips.md) — learned perceptual similarity
- [FID](fid.md) — distributional distance between image sets
- [Inception Score](inception_score.md) — quality and diversity of generated images
