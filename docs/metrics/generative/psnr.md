---
title: "Peak Signal-to-Noise Ratio (PSNR)"
---
# Peak Signal-to-Noise Ratio (PSNR)

## Overview

PSNR measures the ratio between the maximum possible signal power and the power of corrupting noise (distortion). It is derived from Mean Squared Error (MSE) and expressed in decibels (dB). PSNR is the simplest and most widely reported image quality metric. It requires a pixel-aligned reference image and operates purely on intensity differences—no perceptual modeling. Despite poor correlation with human perception for complex distortions, its simplicity, speed, and universality make it a standard baseline metric in image processing.

## Formula

$$
\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\!\left(\frac{\text{MAX}}{\sqrt{\text{MSE}}}\right)
$$

Where:
- $\text{MAX}$: maximum possible pixel value (255 for 8-bit, 1.0 for float [0,1])
- $\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2$: mean squared error between reference and test images
- If MSE = 0, PSNR = ∞ (identical images)

## Visual Diagram

```
Reference Image ──┐
                   ├──► Pixel-wise difference ──► Square ──► Mean (MSE) ──► 10·log₁₀(MAX²/MSE) ──► PSNR (dB)
Test Image ───────┘
```

<!-- IMAGE: Bar chart showing PSNR values for the same image under different distortions (blur, noise, JPEG compression) alongside human quality ratings demonstrating the divergence between PSNR and perception. -->

## Range & Interpretation

| PSNR (dB) | Interpretation |
|-----------|---------------|
| ∞ | Identical images (MSE = 0) |
| > 40 | Excellent; nearly imperceptible distortion |
| 30–40 | Good quality; minor artifacts |
| 20–30 | Acceptable for some applications |
| < 20 | Poor quality; visible degradation |

Range: **[0, ∞) dB**. Higher is better. Values are relative to MAX; comparing across different dynamic ranges is invalid.

## When to Use

- Baseline metric for image/video quality assessment.
- Evaluating denoising, compression, or super-resolution when paired images are available.
- When computational cost must be minimal (PSNR is O(N) with no model inference).
- Reporting alongside perceptual metrics (SSIM, LPIPS) for completeness.
- Standards compliance (many video/image standards specify PSNR thresholds).

## When NOT to Use

- When perceptual quality matters (use [SSIM](ssim.md) or [LPIPS](lpips.md)).
- Comparing unpaired or misaligned images (even sub-pixel shifts tank PSNR).
- Evaluating generative model distributions (use [FID](fid.md)).
- When images have different dynamic ranges without normalization.
- Texture or style evaluation (PSNR penalizes valid but different textures).

## What It Can Tell You

- The magnitude of pixel-level distortion between two aligned images.
- Whether a reconstruction is mathematically lossless (PSNR = ∞).
- Relative ranking of methods when distortion types are similar.
- Whether compression meets bit-rate vs. quality targets.

## What It Cannot Tell You

- Whether the distortion is perceptually significant.
- Which type of distortion is present (blur vs. noise vs. ringing).
- Anything about structural or semantic quality.
- Quality of generated images without a reference.
- Whether higher PSNR actually means better visual quality (it often doesn't for learned models).

## Sensitivity

- **Dynamic range (MAX)**: Must be set correctly; using MAX=255 for float [0,1] images gives meaningless results.
- **Pixel alignment**: Sub-pixel shifts cause massive PSNR drops despite imperceptible visual change.
- **Color space**: PSNR on RGB vs. YCbCr vs. luminance-only gives different values; specify which.
- **Distortion type**: PSNR poorly distinguishes blur from noise from ringing at the same MSE.
- **Image content**: Flat regions tolerate more noise; textured regions hide noise. PSNR doesn't account for content.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [SSIM](ssim.md) | Structural quality matters | Models luminance, contrast, structure |
| [LPIPS](lpips.md) | Perceptual similarity needed | Deep features; correlates with human judgment |
| MS-SSIM | Multi-scale assessment | Better for varying viewing distances |
| [FID](fid.md) | Distributional comparison | No paired reference needed |
| VMAF | Video quality | Multi-feature fusion; designed for video |

## Code Example

```python
import torch
from torchmetrics.image import PeakSignalNoiseRatio

# Initialize PSNR metric
psnr = PeakSignalNoiseRatio(data_range=1.0)

# Input: paired images, (N, C, H, W), float [0, 1]
img_ref = torch.rand(4, 3, 256, 256)
img_noisy = img_ref + 0.05 * torch.randn_like(img_ref)
img_noisy = img_noisy.clamp(0, 1)

# Compute PSNR
score = psnr(img_noisy, img_ref)
print(f"PSNR: {score:.2f} dB")
# Higher is better; ∞ = identical images
```

## Debugging Use Case

**Scenario**: Evaluating denoising model performance across noise levels.

```python
import torch
from torchmetrics.image import PeakSignalNoiseRatio

def evaluate_denoiser(denoiser, clean_images, noise_levels, device="cuda"):
    """Measure PSNR improvement from denoising at various noise levels."""
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    for sigma in noise_levels:
        noisy = (clean_images + sigma * torch.randn_like(clean_images)).clamp(0, 1)
        with torch.no_grad():
            denoised = denoiser(noisy.to(device))

        psnr_noisy = psnr(noisy.to(device), clean_images.to(device))
        psnr_denoised = psnr(denoised, clean_images.to(device))
        gain = psnr_denoised - psnr_noisy

        print(f"σ={sigma:.2f}: Noisy={psnr_noisy:.1f}dB → Denoised={psnr_denoised:.1f}dB (Δ={gain:.1f}dB)")

        # Diagnostic:
        # PSNR gain < 1 dB → denoiser barely helps at this noise level
        # PSNR gain > 5 dB → significant improvement
        # If PSNR is high but images look blurry → model over-smooths; check LPIPS
```

## Related Metrics

- [SSIM](ssim.md) — structural similarity index
- [LPIPS](lpips.md) — learned perceptual similarity
- [FID](fid.md) — distributional distance for generative models
- [Inception Score](inception_score.md) — quality and diversity without reference
