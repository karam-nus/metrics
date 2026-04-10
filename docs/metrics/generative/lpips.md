---
title: "Learned Perceptual Image Patch Similarity (LPIPS)"
---
# Learned Perceptual Image Patch Similarity (LPIPS)

## Overview

LPIPS measures perceptual distance between two images by comparing their deep feature representations extracted from a pretrained network (AlexNet, VGG, or SqueezeNet). Unlike pixel-level metrics (MSE, PSNR), LPIPS correlates strongly with human perceptual judgments. Features are extracted at multiple layers, channel-wise scaled by learned weights, and the spatial average of L2 distances is summed across layers. Lower LPIPS = more perceptually similar. It operates on image pairs, not distributions.

## Formula

$$
\text{LPIPS}(x, x_0) = \sum_{l} \frac{1}{H_l W_l} \sum_{h,w} \left\| w_l \odot \left(\hat{\phi}_l^{x}_{h,w} - \hat{\phi}_l^{x_0}_{h,w}\right) \right\|_2^2
$$

Where:
- $\hat{\phi}_l^{x}$: unit-normalized activations of layer $l$ for image $x$
- $w_l$: learned per-channel scaling weights for layer $l$
- $H_l, W_l$: spatial dimensions at layer $l$
- $\odot$: element-wise multiplication

## Visual Diagram

```
Image A ──► VGG/AlexNet ──► [Layer 1 feats, Layer 2 feats, ..., Layer L feats]
                                    │              │                   │
                              normalize        normalize          normalize
                                    │              │                   │
                              w₁ · ‖Δ‖²      w₂ · ‖Δ‖²        wₗ · ‖Δ‖²
                                    │              │                   │
Image B ──► VGG/AlexNet ──► [Layer 1 feats, Layer 2 feats, ..., Layer L feats]
                                    └──────── spatial avg ─── sum ──► LPIPS
```

<!-- IMAGE: Side-by-side image pairs with LPIPS scores showing that perceptually similar distortions (blur) get lower scores than perceptually different ones (color shift) even at same MSE. -->

## Range & Interpretation

| LPIPS Value | Interpretation |
|-------------|---------------|
| 0.0 | Identical images |
| 0.0–0.1 | Imperceptible differences |
| 0.1–0.3 | Minor perceptible differences |
| 0.3–0.5 | Clearly different but related |
| 0.5–1.0 | Very different images |

Range: **[0, ~1]** (technically unbounded, but values rarely exceed 1 for normalized images). Lower is better.

## When to Use

- Evaluating image reconstruction quality (super-resolution, inpainting, denoising, compression).
- Comparing perceptual similarity when pixel-level metrics disagree with human judgment.
- As a perceptual loss function during training (differentiable).
- Evaluating image-to-image translation models (pix2pix, CycleGAN).

## When NOT to Use

- Comparing image distributions (use [FID](fid.md) instead).
- When images are not spatially aligned (LPIPS assumes pixel correspondence).
- Non-image modalities (text, audio).
- When computational cost matters and pixel metrics suffice (LPIPS requires a forward pass through a deep network).
- Comparing images of very different resolutions without resizing.

## What It Can Tell You

- Whether two images are perceptually similar according to learned human-aligned features.
- Which reconstruction method better preserves perceptual quality.
- Relative perceptual distances for ranking model outputs.
- Per-image quality scores (unlike distributional metrics).

## What It Cannot Tell You

- Distribution-level quality or diversity of a generative model.
- Whether an image is "realistic" in absolute terms.
- Semantic similarity (two images of different dogs may have high LPIPS).
- Quality of non-spatial features (e.g., global color grading).

## Sensitivity

- **Network backbone**: VGG-based LPIPS and AlexNet-based LPIPS give different absolute values; always use the same backbone for comparison.
- **Spatial alignment**: Misaligned images yield inflated LPIPS even if content is identical.
- **Image normalization**: Expects images in [-1, 1]; incorrect normalization corrupts scores.
- **Resolution**: Different resolutions require resizing; resizing method affects results.
- **Learned weights**: The `lin` (learned linear) variant outperforms the `squeeze` variant on human judgment benchmarks.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [SSIM](ssim.md) | Lightweight structural comparison | No learned weights; measures luminance/contrast/structure |
| [PSNR](psnr.md) | Quick pixel-level baseline | Simple, fast; poorly correlated with perception |
| [FID](fid.md) | Distributional comparison | Aggregate metric over image sets |
| DISTS | Texture-sensitive comparison | Better for texture synthesis evaluation |
| MS-SSIM | Multi-scale structural comparison | No learned weights; multi-resolution |

## Code Example

```python
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Initialize LPIPS with VGG backbone (options: 'vgg', 'alex', 'squeeze')
lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)

# Input: paired images, (N, 3, H, W), float [0, 1] when normalize=True
img1 = torch.rand(4, 3, 256, 256)  # 4 reference images
img2 = torch.rand(4, 3, 256, 256)  # 4 compared images

# Compute LPIPS distance
score = lpips(img1, img2)
print(f"LPIPS: {score:.4f}")
# Lower is better; 0 = identical
```

## Debugging Use Case

**Scenario**: Evaluating super-resolution model output quality.

```python
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio

def evaluate_super_resolution(sr_model, test_loader, device="cuda"):
    """Compare LPIPS vs PSNR to find cases where pixel metrics mislead."""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    for lr_img, hr_img in test_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        with torch.no_grad():
            sr_img = sr_model(lr_img)

        lpips_val = lpips(sr_img, hr_img)
        psnr_val = psnr(sr_img, hr_img)
        print(f"PSNR: {psnr_val:.2f} dB | LPIPS: {lpips_val:.4f}")

        # Diagnostic: High PSNR but high LPIPS → pixel-accurate but perceptually poor
        # (e.g., blurry reconstruction that averages over modes)
        # Low PSNR but low LPIPS → sharp output with slight misalignment
```

## Related Metrics

- [SSIM](ssim.md) — structural similarity without learned weights
- [PSNR](psnr.md) — pixel-level signal-to-noise ratio
- [FID](fid.md) — distributional comparison of image sets
- [Inception Score](inception_score.md) — quality and diversity of generated images
