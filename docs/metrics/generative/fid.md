---
title: "Fréchet Inception Distance (FID)"
---
# Fréchet Inception Distance (FID)

## Overview

Fréchet Inception Distance quantifies the distance between the feature distributions of real and generated image sets. Both sets are embedded via a pretrained Inception-v3 network (pool3 layer, 2048-d), and each distribution is modeled as a multivariate Gaussian. FID then computes the Fréchet (Wasserstein-2) distance between these two Gaussians. It captures both fidelity (quality of individual samples) and diversity (coverage of the real distribution). FID is the de-facto standard for evaluating generative image models (GANs, diffusion models, VAEs). Lower is better; a score of 0 indicates identical distributions.

## Formula

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)
$$

Where:
- $\mu_r, \Sigma_r$: mean and covariance of real image features
- $\mu_g, \Sigma_g$: mean and covariance of generated image features
- $\text{Tr}$: matrix trace
- $(\Sigma_r \Sigma_g)^{1/2}$: matrix square root of the product

## Visual Diagram

```
Real Images ──► Inception-v3 (pool3) ──► Features_r ──► (μ_r, Σ_r) ──┐
                                                                       ├──► Fréchet Distance ──► FID
Generated Images ──► Inception-v3 (pool3) ──► Features_g ──► (μ_g, Σ_g) ┘
```

<!-- IMAGE: Two overlapping multivariate Gaussian ellipsoids in 2D feature space; FID measures the distance between their centers and shape mismatch. -->

## Range & Interpretation

| FID Value | Interpretation |
|-----------|---------------|
| 0 | Identical distributions |
| < 10 | Excellent; near-photorealistic generators |
| 10–50 | Good quality; typical well-trained GANs |
| 50–100 | Noticeable artifacts or limited diversity |
| > 100 | Poor quality or severe mode collapse |

Range: **[0, ∞)**. Lower is better. Values are dataset- and resolution-dependent; always compare FID scores computed on the same reference set.

## When to Use

- Evaluating GAN, VAE, diffusion model, or flow-based generative image quality.
- Comparing generative architectures or hyperparameter configurations.
- Tracking training progress of generative models over epochs.
- Benchmarking on standard datasets (CIFAR-10, LSUN, FFHQ, ImageNet).

## When NOT to Use

- Small sample sizes (< 10K images); FID is biased and high-variance with few samples. Use ≥ 50K for stable estimates.
- Comparing models trained on different datasets or resolutions (FID is not cross-domain comparable).
- Evaluating single-image quality (FID is a distributional metric).
- Non-image modalities (text, audio) without an analogous feature extractor.
- When you need per-sample scores (use LPIPS or SSIM instead).

## What It Can Tell You

- Whether two image distributions are statistically similar in Inception feature space.
- Relative ranking of generative models on the same dataset.
- Whether training is improving (decreasing FID) or deteriorating (increasing FID) over time.
- Presence of mode collapse (FID will remain high if diversity is low).

## What It Cannot Tell You

- Which specific failure mode is present (artifacts vs. mode dropping vs. mode collapse).
- Per-image quality; FID is aggregate only.
- Perceptual quality as judged by humans (Inception features may miss texture/style nuances).
- Anything about out-of-distribution generation quality (memorization vs. generalization).
- Whether generated images are novel or copied from the training set.

## Sensitivity

- **Sample size**: FID is biased for small N; bias scales as O(1/N). Use ≥ 50K samples.
- **Image preprocessing**: Resizing method (bilinear vs. bicubic), normalization, and JPEG compression all affect FID significantly.
- **Inception weights**: Different Inception checkpoints yield different FID values; always use the same checkpoint.
- **Mode collapse**: FID captures reduced diversity but may still yield moderate scores if generated samples are high-quality.
- **Resolution**: Resizing images to 299×299 (Inception input) loses information for high-res images.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [Inception Score](inception_score.md) | No real data available | IS uses only generated images; FID compares against real |
| [KID (Kernel Inception Distance)](fid.md) | Small sample sizes | KID is unbiased; FID is biased for small N |
| [LPIPS](lpips.md) | Per-image pair similarity | Perceptual distance between two specific images |
| [SSIM](ssim.md) | Pixel-aligned image pairs | Structural comparison of paired images |
| Precision & Recall | Disentangling quality vs. diversity | Separate scores for fidelity and coverage |
| Clean-FID | Consistent preprocessing | Fixes inconsistencies in standard FID computation |

## Code Example

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

# Initialize FID metric (features=2048 corresponds to Inception pool3 layer)
fid = FrechetInceptionDistance(feature=2048, normalize=True)

# Simulate real and generated image batches: (N, C, H, W), float32 [0, 1]
# In practice, use ≥ 50K images for stable FID estimates
real_images = torch.rand(512, 3, 299, 299)  # 512 real images
fake_images = torch.rand(512, 3, 299, 299)  # 512 generated images

# Update with real and fake distributions
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

# Compute FID
fid_score = fid.compute()
print(f"FID: {fid_score:.4f}")
# Lower is better; 0 = identical distributions
```

## Debugging Use Case

**Scenario**: Tracking GAN training quality over epochs.

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

def evaluate_fid_over_training(generator, real_loader, epochs, device="cuda"):
    """Compute FID at each epoch to monitor GAN training health."""
    for epoch in range(epochs):
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        # Accumulate real features
        for real_batch, _ in real_loader:
            fid.update(real_batch.to(device), real=True)

        # Generate same number of fake images
        with torch.no_grad():
            for real_batch, _ in real_loader:
                z = torch.randn(real_batch.size(0), 128, device=device)
                fake_batch = generator(z)
                fid.update(fake_batch, real=False)

        score = fid.compute()
        print(f"Epoch {epoch}: FID = {score:.2f}")

        # Diagnostic: FID should generally decrease over training
        # Sudden spikes indicate training instability or mode collapse
```

## Related Metrics

- [Inception Score](inception_score.md) — quality + diversity without real data comparison
- [LPIPS](lpips.md) — learned perceptual similarity for image pairs
- [SSIM](ssim.md) — structural similarity for aligned image pairs
- [PSNR](psnr.md) — pixel-level signal-to-noise ratio
