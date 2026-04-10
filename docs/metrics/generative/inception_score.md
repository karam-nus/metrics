---
title: "Inception Score (IS)"
---
# Inception Score (IS)

## Overview

The Inception Score evaluates generative image models by measuring two properties via a pretrained Inception-v3 classifier: **quality** (each generated image should be confidently classified, yielding low-entropy conditional distributions $p(y|x)$) and **diversity** (the marginal distribution $p(y) = \mathbb{E}_x[p(y|x)]$ should have high entropy, indicating coverage of many classes). IS requires only generated images—no real reference set—making it lightweight but limited. Higher is better; a perfect generator matching ImageNet would score ~250.

## Formula

$$
\text{IS} = \exp\!\left(\mathbb{E}_{x \sim p_g}\left[D_{\text{KL}}\!\left(p(y|x) \,\|\, p(y)\right)\right]\right)
$$

Where:
- $p(y|x)$: Inception-v3 softmax output for image $x$ (conditional label distribution)
- $p(y) = \mathbb{E}_{x}[p(y|x)]$: marginal label distribution over all generated images
- $D_{\text{KL}}$: Kullback–Leibler divergence

High KL divergence → confident per-image predictions that differ from the uniform-like marginal → high IS.

## Visual Diagram

```
Generated Image x_i ──► Inception-v3 ──► p(y|x_i) ──┐
                                                       ├──► KL(p(y|x_i) || p(y)) ──► mean ──► exp ──► IS
All p(y|x_i) ──► Average ──► p(y) ─────────────────┘
```

<!-- IMAGE: Left: sharp peaked per-image distributions (high quality). Right: flat marginal distribution (high diversity). Both together = high IS. -->

## Range & Interpretation

| IS Value | Interpretation |
|----------|---------------|
| 1.0 | Worst: uniform/random noise predictions |
| 2–5 | Poor quality or collapsed diversity |
| 5–10 | Moderate quality |
| 10–50 | Good quality and diversity |
| > 50 | Excellent; well-trained on ImageNet-scale data |

Range: **[1, ∞)**. Higher is better. Theoretical maximum equals number of classes if every image is perfectly classified and classes are uniformly covered. On ImageNet (1000 classes), real data scores ~250.

## When to Use

- Quick sanity check during GAN training when real data statistics are unavailable.
- Comparing models trained on ImageNet or class-conditional generators.
- Monitoring for mode collapse (IS will drop if diversity decreases).
- Ablation studies where relative ranking matters more than absolute value.

## When NOT to Use

- Evaluating unconditional generators on non-ImageNet data (IS is biased toward ImageNet classes).
- When you need to compare against a real distribution (use [FID](fid.md) instead).
- For domains other than natural images (medical, satellite, etc.).
- When diversity within a class matters (IS only measures inter-class diversity).
- Small sample sizes (< 5K images); IS variance is high.

## What It Can Tell You

- Whether generated images are recognizable as distinct object classes.
- Whether the generator covers multiple modes (classes).
- Relative improvements across training or hyperparameter sweeps.
- Presence of severe mode collapse (IS drops sharply).

## What It Cannot Tell You

- Whether generated images resemble real data (no reference comparison).
- Intra-class diversity (e.g., variation within "dog" images).
- Perceptual quality beyond what Inception-v3 captures.
- Overfitting or memorization of training data.
- Quality of non-ImageNet domains without fine-tuning.

## Sensitivity

- **Mode collapse**: IS drops significantly because $p(y)$ concentrates on few classes.
- **Image quality**: Blurry or noisy images produce uncertain $p(y|x)$, lowering KL divergence.
- **Number of classes**: IS is bounded by log(num_classes); comparing across datasets with different class counts is invalid.
- **Sample size**: Compute on ≥ 5K images; typically reported as mean ± std over 10 splits of 5K.
- **Inception weights**: Must use the same Inception-v3 checkpoint for comparable results.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [FID](fid.md) | Real reference data available | Compares against real distribution; more reliable |
| KID | Small sample sizes | Unbiased estimator; no Gaussian assumption |
| [LPIPS](lpips.md) | Per-image perceptual similarity | Pairwise distance, not distributional |
| Precision & Recall | Disentangling quality vs. coverage | Separate axes; more interpretable |
| Classification Accuracy Score | Class-conditional evaluation | Trains a separate classifier on generated data |

## Code Example

```python
import torch
from torchmetrics.image.inception import InceptionScore

# Initialize IS metric
inception_score = InceptionScore(normalize=True)

# Generate synthetic images: (N, C, H, W), float [0, 1]
# In practice, use ≥ 5K images; report mean ± std over splits
fake_images = torch.rand(1024, 3, 299, 299)

# Update metric state
inception_score.update(fake_images)

# Compute IS — returns (mean, std) over internal splits
is_mean, is_std = inception_score.compute()
print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
# Higher is better; 1.0 = worst (random/collapsed)
```

## Debugging Use Case

**Scenario**: Detecting mode collapse during GAN training.

```python
import torch
from torchmetrics.image.inception import InceptionScore

def monitor_mode_collapse(generator, num_samples=5000, latent_dim=128, device="cuda"):
    """Track IS over training to detect mode collapse."""
    is_metric = InceptionScore(normalize=True).to(device)

    with torch.no_grad():
        batch_size = 64
        for i in range(0, num_samples, batch_size):
            z = torch.randn(min(batch_size, num_samples - i), latent_dim, device=device)
            fake = generator(z)
            is_metric.update(fake)

    is_mean, is_std = is_metric.compute()
    print(f"IS = {is_mean:.2f} ± {is_std:.2f}")

    # Diagnostic thresholds (ImageNet-trained generators):
    # IS < 3.0  → severe mode collapse or garbage outputs
    # IS 3–8    → partial collapse or low quality
    # IS > 10   → reasonable diversity and quality
    # Sudden IS drop between checkpoints → training instability
    return is_mean.item()
```

## Related Metrics

- [FID](fid.md) — distributional distance using real reference data
- [LPIPS](lpips.md) — learned perceptual image patch similarity
- [SSIM](ssim.md) — structural similarity index
- [PSNR](psnr.md) — peak signal-to-noise ratio
