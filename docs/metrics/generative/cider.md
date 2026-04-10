---
title: "CIDEr (Consensus-based Image Description Evaluation)"
---
# CIDEr (Consensus-based Image Description Evaluation)

## Overview

CIDEr measures the similarity between a candidate caption and a set of reference captions using **TF-IDF weighted n-gram similarity**. Designed specifically for image captioning, CIDEr captures the intuition that n-grams common across multiple human reference captions (high TF) but rare in the corpus (high IDF) are more informative. CIDEr-D is the standard variant, which adds a length-based Gaussian penalty to discourage gaming via length manipulation. The metric naturally rewards consensus among annotators and penalizes generic, corpus-frequent phrases.

## Formula

$$
\text{CIDEr}_n(c_i, S_i) = \frac{1}{M} \sum_{j=1}^{M} \frac{\mathbf{g}^n(c_i) \cdot \mathbf{g}^n(s_{ij})}{\|\mathbf{g}^n(c_i)\| \, \|\mathbf{g}^n(s_{ij})\|}
$$

$$
\text{CIDEr}(c_i, S_i) = \sum_{n=1}^{N} w_n \cdot \text{CIDEr}_n(c_i, S_i)
$$

Where:
- $\mathbf{g}^n(c_i)$: TF-IDF vector of n-grams for candidate caption $c_i$
- $\mathbf{g}^n(s_{ij})$: TF-IDF vector for $j$-th reference caption
- $M$: number of reference captions for image $i$
- $w_n = \frac{1}{N}$: uniform weight across n-gram orders (typically $N=4$)
- TF-IDF: $g_k(s) = \frac{h_k(s)}{\sum_l h_l(s)} \cdot \log\frac{|I|}{\sum_{I_p} \min(1, \sum_q h_k(s_{pq}))}$

## Visual Diagram

```
Candidate caption ──► Tokenize ──► n-gram extraction ──► TF-IDF weighting ──┐
                                                                              ├──► Cosine similarity ──► Average ──► CIDEr
Reference captions ──► Tokenize ──► n-gram extraction ──► TF-IDF weighting ──┘
                                                              ▲
                                                    IDF from full corpus
```

<!-- IMAGE: Example showing how informative n-grams (e.g., "golden retriever") get high TF-IDF weight while generic ones (e.g., "a photo of") get downweighted. -->

## Range & Interpretation

| CIDEr Score | Interpretation |
|-------------|---------------|
| 0.0 | No meaningful n-gram overlap |
| 0.0–0.5 | Poor caption quality |
| 0.5–1.0 | Moderate; captures some content |
| 1.0–1.5 | Good; matches human consensus |
| > 1.5 | Excellent; near-human captioning |

Range: **[0, 10]** (theoretically; typical values are 0–2 on standard benchmarks like MSCOCO). Higher is better. CIDEr values are dataset-dependent due to IDF computation.

## When to Use

- Evaluating image captioning models (the primary and intended use case).
- When multiple reference captions per image are available (leverages consensus).
- Comparing captioning architectures on standard benchmarks (MSCOCO, Flickr30k).
- When you want to downweight generic, uninformative phrases.

## When NOT to Use

- Machine translation (use [BLEU](bleu.md), [METEOR](meteor.md), or COMET).
- Text summarization (use [ROUGE](rouge.md)).
- Single-reference evaluation (CIDEr needs multiple references for meaningful IDF).
- When the corpus is very small (IDF estimates become unreliable).
- When semantic similarity beyond n-gram overlap is needed (use [BERTScore](bertscore.md)).

## What It Can Tell You

- Whether the candidate uses informative, consensus n-grams from reference captions.
- Relative ranking of captioning models on standard benchmarks.
- Whether the model generates generic vs. specific descriptions.
- Content overlap weighted by informativeness.

## What It Cannot Tell You

- Whether the caption is grammatically correct or fluent.
- Semantic similarity for paraphrases not sharing n-grams.
- Whether the caption accurately describes the image (no visual grounding).
- Factual correctness of generated descriptions.

## Sensitivity

- **Number of references**: CIDEr improves with more references; standard MSCOCO uses 5 per image.
- **Corpus size**: IDF weights depend on corpus; small corpora give noisy IDF estimates.
- **N-gram order**: CIDEr-4 (default) balances precision and recall of phrasing.
- **Caption length**: CIDEr-D applies Gaussian penalty for length mismatch; standard CIDEr does not.
- **Tokenization**: Lowercasing, punctuation removal, and tokenizer choice affect n-gram extraction.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [BLEU](bleu.md) | Precision-focused; no IDF needed | No TF-IDF weighting; n-gram precision only |
| [ROUGE](rouge.md) | Recall-oriented evaluation | Recall-focused; no corpus weighting |
| [METEOR](meteor.md) | Synonym/stem matching needed | Alignment-based with linguistic resources |
| [BERTScore](bertscore.md) | Semantic similarity needed | Contextual embeddings; paraphrase-aware |
| SPICE | Scene graph evaluation | Parses captions into scene graphs; tests semantic propositions |
| CLIPScore | Reference-free captioning evaluation | Uses CLIP to compare caption against image directly |

## Code Example

```python
# CIDEr is not in torchmetrics; use pycocoevalcap or manual implementation
# pip install pycocoevalcap

from pycocoevalcap.cider.cider import Cider

# Format: dict mapping image_id → list of captions
# References: multiple captions per image
references = {
    "img_1": ["a cat sitting on a mat", "the cat rests on the mat",
              "a cat is on the mat", "there is a cat on a mat",
              "cat sitting on mat"],
    "img_2": ["a dog running in a park", "the dog runs through the park",
              "a dog is running in the park", "dog running in park",
              "a brown dog runs across the park"]
}

# Candidates: one caption per image
candidates = {
    "img_1": ["a cat is sitting on the mat"],
    "img_2": ["a dog runs in the park"]
}

# Compute CIDEr
cider = Cider()
score, per_image_scores = cider.compute_score(references, candidates)
print(f"CIDEr corpus score: {score:.4f}")
for img_id, s in zip(candidates.keys(), per_image_scores):
    print(f"  {img_id}: CIDEr = {s:.4f}")
# Higher is better
```

## Debugging Use Case

**Scenario**: Comparing image captioning models and identifying generic outputs.

```python
from pycocoevalcap.cider.cider import Cider

def compare_captioning_models(refs, model_outputs_dict):
    """Evaluate multiple captioning models to find which produces
    the most informative, consensus-matching captions."""
    cider = Cider()

    for model_name, candidates in model_outputs_dict.items():
        score, per_image = cider.compute_score(refs, candidates)
        low_scoring = [(img, s) for img, s in zip(candidates.keys(), per_image) if s < 0.5]

        print(f"{model_name}: CIDEr = {score:.4f}")
        if low_scoring:
            print(f"  Low-scoring images ({len(low_scoring)}):")
            for img_id, s in low_scoring[:5]:
                print(f"    {img_id}: {s:.3f} | '{candidates[img_id][0]}'")

    # Diagnostic patterns:
    # Low CIDEr across all images → model generates generic/irrelevant captions
    # High variance in per-image CIDEr → model struggles with specific image types
    # CIDEr much lower than BLEU → model uses common n-grams but misses informative ones
    # CIDEr much higher than BLEU → model captures key concepts with varying phrasing
```

## Related Metrics

- [BLEU](bleu.md) — n-gram precision without corpus weighting
- [ROUGE](rouge.md) — recall-oriented n-gram overlap
- [METEOR](meteor.md) — alignment-based with synonym support
- [BERTScore](bertscore.md) — contextual embedding similarity
