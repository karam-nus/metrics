---
title: "BLEU (Bilingual Evaluation Understudy)"
---
# BLEU (Bilingual Evaluation Understudy)

## Overview

BLEU measures the quality of machine-generated text by computing precision of n-gram overlaps between candidate and reference translations. It uses **modified precision** (clipping n-gram counts to prevent gaming by repetition) across n-grams of order 1 through N (typically N=4), combined via a geometric mean. A **brevity penalty** (BP) discourages overly short translations. BLEU is corpus-level by design, though sentence-level variants exist. It is the standard metric for machine translation and is widely used for other text generation tasks.

## Formula

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

Where:
- $p_n = \frac{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} \text{Count}_{\text{clip}}(\text{n-gram})}{\sum_{C \in \text{Candidates}} \sum_{\text{n-gram} \in C} \text{Count}(\text{n-gram})}$ (clipped n-gram precision)
- $w_n = \frac{1}{N}$: uniform weights (typically $N=4$)
- $\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}$
- $c$: total candidate length, $r$: effective reference length

## Visual Diagram

```
Candidate: "the cat sat on the mat"
Reference: "the cat is on the mat"

1-grams: the(2/2) cat(1/1) sat(0) on(1/1) the(clip) mat(1/1) → p₁ = 5/6
2-grams: "the cat"(1/1) "cat sat"(0) "sat on"(0) "on the"(1/1) "the mat"(1/1) → p₂ = 3/5
3-grams: "the cat sat"(0) "cat sat on"(0) "sat on the"(0) "on the mat"(1/1) → p₃ = 1/4
4-grams: ... → p₄

BP = 1.0 (candidate length ≥ reference length)
BLEU = BP · exp(0.25·(ln p₁ + ln p₂ + ln p₃ + ln p₄))
```

<!-- IMAGE: Venn diagram of n-gram sets from candidate and reference, with the intersection representing clipped matches. -->

## Range & Interpretation

| BLEU Score | Interpretation |
|------------|---------------|
| 0.0 | No n-gram overlap |
| 0.0–0.1 | Very poor; almost no match |
| 0.1–0.3 | Understandable but low quality |
| 0.3–0.5 | Good quality; fluent translations |
| 0.5–0.7 | High quality; near-human |
| > 0.7 | Rare; often indicates data leakage or very close paraphrase |

Range: **[0, 1]** (often reported as 0–100 by scaling ×100). Higher is better.

## When to Use

- Evaluating machine translation systems (the original and primary use case).
- Comparing text generation models on standardized benchmarks.
- Automated evaluation during model development (fast, reproducible).
- When multiple reference translations are available.

## When NOT to Use

- Single-sentence evaluation (BLEU is unreliable at sentence level; use [METEOR](meteor.md) or [BERTScore](bertscore.md)).
- When paraphrase diversity matters (BLEU penalizes valid rephrasings not in the reference).
- Open-ended generation (creative writing, dialogue) where many valid outputs exist.
- When semantic similarity matters more than lexical overlap (use [BERTScore](bertscore.md)).
- Cross-lingual evaluation without matching tokenization.

## What It Can Tell You

- Corpus-level n-gram overlap quality between system output and references.
- Relative ranking of translation/generation systems on the same test set.
- Whether output is too short (brevity penalty activates).
- Precision of surface-level lexical choices.

## What It Cannot Tell You

- Whether the output is fluent or grammatical (high BLEU ≠ readable).
- Semantic correctness (synonyms get no credit).
- Whether meaning is preserved (word order is only partially captured via n-grams).
- Per-sentence quality reliably.
- Recall—BLEU is precision-focused (see [ROUGE](rouge.md) for recall).

## Sensitivity

- **Number of references**: More references → higher BLEU (more chances for n-gram matches).
- **Tokenization**: Case, punctuation, and tokenization scheme (word vs. subword) significantly affect BLEU. Always use standardized tokenization (e.g., sacrebleu).
- **N-gram order**: BLEU-4 is standard; BLEU-1 over-rewards unigram matches.
- **Corpus size**: More stable on larger corpora; noisy for < 100 sentences.
- **Brevity penalty**: Short outputs are heavily penalized even if precise.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [ROUGE](rouge.md) | Recall matters (summarization) | Recall-oriented; n-gram recall |
| [METEOR](meteor.md) | Synonym/paraphrase matching needed | Uses stems, synonyms, word order |
| [BERTScore](bertscore.md) | Semantic similarity needed | Contextual embeddings; soft matching |
| [CIDEr](cider.md) | Image captioning evaluation | TF-IDF weighted; corpus-specific |
| chrF | Character-level evaluation | Robust to morphological variation |
| COMET | Human-correlated MT evaluation | Trained on human judgments; SOTA correlation |

## Code Example

```python
import torch
from torchmetrics.text import BLEUScore

# Initialize BLEU with default n_gram=4
bleu = BLEUScore(n_gram=4)

# Candidate translations (list of strings)
preds = [
    "the cat sat on the mat",
    "there is a cat on the mat"
]

# Reference translations (list of lists of strings — multiple refs per candidate)
targets = [
    ["the cat is on the mat"],
    ["there is a cat on the mat"]
]

# Compute corpus-level BLEU
score = bleu(preds, targets)
print(f"BLEU-4: {score:.4f}")
# Range [0, 1]; higher is better
```

## Debugging Use Case

**Scenario**: Diagnosing machine translation quality issues.

```python
from torchmetrics.text import BLEUScore

def diagnose_translation_quality(preds, targets):
    """Compare BLEU at different n-gram levels to diagnose issues."""
    for n in [1, 2, 3, 4]:
        bleu_n = BLEUScore(n_gram=n)
        score = bleu_n(preds, targets)
        print(f"BLEU-{n}: {score:.4f}")

    # Diagnostic patterns:
    # High BLEU-1, low BLEU-4 → correct words, wrong word order or phrasing
    # Low BLEU-1 → vocabulary mismatch; model uses different words
    # BLEU-4 ≈ 0 → no 4-gram matches; very different phrasing
    # All BLEU-n high → close lexical match to reference

# Example
preds = ["the cat sat on mat"]
targets = [["the cat is sitting on the mat"]]
diagnose_translation_quality(preds, targets)
```

## Related Metrics

- [ROUGE](rouge.md) — recall-oriented n-gram evaluation for summarization
- [METEOR](meteor.md) — alignment-based metric with synonym support
- [BERTScore](bertscore.md) — semantic similarity via contextual embeddings
- [CIDEr](cider.md) — TF-IDF weighted n-gram similarity for captioning
