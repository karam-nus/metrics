---
title: "METEOR (Metric for Evaluation of Translation with Explicit ORdering)"
---
# METEOR (Metric for Evaluation of Translation with Explicit ORdering)

## Overview

METEOR evaluates text generation quality through explicit word-level alignment between candidate and reference, incorporating **exact matches**, **stem matches** (via Porter stemmer), **synonym matches** (via WordNet), and **paraphrase matches**. After alignment, METEOR computes a harmonic mean of unigram precision and recall (recall-weighted), then applies a **fragmentation penalty** that penalizes discontiguous matches—rewarding fluent, well-ordered output. METEOR correlates better with human judgment than BLEU at the sentence level and across many language pairs.

## Formula

$$
\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Penalty})
$$

Where:
- $F_{\text{mean}} = \frac{10 \cdot P \cdot R}{R + 9 \cdot P}$ (harmonic mean weighted toward recall, $\alpha=0.9$)
- $P = \frac{|\text{matched unigrams}|}{|\text{candidate unigrams}|}$, $R = \frac{|\text{matched unigrams}|}{|\text{reference unigrams}|}$
- $\text{Penalty} = 0.5 \cdot \left(\frac{\text{chunks}}{\text{matched unigrams}}\right)^3$
- **chunks**: minimum number of contiguous groups of matched unigrams in candidate order

## Visual Diagram

```
Reference: "the   cat   is   sitting   on   the   mat"
              │     │         │         │    │     │
Candidate: "the   cat   sat   on   the   mat"
           └──┘  └──┘       └──┘  └──┘ └──┘

Alignment stages:
  1. Exact:   the→the, cat→cat, on→on, the→the, mat→mat
  2. Stem:    sat→sitting (sit→sit)
  3. Synonym: (none needed here)

Matched: 6/6 candidate, 6/7 reference → P=1.0, R=0.857
Chunks: 2 ("the cat sat" and "on the mat") → Penalty = 0.5·(2/6)³ = 0.0019
METEOR = F_mean · (1 - 0.0019) ≈ F_mean
```

<!-- IMAGE: Alignment diagram with colored lines connecting matched words by type: green=exact, blue=stem, orange=synonym. -->

## Range & Interpretation

| METEOR Score | Interpretation |
|-------------|---------------|
| 0.0 | No matches |
| 0.0–0.2 | Poor quality |
| 0.2–0.4 | Moderate quality |
| 0.4–0.6 | Good; fluent with content coverage |
| > 0.6 | Excellent; near-human quality |

Range: **[0, 1]**. Higher is better. METEOR typically yields higher absolute values than BLEU for the same system.

## When to Use

- Machine translation evaluation, especially at sentence level.
- When synonym and paraphrase matching matter (BLEU misses these).
- When correlation with human judgment is critical.
- Evaluating text generation where word order and fluency matter.
- Multilingual evaluation (METEOR supports many languages).

## When NOT to Use

- When only corpus-level evaluation is needed and BLEU suffices.
- Languages without WordNet synonym data (unless using universal METEOR).
- When computational cost matters (METEOR is slower than BLEU due to alignment).
- Open-ended creative generation with no meaningful reference.
- When deep semantic understanding is needed (use [BERTScore](bertscore.md)).

## What It Can Tell You

- Whether the candidate captures reference content using exact words, stems, or synonyms.
- Whether the word order is fluent (fragmentation penalty).
- Per-sentence quality scores (more reliable than sentence-level BLEU).
- The balance between precision and recall in generation.

## What It Cannot Tell You

- Deep semantic similarity beyond synonym lookup.
- Factual correctness or consistency.
- Quality of truly novel phrasings not in any synonym database.
- Discourse-level coherence across multiple sentences.

## Sensitivity

- **Language resources**: Quality depends on availability of stemmers and synonym databases. English has the best support.
- **Parameter tuning**: α (precision/recall weight), β (fragmentation penalty), γ (penalty weight) are tuned per language; defaults are for English.
- **Tokenization**: Case-folding and tokenization affect matching; use standardized preprocessing.
- **Synonym coverage**: WordNet coverage is uneven; domain-specific terms may lack synonyms.
- **Multiple references**: METEOR scores the best alignment across references; more references help.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [BLEU](bleu.md) | Corpus-level precision; speed | No synonym/stem matching; precision-only |
| [ROUGE](rouge.md) | Summarization recall | Recall-focused without alignment |
| [BERTScore](bertscore.md) | Deep semantic matching | Contextual embeddings; no explicit synonym DB |
| [CIDEr](cider.md) | Image captioning | TF-IDF weighting; consensus-based |
| COMET | SOTA MT evaluation | Trained on human judgments; better correlation |
| TER | Edit-distance based evaluation | Counts edits; complementary to METEOR |

## Code Example

```python
import nltk
from nltk.translate.meteor_score import meteor_score

# Ensure required NLTK data is downloaded
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Reference and candidate (tokenized as word lists)
reference = "the cat is sitting on the mat".split()
candidate = "the cat sat on the mat".split()

# Compute METEOR for a single sentence
score = meteor_score([reference], candidate)
print(f"METEOR: {score:.4f}")
# Range [0, 1]; higher is better

# Corpus-level METEOR: average over sentences
references_corpus = [
    ["the cat is sitting on the mat".split()],
    ["a dog runs across the field".split()]
]
candidates_corpus = [
    "the cat sat on the mat".split(),
    "the dog ran across a field".split()
]

corpus_meteor = sum(
    meteor_score(refs, cand)
    for refs, cand in zip(references_corpus, candidates_corpus)
) / len(candidates_corpus)
print(f"Corpus METEOR: {corpus_meteor:.4f}")
```

## Debugging Use Case

**Scenario**: Evaluating text generation diversity and fluency.

```python
import nltk
from nltk.translate.meteor_score import meteor_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def diagnose_generation_quality(candidates, references):
    """Use METEOR to assess synonym usage and word ordering."""
    for i, (cand, refs) in enumerate(zip(candidates, references)):
        cand_tokens = cand.split()
        ref_tokens_list = [r.split() for r in refs]

        score = meteor_score(ref_tokens_list, cand_tokens)
        print(f"Sample {i}: METEOR={score:.4f} | Candidate: '{cand}'")

    # Diagnostic patterns:
    # High METEOR, low BLEU → model uses valid synonyms/paraphrases
    # Low METEOR, high BLEU → exact word matches but poor alignment/ordering
    # Low METEOR, low BLEU → poor content coverage overall
    # Compare with BERTScore: if METEOR is low but BERTScore is high,
    #   the generation is semantically correct but uses unseen vocabulary

candidates = [
    "the feline rested on the rug",
    "cat mat the the on sat"
]
references = [
    ["the cat sat on the mat"],
    ["the cat sat on the mat"]
]
diagnose_generation_quality(candidates, references)
```

## Related Metrics

- [BLEU](bleu.md) — precision-oriented n-gram metric
- [ROUGE](rouge.md) — recall-oriented metric for summarization
- [BERTScore](bertscore.md) — semantic similarity via contextual embeddings
- [CIDEr](cider.md) — TF-IDF weighted metric for captioning
