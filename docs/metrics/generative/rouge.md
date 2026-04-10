---
title: "ROUGE (Recall-Oriented Understudy for Gisting Evaluation)"
---
# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

## Overview

ROUGE is a family of metrics for evaluating automatic summarization and, more broadly, text generation. Unlike BLEU (precision-focused), ROUGE emphasizes **recall**: how much of the reference content is captured by the candidate. The primary variants are **ROUGE-1** (unigram overlap), **ROUGE-2** (bigram overlap), and **ROUGE-L** (longest common subsequence). Each variant reports Precision, Recall, and F1. ROUGE-L is particularly useful as it captures sentence-level word ordering without requiring consecutive n-gram matches.

## Formula

**ROUGE-N** (for n-gram order N):

$$
\text{ROUGE-N}_{\text{recall}} = \frac{\sum_{S \in \text{Refs}} \sum_{\text{n-gram} \in S} \text{Count}_{\text{match}}(\text{n-gram})}{\sum_{S \in \text{Refs}} \sum_{\text{n-gram} \in S} \text{Count}(\text{n-gram})}
$$

**ROUGE-L** (based on Longest Common Subsequence):

$$
R_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{m}, \quad P_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{n}, \quad F_{\text{lcs}} = \frac{(1 + \beta^2) R_{\text{lcs}} P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}
$$

Where $m$ = reference length, $n$ = candidate length, $\beta$ typically set to favor recall.

## Visual Diagram

```
Reference: "the cat sat on the mat and slept"
Candidate: "the cat is on the mat"

ROUGE-1 (unigram recall): matched={the, cat, on, the, mat} → 5/8 = 0.625
ROUGE-2 (bigram recall):  matched={"the cat", "on the", "the mat"} → 3/7 = 0.429
ROUGE-L (LCS):            LCS="the cat on the mat" (len=5) → recall=5/8 = 0.625
```

<!-- IMAGE: Alignment diagram showing matching unigrams, bigrams, and the longest common subsequence between candidate and reference text. -->

## Range & Interpretation

| ROUGE-L F1 | Interpretation |
|------------|---------------|
| 0.0 | No overlap |
| 0.0–0.2 | Poor; minimal content coverage |
| 0.2–0.4 | Moderate; partial coverage |
| 0.4–0.6 | Good; captures key content |
| > 0.6 | Excellent overlap (high for summarization) |

Range: **[0, 1]** for all variants and sub-scores (Precision, Recall, F1). Higher is better.

## When to Use

- Evaluating text summarization models (the primary use case).
- Measuring content coverage/recall in text generation.
- Comparing abstractive or extractive summarization approaches.
- When reference summaries are available and surface overlap is meaningful.

## When NOT to Use

- When semantic similarity matters more than lexical overlap (use [BERTScore](bertscore.md)).
- Machine translation (BLEU or COMET are standard).
- Open-ended generation with high valid-output diversity.
- When paraphrasing quality matters (ROUGE penalizes valid rephrasings).
- Single very short references (ROUGE becomes unreliable).

## What It Can Tell You

- How much reference content the candidate captures (recall).
- Whether the candidate contains extraneous content (precision).
- Relative ranking of summarization systems on the same test set.
- Whether the model captures key n-grams and subsequences.

## What It Cannot Tell You

- Whether the summary is fluent, coherent, or grammatically correct.
- Semantic similarity (synonyms get no credit without additional matching).
- Factual consistency (a summary can have high ROUGE but contain hallucinations).
- Quality of novel or abstractive phrasing not present in the reference.

## Sensitivity

- **Variant selection**: ROUGE-1 is lenient; ROUGE-2 penalizes reordering; ROUGE-L captures long-range ordering.
- **Tokenization/stemming**: Porter stemming increases scores; always report preprocessing details.
- **Number of references**: More references → higher recall scores.
- **Summary length**: Longer candidates tend to have higher recall but lower precision.
- **Stopwords**: Including/excluding stopwords significantly affects unigram-based scores.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [BLEU](bleu.md) | Precision-focused (translation) | N-gram precision with brevity penalty |
| [BERTScore](bertscore.md) | Semantic matching needed | Contextual embeddings; paraphrase-aware |
| [METEOR](meteor.md) | Synonym handling needed | Explicit alignment with stems, synonyms |
| [CIDEr](cider.md) | Image captioning | TF-IDF weighting; corpus-specific |
| ROUGE-WE | Word embedding overlap | Extends ROUGE with soft word matching |
| BARTScore | Generation quality | Model-based; evaluates faithfulness |

## Code Example

```python
from torchmetrics.text.rouge import ROUGEScore

# Initialize ROUGE metric
rouge = ROUGEScore()

# Candidate summaries
preds = [
    "the cat sat on the mat",
    "the quick brown fox jumps over the lazy dog"
]

# Reference summaries
targets = [
    "the cat is on the mat and resting",
    "a fast brown fox leaped over a sleepy dog"
]

# Compute all ROUGE variants
scores = rouge(preds, targets)
for key in sorted(scores.keys()):
    print(f"{key}: {scores[key]:.4f}")
# Outputs: rouge1_fmeasure, rouge1_precision, rouge1_recall,
#          rouge2_fmeasure, ..., rougeL_fmeasure, ...
```

## Debugging Use Case

**Scenario**: Evaluating summarization model and diagnosing failure modes.

```python
from torchmetrics.text.rouge import ROUGEScore

def diagnose_summarization(preds, targets):
    """Use ROUGE variants to diagnose summarization quality."""
    rouge = ROUGEScore()
    scores = rouge(preds, targets)

    r1_p = scores["rouge1_precision"]
    r1_r = scores["rouge1_recall"]
    r2_f = scores["rouge2_fmeasure"]
    rl_f = scores["rougeL_fmeasure"]

    print(f"ROUGE-1 P={r1_p:.3f} R={r1_r:.3f}")
    print(f"ROUGE-2 F1={r2_f:.3f}")
    print(f"ROUGE-L F1={rl_f:.3f}")

    # Diagnostic patterns:
    # High recall, low precision → summary too verbose; includes irrelevant content
    # Low recall, high precision → summary too short; misses key content
    # High ROUGE-1, low ROUGE-2 → correct words but wrong phrasing/order
    # High ROUGE-L vs ROUGE-2 → word order preserved but not as bigrams (abstractive style)

preds = ["the important meeting discussed budget cuts and hiring plans"]
targets = ["the meeting covered budget reductions and new hiring strategies"]
diagnose_summarization(preds, targets)
```

## Related Metrics

- [BLEU](bleu.md) — precision-oriented n-gram metric for translation
- [METEOR](meteor.md) — alignment-based metric with synonym matching
- [BERTScore](bertscore.md) — semantic similarity via contextual embeddings
- [CIDEr](cider.md) — TF-IDF weighted metric for image captioning
