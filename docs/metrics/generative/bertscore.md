---
title: "BERTScore"
---
# BERTScore

## Overview

BERTScore evaluates text generation quality by computing **token-level cosine similarities** between candidate and reference using contextual embeddings from pretrained transformers (BERT, RoBERTa, DeBERTa, etc.). Unlike n-gram metrics (BLEU, ROUGE), BERTScore performs soft semantic matching—capturing paraphrases, synonyms, and meaning preservation even when surface forms differ. It reports **Precision** (how much of the candidate is supported by the reference), **Recall** (how much of the reference is covered), and **F1**. Optional IDF weighting downweights common tokens. BERTScore correlates strongly with human judgment across translation, summarization, and generation tasks.

## Formula

$$
P_{\text{BERT}} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{x}} \max_{x_j \in x} \mathbf{x}_j^\top \hat{\mathbf{x}}_i
$$

$$
R_{\text{BERT}} = \frac{1}{|x|} \sum_{x_j \in x} \max_{\hat{x}_i \in \hat{x}} \mathbf{x}_j^\top \hat{\mathbf{x}}_i
$$

$$
F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
$$

Where:
- $\hat{\mathbf{x}}_i$: contextual embedding of token $i$ in the candidate
- $\mathbf{x}_j$: contextual embedding of token $j$ in the reference
- Embeddings are L2-normalized; the dot product equals cosine similarity
- Optional IDF weighting: $\frac{\text{idf}(x_j)}{\sum_k \text{idf}(x_k)} \max_{\hat{x}_i} \mathbf{x}_j^\top \hat{\mathbf{x}}_i$

## Visual Diagram

```
Reference tokens:  [the]  [cat]  [is]  [sitting]  [on]  [the]  [mat]
                     │      │      │       │        │      │      │
                     ▼      ▼      ▼       ▼        ▼      ▼      ▼
                   BERT → contextual embeddings → [e₁, e₂, e₃, e₄, e₅, e₆, e₇]
                                                          ╲ max cosine sim ╱
Candidate tokens:  [the]  [cat]  [sat]  [on]  [the]  [mat]
                     │      │      │      │      │      │
                     ▼      ▼      ▼      ▼      ▼      ▼
                   BERT → contextual embeddings → [ê₁, ê₂, ê₃, ê₄, ê₅, ê₆]

Precision: avg max similarity for each candidate token against all reference tokens
Recall:    avg max similarity for each reference token against all candidate tokens
F1:        harmonic mean of Precision and Recall
```

<!-- IMAGE: Bipartite matching diagram showing token-level cosine similarities with greedy max alignment highlighted. -->

## Range & Interpretation

| BERTScore F1 | Interpretation |
|-------------|---------------|
| > 0.95 | Near-identical meaning |
| 0.90–0.95 | Excellent semantic similarity |
| 0.85–0.90 | Good; minor meaning differences |
| 0.75–0.85 | Moderate; noticeable divergence |
| < 0.75 | Poor semantic overlap |

Range: **[0, 1]** after rescaling (raw cosine similarities can be negative but are typically rescaled using a baseline). Higher is better. Absolute values depend on the model, layer, and baseline rescaling.

## When to Use

- Evaluating paraphrase quality (BERTScore excels at semantic matching).
- Comparing LLM outputs where surface form varies but meaning should be preserved.
- Text summarization evaluation when abstractive rephrasing is common.
- Machine translation evaluation as a complement to BLEU/METEOR.
- Any generation task where synonyms and paraphrases should receive credit.

## When NOT to Use

- When exact lexical matching is required (use [BLEU](bleu.md)).
- For very long documents (token limit of underlying model; truncation loses information).
- When computational cost is prohibitive (requires transformer forward pass per sentence pair).
- Cross-lingual evaluation without a multilingual model.
- When factual consistency matters more than similarity (use NLI-based metrics).

## What It Can Tell You

- Whether candidate and reference are semantically similar despite different wording.
- Per-token alignment quality (which tokens match well).
- Whether the candidate covers the reference content (recall) and stays on-topic (precision).
- Relative ranking of generation systems with better human correlation than n-gram metrics.

## What It Cannot Tell You

- Factual correctness (semantically similar ≠ factually correct).
- Discourse coherence or logical structure.
- Whether generation is creative, engaging, or fluent beyond semantic content.
- Quality differences for domain-specific terminology without domain-adapted models.

## Sensitivity

- **Model choice**: RoBERTa-large is the default recommendation; different models give different scores. Always compare using the same model.
- **Layer selection**: Intermediate layers often outperform the final layer; default settings are tuned per model.
- **IDF weighting**: Improves correlation with human judgment by downweighting stopwords; computed on the test corpus.
- **Baseline rescaling**: Rescaling against random sentence baselines makes scores more interpretable; enabled by default.
- **Sequence length**: Tokens beyond the model's max length (512 for BERT) are truncated.

## Alternatives & When to Prefer Them

| Metric | Prefer When | Key Difference |
|--------|------------|----------------|
| [BLEU](bleu.md) | Exact n-gram precision needed | Lexical matching only; no semantics |
| [ROUGE](rouge.md) | Recall-focused summarization | N-gram recall; no soft matching |
| [METEOR](meteor.md) | Synonym matching with alignment | Explicit WordNet synonyms; simpler |
| BLEURT | Learned human-correlated score | Trained on human judgments; more expensive |
| COMET | SOTA MT evaluation | Trained regression model; best correlation |
| MoverScore | Earth-mover distance semantics | Uses Word Mover's Distance on embeddings |
| BARTScore | Faithfulness evaluation | Uses generation likelihood as score |

## Code Example

```python
from torchmetrics.text.bert import BERTScore

# Initialize BERTScore with a specific model
bertscore = BERTScore(model_name_or_path="roberta-large", lang="en")

# Candidate and reference texts
preds = [
    "the cat is resting on the rug",
    "a quick brown fox leaps over a lazy dog"
]
targets = [
    "the cat sat on the mat",
    "the fast brown fox jumps over the sleepy dog"
]

# Compute BERTScore — returns dict with 'precision', 'recall', 'f1'
scores = bertscore(preds, targets)
print(f"BERTScore Precision: {scores['precision'].mean():.4f}")
print(f"BERTScore Recall:    {scores['recall'].mean():.4f}")
print(f"BERTScore F1:        {scores['f1'].mean():.4f}")
# Range [0, 1] (after rescaling); higher is better
```

## Debugging Use Case

**Scenario**: Evaluating paraphrase quality and comparing LLM outputs.

```python
from torchmetrics.text.bert import BERTScore

def evaluate_paraphrases(originals, paraphrases, model="roberta-large"):
    """Use BERTScore to assess semantic preservation in paraphrasing."""
    bertscore = BERTScore(model_name_or_path=model, lang="en")
    scores = bertscore(paraphrases, originals)

    for i, (orig, para) in enumerate(zip(originals, paraphrases)):
        f1 = scores["f1"][i].item()
        p = scores["precision"][i].item()
        r = scores["recall"][i].item()
        print(f"[{i}] F1={f1:.3f} P={p:.3f} R={r:.3f}")
        print(f"  Original:   {orig}")
        print(f"  Paraphrase: {para}")

        # Diagnostic patterns:
        # High F1 (>0.92) → strong semantic preservation
        # High P, low R → paraphrase covers subset of original meaning
        # Low P, high R → paraphrase adds information not in original
        # Low F1 (<0.80) → significant meaning change; poor paraphrase

originals = [
    "The economy grew by 3.2% in the third quarter.",
    "Machine learning models require large datasets for training."
]
paraphrases = [
    "GDP increased 3.2% during Q3.",
    "Training ML models needs lots of data."
]
evaluate_paraphrases(originals, paraphrases)
```

## Related Metrics

- [BLEU](bleu.md) — n-gram precision for translation
- [ROUGE](rouge.md) — recall-oriented n-gram overlap for summarization
- [METEOR](meteor.md) — alignment-based metric with synonym support
- [CIDEr](cider.md) — TF-IDF weighted metric for image captioning
