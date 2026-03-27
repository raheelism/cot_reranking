# Design Spec: CoT Reranking Extension Paper
**Date:** 2026-03-27
**Status:** Approved
**Target:** arXiv preprint → SIGIR / ECIR 2026 Workshop
**Timeline:** 3–4 weeks
**Compute:** Google Colab Free Tier (T4 GPU, 15 GB VRAM)

---

## 1. Problem Statement

The paper "Rethinking Reasoning in Document Ranking" (arXiv:2510.08985, Lu et al. 2025) shows that CoT-augmented rerankers consistently underperform direct rerankers. However, it uses fine-tuned Qwen3-4B/8B on A100 GPUs — not reproducible by most researchers. Three key questions remain unanswered:

1. **When** does CoT hurt most? (no query-complexity stratification)
2. **Why** quantitatively — how does reasoning length relate to ranking degradation?
3. **Can we fix it** without any retraining?

---

## 2. Central Thesis

> Chain-of-thought consistently degrades zero-shot reranking at 3B scale, hurts most on factual/simple queries, and the degradation correlates with reasoning length — all addressable with a training-free complexity-aware router.

---

## 3. Paper Title (Working)

*"When Does Chain-of-Thought Hurt Reranking? A Zero-Shot Analysis Across Query Complexity with Selective Routing"*

---

## 4. Novelty Claims

| Claim | Prior Work Gap | Our Contribution |
|---|---|---|
| CoT hurts zero-shot reranking | Only shown for fine-tuned models | First 3B zero-shot confirmation |
| CoT hurts more on simple queries | Hypothesized, never measured | First query-complexity stratification |
| Longer CoT = worse ranking | Not quantified | First correlation with Pearson/Spearman + Mann-Whitney |
| Training-free mitigation | None proposed | First complexity-aware router, no GPU needed |
| Colab-reproducible | No prior paper | Full reproducibility on free hardware |

---

## 5. Model Stack

| Component | Model | Hardware | Role |
|---|---|---|---|
| Reranker | Qwen/Qwen2.5-3B-Instruct (FP16) | T4 GPU (~6 GB VRAM) | Direct-Point and Reason-Point scoring |
| Query encoder | BAAI/bge-small-en-v1.5 (33M) | CPU | Query complexity classification |
| Classifier | Logistic Regression (sklearn) | CPU | Route queries to Direct or CoT |

---

## 6. Datasets

| Dataset | Domain | Test Queries | Role |
|---|---|---|---|
| NFCorpus (BEIR) | Medical, keyword queries | 323 | Factual, short |
| SciFact (BEIR) | Scientific claim verification | 300 | Factual, medium |
| TREC-COVID (BEIR) | COVID literature | 50 | Factual, short |
| BRIGHT-Biology | Multi-hop biology reasoning | ~200 | Complex |
| BRIGHT-Economics | Multi-hop economics reasoning | ~200 | Complex |

All datasets downloaded via `pip install beir` and HuggingFace `datasets` library.
BM25 top-100 candidates per query (via `rank_bm25`).

---

## 7. Reranking Variants

### Direct-Point (no reasoning)
```
System: "Determine if the following passage is relevant to the query.
         Answer only with 'true' or 'false'."
User:   "Query: {q}\nPassage: {p[:512]}"
Score:  softmax(logit[true], logit[false])[0]
```

### Reason-Point (CoT reasoning)
```
NOTE: Qwen2.5-3B-Instruct is not a native reasoning model.
Two-step approach required:

Step 1 — Generate CoT trace:
  Prompt: "Think step by step about whether this passage is relevant
           to the query, then answer with only 'true' or 'false'.\n
           Query: {q}\nPassage: {p[:512]}"
  Call: model.generate(inputs, max_new_tokens=256, do_sample=False)
  Extract: CoT text from generated output (everything before true/false)
  Record:  len(tokenizer(cot_text)['input_ids']) → CoT token length

Step 2 — Score using logits:
  Append CoT to original input, run forward pass
  Extract logits at final position for 'true'/'false' tokens
  Score: softmax(logit[true], logit[false])[0]
```

---

## 8. Experiments

### Experiment 1 — Direct vs CoT Comparison (WHAT)
- Run both variants on all 5 datasets
- Compute NDCG@10 per dataset and averaged
- Main table: rows = datasets, columns = BM25 / Direct-Point / Reason-Point
- Expected: Direct-Point > Reason-Point on BEIR; smaller gap on BRIGHT

### Experiment 2 — Query Complexity Stratification (WHEN)
- Label 200 queries across datasets: Simple / Medium / Complex
  - Distribution: 40 queries per dataset (8 Simple, 16 Medium, 16 Complex each)
  - Simple: ≤8 words, keyword-style, no subordinate clause (e.g., "elephant habitat")
  - Complex: multi-hop, causal, comparative, requires inference (e.g., "How does X affect Y given Z?")
- Train logistic regression on bge-small embeddings (CPU, ~2 min)
- Apply to all queries → complexity bin per query
- Report per-bin NDCG@10 for Direct vs CoT (bar chart)
- Statistical test: Mann-Whitney U on Simple vs Complex NDCG distributions

### Experiment 3 — CoT Length vs NDCG Correlation (WHY)
- For each query: compute avg CoT token length across top-100 doc pairs
- Compute per-query NDCG@10 for Reason-Point
- Pearson r + Spearman ρ between CoT length and NDCG@10
- Bin into Short (<50 tokens), Medium (50–150), Long (>150) → box plots
- Expected: negative correlation (longer CoT → worse NDCG)

### Experiment 4 — Selective Routing (FIX)
- Use complexity classifier: Simple → Direct-Point, Complex → Reason-Point
- Compute NDCG@10 for Selective-Router
- Compare: BM25 < Direct-Point ≤ Selective-Router (target)
- Report ∆NDCG@10 gain vs Direct-only and CoT-only

---

## 9. Colab Notebook Architecture

### Notebook 1 — Data + BM25 (~2 hours)
```
- pip install beir rank_bm25 datasets transformers
- Download 5 datasets
- BM25 top-100 retrieval for all queries
- Save: {dataset}_bm25_top100.json → Google Drive
```

### Notebook 2 — Inference (~3 hours)
```
- Load Qwen2.5-3B-Instruct in FP16 (device_map='auto')
- Direct-Point: score all query-doc pairs, save scores
- Reason-Point: score + record CoT token lengths, save scores + lengths
- Compute NDCG@10 per query for both variants
- Save: {dataset}_direct_results.json, {dataset}_reason_results.json → Drive
```

### Notebook 3 — Analysis + Routing (~2 hours, CPU-only)
```
- Load all saved results from Drive
- Label 200 queries (Simple/Medium/Complex)
- Train bge-small + logistic regression classifier
- Apply classifier to all queries
- Exp 2: NDCG@10 by complexity bin → bar charts
- Exp 3: Correlation analysis → scatter plots + box plots
- Exp 4: Selective routing NDCG@10 → summary table
- Export all figures as PDF/PNG for paper
```

---

## 10. Paper Structure (8 pages, ACM format)

| Section | Pages | Content |
|---|---|---|
| 1. Introduction | 1.0 | Problem, gap in literature, 4 contributions |
| 2. Related Work | 0.5 | 6 key papers, our gap statement |
| 3. Methodology | 1.5 | Setup, variants, datasets, classifier design |
| 4. Main Results | 1.5 | Table 1 (NDCG@10 across 5 datasets) |
| 5. Analysis | 2.0 | Fig 1 (complexity bars) + Fig 2 (length scatter) + Table 2 (routing) |
| 6. Conclusion | 0.5 | Summary, limitations, future work |
| References | 1.0 | ~15 citations |

---

## 11. Key Code Pattern (Memory-Safe Inference)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'Qwen/Qwen2.5-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map='auto'
)

def score_direct(query, passage, tokenizer, model):
    prompt = (
        f"Determine if the following passage is relevant to the query. "
        f"Answer only with 'true' or 'false'.\n\n"
        f"Query: {query}\nPassage: {passage[:512]}"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=1024).to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    true_id  = tokenizer.encode('true',  add_special_tokens=False)[-1]
    false_id = tokenizer.encode('false', add_special_tokens=False)[-1]
    score = torch.softmax(logits[[true_id, false_id]], dim=0)[0].item()
    del inputs, out
    torch.cuda.empty_cache()
    return score
```

---

## 12. Novelty Defense (Reviewer Counterarguments)

**"You used an untuned model — of course results differ from the fine-tuned paper."**
Response: That is the contribution. We show the phenomenon exists at the prompting level, which is more practically relevant — most practitioners use zero-shot LLMs, not custom-trained rerankers.

**"Three BEIR datasets is too few."**
Response: We also include two BRIGHT subsets (reasoning-intensive), giving 5 datasets across two qualitatively different domains, with a total of ~1,000+ test queries.

**"The routing gain might not be statistically significant."**
Response: We report Mann-Whitney U tests throughout and include per-query variance. If routing gains are small, that is also a finding worth reporting.

---

## 13. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| OOM during inference | Low | FP16 + passage[:512] + torch.cuda.empty_cache() per pair |
| Colab session timeout | Medium | Save to Drive after each dataset; resume next session |
| BRIGHT download failure | Low | HuggingFace `datasets` as fallback |
| Classifier too weak | Low | Keyword heuristics as zero-compute fallback |
| CoT/Direct similar on BRIGHT | Low-Med | "Findings are dataset-type dependent" is a publishable result |
| Reviewer rejects zero-shot framing | Low | Framed as extension, not contradiction of original paper |

---

## 14. Related Papers to Cite

1. arXiv:2510.08985 — Lu et al. (main paper being extended)
2. arXiv:2505.16886 — Jedidi et al. (Don't Overthink)
3. arXiv:2502.18418 — Weller et al. (Rank1)
4. arXiv:2508.09539 — Fan et al. (TFRank)
5. arXiv:2503.06034 — Zhuang et al. (Rank-R1)
6. arXiv:2508.07050 — Liu et al. (ReasonRank)
7. arXiv:2505.20046 — Zhang et al. (REARANK)
8. arXiv:2410.05168 — Ji et al. (ReasoningRank)
9. arXiv:2505.14432 — Yang et al. (Rank-K)
10. arXiv:2412.14405 — RaCT
11. arXiv:2510.23544 — LIMRANK
12. arXiv:2309.15088 — Pradeep et al. (RankVicuna)
13. arXiv:2312.02724 — Pradeep et al. (RankZephyr)
14. Sun et al. 2023 — RankGPT (ChatGPT as reranker)
15. Webber et al. 2010 — RBO metric
