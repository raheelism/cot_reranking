# When Does Chain-of-Thought Hurt Reranking?

**A Systematic Zero-Shot Analysis Across Query Complexity, Reasoning Length, and Selective Routing**

> Muhammad Raheel Anwar  
> National University of Sciences & Technology (NUST)  
> `mranwar.mscs22seecs@seecs.edu.pk`

[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Colab Ready](https://img.shields.io/badge/Colab-T4%20GPU-orange?logo=googlecolab)](https://colab.research.google.com)

---

## Overview

Chain-of-thought (CoT) prompting has shown strong gains on multi-step reasoning tasks, yet its role in document reranking remains unclear. This paper presents the **first systematic zero-shot study** of CoT in pointwise document reranking.

Using **Qwen2.5-3B-Instruct** on five benchmarks (BEIR and BRIGHT), we show that:

- **Direct logit scoring** (`Direct-Point`) consistently outperforms **reason-then-score** (`Reason-Point`) on **all five datasets** (mean NDCG@10 gap = 0.048)
- CoT degrades ranking uniformly across **simple, medium, and complex** queries — no complexity-conditional benefit exists
- A complexity-aware router that sends complex queries to CoT **still underperforms** pure direct scoring
- All experiments run on a **single free-tier Google Colab T4 GPU**

---

## Main Results

| Dataset     | BM25   | Direct-Point | Reason-Point | Δ (Direct − Reason) |
|-------------|--------|-------------|--------------|---------------------|
| NFCorpus    | 0.2616 | **0.2835**  | 0.2666       | +0.017              |
| SciFact     | 0.5438 | **0.5938**  | 0.4702       | +0.124              |
| TREC-COVID  | 0.4132 | **0.6040**  | 0.5838       | +0.020              |
| BRIGHT-Bio  | 0.0562 | **0.1113**  | 0.1100       | +0.001              |
| BRIGHT-Econ | 0.0591 | **0.0855**  | 0.0678       | +0.018              |
| **Average** | 0.2868 | **0.3356**  | 0.3007       | **+0.048**          |

---

## Repository Structure

```
cot_reranking/
├── notebooks/
│   ├── 01_data_bm25.ipynb          # Dataset download + BM25 retrieval
│   ├── 02_inference.ipynb          # LLM scoring (Direct-Point & Reason-Point)
│   └── 03_analysis_routing.ipynb   # Statistics, figures, selective routing
├── src/
│   ├── reranker.py                 # Direct-Point and Reason-Point scorers
│   ├── data_utils.py               # Dataset loading and preprocessing
│   ├── metrics.py                  # NDCG@10 and statistical tests
│   ├── classifier.py               # Query complexity (length terciles)
│   └── routing.py                  # Selective routing logic
├── figures/
│   ├── fig1.png                    # NDCG by complexity stratum
│   ├── fig2.png                    # CoT length vs. per-query NDCG
│   └── fig_pipeline.png            # System pipeline diagram
├── paper/
│   ├── main.tex                    # LaTeX source
│   └── references.bib              # Bibliography
├── submission/                     # arXiv-ready submission package
├── results/                        # Saved JSON result files
├── requirements.txt
└── README.md
```

---

## Reproducibility

All experiments were run on a **free-tier Google Colab T4 GPU** (15.6 GB VRAM). No proprietary data, fine-tuning, or specialized infrastructure is required.

### Hardware & Software

| Property             | Value                        |
|----------------------|------------------------------|
| GPU                  | NVIDIA Tesla T4 (15.6 GB VRAM) |
| Python               | 3.10                         |
| CUDA                 | 12.2                         |
| PyTorch              | 2.3.1                        |
| Transformers         | 4.43.0                       |
| BEIR                 | 2.0.0                        |
| Peak VRAM (Direct)   | 9.28 GB                      |
| Peak VRAM (Reason)   | 11.4 GB                      |
| Total runtime        | ~50 h (across 3 Colab sessions) |

### Setup

```bash
git clone https://github.com/raheelism/cot_reranking.git
cd cot_reranking
pip install -r requirements.txt
```

### Step 1 — Data & BM25 Retrieval

Open and run `notebooks/01_data_bm25.ipynb`.

This notebook:
- Installs all dependencies
- Downloads NFCorpus, SciFact, TREC-COVID, BRIGHT-Bio, and BRIGHT-Econ via the `beir` package
- Runs BM25Okapi retrieval (top-100 candidates per query, `k1=1.5`, `b=0.75`)
- Saves compressed candidate JSON files to Google Drive

**Runtime:** ~2 hours

### Step 2 — LLM Inference

Open and run `notebooks/02_inference.ipynb`.

This notebook:
- Loads `Qwen/Qwen2.5-3B-Instruct` in FP16 from Hugging Face
- Scores all query–document pairs under both `Direct-Point` and `Reason-Point`
- Records per-query NDCG@10 and CoT chain lengths
- Saves results as JSON (checkpointed per dataset — safe to interrupt and resume)

**Runtime:** ~40 hours total (interruptible via Drive checkpoints)

> **Tip:** Run one dataset at a time across multiple Colab sessions. The checkpoint system resumes from where it left off.

### Step 3 — Analysis & Routing

Open and run `notebooks/03_analysis_routing.ipynb` (CPU only, no GPU needed).

This notebook:
- Loads saved result files
- Computes complexity stratification (query-length terciles)
- Runs Mann–Whitney U tests for statistical significance
- Computes Pearson r and Spearman ρ for CoT length vs. NDCG
- Evaluates the selective router
- Generates all paper figures

**Runtime:** ~15 minutes

---

## Prompt Templates

**Direct-Point** — logit extracted at the final input token, no generation:
```
System: "Determine if the following passage is relevant to the query.
         Answer only with 'true' or 'false'."
User:   "Query: {query}\nPassage: {passage[:512]}"
```

**Reason-Point** — up to 128 tokens generated, logit extracted at the last decision token:
```
System: "Think step by step about whether the passage is relevant to
         the query, then conclude with only 'true' or 'false'."
User:   "Query: {query}\nPassage: {passage[:512]}"
```

Scoring formula (both variants):

$$s(q, d) = \text{softmax}([\ell_{\text{true}},\ \ell_{\text{false}}])[0]$$

---

## Datasets

| Dataset     | Suite  | Corpus    | Queries | BM25 Recall@100 | Domain      |
|-------------|--------|-----------|---------|-----------------|-------------|
| NFCorpus    | BEIR   | 3,633     | 323     | 0.578           | Medical     |
| SciFact     | BEIR   | 5,183     | 300     | 0.836           | Scientific  |
| TREC-COVID  | BEIR   | 171,332   | 50      | 0.642           | Biomedical  |
| BRIGHT-Bio  | BRIGHT | 57,359    | 103     | 0.391           | Biology     |
| BRIGHT-Econ | BRIGHT | 50,220    | 103     | 0.344           | Economics   |

All datasets are loaded automatically via the `beir` Python package and Hugging Face `datasets`. No manual downloads are required.

---

## Citation

If you use this work, please cite:

```bibtex
@article{anwar2025cotrreranking,
  title     = {When Does Chain-of-Thought Hurt Reranking?
               A Systematic Zero-Shot Analysis Across Query Complexity,
               Reasoning Length, and Selective Routing},
  author    = {Anwar, Muhammad Raheel},
  journal   = {arXiv preprint},
  year      = {2025}
}
```

---

## Acknowledgements

This work extends the analysis of [Lu et al. (2025)](https://arxiv.org/abs/2510.08985) to the zero-shot regime and provides the first query-complexity-stratified evaluation of CoT in document reranking.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
