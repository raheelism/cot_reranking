"""
One-time Colab setup — run this first, then run scripts 01, 02, 03 in order.

Paste into a Colab cell and run:

    !git clone https://github.com/raheelism/cot_reranking.git /content/cot_reranking
    %cd /content/cot_reranking
    !pip install -r requirements.txt -q
    %run scripts/01_data_bm25.py

Or run all steps sequentially:
    %run scripts/01_data_bm25.py    # ~2 hrs — data + BM25 (CPU)
    %run scripts/02_inference.py    # ~3-4 hrs — model inference (GPU required)
    %run scripts/03_analysis_routing.py  # ~30 min — analysis + figures (CPU)
"""
