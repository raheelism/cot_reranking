import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'

def load_model(model_name: str = MODEL_NAME):
    """Load model and tokenizer in FP16. Returns (tokenizer, model)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    print(f"✓ Model loaded: {model_name}")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory used: {mem:.2f} GB")
    return tokenizer, model


DIRECT_SYSTEM = (
    "Determine if the following passage is relevant to the query. "
    "Answer only with 'true' or 'false'."
)

def score_direct(query: str, passage: str, tokenizer, model) -> float:
    """Compute Direct-Point relevance score. Returns P(true) in [0,1]."""
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user",   "content": f"Query: {query}\nPassage: {passage[:512]}"}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, max_length=1024
    ).to(model.device)

    with torch.no_grad():
        out = model(**inputs)

    logits   = out.logits[0, -1, :]
    true_id  = tokenizer.encode('true',  add_special_tokens=False)[-1]
    false_id = tokenizer.encode('false', add_special_tokens=False)[-1]
    score = torch.softmax(logits[[true_id, false_id]], dim=0)[0].item()

    del inputs, out
    torch.cuda.empty_cache()
    return score


REASON_SYSTEM = (
    "Think step by step about whether the following passage is relevant to the query, "
    "then conclude your response with only 'true' or 'false' on the final line."
)

def score_reason(query: str, passage: str, tokenizer, model,
                 max_cot_tokens: int = 256) -> tuple:
    """
    Compute Reason-Point relevance score using two-step generate-then-score.
    Returns (score: float, cot_length: int, cot_text: str).
    """
    messages = [
        {"role": "system", "content": REASON_SYSTEM},
        {"role": "user",   "content": f"Query: {query}\nPassage: {passage[:512]}"}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, max_length=1024
    ).to(model.device)

    # Step 1: Generate CoT + answer token
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_cot_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )

    gen_ids  = generated.sequences[0][inputs['input_ids'].shape[1]:]
    cot_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    cot_length = len(gen_ids)

    # Step 2: Find last true/false token in generated sequence, extract its logit
    true_id  = tokenizer.encode('true',  add_special_tokens=False)[-1]
    false_id = tokenizer.encode('false', add_special_tokens=False)[-1]

    answer_pos = None
    for i in range(len(gen_ids) - 1, -1, -1):
        if gen_ids[i].item() in (true_id, false_id):
            answer_pos = i
            break

    if answer_pos is not None:
        answer_logits = generated.scores[answer_pos][0]
    else:
        # Fallback: use last generated token's logits
        answer_logits = generated.scores[-1][0]

    score = torch.softmax(answer_logits[[true_id, false_id]], dim=0)[0].item()

    del inputs, generated
    torch.cuda.empty_cache()
    return score, cot_length, cot_text


def rerank_dataset(corpus: dict, queries: dict, bm25_results: dict,
                   tokenizer, model, mode: str = 'direct') -> tuple:
    """
    Rerank BM25 top-100 using Direct or Reason mode.
    Returns (rerank_results, cot_lengths) where:
      rerank_results: {qid: {did: score}}
      cot_lengths: {qid: avg_cot_length}  (empty dict if mode='direct')
    """
    assert mode in ('direct', 'reason'), f"mode must be 'direct' or 'reason'"
    rerank_results = {}
    cot_lengths    = {}

    for i, (qid, query_text) in enumerate(queries.items()):
        if qid not in bm25_results:
            continue
        top_docs = list(bm25_results[qid].keys())
        scores   = {}
        lengths  = []

        for did in top_docs:
            passage = corpus.get(did, {}).get('text', '')
            if not passage:
                scores[did] = 0.0
                continue

            if mode == 'direct':
                scores[did] = score_direct(query_text, passage, tokenizer, model)
            else:
                s, cot_len, _ = score_reason(query_text, passage, tokenizer, model)
                scores[did]   = s
                lengths.append(cot_len)

        rerank_results[qid] = scores
        if mode == 'reason' and lengths:
            cot_lengths[qid] = sum(lengths) / len(lengths)

        if (i + 1) % 10 == 0:
            print(f"  [{mode}] {i+1}/{len(queries)} queries done")

    return rerank_results, cot_lengths
