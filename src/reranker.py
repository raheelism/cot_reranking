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
