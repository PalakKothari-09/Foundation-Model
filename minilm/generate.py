import torch

EOT = "<|endoftext|>"


@torch.no_grad()
def sample_next(logits, temperature, top_p, generator=None):
    """logits: (V,). temperature <= 0 means greedy."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = (cum - sorted_p) < top_p  # smallest prefix whose mass reaches top_p (always keeps the top token)
        sorted_p = sorted_p * keep
        sorted_p = sorted_p / sorted_p.sum()
        choice = torch.multinomial(sorted_p, 1, generator=generator)
        return int(sorted_i[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate_ids(model, tokenizer, prompt, max_new_tokens=256, temperature=1.0, top_p=1.0, generator=None):
    """Returns only the newly generated ids (stops early on <|endoftext|>)."""
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    eot_id = tokenizer.bytes_to_id.get(EOT.encode())
    new = []
    for _ in range(max_new_tokens):
        window = (ids + new)[-model.context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        nxt = sample_next(logits, temperature, top_p, generator)
        if nxt == eot_id:
            break
        new.append(nxt)
    return new


def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=1.0, top_p=1.0, generator=None):
    new = generate_ids(model, tokenizer, prompt, max_new_tokens, temperature, top_p, generator)
    return prompt + tokenizer.decode(new)
