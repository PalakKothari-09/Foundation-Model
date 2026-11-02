"""Loss, optimizer, LR schedule, gradient clipping, batching, checkpointing."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def lm_loss(logits, targets):
    """Mean next-token cross entropy. logits (B, S, V), targets (B, S)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build_optimizer(model, lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8):
    """AdamW with decay only on matrices (ndim >= 2); RMSNorm weights get none."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def cosine_lr(t, lr_max, lr_min, warmup_steps, cosine_end):
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    if t <= cosine_end:
        progress = (t - warmup_steps) / (cosine_end - warmup_steps)
        return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)
    return lr_min


def build_scheduler(optimizer, lr_max, lr_min, warmup_steps, cosine_end):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda t: cosine_lr(t, lr_max, lr_min, warmup_steps, cosine_end) / lr_max
    )


def clip_gradients(model, max_norm=1.0):
    """Returns the pre-clip global grad norm (call scaler.unscale_ first when using fp16)."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def load_tokens(path):
    return np.load(path, mmap_mode="r")


def get_batch(data, batch_size, context_length, device, rng=None):
    """Random windows: x = data[i:i+L], y = data[i+1:i+L+1]."""
    rng = rng if rng is not None else np.random.default_rng()
    starts = rng.integers(0, len(data) - context_length, size=batch_size)  # i+L <= len-1 -> y fits
    return batch_from_starts(data, starts, context_length, device)


def batch_from_starts(data, starts, context_length, device):
    x = np.stack([data[i:i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + context_length] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def fixed_eval_starts(data, num_batches, batch_size, context_length, seed=1234):
    """Same held-out windows every time -> comparable validation numbers."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, len(data) - context_length, size=batch_size) for _ in range(num_batches)]


@torch.no_grad()
def evaluate(model, data, starts_list, context_length, device, amp=False):
    model.eval()
    total = 0.0
    for starts in starts_list:
        x, y = batch_from_starts(data, starts, context_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            total += lm_loss(model(x), y).item()
    model.train()
    return total / len(starts_list)


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, extra=None):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck["step"], ck.get("extra", {})
