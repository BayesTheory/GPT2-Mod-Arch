# model.py
# Versão “baseline” com RoPE real-valued (sem dtypes complexos), KV-cache, SDPA,
# RMSNorm, SwiGLU e transplante parcial de GPT-2. Sem chunking de CE..
# Rian Costa Ferreira.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    """RMSNorm: normaliza pelo RMS, sem centralizar."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# RoPE real-valued: cache com cos/sin e rotação 2D por pares, sem dtypes complexos.
def build_rope_cache(seq_len: int, n_embd: int, n_head: int, base: int = 10000) -> torch.Tensor:
    head_dim = n_embd // n_head
    assert head_dim % 2 == 0, "head_dim deve ser par para RoPE."
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, theta)             # [T, head_dim//2]
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.stack((cos, sin), dim=0)     # [2, T, head_dim//2]

def apply_rope(x: torch.Tensor, freqs_cos_sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, D]; freqs_cos_sin: [2, T, D//2]
    cos, sin = freqs_cos_sin[0], freqs_cos_sin[1]
    B, T, H, D = x.shape
    x = x.view(B, T, H, D // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    cos = cos.to(x1.device, x1.dtype).unsqueeze(0).unsqueeze(2)  # [1, T, 1, d]
    sin = sin.to(x1.device, x1.dtype).unsqueeze(0).unsqueeze(2)  # [1, T, 1, d]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)                # [B, T, H, D]
    return y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: Optional[int] = None
    use_rope: bool = True
    use_rmsnorm: bool = True
    use_swiglu: bool = True
    no_bias: bool = True
    logit_soft_cap: Optional[float] = 50.0

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head deve ser múltiplo de n_kv_head."
        self.head_dim = config.n_embd // config.n_head
        self.Wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=not config.no_bias)
        self.Wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=not config.no_bias)
        self.Wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=not config.no_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=not config.no_bias)
    def forward(self, x: torch.Tensor, freqs: Optional[torch.Tensor],
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()
        q = self.Wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.Wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.Wv(x).view(B, T, self.n_kv_head, self.head_dim)
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, D]
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        new_cache = (k.detach(), v.detach())
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, new_cache

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu
        if config.use_swiglu:
            hidden_dim = 4 * config.n_embd
            intermediate_dim = int(2 * hidden_dim / 3)
            self.w1 = nn.Linear(config.n_embd, intermediate_dim, bias=not config.no_bias)
            self.w3 = nn.Linear(config.n_embd, intermediate_dim, bias=not config.no_bias)
            self.w2 = nn.Linear(intermediate_dim, config.n_embd, bias=not config.no_bias)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=not config.no_bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=not config.no_bias)
    def forward(self, x: torch.Tensor):
        if self.use_swiglu:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        Norm = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_1 = Norm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = Norm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x: torch.Tensor, freqs: Optional[torch.Tensor],
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x_norm1 = self.ln_1(x)
        x_norm2 = self.ln_2(x)
        attn_out, new_attn_cache = self.attn(x_norm1, freqs, kv_cache)
        mlp_out = self.mlp(x_norm2)
        h = x + attn_out
        out = h + mlp_out
        return out, new_attn_cache

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=(RMSNorm if config.use_rmsnorm else nn.LayerNorm)(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        if not config.use_rope:
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        if config.use_rope:
            rope = build_rope_cache(self.config.block_size, self.config.n_embd, self.config.n_head)
            self.register_buffer("rope_cache", rope, persistent=False)  # [2, T, d]
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Comprimento {T} > block_size {self.config.block_size}"
        x = self.transformer.wte(idx)
        freqs = None
        if self.config.use_rope:
            freqs = self.rope_cache[:, :T].to(x.device, x.dtype)  # [2, T, d]
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x, _ = block(x, freqs)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if self.config.logit_soft_cap is not None and targets is None:
            cap = self.config.logit_soft_cap
            logits = cap * torch.tanh(logits / cap)
        return logits, loss
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        self.eval()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.config.n_layer
        x = self.transformer.wte(idx)
        T_prompt = idx.size(1)
        freqs_prompt = self.rope_cache[:, :T_prompt].to(x.device, x.dtype) if self.config.use_rope else None
        for i, block in enumerate(self.transformer.h):
            x, kv_cache[i] = block(x, freqs_prompt, None)
        for _ in range(max_new_tokens):
            logits = self.lm_head(self.transformer.ln_f(x[:, -1:, :]))[:, -1, :]
            if temperature > 0.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            T_current = idx.size(1)
            idx = torch.cat((idx, idx_next), dim=1)
            x_step = self.transformer.wte(idx_next)
            freqs_step = self.rope_cache[:, T_current:T_current + 1].to(x_step.device, x_step.dtype) if self.config.use_rope else None
            for j, block in enumerate(self.transformer.h):
                x_step, kv_cache[j] = block(x_step, freqs_step, kv_cache[j])
            x = torch.cat([x, x_step], dim=1)
        self.train()
        return idx
    @classmethod
    def from_pretrained_gpt2_partial(cls, model_type: str, override_args: Optional[dict] = None):
        """
        Transplanta pesos parcialmente de GPT-2: embeddings e atenção (Wq/Wk/Wv/c_proj).
        Mantém MLP quando use_swiglu=True (não isomórfico a c_fc/c_proj).
        Compatível apenas quando n_kv_head == n_head.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        mp = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args_hf = mp[model_type]
        if override_args:
            config_args_hf.update(override_args)
        if config_args_hf.get("n_kv_head") is not None and config_args_hf["n_kv_head"] != config_args_hf["n_head"]:
            raise ValueError("Transplante GPT-2 não suporta GQA (n_kv_head != n_head).")
        config = GPTConfig(**config_args_hf)
        model = GPT(config)
        sd = model.state_dict()
        print(f"Carregando pesos do GPT-2 '{model_type}'...")
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd["transformer.wte.weight"].copy_(sd_hf["transformer.wte.weight"])
        for i in range(config.n_layer):
            c_attn_w = sd_hf[f"transformer.h.{i}.attn.c_attn.weight"].t()
            q_w, k_w, v_w = c_attn_w.split(config.n_embd, dim=0)
            sd[f"transformer.h.{i}.attn.Wq.weight"].copy_(q_w)
            sd[f"transformer.h.{i}.attn.Wk.weight"].copy_(k_w)
            sd[f"transformer.h.{i}.attn.Wv.weight"].copy_(v_w)
            sd[f"transformer.h.{i}.attn.c_proj.weight"].copy_(sd_hf[f"transformer.h.{i}.attn.c_proj.weight"].t())
            if not config.use_swiglu:
                sd[f"transformer.h.{i}.mlp.c_fc.weight"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"].t())
                sd[f"transformer.h.{i}.mlp.c_proj.weight"].copy_(sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"].t())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("\nRelatório de Transplante de Pesos:")
        print("  - Tensores faltando (inicializados aleatoriamente):")
        for m in missing:
            print(f"    - {m}")
        print("\n  - Tensores inesperados (ignorados):")
        for u in unexpected:
            print(f"    - {u}")
        print("Transplante concluído.")
        return model
