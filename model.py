# model.py
# Versão final, robusta e de nível de produção, incorporando todas as
# melhores práticas de performance, robustez e engenharia.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Componentes Auxiliares Modernos

class RMSNorm(nn.Module):
    """
    Implementação do RMSNorm (Root Mean Square Layer Normalization).
    RMSNorm normaliza as ativações usando a raiz quadrada da média dos quadrados,
    sem recentralizar os dados (subtrair a média). Isso reduz a sobrecarga
    computacional e provou ser eficaz em modelos Transformer modernos.
    Referência: "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def build_rope_cache(seq_len: int, n_embd: int, n_head: int, base: int = 10000) -> torch.Tensor:
    head_dim = n_embd // n_head
    assert head_dim % 2 == 0, "A dimensão da cabeça (head_dim) deve ser par para RoPE."
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, device=theta.device)
    freqs = torch.outer(t, theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)

# -----------------------------------------------------------------------------
# Arquitetura do Modelo

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
        assert self.n_head % self.n_kv_head == 0, f"n_head ({self.n_head}) deve ser divisível por n_kv_head ({self.n_kv_head})"
        self.head_dim = config.n_embd // self.n_head
        assert self.head_dim % 8 == 0, "head_dim deve ser divisível por 8 para performance ótima em GPUs modernas."
        
        self.Wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=not config.no_bias)
        self.Wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=not config.no_bias)
        self.Wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=not config.no_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=not config.no_bias)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor], kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()
        
        q = self.Wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.Wk(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.Wv(x).view(B, T, self.n_kv_head, self.head_dim)
        
        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        # SDPA: is_causal é True apenas no pré-processamento (prefill), quando o KV-cache
        # está vazio. Isso permite que PyTorch use kernels otimizados como FlashAttention.
        # Não passar uma attn_mask customizada também é crucial para a performance.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        new_cache = (k.detach(), v.detach()) if kv_cache is not None else None
        return y, new_cache

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.use_swiglu = config.use_swiglu
        if config.use_swiglu:
            # Ativação SwiGLU, como em Llama e PaLM.
            # Referência: "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
            # A dimensão intermediária é geralmente 2/3 da dimensão oculta de 4*D,
            # uma escolha empírica para manter a contagem de parâmetros similar à de um MLP com GELU.
            hidden_dim = 4 * config.n_embd
            intermediate_dim = int(2 * hidden_dim / 3)
            self.w1 = nn.Linear(config.n_embd, intermediate_dim, bias=not config.no_bias)
            self.w3 = nn.Linear(config.n_embd, intermediate_dim, bias=not config.no_bias)
            self.w2 = nn.Linear(intermediate_dim, config.n_embd, bias=not config.no_bias)
        else: # MLP original
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

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor], kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        h, new_attn_cache = self.attn(self.ln_1(x), freqs_cis, kv_cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_attn_cache

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = (RMSNorm if config.use_rmsnorm else nn.LayerNorm)(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        if not config.use_rope:
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # RoPE precomputado como um buffer. Ele é criado no CPU e movido para o device/dtype
        # corretos no forward, garantindo flexibilidade e performance.
        if config.use_rope:
            rope = build_rope_cache(self.config.block_size, self.config.n_embd, self.config.n_head)
            self.register_buffer("rope_cache", rope, persistent=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequência de entrada ({T}) excede o block_size ({self.config.block_size})"
        
        x = self.transformer.wte(idx)
        
        freqs_cis = None
        if self.config.use_rope:
            freqs_cis = self.rope_cache[:T].to(device=x.device, dtype=x.dtype)
        else: # Usa embeddings posicionais absolutos
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        
        for block in self.transformer.h:
            x, _ = block(x, freqs_cis)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Aplica soft cap apenas durante a inferência
        if self.config.logit_soft_cap is not None and targets is None:
            cap = self.config.logit_soft_cap
            logits = cap * torch.tanh(logits / cap)
            
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, use_soft_cap: bool = True):
        self.eval()
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.config.n_layer

        # --- Fase 1: Pré-processamento do Prompt (Prefill) ---
        x = self.transformer.wte(idx)
        T_prompt = idx.size(1)
        freqs_cis_prompt = self.rope_cache[:T_prompt].to(device=x.device, dtype=x.dtype) if self.config.use_rope else None
        for i, block in enumerate(self.transformer.h):
            x, kv_cache[i] = block(x, freqs_cis_prompt, kv_cache[i])

        # --- Fase 2: Geração Incremental (Decoding) ---
        for i in range(max_new_tokens):
            # Pega o último token para prever o próximo
            x_last = x[:, -1:, :]
            
            # Normalização e projeção para logits
            x_last = self.transformer.ln_f(x_last)
            logits = self.lm_head(x_last)
            logits = logits[:, -1, :] # (B, vocab_size)

            # Aplica soft-capping se solicitado
            if use_soft_cap and self.config.logit_soft_cap is not None:
                cap = self.config.logit_soft_cap
                logits = cap * torch.tanh(logits / cap)
            
            # Amostragem (temperatura e top-k)
            if temperature > 0.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Anexa o novo token à sequência
            idx = torch.cat((idx, idx_next), dim=1)
            
            # --- Forward pass eficiente: apenas para o novo token ---
            T_current = idx.size(1)
            x = self.transformer.wte(idx_next)
            freqs_cis_step = self.rope_cache[T_current-1:T_current].to(x.device, x.dtype) if self.config.use_rope else None
            for j, block in enumerate(self.transformer.h):
                x, kv_cache[j] = block(x, freqs_cis_step, kv_cache[j])

        self.train()
        return idx

    @classmethod
    def from_pretrained_gpt2_partial(cls, model_type: str, override_args: Optional[dict] = None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Carregando e adaptando pesos do GPT-2 '{model_type}'...")

        config_args_hf = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args_hf['vocab_size'] = 50257
        config_args_hf['block_size'] = 1024
        
        if override_args: config_args_hf.update(override_args)
        
        config = GPTConfig(**config_args_hf)
        model = GPT(config)
        sd = model.state_dict()

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # --- Transplante de Pesos ---
        # 1. Divide a camada c_attn do GPT-2 em Wq, Wk, Wv para o nosso modelo
        for i in range(config.n_layer):
            c_attn_w = sd_hf[f'transformer.h.{i}.attn.c_attn.weight'].t()
            q_w, k_w, v_w = c_attn_w.split(config.n_embd, dim=0)
            sd[f'transformer.h.{i}.attn.Wq.weight'].copy_(q_w)
            sd[f'transformer.h.{i}.attn.Wk.weight'].copy_(k_w)
            sd[f'transformer.h.{i}.attn.Wv.weight'].copy_(v_w)
            
            if not config.no_bias:
                c_attn_b = sd_hf[f'transformer.h.{i}.attn.c_attn.bias']
                q_b, k_b, v_b = c_attn_b.split(config.n_embd, dim=0)
                sd[f'transformer.h.{i}.attn.Wq.bias'].copy_(q_b)
                sd[f'transformer.h.{i}.attn.Wk.bias'].copy_(k_b)
                sd[f'transformer.h.{i}.attn.Wv.bias'].copy_(v_b)

        # 2. Copia as outras camadas compatíveis
        model.transformer.wte.load_state_dict({'weight': sd_hf['transformer.wte.weight']})
        for i in range(config.n_layer):
            model.transformer.h[i].mlp.load_state_dict({
                'w1.weight': sd_hf[f'transformer.h.{i}.mlp.c_fc.weight'].t() if config.use_swiglu else None, # Ajustar se não for SwiGLU
                'w2.weight': sd_hf[f'transformer.h.{i}.mlp.c_proj.weight'].t(),
            }, strict=False) # strict=False é útil aqui
        model.transformer.ln_f.load_state_dict({
            'weight': sd_hf['transformer.ln_f.weight'],
            'bias': sd_hf.get('transformer.ln_f.bias'), # GPT-2 usa LayerNorm com bias
        }, strict=False)

        # Usar strict=False e relatar o que não foi carregado
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Relatório de Transplante de Pesos:")
        print(f"  - Tensores faltando no checkpoint (serão inicializados aleatoriamente):")
        for m in missing: print(f"    - {m}")
        print(f"  - Tensores inesperados no checkpoint (ignorados):")
        for u in unexpected: print(f"    - {u}")
        
        return model