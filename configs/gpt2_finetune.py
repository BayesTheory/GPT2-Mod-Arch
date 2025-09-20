# configs/gpt2_finetune.py
# Configuração para fine-tuning partindo do GPT-2 Small pré-treinado (compatível com pesos originais).
# Mantém arquitetura idêntica ao GPT-2 Small para carregar todos os pesos; modernizações ficam desativadas.
# Rian Costa Ferreira.

class Config:
    # --- Ponto de Entrada ---
    # 'scratch' para treinar do zero, 'resume' para continuar, 'gpt2' para fine-tuning
    init_from = 'gpt2'  # carrega pesos do GPT-2 Small pré-treinado [compatível]

    # --- Arquitetura do Modelo (DEVE ser idêntica ao GPT-2 Small) ---
    n_layer = 12
    n_head = 12
    n_embd = 384
    # Para compatibilidade total com GPT-2 Small, não usar GQA; manter n_kv_head = n_head
    n_kv_head = 12  # igual a n_head para corresponder aos pesos pré-treinados
    block_size = 1024
    vocab_size = 50257

    # Modernizações (desligadas para compatibilidade com checkpoint original)
    use_rope = False     # GPT-2 usa embeddings posicionais absolutos; RoPE mudaria o formato [incompatível]
    use_rmsnorm = False  # GPT-2 usa LayerNorm, não RMSNorm
    use_swiglu = False   # GPT-2 usa MLP com GELU, não SwiGLU
    no_bias = False      # GPT-2 original usa bias nos lineares
    logit_soft_cap = None  # desativado para manter o head padrão

    # --- Treinamento (ajustado para FT estável) ---
    # ATENÇÃO: total_batch_size é número de sequências por passo no seu train.py (não tokens).
    # Se deseja ~2**19 tokens/step com block_size=1024: 524288 / 1024 = 512 sequências.
    total_batch_size = 512   # 512 seq/step ≈ 524288 tokens/step com T=1024
    batch_size = 16          # micro-batch inicial; o calibrador ajusta para caber em VRAM
    max_steps = 2000         # FT curto

    # --- Otimizador (AdamW) + Scheduler ---
    # Prática comum em FT: LR baixo (e.g., 2e-5–5e-5), warmup curto e decaimento suave (cosine/linear) [HF]
    lr_schedule = 'cosine'   # decaimento contínuo por passo após warmup (recomendado p/ FT)
    learning_rate = 3e-5     # LR base/pico do warmup (pequeno para FT estável)
    max_lr = 3e-5            # pico após warmup
    min_lr = 3e-6            # piso para refinamento no fim
    warmup_steps = 120       # ~6% de 2000 passos (warmup curto recomendado em Transformers)
    weight_decay = 0.01      # WD moderado típico em FT de Transformers
    beta1 = 0.9
    beta2 = 0.999            # beta2 mais alto é comum em FT estável

    # --- Estabilizações (se suportadas pelo train.py) ---
    label_smoothing = 0.05   # suaviza variância da CE por minibatch em FT
    use_ema = True           # avalia/salva média exponencial de pesos (EMA)
    ema_decay = 0.999

    # --- Logging e Salvamento ---
    log_dir = 'log_finetune'  # diretório para os logs de fine-tuning
    eval_interval = 100       # valida com boa cadência para monitorar FT
    save_interval = 500       # salva periodicamente
    eval_steps = 40           # média mais robusta da val (reduz variância)

    # --- Misc ---
    seed = 1337
    use_compile = True        # aceleração com torch.compile

def get_config():
    return Config()
