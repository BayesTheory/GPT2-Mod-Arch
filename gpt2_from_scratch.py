# configs/gpt2_from_scratch.py  (perfil tiny para TinyStories, 1x A100 40GB)

class Config:
    # --- Ponto de Entrada ---
    init_from = 'scratch'  # treino do zero

    # --- Arquitetura tiny (na faixa TinyStories) ---
    n_layer = 6            # camadas
    n_head = 6             # cabeças (d_head = 384/6 = 64)
    n_embd = 384           # dimensão do embedding
    n_kv_head = 3          # GQA (reduz custo/memória de KV)
    block_size = 256       # contexto curto adequado ao TinyStories
    vocab_size = 50257     # GPT-2 BPE compatível

    # Modernizações
    use_rope = True        # posicionais rotacionais (RoPE)
    use_rmsnorm = True     # norma moderna estável
    use_swiglu = True      # MLP com SwiGLU
    no_bias = True
    logit_soft_cap = 50.0

    # --- Treinamento (passo curto e estável) ---
    total_batch_size = 2048   # sequências por passo (tok/s escala com block_size)
    batch_size = 64           # micro-batch; grad_accum_steps = 2048/(64*1) = 32
    max_steps = 500           # run curto (~3h a ~20.5s/step)

    # --- Otimizador (AdamW) + Scheduler ---
    # selecione via campo abaixo ou env LR_SCHEDULE={'plateau','cosine','linear','constant'}
    lr_schedule = 'plateau'   # padrão: dinâmico por validação; troque para 'cosine' se quiser decaimento contínuo
    learning_rate = 2.0e-4    # alvo/pico após warmup
    max_lr = 2.0e-4           # usado no warmup e nos schedulers temporais
    min_lr = 5.0e-6           # piso (para cosine/linear) e limite inferior no plateau
    warmup_steps = 100        # warmup curto (sobe até o pico e libera o scheduler)

    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95

    # --- Hiperparâmetros do 'plateau' (dinâmico por val_loss) ---
    plateau_factor = 0.5          # reduz LR pela metade quando plateia
    plateau_patience = 1          # nº de avaliações sem melhora antes de reduzir LR
    plateau_cooldown = 1          # evita reduções em sequência por ruído
    plateau_threshold = 1e-3      # ignora micro variações (threshold de significância)
    plateau_threshold_mode = 'rel' # 'rel' ou 'abs' (rel = melhora relativa)

    # --- Early Stopping (opcional) ---
    use_early_stopping = True
    early_stopping_patience = 3    # nº de avaliações sem melhora após warmup

    # --- Logging e Salvamento ---
    log_dir = 'log_scratch'
    eval_interval = 150            # mais frequente para o Plateau reagir cedo
    save_interval = 300            # salva ao final do run curto
    eval_steps = 20

    # --- Misc ---
    seed = 1337
    use_compile = True             # backend 'aot_eager' (estável)

def get_config():
    return Config()
