# configs/gpt2_from_scratch.py
# Configuração para treinar um modelo do zero com a arquitetura moderna.

class Config:
    # --- Ponto de Entrada ---
    # 'scratch' para treinar do zero, 'resume' para continuar, 'gpt2' para fine-tuning
    init_from = 'scratch'

    # --- Arquitetura do Modelo (GPT-2 Small 124M com melhorias) ---
    n_layer = 12
    n_head = 12
    n_embd = 768
    n_kv_head = 12  # MHA (n_kv_head == n_head) é um baseline robusto para treinar do zero
    block_size = 1024
    vocab_size = 50257
    # Modernizações
    use_rope = True
    use_rmsnorm = True
    use_swiglu = True
    no_bias = True
    logit_soft_cap = 50.0

    # --- Configuração do Treinamento ---
    total_batch_size = 524288  # 2**19 tokens
    batch_size = 64            # Micro-batch size
    max_steps = 20000          # Número total de passos de treinamento
    
    # --- Otimizador (AdamW) ---
    learning_rate = 6e-4       # Taxa de aprendizado inicial (não usada com scheduler de cosseno)
    max_lr = 6e-4              # Taxa de aprendizado máxima para o scheduler
    min_lr = 6e-5              # Taxa de aprendizado mínima
    warmup_steps = 1000        # Passos de aquecimento
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    
    # --- Logging e Salvamento ---
    log_dir = 'log_scratch'    # Diretório para salvar logs e checkpoints
    eval_interval = 250        # Intervalo para rodar a validação
    save_interval = 1000       # Intervalo para salvar checkpoints
    eval_steps = 20            # Número de batches para a validação

    # --- Misc ---
    seed = 1337
    use_compile = True         # Usa torch.compile para acelerar (requer PyTorch 2.0+)

def get_config():
    return Config()


# Rian Costa Ferreira