# configs/gpt2_finetune.py
# Configuração para fine-tuning a partir do GPT-2 pré-treinado.
# Rian Costa Ferreiras
class Config:
    # --- Ponto de Entrada ---
    # 'scratch' para treinar do zero, 'resume' para continuar, 'gpt2' para fine-tuning
    init_from = 'gpt2'

    # --- Arquitetura do Modelo (DEVE ser compatível com o GPT-2 Small) ---
    n_layer = 12
    n_head = 12
    n_embd = 768
    # n_kv_head pode ser diferente, pois as camadas de atenção serão inicializadas do zero
    # Usar GQA (n_kv_head < n_head) pode ser uma boa otimização aqui.
    n_kv_head = 4 # Exemplo de GQA com 3 grupos de query por K/V head (12/4=3)
    block_size = 1024
    vocab_size = 50257
    # Modernizações ativadas
    use_rope = True
    use_rmsnorm = True
    use_swiglu = True
    no_bias = True
    logit_soft_cap = 50.0

    # --- Configuração do Treinamento (Ajustada para Fine-tuning) ---
    total_batch_size = 524288  # 2**19 tokens
    batch_size = 64            # Micro-batch size
    max_steps = 2000           # Fine-tuning requer muito menos passos
    
    # --- Otimizador (AdamW) ---
    # A TAXA DE APRENDIZADO É O PARÂMETRO MAIS IMPORTANTE NO FINE-TUNING.
    # Deve ser ~10x menor que no treinamento do zero.
    learning_rate = 3e-5
    max_lr = 3e-5              # Taxa de aprendizado máxima (constante ou com decay suave)
    min_lr = 3e-6              # Taxa de aprendizado mínima
    warmup_steps = 200         # Aquecimento mais curto
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    
    # --- Logging e Salvamento ---
    log_dir = 'log_finetune'   # Diretório separado para os logs de fine-tuning
    eval_interval = 100        # Avalia com mais frequência
    save_interval = 500        # Salva com mais frequência
    eval_steps = 20

    # --- Misc ---
    seed = 1337
    use_compile = True         # Usa torch.compile para acelerar

def get_config():
    return Config()