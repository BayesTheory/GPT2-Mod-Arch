# generate.py
# Versão final e robusta. Carrega um checkpoint e gera texto de forma
# eficiente e segura, chamando o método model.generate() otimizado.
# Rian Costa Ferreira.

import sys
import torch
import tiktoken
from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Configurações de Geração
# Uso: python generate.py /path/to/checkpoint.pt "Seu prompt..."
# -----------------------------------------------------------------------------
checkpoint_path = "log/ckpt.pt"
prompt = "Hello, I'm a language model,"
max_new_tokens = 100
temperature = 0.8
top_k = 50
# -----------------------------------------------------------------------------

# Sobrescreve padrões com argumentos de linha de comando
if len(sys.argv) > 1:
    checkpoint_path = sys.argv[1]
if len(sys.argv) > 2:
    prompt = sys.argv[2]

# Setup do device e precisão
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando device: {device}")
torch.set_float32_matmul_precision('high')

# Carregamento do modelo
print(f"Carregando checkpoint de {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint_model_args = checkpoint.get('model_args', None)

if checkpoint_model_args is None:
    raise ValueError("Checkpoint inválido: 'model_args' não encontrado.")

model_config = GPTConfig(**checkpoint_model_args)
model = GPT(model_config)

state_dict = checkpoint['model']

# CORREÇÃO: Trata múltiplos prefixos comuns de DDP e compile
unwanted_prefixes = ['_orig_mod.', 'module.']
for p in unwanted_prefixes:
    state_dict = { (k[len(p):] if k.startswith(p) else k): v for k, v in state_dict.items() }
model.load_state_dict(state_dict)

model.eval()
model.to(device)
print("Modelo carregado com sucesso.")

# Tokenização e proteção contra prompts grandes
enc = tiktoken.get_encoding("gpt2") # "r50k_base" é equivalente para GPT-2
prompt_ids = enc.encode(prompt)
idx = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])

# CORREÇÃO: Trunca o prompt se ele for maior que o block_size do modelo
if idx.shape[1] > model.config.block_size:
    print(f"Atenção: O prompt ({idx.shape[1]} tokens) é maior que o block_size ({model.config.block_size}). Truncando...")
    idx = idx[:, -model.config.block_size:]

# Geração de texto
print(f"Gerando a partir do prompt: \"{prompt}\"")
print("---")

# CORREÇÃO: Lógica de autocast robusta que verifica o suporte de hardware
autocast_device = "cuda" if device.startswith("cuda") else "cpu"
dtype_autocast = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
ctx = torch.autocast(device_type=autocast_device, dtype=dtype_autocast)

with torch.no_grad():
    with ctx:
        # CORREÇÃO: Chama o método .generate() correto, sem o parâmetro inexistente
        output_ids = model.generate(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        generated_text = enc.decode(output_ids[0].tolist())
        print(generated_text)

print("---")