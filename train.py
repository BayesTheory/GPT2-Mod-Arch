# train.py
# Versão final e robusta. Orquestrador de treinamento para o modelo GPT moderno,
# com suporte a DDP, retomada estável e métricas fiéis.

import os
import sys
import time
import math
import importlib.util
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from data import DataLoaderLite

# -----------------------------------------------------------------------------
# Carregamento da Configuração
if len(sys.argv) < 2:
    print("Erro: Forneça o caminho para o arquivo de configuração.", file=sys.stderr)
    sys.exit(1)

config_file_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("config", config_file_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.get_config()
# -----------------------------------------------------------------------------
# Setup do Treinamento Distribuído (DDP)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "Treinamento distribuído requer CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Usando device: {device}")
device_type = "cuda" if "cuda" in device else "cpu"

# Setup do Autocast Robusto
amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
amp_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
# -----------------------------------------------------------------------------
# Setup Geral
torch.manual_seed(config.seed + ddp_rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed + ddp_rank)
torch.set_float32_matmul_precision('high')

if master_process:
    os.makedirs(config.log_dir, exist_ok=True)

# DataLoader
train_loader = DataLoaderLite(B=config.batch_size, T=config.block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=config.batch_size, T=config.block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# Cálculo correto de grad_accum_steps
global_bs = config.batch_size * ddp_world_size
assert config.total_batch_size % global_bs == 0, f"total_batch_size ({config.total_batch_size}) deve ser múltiplo de batch_size * ddp_world_size ({global_bs})"
grad_accum_steps = config.total_batch_size // global_bs
if master_process:
    print(f"Batch size total desejado: {config.total_batch_size}")
    print(f"=> Passos de acumulação de gradiente calculados: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# Inicialização do Modelo
model_args = dict(
    vocab_size=config.vocab_size, n_layer=config.n_layer, n_head=config.n_head,
    n_embd=config.n_embd, block_size=config.block_size, n_kv_head=config.n_kv_head,
    use_rope=config.use_rope, use_rmsnorm=config.use_rmsnorm, use_swiglu=config.use_swiglu,
    no_bias=config.no_bias, logit_soft_cap=config.logit_soft_cap
)
start_step = 0
optimizer_state = None

if config.init_from == 'scratch':
    if master_process: print("Inicializando um novo modelo do zero.")
    model = GPT(GPTConfig(**model_args))
elif config.init_from == 'resume':
    ckpt_path = os.path.join(config.log_dir, "latest.pt")
    if master_process: print(f"Retomando treinamento do checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd']:
        if model_args[k] != checkpoint_model_args[k]:
            raise ValueError(f"Inconsistência: '{k}' é {model_args[k]} na config, mas {checkpoint_model_args[k]} no checkpoint.")
    model = GPT(GPTConfig(**checkpoint_model_args))
    model.load_state_dict(checkpoint['model'])
    start_step = checkpoint['step'] + 1
    optimizer_state = checkpoint['optimizer']
elif config.init_from == 'gpt2':
    if master_process: print("Inicializando com pesos pré-treinados do GPT-2.")
    model = GPT.from_pretrained_gpt2_partial('gpt2', override_args=model_args)

model.to(device)
if config.use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
# -----------------------------------------------------------------------------
# Otimizador
optimizer = torch.optim.AdamW(
    raw_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2), eps=1e-8
)
if optimizer_state:
    optimizer.load_state_dict(optimizer_state)
# -----------------------------------------------------------------------------
# Agendador de Taxa de Aprendizado
def get_lr(it):
    if it < config.warmup_steps: return config.max_lr * (it + 1) / config.warmup_steps
    if it > config.max_steps: return config.min_lr
    decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)
# -----------------------------------------------------------------------------
# Loop de Treinamento
log_file_path = os.path.join(config.log_dir, "log.txt")
if master_process and start_step == 0:
    with open(log_file_path, "w") as f: f.write(f"Iniciando treinamento com a config: {config_file_path}\n")

# Usa 'it' para o iterador do loop para evitar conflito com 'step' do checkpoint
for it in range(start_step, config.max_steps):
    t0 = time.time()
    last_step = (it == config.max_steps - 1)

    # Avaliação periódica
    if it % config.eval_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = torch.zeros((), device=device, dtype=torch.float32)
            for _ in range(config.eval_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with amp_ctx:
                    _, loss = model(x, y)
                val_loss_accum += loss.detach()
            val_loss_accum /= config.eval_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Passo {it:5d} | Perda de validação: {val_loss_accum.item():.4f}")
            with open(log_file_path, "a") as f: f.write(f"{it} val {val_loss_accum.item():.4f}\n")
            if it > 0 and (it % config.save_interval == 0 or last_step):
                checkpoint = {
                    'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'model_args': model_args, 'step': it, 'val_loss': val_loss_accum.item(),
                    'config': vars(config)
                }
                ckpt_path = os.path.join(config.log_dir, f"model_{it:05d}.pt")
                latest_path = os.path.join(config.log_dir, "latest.pt")
                print(f"Salvando checkpoint em {ckpt_path} e {latest_path}")
                torch.save(checkpoint, ckpt_path)
                torch.save(checkpoint, latest_path)
        if ddp:
            dist.barrier() # Garante que todos os processos esperem o master salvar

    # Passo de otimização
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    sync_context = model.no_sync if ddp and (it + 1) % grad_accum_steps != 0 else nullcontext
    
    with sync_context():
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

    if ddp: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(it)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    optimizer.step()
    
    if device_type == "cuda": torch.cuda.synchronize()
    
    # Logging
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    
    if master_process:
        print(f"Passo {it:5d} | Perda de treino: {loss_accum.item():.6f} | lr: {lr:.4e} | dt: {dt:.2f}ms | throughput: {tokens_per_sec:.2f} tokens/s")
        with open(log_file_path, "a") as f: f.write(f"{it} train {loss_accum.item():.6f}\n")

if ddp: destroy_process_group()