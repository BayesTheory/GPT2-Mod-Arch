# train.py
# Melhorias focadas em 1x A100 40GB:
# - calibração do micro-batch em eager (antes de compilar/DDP)
# - torch.compile padrão backend='aot_eager' (estável); altere via TORCH_COMPILE_BACKEND
# - SDPA rápido (Flash/Efficient) via sdpa_kernel
# - AdamW fused=True quando suportado (fallback automático)
# - prefetch assíncrono CPU->GPU com CUDA stream para sobrepor I/O e compute
# - backward "chunked" da CrossEntropy para reduzir pico de memória
# - barra de progresso (tqdm) no laço de micro-steps para it/s e ETA
# - seleção de scheduler: LR_SCHEDULE ∈ {'cosine','linear','constant','plateau'} (env) ou config.lr_schedule
# - throughput correto: loga seq/s e tok/s (= seq/s * block_size)

import os
import time
import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from data import DataLoaderLite
from utils import MLflowLogger
from tqdm import trange  # pip install tqdm

# SDPA rápido (seleciona backends ótimos para atenção)
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel_ctx, SDPBackend as _SDPBackend
    def _sdpa_fast_ctx():
        # Prioriza Flash e Efficient; evita backend Math
        return _sdpa_kernel_ctx([_SDPBackend.FLASH_ATTENTION, _SDPBackend.EFFICIENT_ATTENTION], set_priority=False)
except Exception:
    try:
        # Fallback para API legada
        from torch.backends.cuda import sdp_kernel as _legacy_sdp_kernel
        def _sdpa_fast_ctx():
            return _legacy_sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    except Exception:
        def _sdpa_fast_ctx():
            return nullcontext()

class CUDAPrefetcher:
    """
    Prefetch assíncrono de lotes: sobrepõe HtoD e compute usando uma stream dedicada.
    """
    def __init__(self, loader, device):
        assert isinstance(device, str) and device.startswith("cuda")
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_x = None
        self.next_y = None
        self._preload()

    def _preload(self):
        x, y = self.loader.next_batch()
        with torch.cuda.stream(self.stream):
            # H->D não bloqueante
            self.next_x = x.to(self.device, non_blocking=True)
            self.next_y = y.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        x = self.next_x
        y = self.next_y
        self._preload()
        return x, y

def _calibrate_micro_batch(model, example_batch, device, amp_ctx, initial_mb):
    """
    Reduz o micro-batch até um fwd+bwd caber em VRAM.
    Mantém o total_batch_size via grad_accum_steps depois.
    """
    mb = int(initial_mb)
    x, y = example_batch
    while mb > 0:
        try:
            xb, yb = x[:mb].to(device), y[:mb].to(device)
            model.train()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            with amp_ctx:
                # forward "leve": pega logits e testa CE em uma pequena fatia
                logits, _ = model(xb, None)
                N = xb.size(0) * xb.size(1)
                V = logits.size(-1)
                step = min(2048, N)
                loss_test = F.cross_entropy(logits.view(N, V)[:step].float(), yb.view(N)[:step], reduction="mean")
            loss_test.backward()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return mb
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            mb //= 2
    return max(1, mb)

def train(config, logger: MLflowLogger, model_type: str):
    # Preferências de performance seguras (sem afetar qualidade)
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # DDP
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP requer CUDA"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
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

    # AMP (bf16 onde suportado)
    use_bf16 = (device_type == "cuda" and torch.cuda.is_bf16_supported()) or \
               (device_type == "cpu" and torch.backends.cpu.is_cpu_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else None
    amp_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    if master_process:
        print(f"Usando precisão mista (bfloat16): {use_bf16}")

    # Seeds, precisão e dirs
    torch.manual_seed(config.seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + ddp_rank)
    torch.set_float32_matmul_precision("high")
    if master_process:
        os.makedirs(config.log_dir, exist_ok=True)

    # Data
    train_loader = DataLoaderLite(B=config.batch_size, T=config.block_size,
                                  process_rank=ddp_rank, num_processes=ddp_world_size,
                                  split="train", data_root="tinystories_npy")
    val_loader = DataLoaderLite(B=config.batch_size, T=config.block_size,
                                process_rank=ddp_rank, num_processes=ddp_world_size,
                                split="val", data_root="tinystories_npy")

    # Modelo em eager até calibrar
    if model_type == "modern":
        from model import GPT, GPTConfig
        model_args = {k: v for k, v in vars(config).items() if k in GPTConfig.__annotations__}
    elif model_type == "original":
        from model_original import GPT, GPTConfig
        model_args = {k: v for k, v in vars(config).items() if k in GPTConfig.__annotations__}
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
    model = GPT(GPTConfig(**model_args)).to(device)
    raw_model = model

    # Calibração do micro-batch (eager)
    x_cal, y_cal = train_loader.next_batch()
    micro_batch = _calibrate_micro_batch(raw_model, (x_cal, y_cal), device, amp_ctx, config.batch_size)

    # grad_accum baseado no micro-batch calibrado
    global_bs = micro_batch * ddp_world_size
    assert config.total_batch_size % global_bs == 0, "total_batch_size deve ser múltiplo de micro_batch * world_size"
    grad_accum_steps = config.total_batch_size // global_bs

    # Compilação (estável por padrão)
    if config.use_compile:
        backend = os.environ.get("TORCH_COMPILE_BACKEND", "aot_eager")
        print(f"[compile] backend='{backend}'")
        model = torch.compile(model, backend=backend, mode="default")

    # DDP após compile
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    # Logs claros
    if master_process:
        print(f"Batch size total (tokens): {config.total_batch_size}")
        print(f"-> Micro-batch por GPU: {micro_batch}")
        print(f"-> GPUs (world_size): {ddp_world_size}")
        print(f"=> Passos de acumulação (grad_accum_steps): {grad_accum_steps}")

    # Otimizador (tenta AdamW fused; fallback automático)
    try:
        optimizer = torch.optim.AdamW(raw_model.parameters(),
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay,
                                      betas=(config.beta1, config.beta2),
                                      fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(raw_model.parameters(),
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay,
                                      betas=(config.beta1, config.beta2))

    # Seleção de scheduler (env > config > default cosine)
    scheduler_type = os.environ.get("LR_SCHEDULE", getattr(config, "lr_schedule", "cosine")).lower()

    # Plateau (dinâmico por validação)
    if scheduler_type == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        plateau_factor = float(getattr(config, "plateau_factor", 0.5))
        plateau_patience = int(getattr(config, "plateau_patience", 3))
        plateau_cooldown = int(getattr(config, "plateau_cooldown", 1))
        min_lr_cfg = float(getattr(config, "min_lr", 0.0))
        plateau_threshold = float(getattr(config, "plateau_threshold", 1e-3))
        plateau_threshold_mode = str(getattr(config, "plateau_threshold_mode", "rel"))
        scheduler_plateau = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            cooldown=plateau_cooldown,
            min_lr=min_lr_cfg,
            threshold=plateau_threshold,
            threshold_mode=plateau_threshold_mode,
            verbose=True,
        )
    else:
        scheduler_plateau = None

    def get_lr(it):
        warmup_steps = int(getattr(config, "warmup_steps", 0))
        max_steps = int(getattr(config, "max_steps", 1))
        max_lr = float(getattr(config, "max_lr", getattr(config, "learning_rate", 1e-3)))
        min_lr = float(getattr(config, "min_lr", 0.0))
        if scheduler_type == "constant":
            # warmup linear até max_lr e depois mantém constante
            if warmup_steps > 0 and it < warmup_steps:
                return max_lr * (it + 1) / warmup_steps
            return max_lr
        elif scheduler_type == "linear":
            # warmup linear, depois decaimento linear até min_lr
            if warmup_steps > 0 and it < warmup_steps:
                return max_lr * (it + 1) / warmup_steps
            if it <= max_steps:
                denom = max(1, (max_steps - warmup_steps))
                decay_ratio = (it - warmup_steps) / denom
                return max_lr - decay_ratio * (max_lr - min_lr)
            return min_lr
        elif scheduler_type == "plateau":
            # durante warmup, subir linear; após warmup, Plateau ajusta nas avaliações
            if warmup_steps > 0 and it < warmup_steps:
                return max_lr * (it + 1) / warmup_steps
            return None
        else:
            # default: warmup + cosine decay
            if warmup_steps > 0 and it < warmup_steps:
                return max_lr * (it + 1) / warmup_steps
            if it > max_steps:
                return min_lr
            denom = max(1, (max_steps - warmup_steps))
            decay_ratio = (it - warmup_steps) / denom
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)

    # Prefetch assíncrono no treino para sobrepor H2D e compute
    use_prefetch = (device_type == "cuda")
    if use_prefetch:
        train_pf = CUDAPrefetcher(train_loader, device)

    # Chunk para backward da CE (em tokens)
    ce_bw_chunk = int(os.environ.get("CE_BW_CHUNK_TOKENS", "4096"))

    # Early stopping opcional
    use_es = bool(getattr(config, "use_early_stopping", False))
    es_patience = int(getattr(config, "early_stopping_patience", 0))
    best_val = float("inf")
    no_improve = 0

    # Loop
    for it in range(config.max_steps):
        last_step = (it == config.max_steps - 1)

        # Avaliação
        if it % config.eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad(), _sdpa_fast_ctx():
                val_loss_accum = torch.zeros((), device=device)
                for _ in range(config.eval_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with amp_ctx:
                        # caminho padrão do modelo (CE normal) só para avaliar
                        _, loss = model(x[:micro_batch], y[:micro_batch])
                    val_loss_accum += loss
                val_loss_accum /= config.eval_steps
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                val_loss = val_loss_accum.item()
                print(f"Passo {it:5d} | Val loss: {val_loss:.4f}")
                logger.log_metric("val_loss", val_loss, it)
                if it > 0 and (it % config.save_interval == 0 or last_step):
                    ckpt = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "step": it,
                        "val_loss": val_loss
                    }
                    ckpt_path = os.path.join(config.log_dir, f"model_{it:05d}.pt")
                    torch.save(ckpt, ckpt_path)
                    logger.log_artifact(ckpt_path)
                # ReduceLROnPlateau reage à validação
                if scheduler_plateau is not None and it >= getattr(config, "warmup_steps", 0):
                    scheduler_plateau.step(val_loss)  # chamada deve acontecer após a validação
                # early stopping
                if use_es:
                    if val_loss + 1e-6 < best_val:
                        best_val = val_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if it >= getattr(config, "warmup_steps", 0) and no_improve >= es_patience > 0:
                            print(f"[early stopping] sem melhora em {es_patience} avaliações; encerrando no passo {it}.")
                            break
            # Limpeza de VRAM para reduzir fragmentação
            if device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        # Treino
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)

        with _sdpa_fast_ctx():
            # Barra de progresso por micro-steps (it/s e ETA)
            bar = trange(grad_accum_steps, desc=f"step {it} | micro-steps", leave=False)
            t_block = time.time()
            block = 256  # imprime ETA a cada 256 micro-steps

            for micro_step in bar:
                sync_context = model.no_sync if ddp and (micro_step + 1) != grad_accum_steps else nullcontext
                with sync_context():
                    if use_prefetch:
                        x, y = train_pf.next()
                    else:
                        x, y = train_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                    Bm = micro_batch
                    with amp_ctx:
                        # 1) forward sem targets para obter logits
                        logits, _ = model(x[:Bm], None)        # [B, T, V]
                        # 2) backward "chunked" em tokens para reduzir pico de memória
                        N = Bm * x.size(1)
                        V = logits.size(-1)
                        logits_flat = logits.view(N, V)
                        y_flat = y[:Bm].contiguous().view(N)
                        total = 0.0
                        for s in range(0, N, ce_bw_chunk):
                            e = min(s + ce_bw_chunk, N)
                            loss_chunk = F.cross_entropy(logits_flat[s:e].float(), y_flat[s:e], reduction="sum")
                            # normaliza por N e por grad_accum_steps; backward em pedaços
                            (loss_chunk / N / grad_accum_steps).backward(retain_graph=(e < N))
                            total += loss_chunk.detach()
                        loss = (total / N)
                    loss_accum += (loss / grad_accum_steps)

                # Atualiza ETA/it/s no cabeçalho da barra a cada bloco
                if (micro_step + 1) % block == 0:
                    dt_block = time.time() - t_block
                    it_rate = block / max(dt_block, 1e-6)
                    remain = grad_accum_steps - (micro_step + 1)
                    eta_s = remain / max(it_rate, 1e-6)
                    bar.set_description(f"step {it} | {micro_step+1}/{grad_accum_steps} | {it_rate:.1f} it/s | ETA {eta_s/60:.1f}m")
                    t_block = time.time()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Atualização do LR
        set_lr = get_lr(it)
        if scheduler_type == "plateau":
            # Durante warmup, ajusta; após warmup, Plateau ajusta nas avaliações
            if set_lr is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = set_lr
            lr_cur = optimizer.param_groups[0]["lr"]
        else:
            # Schedulers temporais (cosine/linear/constant)
            lr_cur = set_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_cur

        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        seq_per_sec = (config.total_batch_size) / dt
        tokens_per_sec = (config.total_batch_size * config.block_size) / dt  # correto: tok/s = seq/s * T
        if master_process:
            train_loss = loss_accum.item()
            print(f"Passo {it:5d} | Train loss: {train_loss:.6f} | lr: {lr_cur:.4e} | dt: {dt*1000:.2f}ms | seq/s: {seq_per_sec:.2f} | tok/s: {tokens_per_sec:.0f}")
            logger.log_metric("train_loss", train_loss, it)
            logger.log_metric("learning_rate", lr_cur, it)
            logger.log_metric("tokens_per_sec", tokens_per_sec, it)

    if ddp:
        destroy_process_group()

# Dicas:
# - Para reduzir fragmentação de VRAM, exporte PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128" antes de importar torch.
# - Ajuste CE_BW_CHUNK_TOKENS (ex.: 2048/4096/8192) conforme a VRAM disponível.
# - Escolha o scheduler via env LR_SCHEDULE={'cosine','linear','constant','plateau'} ou config.lr_schedule.
