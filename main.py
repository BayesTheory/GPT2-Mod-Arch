# main.py
# Ponto de entrada: escolhe config em configs/, prepara TinyStories, inicia MLflow e chama o treino.
# Suporta: init_from = 'scratch' (treino do zero) ou 'gpt2' (transplante parcial seguro).
# Observação: transplante GPT-2 -> arquitetura "modern" só é direto quando n_kv_head == n_head.
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import argparse
import importlib
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import torch
from utils import MLflowLogger, set_seed, prepare_tinystories_dataset
from train import train


def to_dict(cfg_obj) -> Dict[str, Any]:
    if is_dataclass(cfg_obj):
        return asdict(cfg_obj)
    return {k: getattr(cfg_obj, k) for k in dir(cfg_obj) if not k.startswith("__") and not callable(getattr(cfg_obj, k))}

def pretty_print_config(cfg: Dict[str, Any]):
    print("\n================ CONFIG ATIVA ================")
    print(json.dumps(cfg, indent=2, sort_keys=True))
    print("=============================================\n")

def build_model_with_optional_gpt2(config, model_type: str):
    """
    Constrói o modelo conforme 'model_type' e lida com init_from:
      - 'scratch': retorna o modelo do zero
      - 'gpt2': tenta transplante parcial seguro
        * modern + (n_kv_head == n_head): usa from_pretrained_gpt2_partial
        * modern + (n_kv_head != n_head): fallback: copia apenas embeddings do GPT-2
        * original: não implementado aqui (mantém do zero)
    Retorna (model, used_gpt2_transplant: bool)
    """
    if model_type == "modern":
        from model import GPT, GPTConfig
        model_args = {k: v for k, v in vars(config).items() if k in GPTConfig.__annotations__}
        init_from = getattr(config, "init_from", "scratch")

        if init_from == "scratch":
            print("Inicializando modelo moderno do zero.")
            model = GPT(GPTConfig(**model_args))
            return model, False

        elif init_from == "gpt2":
            # Caminho A: transplante direto se não houver GQA
            if model_args.get("n_kv_head", model_args["n_head"]) == model_args["n_head"]:
                print("Inicializando modelo moderno a partir de GPT-2 (transplante parcial compatível).")
                model = GPT.from_pretrained_gpt2_partial("gpt2", override_args=model_args)
                return model, True
            else:
                # Caminho B: fallback embeddings-only
                print("AVISO: n_kv_head != n_head (GQA) torna o transplante de GPT-2 incompatível; aplicando fallback de embeddings-only.")
                model = GPT(GPTConfig(**model_args))
                try:
                    from transformers import GPT2LMHeadModel
                    sd_hf = GPT2LMHeadModel.from_pretrained("gpt2").state_dict()
                    with torch.no_grad():
                        model.transformer.wte.weight.copy_(sd_hf["transformer.wte.weight"])
                    print("Embeddings do GPT-2 copiados (lm_head atado herda os mesmos pesos).")
                except Exception as e:
                    print(f"Falha ao copiar embeddings do GPT-2 (continuando com init aleatório): {e}")
                return model, False

        else:
            print(f"init_from='{init_from}' não suportado explicitamente; iniciando do zero.")
            model = GPT(GPTConfig(**model_args))
            return model, False

    elif model_type == "original":
        # Modelo GPT-2 "original" do projeto (sem modernizações)
        from model_original import GPT, GPTConfig
        model_args = {k: v for k, v in vars(config).items() if k in GPTConfig.__annotations__}
        init_from = getattr(config, "init_from", "scratch")
        if init_from != "scratch":
            print(f"init_from='{init_from}' não implementado para 'original'; iniciando do zero.")
        model = GPT(GPTConfig(**model_args))
        return model, False

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Treinar TinyModel no TinyStories com SDPA/KV-cache e MLflow.")
    parser.add_argument("--config", type=str, default="configs.gpt2_from_scratch", help="Módulo de config (ex.: configs.gpt2_from_scratch ou configs.gpt2_finetune).")
    parser.add_argument("--model-type", type=str, default="modern", choices=["modern", "original"], help="Arquitetura alvo.")
    parser.add_argument("--experiment", type=str, default="TinyStories", help="Nome do experimento MLflow.")
    parser.add_argument("--run-name", type=str, default="run", help="Nome da run MLflow.")
    parser.add_argument("--prepare-data", action="store_true", help="Baixar/Converter TinyStories antes do treino.")
    parser.add_argument("--skip-mlflow", action="store_true", help="Pular tracking de MLflow (debug local).")
    args = parser.parse_args()

    # Carrega config do módulo
    try:
        cfg_mod = importlib.import_module(args.config)
        config = cfg_mod.get_config()
    except Exception as e:
        print(f"Erro ao importar config '{args.config}': {e}")
        sys.exit(1)

    # Preparação de dataset (opcional)
    if args.prepare_data:
        prepare_tinystories_dataset(data_dir="tinystories", npy_dir="tinystories_npy")

    # Semeadura e impressão
    set_seed(getattr(config, "seed", 1337))
    cfg_dict = to_dict(config)
    cfg_dict["model_type"] = args.model_type
    pretty_print_config(cfg_dict)

    # Inicializa MLflow
    logger = MLflowLogger(args.experiment, args.run_name, config) if not args.skip_mlflow else None

    # Build do modelo (com eventual transplante) e despacho para train()
    try:
        # O train() atual instancia o modelo internamente;
        # aqui apenas detectamos e informamos sobre o caminho de init para logs.
        model_for_log, used_transplant = build_model_with_optional_gpt2(config, args.model_type)
        del model_for_log  # Apenas para verificar e logar a rota de inicialização
        if logger is not None:
            logger.log_metric("used_gpt2_transplant", 1.0 if used_transplant else 0.0, step=0)
    except Exception as e:
        print(f"Falha na etapa de construção/diagnóstico do modelo: {e}")
        if logger is not None:
            logger.finish_run()
        sys.exit(1)

    # Executa treino
    try:
        if logger is None:
            # constrói um logger "no-op" compatível
            class _NoLogger:
                def log_metric(self, *a, **k): pass
                def log_artifact(self, *a, **k): pass
                def finish_run(self): pass
            no_logger = _NoLogger()
            train(config, no_logger, args.model_type)
        else:
            train(config, logger, args.model_type)
    finally:
        if logger is not None:
            logger.finish_run()

if __name__ == "__main__":
    main()
