# utils.py
# Funções auxiliares para o projeto, incluindo logging de experimentos com MLflow.
# Rian Costa Ferreira.

import os
import random
import numpy as np
import torch
import mlflow
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

def set_seed(seed: int):
    """Define seeds e ativa caminhos determinísticos quando possível."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # determinismo (pode impactar performance dependendo do hardware)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLflowLogger:
    """
    Wrapper simples para MLflow: registra params, métricas e artefatos.
    """
    def __init__(self, experiment_name: str, run_name: str, config: object):
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        params = {k: v for k, v in vars(config).items() if not k.startswith("__")}
        mlflow.log_params(params)
        print(f"MLflow run started. Experiment='{experiment_name}', Run='{run_name}'")

    def log_metric(self, key: str, value: float, step: int):
        mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, local_path: str):
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path)

    def finish_run(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()
            print("MLflow run finished.")

def _session_with_retries(total=5, backoff=0.5):
    s = requests.Session()
    retries = Retry(total=total, backoff_factor=backoff, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def prepare_tinystories_dataset(data_dir="tinystories", npy_dir="tinystories_npy"):
    """
    Baixa TinyStories (bins prontos) e converte para shards .npy (int32).
    """
    if os.path.exists(npy_dir):
        print(f"Dataset já preparado em '{npy_dir}'.")
        return
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    def download_file(url, out_path, timeout=60):
        s = _session_with_retries()
        print(f"Baixando {url} -> {out_path} ...")
        with s.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    def convert_bin_to_npy(bin_file, npy_file, header_bytes: int = 0):
        raw = np.fromfile(bin_file, dtype=np.uint8)
        if header_bytes:
            raw = raw[header_bytes:]
        assert raw.nbytes % 2 == 0, "Arquivo .bin deve ter múltiplo de 2 bytes (uint16)."
        tokens = raw.view(np.uint16)
        np.save(npy_file, tokens.astype(np.int32))
        print(f"Convertido {bin_file} -> {npy_file} (tokens={tokens.size})")

    train_bin = os.path.join(data_dir, "TinyStories_train.bin")
    val_bin = os.path.join(data_dir, "TinyStories_val.bin")

    if not os.path.exists(train_bin):
        download_file(
            "https://huggingface.co/datasets/karpathy/tinystories/resolve/main/TinyStories_train.bin?download=true",
            train_bin
        )
    if not os.path.exists(val_bin):
        download_file(
            "https://huggingface.co/datasets/karpathy/tinystories/resolve/main/TinyStories_val.bin?download=true",
            val_bin
        )

    convert_bin_to_npy(train_bin, os.path.join(npy_dir, "tinystories_train_000000.npy"))
    convert_bin_to_npy(val_bin, os.path.join(npy_dir, "tinystories_val_000000.npy"))
    print("Preparação do dataset TinyStories concluída.")
