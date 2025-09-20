# dockerize.py
# Uso em notebook:
#   from dockerize import write_cpu_image, write_gpu_image, build_image, run_container, verify_local
#   write_cpu_image()
#   build_image("nanogpt-proof", dockerfile="Dockerfile.cpu")
#   run_container("nanogpt-proof", artifacts_dir="./artifacts", ckpt_name="model.pt", gpus=False)
#
# Sem Docker (verificação local):
#   from dockerize import verify_local
#   print(verify_local("./artifacts/model.pt"))

import os
import sys
import json
import shutil
import textwrap
import subprocess
import hashlib
from pathlib import Path
from typing import Optional, Dict, List

HERE = Path(__file__).resolve().parent

CPU_DOCKERFILE = "Dockerfile.cpu"
GPU_DOCKERFILE = "Dockerfile.gpu"
DOCKERIGNORE = ".dockerignore"
VERIFY_MIN = "verify_ckpt.py"
VERIFY_TORCH = "torch_verify.py"

def _check_docker():
    if shutil.which("docker") is None:
        raise RuntimeError("Docker não encontrado no PATH; instale e reinicie o shell antes de prosseguir.")

def write_cpu_image():
    """
    Gera Dockerfile mínimo (CPU) para provar checkpoint (.pt/.pth) sem PyTorch na imagem.
    """
    df = textwrap.dedent("""\
        # syntax=docker/dockerfile:1
        FROM python:3.12-slim

        WORKDIR /app
        COPY verify_ckpt.py /app/verify_ckpt.py

        # utilitário leve para sha256
        RUN apt-get update && apt-get install -y --no-install-recommends openssl \\
            && rm -rf /var/lib/apt/lists/*

        ENV CKPT_PATH=/artifacts/model.pt

        ENTRYPOINT ["python", "-u", "verify_ckpt.py"]
    """)
    (HERE / CPU_DOCKERFILE).write_text(df, encoding="utf-8")

    verify = textwrap.dedent("""\
        import os, sys, subprocess, json

        ckpt = os.environ.get("CKPT_PATH", "/artifacts/model.pt")
        out = {"ckpt_path": ckpt, "exists": os.path.exists(ckpt)}
        if out["exists"]:
            try:
                sha = subprocess.check_output(
                    ["sh", "-lc", f"openssl dgst -sha256 {ckpt} | awk '{'{'}print $2{'}'}'"]
                ).decode().strip()
                out["sha256"] = sha
                out["size_bytes"] = os.path.getsize(ckpt)
            except Exception as e:
                out["error"] = f"sha256_failed: {e}"
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0 if out["exists"] else 1)
    """)
    (HERE / VERIFY_MIN).write_text(verify, encoding="utf-8")

    ignore = textwrap.dedent("""\
        __pycache__/
        *.pt
        *.pth
        *.bin
        data/
        datasets/
        tinystories_npy/
        .env
        .git
        *.ipynb_checkpoints
    """)
    (HERE / DOCKERIGNORE).write_text(ignore, encoding="utf-8")
    print(f"[ok] Arquivos CPU gerados: {CPU_DOCKERFILE}, {VERIFY_MIN}, {DOCKERIGNORE}")

def write_gpu_image():
    """
    Gera Dockerfile (GPU/NGC PyTorch) e script torch_verify.py para torch.load opcional.
    Requer NVIDIA Container Toolkit no host para --gpus all.
    """
    df = textwrap.dedent("""\
        # syntax=docker/dockerfile:1
        FROM nvcr.io/nvidia/pytorch:24.06-py3

        WORKDIR /app
        COPY torch_verify.py /app/torch_verify.py

        ENV CKPT_PATH=/artifacts/model.pt
        ENV TORCH_DEVICE=cuda

        # Executar com: docker run --rm --gpus all -v $PWD/artifacts:/artifacts <image>
        ENTRYPOINT ["python", "-u", "torch_verify.py"]
    """)
    (HERE / GPU_DOCKERFILE).write_text(df, encoding="utf-8")

    torch_verify = textwrap.dedent("""\
        import os, json, torch

        ckpt = os.environ.get("CKPT_PATH", "/artifacts/model.pt")
        want = os.environ.get("TORCH_DEVICE", "cuda")
        has_cuda = torch.cuda.is_available()
        dev = "cuda" if (want == "cuda" and has_cuda) else "cpu"
        out = {"ckpt_path": ckpt, "exists": os.path.exists(ckpt), "cuda_available": has_cuda, "load_device": dev}
        if out["exists"]:
            try:
                obj = torch.load(ckpt, map_location=dev)
                out["loaded_type"] = type(obj).__name__
                if isinstance(obj, dict):
                    out["keys"] = list(obj.keys())[:12]
                    if "model" in obj and hasattr(obj["model"], "keys"):
                        out["model_state_dict_keys"] = list(obj["model"].keys())[:12]
            except Exception as e:
                out["error"] = str(e)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    """)
    (HERE / VERIFY_TORCH).write_text(torch_verify, encoding="utf-8")

    # .dockerignore igual ao CPU
    if not (HERE / DOCKERIGNORE).exists():
        ignore = textwrap.dedent("""\
            __pycache__/
            *.pt
            *.pth
            *.bin
            data/
            datasets/
            tinystories_npy/
            .env
            .git
            *.ipynb_checkpoints
        """)
        (HERE / DOCKERIGNORE).write_text(ignore, encoding="utf-8")
    print(f"[ok] Arquivos GPU gerados: {GPU_DOCKERFILE}, {VERIFY_TORCH}")

def build_image(image_name: str, dockerfile: str):
    """
    Constrói a imagem Docker a partir do Dockerfile indicado.
    """
    _check_docker()
    cmd = ["docker", "build", "-t", image_name, "-f", dockerfile, "."]
    print("[build]", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[ok] Imagem '{image_name}' construída.")

def run_container(image_name: str, artifacts_dir: str, ckpt_name: str = "model.pt",
                  gpus: bool = False, extra_env: Optional[Dict[str, str]] = None,
                  extra_args: Optional[List[str]] = None):
    """
    Executa o contêiner montando o diretório de artefatos em /artifacts e definindo CKPT_PATH.
    """
    _check_docker()
    artifacts = Path(artifacts_dir).resolve()
    assert artifacts.exists(), f"Diretório de artefatos não encontrado: {artifacts}"
    env = {"CKPT_PATH": f"/artifacts/{ckpt_name}"}
    if extra_env:
        env.update(extra_env)
    args = ["docker", "run", "--rm", "-v", f"{artifacts}:/artifacts:ro"]
    if gpus:
        args += ["--gpus", "all"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    if extra_args:
        args += extra_args
    args += [image_name]
    print("[run]", " ".join(args))
    subprocess.check_call(args)

def verify_local(path: str) -> str:
    """
    Verifica localmente (sem Docker) existência, sha256 e tamanho do checkpoint.
    Retorna uma string JSON.
    """
    p = Path(path)
    out = {"ckpt_path": str(p), "exists": p.exists()}
    if p.exists():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
        out["size_bytes"] = p.stat().st_size
    return json.dumps(out, ensure_ascii=False, indent=2)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cpu", "gpu"], default="cpu", help="cpu=minimal (sem torch); gpu=NGC PyTorch")
    p.add_argument("--image", required=False, default="nanogpt-proof", help="nome da imagem, ex.: nanogpt-proof")
    p.add_argument("--artifacts", default="./artifacts", help="pasta com o checkpoint a montar")
    p.add_argument("--ckpt", default="model.pt", help="nome do arquivo do checkpoint dentro de artifacts/")
    p.add_argument("--gpus", default="", help="use 'all' para expor GPUs (requer NVIDIA Container Toolkit)")
    p.add_argument("--build-only", action="store_true", help="apenas construir a imagem (não executar)")
    p.add_argument("--no-docker", action="store_true", help="pular Docker e apenas verificar localmente o checkpoint")
    args = p.parse_args()

    if args.no_docker:
        print(verify_local(os.path.join(args.artifacts, args.ckpt)))
        return

    if args.mode == "cpu":
        write_cpu_image()
        df = CPU_DOCKERFILE
    else:
        write_gpu_image()
        df = GPU_DOCKERFILE

    build_image(args.image, df)
    if not args.build_only:
        run_container(args.image, args.artifacts, ckpt_name=args.ckpt, gpus=(args.gpus == "all"))

if __name__ == "__main__":
    main()
