# data.py
# Carregador leve de tokens .npy com suporte a DDP, memmap, shuffle por época e checagens robustas.
# Rian Costa Ferreira.

import os
import numpy as np
import torch
from typing import List, Optional

def load_tokens(filename: str, mmap: bool = True) -> np.ndarray:
    """
    Carrega tokens de um arquivo .npy como ndarray (int32/int64).
    Usa memmap para reduzir RAM quando possível.
    """
    if mmap:
        npt = np.load(filename, mmap_mode='r')
    else:
        npt = np.load(filename)
    if npt.dtype not in (np.int32, np.int64):
        npt = npt.astype(np.int32, copy=False)
    return npt

class DataLoaderLite:
    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_root: str = "tinystories_npy",
        seed: int = 1337,
        mmap: bool = True,
        shuffle: bool = True,
    ):
        assert split in {"train", "val"}
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_root = data_root
        self.seed = seed
        self.shuffle = shuffle
        self.mmap = mmap

        shards = sorted(
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if split in s and s.endswith(".npy")
        )
        assert len(shards) > 0, f"Nenhum shard '{split}' em {data_root}"
        self.shards: List[str] = shards

        if self.process_rank == 0:
            print(f"Encontrados {len(shards)} shards para {split}")

        self._rng = np.random.RandomState(self.seed)
        self.epoch = 0
        self._order = np.arange(len(self.shards), dtype=np.int32)
        self._maybe_shuffle_order()
        self._load_shard(0)
        self.current_position = self.B * self.T * self.process_rank

    def _maybe_shuffle_order(self):
        if self.shuffle and self.split == "train":
            self._rng.seed(self.seed + self.epoch)
            self._rng.shuffle(self._order)

    def _load_shard(self, order_idx: int):
        self.current_shard_order_idx = order_idx % len(self.shards)
        shard_idx = int(self._order[self.current_shard_order_idx])
        self.current_shard_path = self.shards[shard_idx]
        self.tokens_np = load_tokens(self.current_shard_path, mmap=self.mmap)
        self.tokens_len = int(self.tokens_np.shape[0])

    def _advance_shard(self):
        self._load_shard(self.current_shard_order_idx + 1)
        self.current_position = self.B * self.T * self.process_rank

    def reset(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.epoch = int(epoch)
        self._maybe_shuffle_order()
        self._load_shard(0)
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        need = B * T + 1

        while self.current_position + need > self.tokens_len:
            self._advance_shard()

        buf_np = self.tokens_np[self.current_position : self.current_position + need]
        buf = torch.tensor(np.asarray(buf_np), dtype=torch.long)


        x = buf[:-1].reshape(B, T)
        y = buf[1:].reshape(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + need > self.tokens_len:
            self._advance_shard()

        return x, y
