# data.py
# Contém o DataLoaderLite, responsável por carregar os dados pré-tokenizados
# de forma eficiente para o treinamento, com suporte a DDP.

import os
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Carregador de Dados

def load_tokens(filename):
    """ Carrega tokens de um arquivo .npy e converte para um tensor PyTorch. """
    npt = np.load(filename)
    npt = npt.astype(np.int32) # Garante o tipo de dado correto
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root="edu_fineweb10B"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Encontra os arquivos shard para o split (train/val)
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"Nenhum shard encontrado para o split '{split}' no diretório '{data_root}'"
        
        # Imprime o número de shards encontrados apenas no processo mestre
        if self.process_rank == 0:
            print(f"Encontrados {len(shards)} shards para o split {split}")
        
        self.reset()

    def reset(self):
        # Reseta o estado, começando do primeiro shard
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # A posição inicial é deslocada para cada processo, para que eles leiam partes diferentes dos dados
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # Carrega um buffer de tokens um pouco maior (B*T + 1) para garantir que temos x e y
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        
        # Avança a posição no tensor para a próxima leitura
        self.current_position += B * T * self.num_processes
        
        # Se o próximo batch estourar os limites do shard atual, avança para o próximo shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            # Carrega os tokens do novo shard
            self.tokens = load_tokens(self.shards[self.current_shard])
            # Reseta a posição inicial para o rank do processo atual
            self.current_position = self.B * self.T * self.process_rank
            
        return x, y