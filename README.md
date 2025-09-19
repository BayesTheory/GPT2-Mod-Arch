# nanoGPT-moderno: Uma Versão Avançada do nanoGPT

> Uma modernização extensiva do aclamado [nanoGPT](https://github.com/karpathy/nanoGPT) de Andrej Karpathy, atualizado com a arquitetura e otimizações de LLMs de última geração (padrão 2024).

[![Licença: MIT](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Status do Projeto](https://img.shields.io/badge/status-desenvolvimento-orange)]()

Este projeto pega a base de código simples e educacional do nanoGPT, que implementa a arquitetura original do GPT-2, e a eleva com as melhorias que definem modelos como Llama, Gemma e Mistral. O resultado é uma base de código que permanece clara, mas agora é capaz de treinar modelos significativamente mais rápidos, eficientes e com maior capacidade de contexto.

---

## Principais Modernizações Implementadas

A tabela abaixo resume a transição da arquitetura clássica do GPT-2 para a nossa implementação moderna.

| Componente | Arquitetura Original (GPT-2) | Arquitetura Modernizada (Este Projeto) |
| :--- | :--- | :--- |
| **Posicionamento** | Embeddings Absolutos (`wpe`) | **Embeddings Rotatórios (RoPE)** |
| **Normalização** | `LayerNorm` | **RMSNorm** |
| **Atenção (Mecanismo)** | Multi-Head Attention (MHA) | **Grouped-Query Attention (GQA)** |
| **Atenção (Kernel)** | Implementação Manual | **SDPA (Otimizado com FlashAttention)** |
| **MLP (Ativação)** | `GELU` | **`SwiGLU`** |
| **Inferência** | Recálculo Completo (O(T²)) | **KV-Cache Incremental (O(T))** |
| **Camadas Lineares** | Com `bias` | **Sem `bias`** (mais estável com RMSNorm) |
| **Saída** | Logits brutos | **Logit Soft-Capping** |
| **Treinamento** | Script monolítico | **Estrutura modular e robusta (DDP)** |
| **Inicialização** | Do zero ou completo | **Transferência parcial de pesos do GPT-2** |

---

## Diagramas da Arquitetura

### Diagrama do Sistema Completo
Este diagrama mostra o fluxo de trabalho do projeto, desde a configuração até o treinamento e a geração de texto.

![Diagrama do Sistema](img/Untitled%20diagram%20_%20Mermaid%20Chart-2025-09-19-023050.png)

### Diagrama da Arquitetura do Modelo
Este diagrama oferece um "raio-x" da arquitetura interna do modelo, detalhando o fluxo de dados através dos blocos Transformer modernizados.

![Diagrama do Modelo](img/Untitled%20diagram%20_%20Mermaid%20Chart-2025-09-19-024130.png)

---

## Estrutura do Projeto

O projeto foi organizado de forma modular para clareza e manutenção:

```
nanoGPT-moderno/
├── model.py            # Definição da arquitetura do modelo GPT moderno.
├── train.py            # Script principal para orquestrar o treinamento.
├── generate.py         # Script para gerar texto usando um modelo treinado.
├── data.py             # Carregador de dados eficiente (DataLoaderLite).
├── configs/
│   ├── gpt2_from_scratch.py  # Config para treinar do zero.
│   └── gpt2_finetune.py      # Config para fine-tuning a partir do GPT-2.
└── README.md           # Este arquivo.
```

---

## Guia de Início Rápido

### 1. Preparação do Ambiente
**Clone o Repositório:**
```bash
git clone https://github.com/BayesTheory/GPT2-Mod-Arch.git
cd GPT2-Mod-Arch
```

**Crie um arquivo `requirements.txt`** com o seguinte conteúdo:
```txt
torch>=2.0
transformers
tiktoken
numpy
```

**Crie um Ambiente Virtual e Instale as Dependências:**
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preparação dos Dados
Este projeto espera dados pré-tokenizados no formato de arquivos `.npy`. Coloque seus shards em um diretório (ex: `edu_fineweb10B/`).
```
edu_fineweb10B/
├── fineweb_edu_train_000000.npy
└── fineweb_edu_val_000000.npy
```

---

## Como Usar

### 1. Treinar um Modelo do Zero
Este comando iniciará um treinamento usando a arquitetura moderna.
```bash
# Treino em uma única GPU
python train.py configs/gpt2_from_scratch.py

# Treinamento distribuído em 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py configs/gpt2_from_scratch.py
```

### 2. Fine-tuning a partir do GPT-2
Carrega os pesos compatíveis do GPT-2 e inicia um fine-tuning.
```bash
python train.py configs/gpt2_finetune.py
```

### 3. Retomar um Treinamento
Modifique a linha `init_from = 'resume'` no seu arquivo de configuração. O script procurará automaticamente pelo checkpoint `latest.pt` no diretório de log.

### 4. Gerar Texto
Use o script `generate.py` para ver seu modelo em ação.
```bash
# Exemplo usando o último checkpoint salvo de um treino
python generate.py log_scratch/latest.pt "The future of artificial intelligence is"
```

---

## Agradecimentos
Este projeto é uma evolução e homenagem ao [nanoGPT](https://github.com/karpathy/nanoGPT) de Andrej Karpathy, cujo trabalho tornou a educação em IA profunda acessível a muitos.

## Licença
Este projeto é licenciado sob a Licença MIT.
