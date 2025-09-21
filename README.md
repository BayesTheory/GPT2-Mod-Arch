# nanoGPT-moderno

[![Licença: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)

Um fork do `nanoGPT` de Karpathy, atualizado com a arquitetura e as melhores práticas de LLMs de 2024–2025, como RoPE, GQA, SwiGLU e Flash Attention.

Este projeto moderniza o `nanoGPT` para ser uma base de código clara e flexível, focada em arquitetura robusta e inferência otimizada. É a ferramenta ideal para quem quer aprender, experimentar e entender a ponte entre o GPT-2 clássico e arquiteturas modernas como Llama 3.

---

## 🎯 Para Quem é Este Projeto?

*   **Estudantes e Pesquisadores:** Uma base de código limpa para entender na prática os componentes de Transformers modernos.
*   **Desenvolvedores:** Um ponto de partida sólido e minimalista para prototipar e experimentar com novas arquiteturas.
*   **Entusiastas:** Para qualquer pessoa que queira treinar seu próprio "GPT" do zero com tecnologia de ponta.

## ✨ Destaques da Arquitetura

Esta versão implementa otimizações cruciais que se tornaram padrão em modelos de linguagem de alta performance.

| Componente                 | Versão Clássica (GPT-2)     | Versão Moderna (nanoGPT-moderno)        |
| :------------------------- | :-------------------------- | :-------------------------------------- |
| **Embeddings Posicionais** | Absolutos (`wpe`)           | **RoPE** (Rotational Position Embeddings) |
| **Normalização**           | `LayerNorm`                 | **RMSNorm** (Root Mean Square Norm)     |
| **Atenção (Mecanismo)**    | MHA (Multi-Head Attention)  | **GQA** (Grouped-Query Attention)       |
| **Atenção (Kernel)**       | Implementação manual        | **SDPA** (Flash/Efficient Attention)    |
| **Ativação (MLP)**         | `GELU`                      | **SwiGLU**                              |
| **Bias em Camadas Densas** | Com bias                    | **Sem bias** para maior estabilidade    |
| **Inferência**             | Recomputação completa       | **KV-cache** incremental                |
| **Saída**                  | Logits brutos               | **Logit soft-capping**                  |
| **Inicialização**          | Padrão GPT-2                | Otimizada para arquitetura moderna      |
| **Estrutura de Código**    | Monolítica                  | Modular e extensível                    |

## 🚀 Comece a Usar em Minutos

###. Modelo
*  `https://huggingface.co/rianagario/GPT2-Mod-Arch`

###. Requisitos

*   Python 3.9+
*   PyTorch 2.x (com suporte a CUDA para melhor performance)
*   Dependências adicionais: `numpy`, `tiktoken`, `transformers`, `tqdm`, `mlflow` (opcional).

## Relatório Técnico

*  `https://www.overleaf.com/read/prvwjxcjfxfn#1ebc57`

## Resumo Resultado

Treinado do zero em um dataset de 2 GB, o modelo validou a eficácia da arquitetura modernizada ao aprender a gerar sentenças gramaticalmente corretas. No entanto, como esperado para o volume de dados, o treinamento levou a um overfitting, fazendo com que o modelo se especializasse excessivamente no conteúdo do corpus. Isso se manifestou em uma forte tendência a associar conceitos a personagens específicos, tornando-o um "gerador de histórias" com viés narrativo, em vez de um modelo de linguagem generalista. O resultado sublinha que a performance da arquitetura é diretamente limitada pela escala e diversidade do dataset de treinamento.

## Teste Técnico

![Teste](https://raw.githubusercontent.com/BayesTheory/GPT2-Mod-Arch/main/img/print.jpeg)

![Teste](https://raw.githubusercontent.com/BayesTheory/GPT2-Mod-Arch/main/img/print%20(2).jpeg)

