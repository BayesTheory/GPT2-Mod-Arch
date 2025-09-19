# nanoGPT-moderno: Uma Versão Avançada do GPT-2

Uma modernização extensiva do nanoGPT (Karpathy), atualizada com práticas e otimizações de LLMs padrão 2024, mantendo o código claro e hackável. O foco é a arquitetura e a compatibilidade com técnicas modernas de inferência. Treinamento e benchmarks não foram realizados por serem computacionalmente pesados.

### Principais melhorias

| Componente            | GPT-2 Clássico                | Versão Moderna (este projeto)      |
| :-------------------- | :---------------------------- | :--------------------------------- |
| Posicionamento        | Embeddings absolutos (wpe)    | Embeddings rotatórios (RoPE)       |
| Normalização          | LayerNorm                      | RMSNorm                            |
| Atenção (mecanismo)   | MHA                           | Grouped-Query Attention (GQA)      |
| Atenção (kernel)      | Implementação manual           | SDPA com FlashAttention (quando disponível) |
| MLP (ativação)        | GELU                          | SwiGLU                             |
| Inferência            | Recomputação O(T²)            | KV-Cache incremental O(T)          |
| Camadas lineares      | Com bias                      | Sem bias (mais estável com RMSNorm) |
| Saída                 | Logits brutos                 | Logit soft-capping                 |
| Inicialização         | Do zero/compatível GPT-2      | Transferência parcial de pesos GPT-2 |
| Estrutura de código   | Monolítica                    | Modular, clara e extensível        |

### Estado do projeto
- Foco em arquitetura moderna e utilitários de inferência; sem resultados de treino ou ablações empíricas.
- Suporte a RoPE, GQA, SwiGLU, KV-cache e SDPA/FlashAttention via PyTorch 2.x.
- Transferência parcial de pesos a partir de checkpoints GPT-2 para facilitar experimentos de compatibilidade.
- Código sem vieses de treino: camadas lineares sem bias, normalização via RMSNorm e soft-capping de logits para estabilidade.

### Arquivos principais
- model.py: definição do Transformer moderno (RoPE, GQA, SwiGLU, RMSNorm, KV-cache).
- generate.py: utilitário de geração com KV-cache e SDPA.
- data.py: DataLoader leve para shards tokenizados (.npy).
- configs/: parâmetros de arquitetura e inicialização (inclui presets compatíveis com GPT-2).

### Requisitos
- Python 3.9+
- PyTorch 2.x (com suporte a SDPA/FlashAttention quando disponível)
- numpy, tiktoken, transformers

### Documento técnico (Overleaf)
Manuscrito em preparação no Overleaf detalhando decisões de projeto, compatibilidade com GPT-2, e análise teórica das escolhas (RoPE, GQA, SwiGLU, RMSNorm).
*(ttps://www.overleaf.com/read/prvwjxcjfxfn#1ebc57*

### Licença
MIT
