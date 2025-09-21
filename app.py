# app_chat.py

import streamlit as st
import torch
import tiktoken
import time
import gc

# Importe suas classes de modelo e config
from model import GPT, GPTConfig

# --- CONFIGURAÇÕES DA PÁGINA E DO MODELO ---
# st.set_page_config define o título que aparece na aba do navegador, o ícone e o layout
st.set_page_config(
    page_title="Meu GPT Pessoal",
    page_icon="🧠",
    layout="centered"
)

# CORREÇÃO: Adicionado 'r' antes da string para tratar o caminho como "raw" (cru)
CHECKPOINT_PATH = r"C:\Users\Rian\Desktop\gpt\mlruns\706298003787278334\2ab5a67b5c35444eb888c19ca8f8be80\artifacts\model_00499.pt"
DEVICE = 'cpu'

# --- LÓGICA DO MODELO (A mesma função com cache, pois é essencial) ---

@st.cache_resource
def load_model_and_tokenizer():
    """
    Carrega o modelo e o tokenizador uma única vez.
    """
    with st.spinner("Inicializando o cérebro digital... 🧠 Por favor, aguarde."):
        # Carregar o modelo na CPU
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        model.eval()
        model.to(DEVICE)
        
        del checkpoint, state_dict
        gc.collect()
        
        # Carregar tokenizador
        tokenizer = tiktoken.get_encoding("gpt2")
        
    return model, tokenizer

# --- FUNÇÃO DE GERAÇÃO COM EFEITO DE STREAMING ---

def generate_response_stream(model, tokenizer, prompt_text, max_new_tokens, temperature):
    """
    Gera a resposta completa e depois a "transmite" palavra por palavra.
    """
    start_ids = tokenizer.encode(prompt_text, allowed_special={"<|endoftext|>"})
    x = (torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...])
    
    # Gerar a resposta completa (ainda é o passo lento)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
        full_response_tokens = y[0].tolist()
    
    # Decodifica apenas a parte nova da resposta
    new_response_tokens = full_response_tokens[len(start_ids):]
    response_text = tokenizer.decode(new_response_tokens)
    
    # Simula o streaming palavra por palavra
    for word in response_text.split():
        yield word + " "
        time.sleep(0.05) # Pequeno atraso para o efeito de digitação

# --- INTERFACE GRÁFICA (O FRONTEND BONITINHO) ---

# Título principal
st.title("Meu GPT Pessoal 🧠")
st.caption("Uma interface de chat elegante para o seu modelo de linguagem.")

# Carrega o modelo e o tokenizador (pode mostrar o spinner na primeira vez)
try:
    model, tokenizer = load_model_and_tokenizer()

    # Configurações na barra lateral
    with st.sidebar:
        st.header("🛠️ Configurações")
        max_new_tokens = st.slider("Máximo de novos tokens:", 50, 1000, 150)
        temperature = st.slider("Temperatura (Criatividade):", 0.1, 1.0, 0.7, 0.05)
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.messages = []
            st.rerun()

    # Inicializa o histórico de chat na "memória" da sessão do Streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe as mensagens do histórico a cada atualização da tela
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Campo de input do usuário (fica fixo no final da página)
    if prompt := st.chat_input("Como posso te ajudar?"):
        # Adiciona a mensagem do usuário ao histórico e exibe na tela
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gera e exibe a resposta do assistente
        with st.chat_message("assistant"):
            # O spinner agora fica dentro do balão de chat do assistente
            with st.spinner("Pensando..."):
                response_generator = generate_response_stream(model, tokenizer, prompt, max_new_tokens, temperature)
                # st.write_stream simula o efeito de digitação
                full_response = st.write_stream(response_generator)
        
        # Adiciona a resposta completa do assistente ao histórico
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Um pequeno efeito de comemoração :)
        st.balloons()

except FileNotFoundError:
    st.error(f"ERRO: Arquivo do modelo não encontrado em '{CHECKPOINT_PATH}'.")
    st.info("Verifique o caminho da variável `CHECKPOINT_PATH` no arquivo `app_chat.py`.")
except Exception as e:
    st.error(f"Ocorreu um erro inesperado: {e}")
    st.exception(e) # Mostra mais detalhes do erro para debug
