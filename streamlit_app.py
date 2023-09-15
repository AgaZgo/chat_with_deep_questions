from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

import streamlit as st
import os 

from utils import load_transcripts, build_vectordb, build_prompt, initialize_chain


st.set_page_config(page_title="DeepQuestionsChat", page_icon=":question:", layout="wide")
st.title('Chat with Deep Questions podcast')

with st.sidebar:
    gpt_model = st.selectbox('GPT model', ['gpt-3.5-turbo', 'gpt-4'], index=0)
    openai_api_key = st.text_input("Your OpenAI API key")

    if openai_api_key:    
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.stop()

if 'chunks' not in st.session_state:
    st.session_state['chunks'] = load_transcripts('transcripts/', 1000, 50)
    
embedding = OpenAIEmbeddings()

if 'vectordb' not in st.session_state:
    st.session_state['vectordb'] = build_vectordb('chroma/', st.session_state['chunks'], embedding)
    
prompt_template = build_prompt()
llm = ChatOpenAI(model_name=gpt_model, temperature=0)

if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = initialize_chain(llm, st.session_state['vectordb'], prompt_template)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if question := st.chat_input():
    st.chat_message("user").write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message("assistant"):
        response = st.session_state['qa_chain']({"query": question})
        st.session_state.messages.append({'role': 'assistant', 'content': response['result']})
        st.write(response['result'])