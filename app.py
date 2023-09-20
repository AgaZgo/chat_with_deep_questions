import streamlit as st

from utils.sidebar import sidebar
from utils.chunks import get_chunks
from utils.vectordb import build_vectordb
from utils.prompt import build_prompt
from utils.chains import initialize_chain


st.set_page_config(page_title="DeepQuestionsChat", page_icon=":question:", layout="wide")
st.title('Chat with Deep Questions podcast')
sidebar()

openai_api_key = st.session_state.get("OPENAI_API_KEY")


if "OPENAI_API_KEY" not in st.session_state:
    st.warning(
        "Enter your OpenAI API key in the sidebar to start a chat."
    )
    st.stop()

    
episode = st.session_state['episode']
gpt_model = st.session_state['gpt_model']

if 'chunks' not in st.session_state:
    st.session_state['chunks'] = get_chunks()

if 'vectordb' not in st.session_state:
    st.session_state['vectordb'] = build_vectordb('chroma/', st.session_state['chunks'])
    
prompt_template = build_prompt()

if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = initialize_chain(prompt_template)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", 
         "content": "Hi, I'm a Deep Questions chatbot. I will answer your questions about the podcast. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if question := st.chat_input():
    st.chat_message("user").write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message("assistant"):
        response = st.session_state['qa_chain']({"query": question})
        st.session_state.messages.append({'role': 'assistant', 'content': response['result']})
        st.write(response['result'])