import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA


def initialize_chain(prompt_template):
    llm = ChatOpenAI(
        model_name=st.session_state['gpt_model'], 
        temperature=0, 
        openai_api_key=st.session_state['OPENAI_API_KEY'])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=st.session_state['vectordb'].as_retriever(search_kwargs={'k':5}),
        chain_type="stuff",
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={
            "verbose": False,
            "prompt": prompt_template,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question")
        }
    )
    return qa_chain