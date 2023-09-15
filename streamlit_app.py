from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler

import streamlit as st
import os 
import shutil


st.set_page_config(page_title="DeepQuestionsChat", page_icon=":question:", layout="wide")
st.title('Chat with Deep Questions podcast')

with st.sidebar:
    gpt_model = st.selectbox('GPT model', ['gpt-3.5-turbo', 'gpt-4'], index=0)
    openai_api_key = st.text_input("Your OpenAI API key")

    if openai_api_key:    
        os.environ['OPENAI_API_KEY'] = openai_api_key
    else:
        st.stop()

    
def load_transcripts(data_path, chunk_size, chunk_overlap):
    loader = DirectoryLoader(data_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    return chunks

def build_vectordb(persist_directory, chunks, embedding):
    shutil.rmtree(persist_directory)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

def build_prompt():
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    return prompt_template

def initialize_chain(llm, vectordb, prompt_template):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={'k':3}),
        chain_type="stuff",
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt_template,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question")
        }
    )
    return qa_chain

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