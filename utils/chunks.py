import streamlit as st

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chunks(chunk_size=2000, chunk_overlap=50): 
    data_path = 'data/' 
    if st.session_state['episode']!='all':
        data_path += st.session_state['episode']
    loader = TextLoader(data_path) if data_path[-4:]=='.txt' else DirectoryLoader(data_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    
    return chunks
    