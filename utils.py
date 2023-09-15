from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chains import RetrievalQA

import shutil


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