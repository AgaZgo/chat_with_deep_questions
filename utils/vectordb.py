from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import shutil


def build_vectordb(persist_directory, chunks):
    shutil.rmtree(persist_directory)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb