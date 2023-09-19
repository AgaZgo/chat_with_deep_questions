from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import shutil
import os


def build_vectordb(persist_directory, chunks):
    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)
    else:
        shutil.rmtree(persist_directory)
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb