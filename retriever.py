# retriever.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)

def retrieve(query, k=10):
    return vectorstore.similarity_search(query, k=k)


