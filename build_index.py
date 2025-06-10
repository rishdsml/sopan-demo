# build_index.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

df = pd.read_csv("sopan_demo_final1.csv")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("N/A")

def row_to_document(row):
    content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    return Document(page_content=content)

documents = [row_to_document(row) for _, row in df.iterrows()]
chunk_size = 100
chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]

vectorstore = None
for i, chunk in enumerate(chunks):
    print(f"Embedding batch {i + 1}/{len(chunks)}")
    vs = FAISS.from_documents(chunk, embeddings)
    if vectorstore is None:
        vectorstore = vs
    else:
        vectorstore.merge_from(vs)

vectorstore.save_local("faiss_store")
print(" FAISS store built and saved.")
print("Total documents indexed:", len(documents))
