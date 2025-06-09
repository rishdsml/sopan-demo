from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

# Load FAISS
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)

# Test query
query = "What is the NCR category for formation damage during BHA operations?"
results = vectorstore.similarity_search(query, k=5)

# Display
print(" [FAISS Search Results]")
for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---\n{doc.page_content}")
