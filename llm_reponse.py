import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from prompt import RAG_BOT_PROMPT
from web_search import search_web
from retriever import retrieve
from langchain.schema import HumanMessage
import time

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def generate_reply(query: str) -> str:
    print("\n [RAG BOT] Query received:")
    print(query.strip())

    start = time.time()

    # Step 1: FAISS Retrieval
    faiss_docs = retrieve(query, k=5)
    faiss_text = "\n\n".join([doc.page_content for doc in faiss_docs])

    print(f"\n [Memory] Retrieved {len(faiss_docs)} internal documents.")
    
    # Step 2: Decide if fallback needed
    if not faiss_docs or faiss_text.count("N/A") > 15:
        print(" [Fallback] Memory weak. Triggering web search...")
        web_snippets = search_web(query)
    else:
        print(" [Memory] Sufficient context found. Skipping web search.")
        web_snippets = []

    web_text = "\n".join(web_snippets)

    # Step 3: Prepare prompt
    final_prompt = RAG_BOT_PROMPT.format(
        query=query,
        faiss_docs=faiss_text,
        web_snippets=web_text
    )

    print("\n [LLM] Sending prompt to GPT-4o...")
    response = llm.invoke([HumanMessage(content=final_prompt)])

    end = time.time()
    print(f"\n⏱ [Done] Response generated in {round(end - start, 2)} seconds.")

    return response.content
