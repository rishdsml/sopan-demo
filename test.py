import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from llm_reponse import generate_reply
from prompt import RAG_BOT_PROMPT, GENERAL_QA_PROMPT
from web_search import search_web
import os
from dotenv import load_dotenv
load_dotenv()

# Load vectorstore + retriever
vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY")),
    retriever=retriever,
    return_source_documents=True
)

# Page setup
st.set_page_config(page_title="NCR Chatbot", layout="centered")
st.markdown("### FieldBot")
st.markdown("Ask me anything about reports, root causes, or field issues.")
st.info("For detailed field analysis, prefix your query with `analyze:`")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []

# Display message history
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask your NCR question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(" Agent is thinking..."):
            try:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))

                # Grab recent chat history (last 2 rounds: user + assistant)
                history = [
                    f"{m['role'].capitalize()}: {m['content']}"
                    for m in st.session_state.messages[-4:-1]
                ]
                history_text = "\n".join(history)

                # CASE 1: Structured analysis mode
                if user_input.lower().startswith("analyse:"):
                    query = user_input.replace("analyse:", "").strip()
                    memory_context = "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])
                    web_snips = search_web(query)
                    web_context = "\n".join(web_snips)

                    formatted_prompt = RAG_BOT_PROMPT.format(
                        query=query,
                        faiss_docs=memory_context,
                        web_snippets=web_context
                    )
                    final_answer = llm.invoke(formatted_prompt).content

                # CASE 2: Normal query with memory + web fallback
                else:
                    response = qa_chain(user_input)
                    answer = response["result"].strip()
                    retrieved_docs = response.get("source_documents", [])

                    if answer.lower() in ["", "n/a", "not_found"]:
                        memory_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                        web_snips = search_web(user_input)
                        web_context = "\n".join(web_snips)

                        formatted_prompt = GENERAL_QA_PROMPT.format(
                            history=history_text,
                            memory=memory_context,
                            web=web_context,
                            question=user_input
                        )
                        final_answer = llm.invoke(formatted_prompt).content
                    else:
                        final_answer = answer

            except Exception as e:
                final_answer = f"Error processing your request: {str(e)}. Please try again."

        # Save + show assistant reply
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.markdown(final_answer)
