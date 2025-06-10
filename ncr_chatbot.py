import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


def search_web(query): return [f"Web snippet for {query}"]
RAG_BOT_PROMPT = "Query: {query}\nFAISS Docs: {faiss_docs}\nWeb Snippets: {web_snippets}"
GENERAL_QA_PROMPT = "History: {history}\nMemory: {memory}\nWeb: {web}\nQuestion: {question}"

if os.path.exists("faiss_store"):
    vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY")),
        retriever=retriever,
        return_source_documents=True
    )
else:
    
    retriever = None
    qa_chain = None


st.set_page_config(page_title="FieldBot", layout="centered")

# Create columns to position the button in the top-right
col1, col2 = st.columns([4, 1])
with col1:
    st.write("Ask me anything about reports, root causes, or field issues.")
    st.info("For detailed field analysis, prefix your query with `analyze:`")
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.write(msg["content"])

# The button logic has been removed from the bottom and placed at the top.
user_input = st.chat_input("Ask your NCR question here...")


if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            
            if not qa_chain:
                final_answer = "Error: FAISS vector store not found. Please ensure 'faiss_store' directory is present."
            else:
                try:
                    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))

                    history = [
                        f"{m['role'].capitalize()}: {m['content']}"
                        for m in st.session_state.messages[-4:-1]
                    ]
                    history_text = "\n".join(history)

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

                    else:
                        response = qa_chain(user_input)
                        answer = response["result"].strip()
                        retrieved_docs = response.get("source_documents", [])

                        if not answer or answer.lower() in ["n/a", "not_found"]:
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

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.write(final_answer)


