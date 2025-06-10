import streamlit as st
import pandas as pd
import re
import os
import json
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent, tool

# --- Import from our custom files ---
from prompt import AGENT_SYSTEM_PROMPT, PANDAS_PARSER_PROMPT, ANALYST_PROMPT_TEMPLATE
from web_search import web_search_tool

# --- SET PAGE CONFIG AS THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="FieldBot", layout="centered")

load_dotenv()

# --- 1. Load Data and Models ---

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sopan_demo_final1.csv')
        df.columns = df.columns.str.strip()
        for col in ['NCR ID', 'Well_Id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return df
    except FileNotFoundError:
        st.error("Fatal Error: 'sopan_demo_final1.csv' not found.")
        return None

@st.cache_resource
def load_faiss_retriever():
    if os.path.exists("faiss_store"):
        try:
            embeddings = OpenAIEmbeddings()
            return FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 20})
        except Exception as e:
            st.error(f"Error loading FAISS store: {e}")
    return None

df = load_data()
faiss_retriever = load_faiss_retriever()
llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY"))

# --- 2. Define the Tools for the Agent ---

@tool
def ncr_database_tool(query: str) -> str:
    """
    <--- THIS DOCSTRING IS REQUIRED --->
    Use this tool for any questions that require looking up specific data, counting items, or listing items from the internal SOPAN NCR database.
    This is the best tool for any query that mentions a specific column name, an ID number (like NCR ID or Well ID), or asks for a count or a list of items.
    """
    print(f"--- Calling Intelligent Database Tool with query: {query} ---")
    if df is None: return "Error: DataFrame not loaded."
    parser_prompt = PANDAS_PARSER_PROMPT.format(df_columns=df.columns.tolist(), query=query)
    try:
        response_str = llm.invoke(parser_prompt).content
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not json_match: return "Error: The AI parser did not return a valid command."
        command = json.loads(json_match.group(0))
        operation, column, value = command.get("operation"), command.get("column"), command.get("value")
        if not column or column not in df.columns: return f"Error: Invalid column '{column}' identified."
        if operation == "count_unique":
            return f"Count of unique values for {column}: {df[column].nunique()}"
        elif operation == "lookup":
            if value is None: return "Error: A value is required for a lookup."
            results = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
            if results.empty: return f"No records found where '{column}' contains '{value}'."
            return f"Found {len(results)} records. Data for best match:\n{results.iloc[0].to_string()}"
        elif operation == "list_unique":
            return f"Unique values for '{column}':\n{df[column].dropna().unique().tolist()[:50]}"
        else: return f"Unsupported operation '{operation}'."
    except Exception as e: return f"A technical error occurred: {e}"

@tool
def semantic_search_tool(query: str) -> str:
    """
    <--- THIS DOCSTRING IS REQUIRED --->
    Use this tool for open-ended, conceptual, or semantic questions about root causes, resolutions, or summaries based on the internal SOPAN knowledge base of past NCR reports.
    Do NOT use this tool for counting, specific data lookups, or questions about public information.
    """
    print(f"--- Calling Semantic Search Tool with query: {query} ---")
    if faiss_retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=faiss_retriever)
        response = qa_chain.invoke({"query": query})
        return response.get("result", "No relevant information found.")
    return "FAISS knowledge base is not available."

# --- 3. Create the Conversational Agent ---

# The web_search_tool is imported from web_search.py and also needs its docstring
tools = [ncr_database_tool, semantic_search_tool, web_search_tool]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. Streamlit UI ---

st.title("FieldBot NCR Assistant")
st.info("For a detailed structured analysis, prefix your query with `analyse:`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask your NCR data..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            if df is None:
                response_content = "Error: The main data file could not be loaded."
            else:
                try:
                    query_to_agent = user_input
                    use_analyst_prompt = False

                    if user_input.lower().startswith("analyse:"):
                        query_to_agent = user_input[len("analyse:"):].strip()
                        use_analyst_prompt = True

                    agent_response = agent_executor.invoke({
                        "input": query_to_agent,
                        "chat_history": st.session_state.messages[:-1]
                    })
                    raw_tool_output = agent_response["output"]
                    
                    if use_analyst_prompt:
                        final_prompt = ANALYST_PROMPT_TEMPLATE.format(question=query_to_agent, tool_output=raw_tool_output)
                        final_synthesized_answer = llm.invoke(final_prompt).content
                        response_content = final_synthesized_answer
                    else:
                        # For direct answers, we can often just use the tool's output directly
                        # Or use a simpler prompt if needed. Let's just use the direct output for now.
                        response_content = raw_tool_output

                except Exception as e:
                    response_content = f"An error occurred: {e}"
            
            st.write(response_content)
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})

