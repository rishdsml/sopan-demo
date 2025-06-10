# prompts.py

from langchain.prompts import PromptTemplate

# Prompt for the main agent to define its persona and instructions
AGENT_SYSTEM_PROMPT = "You are a helpful assistant for analyzing NCR reports. You must choose the best tool for each user query and you have access to the conversation history."

# Prompt for the parser LLM call inside the database tool
PANDAS_PARSER_PROMPT = """
Based on the user's query, create a JSON object to query a pandas DataFrame.
The JSON must have "operation", "column", and an optional "value".
The list of available columns is: {df_columns}

Supported operations:
1. 'count_unique': For questions like "how many unique [column_name] are there?".
2. 'lookup': For questions like "find rows where [column_name] is [value]".
3. 'list_unique': For questions like "list all unique values for [column_name]".

Analyze the user's query and find the best matching column from the list. If a specific ID is mentioned (e.g., "well id 1445"), use the appropriate ID column ("Well_Id").

USER QUERY: "{query}"

JSON:
"""

# NEW: A simpler prompt for direct, clean answers
DIRECT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "tool_output"],
    template=(
        "You are a helpful assistant. Use the provided data to directly and concisely answer the user's question.\n"
        "Use markdown for formatting, such as bolding or bullet points, to keep the answer clean and readable.\n\n"
        "User's Question: {question}\n\n"
        "Data from Tool:\n{tool_output}\n\n"
        "Answer:"
    )
)

# The detailed prompt for the "analyse:" mode
ANALYST_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "tool_output"],
    template=(
        "You are a senior technical analyst at SOPAN Oilfield Services.\n"
        "Your task is to provide a detailed analysis based on a user's query and the raw data retrieved from internal tools.\n"
        "Generate a structured response in four parts: Field Report Summary, Ground Truth, Probable Analysis, and Recommended Action.\n\n"
        "---\n\n"
        "**Field Report (User's Query):**\n"
        "{question}\n\n"
        "**Internal Memory (Retrieved Data):**\n"
        "{tool_output}\n\n"
        "---\n\n"
        "**Your Structured Analysis:**\n\n"
        "**Field Report Summary:**\n"
        "Based on the user's query and the retrieved data, summarize the core issue in one to two lines.\n\n"
        "**Ground Truth:**\n"
        "What is happening? Extract and present the key facts, figures, and direct observations from the retrieved data.\n\n"
        "**Probable Analysis:**\n"
        "Provide your expert interpretation of the situation based on the Ground Truth.\n\n"
        "**Recommended Action:**\n"
        "Suggest clear, actionable next steps for the relevant team."
    )
)
