import os
from serpapi import GoogleSearch
from typing import List
from dotenv import load_dotenv
from langchain.agents import tool

load_dotenv()

@tool
def web_search_tool(query: str) -> str:
    """
    Use this tool to find up-to-date information on the public internet. 
    Best for questions about external companies, general technical knowledge, current events, or anything that would not be found in the internal NCR database.
    """
    print(f"--- Calling Web Search Tool with query: {query} ---")
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": 3,  # Get top 3 results
        "engine": "google",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        snippets = []

        for result in results.get("organic_results", []):
            snippet = result.get("snippet") or result.get("title")
            if snippet:
                snippets.append(snippet)
        
        if not snippets:
            return "No relevant information found on the web."

        return "\n".join(snippets)

    except Exception as e:
        print(f" [ERROR] Web search failed: {e}")
        return "There was an error performing the web search."