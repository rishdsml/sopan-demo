import os
from serpapi.google_search import GoogleSearch
from typing import List
from dotenv import load_dotenv
load_dotenv()

import os


SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def search_web(query: str, num_results: int = 3) -> List[str]:
    """
    Search the web using SerpAPI and return top N snippets.

    Args:
        query (str): The user query to search for.
        num_results (int): Number of snippets to return.

    Returns:
        List[str]: List of top snippets from search results.
    """
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num_results,
        "engine": "google",
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        snippets = []

        for result in results.get("organic_results", [])[:num_results]:
            snippet = result.get("snippet") or result.get("title")
            if snippet:
                snippets.append(snippet)

        return snippets

    except Exception as e:
        print(f" [ERROR] Web search failed: {e}")
        return []