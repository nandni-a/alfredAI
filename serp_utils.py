import os
import requests

SERP_API_KEY = os.getenv("bfb5d94186eb8fafd3c92f3ced3d259ba53559d58e4b9407ce358727ae7517a1")

def serp_search(query):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "engine": "google"
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get("organic_results", [])
    return "\n\n".join(f"{r['title']}\n{r['link']}" for r in results[:5])
