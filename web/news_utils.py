import requests

NEWS_API_KEY = "bfa14549cdc84faf88c40e947da98d4d"

def fetch_news(query, api_key=NEWS_API_KEY):
    url=f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&apiKey={api_key}"
    try:
        resp = requests.get(url).json()
        articles = resp.get("articles", [])
        return [{"text": (a.get("title") or "") + " " + (a.get("description") or ""),
                 "published": a.get("publishedAt"), "url": a.get("url")} for a in articles]
    except Exception as e:
        print(f"Error fetching news for {query}: {e}")
        return []

def summarize_news_for_investor(news_list):
    if not news_list: return "No recent news."
    combined_news = " ".join([n["text"] for n in news_list[:5]])
    prompt=f"Summarize the following news for an investor in 2-3 sentences: {combined_news}"
    try:
        resp = requests.post("http://localhost:11434/v1/chat/completions",
                             json={"model":"llama3","messages":[{"role":"user","content":prompt}]})
        if resp.status_code!=200: return "Failed to generate summary."
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return "Failed to generate summary."