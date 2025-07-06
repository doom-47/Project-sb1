import os
from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from bs4 import BeautifulSoup, Comment, NavigableString
import json
import html2text
from youtube_transcript_api import YouTubeTranscriptApi
import re
from groq import Groq, BadRequestError
import sys
import numpy as np
from numpy.linalg import norm
import asyncio

load_dotenv()

# Configuration
SEARXNG_URL = os.getenv('SEARXNG_URL')
BROWSERLESS_URL = os.getenv('BROWSERLESS_URL')
TOKEN = os.getenv('TOKEN')
PROXY_PROTOCOL = os.getenv('PROXY_PROTOCOL', 'http')
PROXY_URL = os.getenv('PROXY_URL')
PROXY_USERNAME = os.getenv('PROXY_USERNAME')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
PROXY_PORT = os.getenv('PROXY_PORT')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://ollama:11434')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'llama3')

REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
FILTER_SEARCH_RESULT_BY_AI = os.getenv('FILTER_SEARCH_RESULT_BY_AI', 'false').lower() == 'true'
AI_ENGINE = os.getenv('AI_ENGINE', 'openai')

domains_only_for_browserless = ["twitter", "x", "facebook", "ucarspro"]

app = FastAPI()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class EmbeddingService:
    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.ollama_host = OLLAMA_HOST
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embeddings from Groq or Ollama fallback"""
        try:
            if self.groq_client:
                response = await asyncio.to_thread(
                    lambda: self.groq_client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small"
                    )
                )
                return response.data[0].embedding
            
            if self.ollama_host:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.ollama_host}/api/embeddings",
                        json={"model": EMBEDDING_MODEL, "prompt": text},
                        timeout=30.0
                    )
                    return response.json().get("embedding")
            return None
        except Exception as e:
            print(f"Embedding error: {e}", file=sys.stderr)
            return None

embedder = EmbeddingService()

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between vectors"""
    return np.dot(a, b) / (norm(a) * norm(b))

async def semantic_rerank(query: str, results: List[Dict]) -> List[Dict]:
    """Re-rank results by semantic similarity"""
    if not results:
        return results
    
    print(f"Starting semantic reranking for query: {query[:50]}...")
    query_embedding = await embedder.get_embedding(query)
    if query_embedding is None:
        print("Warning: Failed to generate query embedding")
        return results
    
    # Process embeddings in parallel
    tasks = []
    for result in results:
        content = f"{result.get('title', '')} {result.get('content', '')}"
        tasks.append(embedder.get_embedding(content))
    
    embeddings = await asyncio.gather(*tasks)
    
    # Score and sort results
    scored = []
    for result, emb in zip(results, embeddings):
        if emb is not None:
            score = cosine_similarity(query_embedding, emb)
            scored.append((score, result))
        else:
            scored.append((0.0, result))  # Default score if embedding fails
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return [result for (_, result) in scored]

def get_proxies(without=False):
    """Original proxy function"""
    if PROXY_URL and PROXY_USERNAME and PROXY_PASSWORD and PROXY_PORT:
        proxy_string = f"{PROXY_PROTOCOL}://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_URL}:{PROXY_PORT}"
        if without:
            return {
                "http": proxy_string,
                "https": proxy_string
            }
        return proxy_string
    return None

def fetch_normal_content(url):
    """Original content fetcher"""
    print(f"DEBUG: Attempting normal HTTP fetch for: {url}", file=sys.stderr)
    try:
        proxy_url = get_proxies()
        
        transport = None
        if proxy_url:
            transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
        
        with httpx.Client(
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
            follow_redirects=True,
            transport=transport
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            print(f"DEBUG: Normal HTTP fetch SUCCESS for: {url}", file=sys.stderr)
            return response.text
    except httpx.RequestError as e:
        print(f"DEBUG: [httpx] Request error for {url}: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"DEBUG: [httpx] HTTP error for {url} (Status {e.response.status_code}): {e.response.text}", file=sys.stderr)
    return None

def fetch_browserless_content(url):
    """Original browserless fetcher"""
    print(f"DEBUG: Attempting Browserless fetch for: {url}", file=sys.stderr)
    try:
        browserless_url = f"{BROWSERLESS_URL}/content"
        params = {}
        
        if TOKEN:
            params['token'] = TOKEN

        proxy_url_for_browserless = f"{PROXY_PROTOCOL}://{PROXY_URL}:{PROXY_PORT}" if PROXY_URL and PROXY_PORT else None
        if proxy_url_for_browserless:
            params['--proxy-server'] = proxy_url_for_browserless

        browserless_data = {
            "url": url,
            "rejectResourceTypes": ["image", "stylesheet", "font", "media"],
            "gotoOptions": {
                "waitUntil": "networkidle0",
                "timeout": REQUEST_TIMEOUT * 1000,
                "headers": HEADERS
            },
            "bestAttempt": True,
            "setJavaScriptEnabled": True,
        }
        
        if PROXY_USERNAME and PROXY_PASSWORD:
            browserless_data["authenticate"] = {
                "username": PROXY_USERNAME,
                "password": PROXY_PASSWORD
            }

        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json'
        }

        with httpx.Client(timeout=REQUEST_TIMEOUT * 2) as client:
            response = client.post(
                browserless_url,
                params=params,
                headers=headers,
                data=json.dumps(browserless_data)
            )
            response.raise_for_status()
            print(f"DEBUG: Browserless fetch SUCCESS for: {url}", file=sys.stderr)
            return response.text
    except httpx.RequestError as e:
        print(f"DEBUG: [browserless] Request error for {url}: {e}", file=sys.stderr)
    except httpx.HTTPStatusError as e:
        print(f"DEBUG: [browserless] HTTP error for {url} (Status {e.response.status_code}): {e.response.text}", file=sys.stderr)
    return None

def fetch_content(url):
    """Original content fetcher with fallback"""
    if any(domain in url for domain in domains_only_for_browserless):
        print(f"DEBUG: Domain '{url}' is in browserless-only list. Forcing Browserless fetch.", file=sys.stderr)
        return fetch_browserless_content(url)

    content = fetch_normal_content(url)
    if content is None:
        print(f"DEBUG: Normal HTTP fetch FAILED for '{url}'. Falling back to Browserless.", file=sys.stderr)
        content = fetch_browserless_content(url)
    else:
        print(f"DEBUG: Normal HTTP fetch SUCCEEDED for '{url}'. No Browserless fallback needed.", file=sys.stderr)
    return content

def get_transcript(video_id: str, format: str = "markdown"):
    """Original transcript fetcher"""
    print(f"DEBUG: Attempting to get YouTube transcript for video ID: {video_id}", file=sys.stderr)
    try:
        proxies_for_youtube = get_proxies(without=True)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies_for_youtube)
        transcript = " ".join([entry['text'] for entry in transcript_list])

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video_page = fetch_content(video_url)
        title = extract_title(video_page)

        print(f"DEBUG: YouTube transcript fetch SUCCESS for video ID: {video_id}", file=sys.stderr)
        if format == "json":
            return JSONResponse({"url": video_url, "title": title, "transcript": transcript})
        return PlainTextResponse(f"Title: {title}\n\nURL Source: {video_url}\n\nTranscript:\n{transcript}")
    except Exception as e:
        print(f"DEBUG: Failed to retrieve transcript for video ID {video_id}: {str(e)}", file=sys.stderr)
        return PlainTextResponse(f"Failed to retrieve transcript: {str(e)}")

def extract_title(html_content):
    """Original title extractor"""
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.find("title")
        return title.string.replace(" - YouTube", "") if title else 'No title'
    return 'No title'

def clean_html(html):
    """Original HTML cleaner"""
    soup = BeautifulSoup(html, 'html.parser')
    
    for script_or_style in soup(["script", "style", "header", "footer", "noscript", "form", "input", 
                               "textarea", "select", "option", "button", "svg", "iframe", "object", 
                               "embed", "applet", "nav", "navbar"]):
        script_or_style.decompose()

    ids = ['layers']
    for id_ in ids:
        tag = soup.find(id=id_)
        if tag:
            tag.decompose()
    
    for tag in soup.find_all(True):
        tag.attrs = {key: value for key, value in tag.attrs.items() if key not in ['class', 'id', 'style']}
    
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    return str(soup)

def parse_html_to_markdown(html, url, title=None):
    """Original HTML to Markdown converter"""
    print(f"DEBUG: Cleaning HTML and converting to Markdown for: {url}", file=sys.stderr)
    cleaned_html = clean_html(html)
    title_ = title or extract_title(html)

    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = False
    text_maker.ignore_tables = False
    text_maker.bypass_tables = False
    text_maker.ignore_images = False
    text_maker.protect_links = True
    text_maker.mark_code = True
    
    markdown_content = text_maker.handle(cleaned_html)
    
    markdown_content = re.sub(r'\n\s*\n', '\n\n', markdown_content)
    markdown_content = re.sub(r'^\s*[\-\*]\s*$', '', markdown_content, flags=re.MULTILINE)
    markdown_content = re.sub(r'\[\]\s*\(#\)', '', markdown_content)

    print(f"DEBUG: Markdown conversion complete for: {url}", file=sys.stderr)
    return {
        "title": title_,
        "url": url,
        "markdown_content": markdown_content
    }

def rerenker_ai(data: Dict[str, List[dict]], max_token: int = 2000) -> Dict[str, List[dict]]:
    """Original AI reranker"""
    print(f"DEBUG: AI re-ranking initiated. AI_ENGINE: {AI_ENGINE}", file=sys.stderr)
    client = None
    model = None
    
    class ResultItem(BaseModel):
        title: str
        url: str
        content: str
    class SearchResult(BaseModel):
        results: List[ResultItem]
    
    system_message = (
        'You are an AI assistant that filters and re-ranks search results based on relevance to a given query. '
        'Your output MUST be a JSON object with a single key: "results". '
        'The value of "results" MUST be a JSON array of objects. '
        'Each object in the "results" array MUST have "title", "url", and "content" keys. '
        'Only include results that are directly relevant to the query. '
        'If a result\'s original "content" field is empty, you may use the "title" or "url" to infer relevance.'
    )
    
    if AI_ENGINE == "groq":
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            print("ERROR: GROQ_API_KEY not found in .env file.", file=sys.stderr)
            return data
        client = Groq(api_key=groq_api_key)
        model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        print(f"DEBUG: Using Groq model: {model}", file=sys.stderr)
    else:
        import openai
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("ERROR: OPENAI_API_KEY not found in .env file.", file=sys.stderr)
            return data
        client = openai
        openai.api_key = openai_api_key
        model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo-0125')
        print(f"DEBUG: Using OpenAI model: {model}", file=sys.stderr)
    
    filtered_results = []
    batch_size = 10
    query = data["query"]
    results = data["results"]
    
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]
        processed_batch = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            } 
            for item in batch
        ]

        try:
            print(f"DEBUG: Sending batch to LLM for re-ranking (batch {i//batch_size + 1})", file=sys.stderr)
            response = client.chat.completions.create(
                model=model,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": f"Query: \"{query}\"\n\nSearch Results:\n{json.dumps(processed_batch, indent=2)}\n\nPlease filter and re-rank these results based on the query."
                    }
                ],
                temperature=0.5,
                max_tokens=max_token,
                response_format={"type":"json_object"}
            )
            
            llm_response_content = response.choices[0].message.content
            print(f"DEBUG: LLM Raw Response Content: {llm_response_content}", file=sys.stderr)

            json_match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```|(\{.*\}|\[.*\])', llm_response_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(1) or json_match.group(2)
                batch_filtered_data = json.loads(json_string)
            else:
                batch_filtered_data = json.loads(llm_response_content)

            if 'results' in batch_filtered_data and isinstance(batch_filtered_data['results'], list):
                filtered_results.extend(batch_filtered_data['results'])
                print(f"DEBUG: LLM re-ranking SUCCESS for batch {i//batch_size + 1}", file=sys.stderr)
            else:
                print(f"WARNING: 'results' key missing in LLM response", file=sys.stderr)
                filtered_results.extend(processed_batch)

        except Exception as e:
            print(f"ERROR: AI processing error: {e}", file=sys.stderr)
            filtered_results.extend(processed_batch)

    print(f"DEBUG: AI re-ranking complete. Total filtered results: {len(filtered_results)}", file=sys.stderr)
    return {"results": filtered_results, "query": query}

def searxng(query: str, categories: str = "general") -> dict:
    """Original SearXNG searcher"""
    print(f"DEBUG: Calling SearXNG for query: '{query}'", file=sys.stderr)
    searxng_url = f"{SEARXNG_URL}/search?q={query}&categories={categories}&format=json"
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get(searxng_url, headers=HEADERS)
            response.raise_for_status()
            search_results = response.json()
            if "results" not in search_results:
                search_results["results"] = []
            print(f"DEBUG: SearXNG call SUCCESS. Found {len(search_results['results'])} results.", file=sys.stderr)
            return search_results
    except httpx.RequestError as e:
        print(f"ERROR: SearXNG request failed: {e}", file=sys.stderr)
        return {"results": [], "error": f"Search query failed: {e}"}
    except httpx.HTTPStatusError as e:
        print(f"ERROR: SearXNG HTTP error: {e}", file=sys.stderr)
        return {"results": [], "error": f"Search query failed: {e}"}

async def search(query: str, num_results: int, json_response: bool = False, semantic: bool = True) -> JSONResponse | PlainTextResponse:
    """Enhanced search function with semantic ranking"""
    print(f"DEBUG: Starting search for: '{query}'", file=sys.stderr)
    search_results = searxng(query)
    
    if "error" in search_results:
        error_msg = search_results["error"]
        print(f"ERROR: {error_msg}", file=sys.stderr)
        if json_response:
            return JSONResponse({"error": error_msg}, status_code=500)
        return PlainTextResponse(f"Error: {error_msg}", status_code=500)

    print(f"Semantic ranking {'enabled' if semantic else 'disabled'} for query: {query}")
    if semantic and search_results.get('results'):
        print(f"First result before ranking: {search_results['results'][0]['title'][:50]}...")

    if FILTER_SEARCH_RESULT_BY_AI:
        print("DEBUG: Applying AI filtering", file=sys.stderr)
        search_results = rerenker_ai(search_results)
    elif semantic:
        print("DEBUG: Applying semantic re-ranking", file=sys.stderr)
        search_results["results"] = await semantic_rerank(query, search_results["results"])

    json_return = []
    markdown_return = ""
    
    for result in search_results.get("results", [])[:num_results]:
        url = result.get("url")
        title = result.get("title")
        content_snippet = result.get("content", "")

        if not url:
            continue

        if "youtube" in url:
            video_id_match = re.search(r"v=([^&]+)", url)
            if video_id_match:
                transcript_response = get_transcript(video_id_match.group(1), "json" if json_response else "markdown")
                if json_response:
                    json_return.append(json.loads(transcript_response.body.decode('utf-8')))
                else:
                    markdown_return += transcript_response.body.decode('utf-8') + "\n\n ---------------- \n\n"
            continue

        html_content = fetch_content(url)
        if html_content:
            markdown_data = parse_html_to_markdown(html_content, url, title=title)
            if markdown_data["markdown_content"].strip():
                if json_response:
                    json_return.append(markdown_data)
                else:
                    markdown_return += (
                        f"Title: {markdown_data['title']}\n\n"
                        f"URL Source: {markdown_data['url']}\n\n"
                        f"Markdown Content:\n{markdown_data['markdown_content']}"
                    ) + "\n\n ---------------- \n\n"
            elif content_snippet:
                if json_response:
                    json_return.append({
                        "title": title,
                        "url": url,
                        "markdown_content": content_snippet
                    })
                else:
                    markdown_return += (
                        f"Title: {title}\n\n"
                        f"URL Source: {url}\n\n"
                        f"Markdown Content (Snippet):\n{content_snippet}"
                    ) + "\n\n ---------------- \n\n"

    print(f"DEBUG: Search completed for: '{query}'", file=sys.stderr)
    if json_response:
        return JSONResponse(json_return)
    return PlainTextResponse(markdown_return)

@app.get("/images")
def get_search_images(
    q: str = Query(..., description="Search images"),
    num_results: int = Query(5, description="Number of results")
    ):
    result_list = searxng(q, categories="images")
    return JSONResponse(result_list.get("results", [])[:num_results])

@app.get("/videos")
def get_search_videos(
    q: str = Query(..., description="Search videos"),
    num_results: int = Query(5, description="Number of results")
    ):
    result_list = searxng(q, categories="videos")
    return JSONResponse(result_list.get("results", [])[:num_results])

@app.get("/")
async def get_search_results(
    q: str = Query(..., description="Search query"), 
    num_results: int = Query(5, description="Number of results"),
    format: str = Query("markdown", description="Output format (markdown or json)"),
    semantic: bool = Query(True, description="Enable semantic re-ranking")
):
    return await search(q, num_results, format == "json", semantic)

@app.get("/r/{url:path}")
def fetch_url(request: Request, url: str, format: str = Query("markdown", description="Output format (markdown or json)")):
    if "youtube.com/watch" in url:
        video_id_match = re.search(r"v=([^&]+)", url)
        if video_id_match:
            return get_transcript(video_id_match.group(1), format)
        return PlainTextResponse("Invalid YouTube URL", status_code=400)
    
    html_content = fetch_content(url)
    if html_content:
        markdown_data = parse_html_to_markdown(html_content, url)
        if format == "json":
            return JSONResponse(markdown_data)
        return PlainTextResponse(
            f"Title: {markdown_data['title']}\n\n"
            f"URL Source: {markdown_data['url']}\n\n"
            f"Markdown Content:\n{markdown_data['markdown_content']}"
        )
    return PlainTextResponse("Failed to retrieve content", status_code=500)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)