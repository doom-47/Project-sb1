import os
import asyncio
import re
import sys
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Tuple, Any
import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from bs4 import BeautifulSoup, Comment, NavigableString
import json
import html2text
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from groq import Groq, BadRequestError
from dotenv import load_dotenv

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
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '30'))
FILTER_SEARCH_RESULT_BY_AI = os.getenv('FILTER_SEARCH_RESULT_BY_AI', 'false').lower() == 'true'
AI_ENGINE = os.getenv('AI_ENGINE', 'openai')

# JS-heavy sites that should always use Browserless first
JS_HEAVY_SITES = {
   'ndtv.com', 'cnn.com', 'msn.com', 'indiatimes.com', 'reuters.com',
   'bbc.com', 'bbc.co.uk', 'forbes.com', 'bloomberg.com', 'wsj.com', 'ft.com',
   'guardian.co.uk', 'theguardian.com', 'nytimes.com', 'washingtonpost.com',
   'cnbc.com', 'espn.com', 'tiktok.com', 'instagram.com', 'linkedin.com',
   'medium.com', 'substack.com', 'twitter.com', 'x.com', 'facebook.com',
   'timesofindia.indiatimes.com', 'hindustantimes.com', 'indianexpress.com',
   'news18.com', 'firstpost.com', 'theprint.in', 'scroll.in', 'wire.in',
   'livemint.com', 'economictimes.indiatimes.com', 'financialexpress.com',
   'businesstoday.in', 'moneycontrol.com', 'zeenews.india.com'
}

# Cache for dynamically discovered JS-heavy sites
js_heavy_cache = set()

# Sites that should only use Browserless (legacy compatibility)
domains_only_for_browserless = ["twitter", "x", "facebook", "ucarspro"]

# Index page patterns that likely contain article links
INDEX_PATTERNS = [
   r'/world/?$', r'/politics/?$', r'/business/?$', r'/technology/?$',
   r'/sports/?$', r'/entertainment/?$', r'/health/?$', r'/science/?$',
   r'/category/', r'/section/', r'/topic/', r'/tag/', r'/news/?$',
   r'/latest/?$', r'/trending/?$', r'/breaking/?$', r'/headlines/?$',
   r'/$'  # Homepage
]

app = FastAPI()

HEADERS = {
   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class TextCleaner:
   @staticmethod
   def clean_text(text: str) -> str:
       if not text:
           return ""
       text = re.sub(r'\s+', ' ', text)
       text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
       text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
       text = re.sub(r'[!]{2,}', '!', text)
       text = re.sub(r'[?]{2,}', '?', text)
       text = re.sub(r'[.]{3,}', '...', text)
       text = re.sub(r'[^\w\s.,!?;:()\-"\']', ' ', text)
       text = re.sub(r'\s+', ' ', text)
       return text.strip()
   
   @staticmethod
   def clean_html_content(html_content: str) -> str:
       if not html_content:
           return ""
       
       soup = BeautifulSoup(html_content, 'html.parser')
       
       unwanted_tags = [
           'script', 'style', 'header', 'footer', 'noscript', 'form', 'input',
           'textarea', 'select', 'option', 'button', 'svg', 'iframe', 'object',
           'embed', 'applet', 'nav', 'navbar', 'aside', 'advertisement', 'ads',
           'cookie', 'popup', 'modal', 'overlay', 'social-share', 'comment'
       ]
       
       for tag in soup(unwanted_tags):
           tag.decompose()
       
       unwanted_selectors = [
           '[class*="ad"]', '[class*="advertisement"]', '[class*="banner"]',
           '[class*="popup"]', '[class*="modal"]', '[class*="overlay"]',
           '[class*="social"]', '[class*="share"]', '[class*="cookie"]',
           '[id*="ad"]', '[id*="advertisement"]', '[id*="banner"]',
           '[id*="popup"]', '[id*="modal"]', '[id*="tracking"]'
       ]
       
       for selector in unwanted_selectors:
           for element in soup.select(selector):
               element.decompose()
       
       for element in soup.find_all():
           if not element.get_text(strip=True) and not element.find_all(['img', 'video', 'audio']):
               element.decompose()
       
       for tag in soup.find_all(True):
           allowed_attrs = ['href', 'src', 'alt', 'title']
           tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed_attrs}
       
       for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
           comment.extract()
       
       return str(soup)
   
   @staticmethod
   def clean_markdown_content(markdown_content: str) -> str:
       if not markdown_content:
           return ""
       
       markdown_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', markdown_content)
       markdown_content = re.sub(r'^\s*[\-\*]\s*$', '', markdown_content, flags=re.MULTILINE)
       markdown_content = re.sub(r'\[\]\s*\(#?\)', '', markdown_content)
       markdown_content = re.sub(r'^[-_]{4,}$', '', markdown_content, flags=re.MULTILINE)
       markdown_content = re.sub(r'\|\s*\|', '|', markdown_content)
       markdown_content = re.sub(r'^\s*[#*\-_|]+\s*$', '', markdown_content, flags=re.MULTILINE)
       
       return markdown_content.strip()

class EmbeddingService:
   def __init__(self):
       try:
           self.model = SentenceTransformer(EMBEDDING_MODEL)
       except Exception as e:
           print(f"Error loading embedding model: {e}")
           self.model = None
   
   async def get_embedding(self, text: str) -> Optional[List[float]]:
       if not self.model or not text:
           return None
       
       try:
           cleaned_text = TextCleaner.clean_text(text)
           if not cleaned_text:
               return None
           
           if len(cleaned_text) > 2000:
               cleaned_text = cleaned_text[:2000]
           
           embedding = await asyncio.to_thread(
               lambda: self.model.encode(cleaned_text, convert_to_tensor=False)
           )
           
           return embedding.tolist()
       except Exception as e:
           print(f"Embedding error: {e}")
           return None

embedder = EmbeddingService()
text_cleaner = TextCleaner()

def get_domain(url: str) -> str:
   try:
       return urlparse(url).netloc.lower()
   except:
       return ""

def is_js_heavy_site(url: str) -> bool:
   domain = get_domain(url)
   domain_parts = domain.split('.')
   for i in range(len(domain_parts)):
       check_domain = '.'.join(domain_parts[i:])
       if check_domain in JS_HEAVY_SITES or check_domain in js_heavy_cache:
           return True
   return False

def is_likely_index_page(url: str) -> bool:
   path = urlparse(url).path.lower()
   return any(re.search(pattern, path) for pattern in INDEX_PATTERNS)

def is_content_substantial(content: str, min_length: int = 500) -> bool:
   if not content or len(content.strip()) < min_length:
       return False
   
   low_quality_indicators = [
       'javascript is disabled', 'enable javascript', 'browser not supported',
       'page not found', 'access denied', 'loading...', 'please wait',
       'cookies required', 'subscription required'
   ]
   
   content_lower = content.lower()
   return not any(indicator in content_lower for indicator in low_quality_indicators)

def get_proxies(without=False):
   if PROXY_URL and PROXY_USERNAME and PROXY_PASSWORD and PROXY_PORT:
       proxy_string = f"{PROXY_PROTOCOL}://{PROXY_USERNAME}:{PROXY_PASSWORD}@{PROXY_URL}:{PROXY_PORT}"
       if without:
           return {"http": proxy_string, "https": proxy_string}
       return proxy_string
   return None

async def fetch_with_httpx(url: str, timeout: int = 15) -> Optional[str]:
   try:
       proxy_url = get_proxies()
       transport = None
       if proxy_url:
           transport = httpx.HTTPTransport(proxy=httpx.Proxy(url=proxy_url))
       
       async with httpx.AsyncClient(
           timeout=timeout,
           headers=HEADERS,
           follow_redirects=True,
           transport=transport
       ) as client:
           response = await client.get(url)
           response.raise_for_status()
           return response.text
   except Exception as e:
       print(f"httpx fetch failed for {url}: {e}")
       return None

async def fetch_with_browserless(url: str, timeout: int = 30) -> Optional[str]:
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
               "timeout": timeout * 1000,
           },
           "setExtraHTTPHeaders": HEADERS,
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

       async with httpx.AsyncClient(timeout=timeout * 2) as client:
           response = await client.post(
               browserless_url,
               params=params,
               headers=headers,
               data=json.dumps(browserless_data)
           )
           response.raise_for_status()
           return response.text
   except Exception as e:
       print(f"Browserless fetch failed for {url}: {e}")
       return None

async def smart_fetch_content(url: str) -> Optional[str]:
   domain = get_domain(url)
   
   if any(domain_part in url for domain_part in domains_only_for_browserless):
       return await fetch_with_browserless(url)
   
   if is_js_heavy_site(url):
       content = await fetch_with_browserless(url)
       if content and is_content_substantial(content):
           return content
       
       content = await fetch_with_httpx(url)
       if content and is_content_substantial(content):
           return content
   else:
       content = await fetch_with_httpx(url)
       if content and is_content_substantial(content):
           return content
       
       content = await fetch_with_browserless(url)
       if content and is_content_substantial(content):
           js_heavy_cache.add(domain)
           return content
   
   return None

def extract_article_links(html_content: str, base_url: str, max_links: int = 10) -> List[str]:
   if not html_content:
       return []
   
   soup = BeautifulSoup(html_content, 'html.parser')
   links = []
   
   article_selectors = [
       'article a[href]', '.article a[href]', '.story a[href]', '.news-item a[href]',
       'h1 a[href]', 'h2 a[href]', 'h3 a[href]', '.headline a[href]',
       '.title a[href]', '.post-title a[href]', 'a[href*="/article/"]',
       'a[href*="/story/"]', 'a[href*="/news/"]', 'a[href*="/post/"]',
       'a[href*="/2024/"]', 'a[href*="/2025/"]'
   ]
   
   for selector in article_selectors:
       for link in soup.select(selector):
           href = link.get('href')
           if href:
               full_url = urljoin(base_url, href)
               
               if (full_url not in links and 
                   not any(skip in full_url.lower() for skip in ['javascript:', 'mailto:', '#', 'search', 'login', 'register', 'subscribe', 'contact']) and
                   len(urlparse(full_url).path) > 1):
                   links.append(full_url)
                   
                   if len(links) >= max_links:
                       break
       
       if len(links) >= max_links:
           break
   
   return links[:max_links]

def extract_title(html_content):
   if html_content:
       soup = BeautifulSoup(html_content, 'html.parser')
       title = soup.find("title")
       if title:
           cleaned_title = text_cleaner.clean_text(title.string.replace(" - YouTube", ""))
           return cleaned_title if cleaned_title else 'No title'
   return 'No title'

def parse_html_to_markdown(html, url, title=None):
   cleaned_html = text_cleaner.clean_html_content(html)
   title_ = title or extract_title(html)
   cleaned_title = text_cleaner.clean_text(title_)

   text_maker = html2text.HTML2Text()
   text_maker.ignore_links = False
   text_maker.ignore_tables = False
   text_maker.bypass_tables = False
   text_maker.ignore_images = False
   text_maker.protect_links = True
   text_maker.mark_code = True
   
   markdown_content = text_maker.handle(cleaned_html)
   cleaned_markdown = text_cleaner.clean_markdown_content(markdown_content)

   return {
       "title": cleaned_title,
       "url": url,
       "markdown_content": cleaned_markdown
   }

def get_transcript(video_id: str, format: str = "markdown"):
   try:
       proxies_for_youtube = get_proxies(without=True)
       transcript_list = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies_for_youtube)
       
       transcript_parts = []
       for entry in transcript_list:
           cleaned_text = text_cleaner.clean_text(entry['text'])
           if cleaned_text:
               transcript_parts.append(cleaned_text)
       
       transcript = " ".join(transcript_parts)

       video_url = f"https://www.youtube.com/watch?v={video_id}"
       
       try:
           with httpx.Client(timeout=15, headers=HEADERS) as client:
               response = client.get(video_url)
               video_page = response.text
       except:
           video_page = None
       
       title = extract_title(video_page)
       cleaned_title = text_cleaner.clean_text(title)

       if format == "json":
           return JSONResponse({
               "url": video_url, 
               "title": cleaned_title, 
               "transcript": transcript
           })
       return PlainTextResponse(f"Title: {cleaned_title}\n\nURL Source: {video_url}\n\nTranscript:\n{transcript}")
   except Exception as e:
       return PlainTextResponse(f"Failed to retrieve transcript: {str(e)}")

async def fetch_article_content(url: str) -> Optional[Dict]:
   if "youtube" in url:
       video_id_match = re.search(r"v=([^&]+)", url)
       if video_id_match:
           transcript_response = get_transcript(video_id_match.group(1), "json")
           return json.loads(transcript_response.body.decode('utf-8'))
       return None
   
   html_content = await smart_fetch_content(url)
   if html_content:
       markdown_data = parse_html_to_markdown(html_content, url)
       if markdown_data["markdown_content"].strip():
           return markdown_data
   
   return None

async def fetch_multiple_articles(urls: List[str]) -> List[Dict]:
   tasks = [fetch_article_content(url) for url in urls]
   results = await asyncio.gather(*tasks, return_exceptions=True)
   
   valid_results = []
   for result in results:
       if isinstance(result, dict) and result.get("markdown_content"):
           valid_results.append(result)
   
   return valid_results

def cosine_similarity(a: List[float], b: List[float]) -> float:
   try:
       return np.dot(a, b) / (norm(a) * norm(b))
   except:
       return 0.0

async def semantic_rerank(query: str, results: List[Dict]) -> List[Dict]:
   if not results:
       return results
   
   cleaned_query = text_cleaner.clean_text(query)
   query_embedding = await embedder.get_embedding(cleaned_query)
   
   if query_embedding is None:
       return results
   
   tasks = []
   for result in results:
       title = text_cleaner.clean_text(result.get('title', ''))
       content = text_cleaner.clean_text(result.get('content', ''))
       combined_content = f"{title} {content}"
       tasks.append(embedder.get_embedding(combined_content))
   
   embeddings = await asyncio.gather(*tasks)
   
   scored = []
   for result, emb in zip(results, embeddings):
       if emb is not None:
           score = cosine_similarity(query_embedding, emb)
           scored.append((score, result))
       else:
           scored.append((0.0, result))
   
   scored.sort(key=lambda x: x[0], reverse=True)
   
   return [result for (_, result) in scored]

def searxng(query: str, categories: str = "general") -> dict:
   searxng_url = f"{SEARXNG_URL}/search?q={query}&categories={categories}&format=json"
   try:
       with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
           response = client.get(searxng_url, headers=HEADERS)
           response.raise_for_status()
           search_results = response.json()
           if "results" not in search_results:
               search_results["results"] = []
           return search_results
   except Exception as e:
       return {"results": [], "error": f"Search query failed: {e}"}

def rerenker_ai(data: Dict[str, List[dict]], max_token: int = 2000) -> Dict[str, List[dict]]:
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
       'Focus on high-quality, informative content and remove spam or low-quality results.'
   )
   
   if AI_ENGINE == "groq":
       groq_api_key = os.getenv('GROQ_API_KEY')
       if not groq_api_key:
           return data
       client = Groq(api_key=groq_api_key)
       model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
   else:
       import openai
       openai_api_key = os.getenv('OPENAI_API_KEY')
       if not openai_api_key:
           return data
       client = openai
       openai.api_key = openai_api_key
       model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo-0125')
   
   filtered_results = []
   batch_size = 10
   query = data["query"]
   results = data["results"]
   
   for i in range(0, len(results), batch_size):
       batch = results[i:i+batch_size]
       
       processed_batch = []
       for item in batch:
           cleaned_item = {
               "title": text_cleaner.clean_text(item.get("title", "")),
               "url": item.get("url", ""),
               "content": text_cleaner.clean_text(item.get("content", ""))
           }
           processed_batch.append(cleaned_item)

       try:
           response = client.chat.completions.create(
               model=model,
               stream=False,
               messages=[
                   {"role": "system", "content": system_message},
                   {"role": "user", "content": f"Query: \"{query}\"\n\nSearch Results:\n{json.dumps(processed_batch, indent=2)}\n\nPlease filter and re-rank these results based on the query, focusing on high-quality, relevant content."}
               ],
               temperature=0.3,
               max_tokens=max_token,
               response_format={"type":"json_object"}
           )
           
           llm_response_content = response.choices[0].message.content
           json_match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```|(\{.*\}|\[.*\])', llm_response_content, re.DOTALL)
           if json_match:
               json_string = json_match.group(1) or json_match.group(2)
               batch_filtered_data = json.loads(json_string)
           else:
               batch_filtered_data = json.loads(llm_response_content)

           if 'results' in batch_filtered_data and isinstance(batch_filtered_data['results'], list):
               filtered_results.extend(batch_filtered_data['results'])
           else:
               filtered_results.extend(processed_batch)

       except Exception as e:
           print(f"AI processing error: {e}")
           filtered_results.extend(processed_batch)

   return {"results": filtered_results, "query": query}

@app.get("/")
async def get_search_results(
   q: str = Query(..., description="Search query"), 
   num_results: int = Query(5, description="Number of results"),
   format: str = Query("markdown", description="Output format (markdown or json)"),
   semantic: bool = Query(True, description="Enable semantic re-ranking")
):
   search_results = searxng(q)
   
   if "error" in search_results:
       error_msg = search_results["error"]
       if format == "json":
           return JSONResponse({"error": error_msg}, status_code=500)
       return PlainTextResponse(f"Error: {error_msg}", status_code=500)

   if FILTER_SEARCH_RESULT_BY_AI:
       search_results = rerenker_ai(search_results)
   elif semantic:
       search_results["results"] = await semantic_rerank(q, search_results["results"])

   fetch_tasks = []
   for result in search_results.get("results", [])[:num_results]:
       url = result.get("url")
       if url:
           fetch_tasks.append(fetch_article_content(url))

   articles = await asyncio.gather(*fetch_tasks, return_exceptions=True)
   
   json_return = []
   markdown_return = ""
   
   for article in articles:
       if isinstance(article, dict) and article.get("markdown_content"):
           if format == "json":
               json_return.append(article)
           else:
               markdown_return += (
                   f"Title: {article['title']}\n\n"
                   f"URL Source: {article['url']}\n\n"
                   f"Markdown Content:\n{article['markdown_content']}"
               ) + "\n\n ---------------- \n\n"

   if format == "json":
       return JSONResponse(json_return)
   return PlainTextResponse(markdown_return)

@app.get("/r/{url:path}")
async def fetch_url(
   request: Request, 
   url: str, 
   format: str = Query("markdown", description="Output format (markdown or json)"),
   extract_links: bool = Query(False, description="Extract article links from index pages")
):
   if "youtube.com/watch" in url:
       video_id_match = re.search(r"v=([^&]+)", url)
       if video_id_match:
           return get_transcript(video_id_match.group(1), format)
       return PlainTextResponse("Invalid YouTube URL", status_code=400)
   
   if extract_links or is_likely_index_page(url):
       html_content = await smart_fetch_content(url)
       if html_content:
           article_links = extract_article_links(html_content, url)
           
           if article_links:
               articles = await fetch_multiple_articles(article_links)
               
               if format == "json":
                   return JSONResponse({
                       "index_url": url,
                       "articles_found": len(articles),
                       "articles": articles
                   })
               else:
                   markdown_return = f"Index Page: {url}\n\nArticles Found: {len(articles)}\n\n"
                   markdown_return += "=" * 50 + "\n\n"
                   for article in articles:
                       markdown_return += (
                           f"Title: {article['title']}\n\n"
                           f"URL Source: {article['url']}\n\n"
                           f"Markdown Content:\n{article['markdown_content']}"
                       ) + "\n\n ---------------- \n\n"
                   return PlainTextResponse(markdown_return)
   
   html_content = await smart_fetch_content(url)
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

@app.get("/health")
async def health_check():
   return {
       "status": "healthy", 
       "embedding_model": EMBEDDING_MODEL,
       "ai_engine": AI_ENGINE,
       "semantic_search": embedder.model is not None,
       "js_heavy_sites": len(JS_HEAVY_SITES),
       "cached_js_sites": len(js_heavy_cache)
   }

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)
