import subprocess
import httpx
import sys
import re
import asyncio

FASTAPI_APP_URL = "http://localhost:8000"

def generate_search_query_ollama(keywords: str, model: str = "llama3") -> str:
    system_prompt = (
        "You are a helpful assistant that generates precise and semantically rich "
        "search queries based on user-provided keywords. Your output should be "
        "a list of 10 concise and effective search query combinations, one per line. "
        "Do not include any introductory or concluding remarks, just the queries. "
        "Each query should be a standalone string, ready to be used in a search engine."
    )

    user_prompt = f"Keywords: {keywords}\n\nGenerate 10 relevant search query combinations."

    full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=full_prompt,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        ollama_raw_output = result.stdout.strip()
        
        match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)(?=<\|eot_id\|>|$)', ollama_raw_output, re.DOTALL)
        
        if match:
            extracted_queries = match.group(1).strip()
            extracted_queries = re.sub(r'<\|eot_id\|>|<\|end_of_text\|>', '', extracted_queries).strip()
            return extracted_queries
        else:
            print("Warning: Could not parse Ollama output with regex. Returning raw output.", file=sys.stderr)
            return ollama_raw_output

    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama (exit code {e.returncode}): {e}", file=sys.stderr)
        print(f"Ollama stdout: {e.stdout}", file=sys.stderr)
        print(f"Ollama stderr: {e.stderr}", file=sys.stderr)
        return f"Error generating query with Ollama: {e.stderr or e.stdout or 'Unknown error'}"
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.", file=sys.stderr)
        return "Error: Ollama not installed or not in PATH."
    except Exception as e:
        print(f"An unexpected error occurred during Ollama generation: {e}", file=sys.stderr)
        return f"Unexpected error with Ollama: {e}"

async def fetch_search_results_from_fastapi(query: str, num_results: int = 5, output_format: str = "markdown") -> str:
    params = {
        "q": query,
        "num_results": num_results,
        "format": output_format
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(FASTAPI_APP_URL, params=params, timeout=300) 
            response.raise_for_status()
            return response.text
    except httpx.RequestError as e:
        print(f"Error connecting to FastAPI app at {FASTAPI_APP_URL}: {e}", file=sys.stderr)
        return f"Error: Could not connect to FastAPI app. Is it running at {FASTAPI_APP_URL}? Details: {e}"
    except httpx.HTTPStatusError as e:
        print(f"Error from FastAPI app (Status {e.response.status_code}): {e.response.text}", file=sys.stderr)
        return f"Error from FastAPI app: Status {e.response.status_code} - {e.response.text}"
    except Exception as e:
        print(f"An unexpected error occurred while fetching from FastAPI: {e}", file=sys.stderr)
        return f"Unexpected error fetching from FastAPI: {e}"

async def main():
    print("📚 Local Search Query Generator & Scraper (Ollama LLM & FastAPI)\n", file=sys.stderr)
    
    keywords = input("Enter your keywords (e.g., social media, education): ").strip()

    if not keywords:
        print("No keywords entered. Exiting.", file=sys.stderr)
        return

    print("\n🔹 Generating search queries with Ollama...\n", file=sys.stderr)
    ollama_queries_raw_output = generate_search_query_ollama(keywords)
    
    ollama_queries = [
        q.strip() for q in ollama_queries_raw_output.split('\n') 
        if q.strip() and not q.strip().startswith('<|') and not q.strip().endswith('|>')
    ]

    if not ollama_queries:
        print(f"Failed to generate valid queries from Ollama. Raw output: \n{ollama_queries_raw_output}", file=sys.stderr)
        return

    print("🔹 Generated Queries:\n", file=sys.stderr)
    for i, query in enumerate(ollama_queries):
        print(f"{i+1}. {query}", file=sys.stderr)

    print("\n--- Fetching results for each query from FastAPI scraper ---\n", file=sys.stderr)

    for i, query in enumerate(ollama_queries):
        print(f"\n--- Fetching results for Query {i+1}: \"{query}\" ---", file=sys.stderr)
        
        print(f"--- Results for Query {i+1}: \"{query}\" ---")
        scraper_output = await fetch_search_results_from_fastapi(query, num_results=3, output_format="markdown")
        
        print(scraper_output)
        print("\n------------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())