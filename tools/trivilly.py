import os
import httpx
import asyncio
from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")

#trivally api endpoint
BASE = "https://api.tavily.com/search"


##passing json to base endpoint and get result block from reponse json
async def tavily_search(query: str,topic:str="general",max_results:int=5)->list[dict]:

    my_data={
        "api_key":TAVILY_KEY,
        "query":query,
        "topic":topic,
        "max_results":max_results,
        "include_raw_content":False,
    }

    async with httpx.AsyncClient() as client:
        print("Sending data...")
        try:
            response=await client.post(BASE,json=my_data)
            response.raise_for_status()
            print("REQ was successful")
            return response.json().get("results", [])

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error! The server rejected our request.")
            print(f"Status Code: {e.response.status_code}")
            print(f"Message from server: {e.response.text}")
            return []

        except httpx.RequestError as e:
            print(f"Connection Error while trying to reach {e.request.url}")
            return []

    

## formats results for good will pass to gemini api
def format_results(results: list[dict]) -> str:
    if not results:
        return "(no results found)"
    formatted_blocks = []
    for article in results:
        block = (
            f"SOURCE: {article.get('url', 'N/A')}\n"
            f"TITLE: {article.get('title', 'N/A')}\n"
            f"CONTENT: {article.get('content', 'N/A')}"
        )
        formatted_blocks.append(block)
        
    return "\n\n".join(formatted_blocks)

if __name__ == "__main__":
    async def run_test():
        raw_results = await tavily_search("Stripe revenue 2024")
        final_string = format_results(raw_results)
        print(" FINAL STRING FOR AGENT ---")
        print(final_string)
        
    asyncio.run(run_test())

