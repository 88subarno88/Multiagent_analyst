import asyncio
import os
import json
from dotenv import load_dotenv;
from google import genai
from google.genai import types
from tools.tavily_client import tavily_search
from tools.web_scraper import scrape_article
load_dotenv()


client = genai.Client(
      api_key=os.environ["GEMINI_API_KEY"],
      http_options=types.HttpOptions(api_version="v1beta"),
)


NEWS_EXTRACTION_PROMPT = """
You are a news analyst. Extract structured events from the news articles provided.

Return ONLY a JSON object. No explanation. No markdown. No extra text.

{
  "findings": [
    {
      "headline": "Short headline of the news event",
      "date": "Date if mentioned in the article, else write 'recent'",
      "event_type": "one of: funding | product_launch | leadership | partnership | controversy | earnings | other",
      "summary": "2-3 sentences explaining what happened",
      "impact": "1 sentence on why this matters for the company",
      "source": "URL of the article this came from"
    }
  ],
  "confidence": "high | medium | low",
  "confidence_reason": "One sentence explaining why you rated the confidence this way."
}

Rules:
- Only include events explicitly stated in the articles. No guessing.
- If two articles cover the same event, merge them into one finding.
- If an article has no news event worth including, skip it entirely.
- If nothing useful was found at all, return findings as an empty list.
"""



async def scrape_with_fallback(result: dict) -> str:
    url_=result.get('url','N/A')
    title_=result.get('title','N/A')
    snippet_=result.get('content','N/A')
    scraped=await scrape_article(url_, max_chars=3000)
    if  len(scraped) < 100:
        return f"SOURCE: {url_}\nTITLE: {title_}\n{snippet_}"
    else:
        return f"SOURCE: {url_}\nTITLE: {title_}\n\n{scraped}"



async def run_news_agent(queries: list[str]) -> str:
    res = await asyncio.gather(*[tavily_search(q,topic="news", max_results=4) for q in queries])
    all_res = [item for batch in res for item in batch]
    if len(all_res)==0:
            fallback_data ={
            "results": [],
            "message": "No results found"
            }
            return json.dumps(fallback_data)
    scraped = await asyncio.gather(*[scrape_with_fallback(r) for r in all_res])
    separator = "\n\n---\n\n"
    combined_text = separator.join(scraped)
    print("calling gemini api..")
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash" ,        
                contents= [combined_text],   
                config=types.GenerateContentConfig(
                    system_instruction=NEWS_EXTRACTION_PROMPT,
                    temperature=0.1,   
                    response_mime_type="application/json"    
                )
            )
            return response.text
        except Exception as e:
        
            error_msg = str(e)
            if "503" in error_msg and attempt < 2:
                print(f"Google servers busy. Retrying in 5 seconds... (Attempt {attempt + 1}/3)")
                await asyncio.sleep(5)
                continue 
                
            
            print(f"GEMINI API ERROR: {e}")
            fallback_data = {
                "error": True,
                "message": f"The AI model failed to respond: {error_msg}",
                "results": [] 
            }
            return json.dumps(fallback_data)


