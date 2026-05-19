import asyncio
import os
import json
from google import genai
from google.genai import types
from tools.tavily_client import tavily_search
from tools.web_scraper import scrape_article

client = genai.Client(
      api_key=os.environ["GEMINI_API_KEY"],
      http_options=types.HttpOptions(api_version="v1beta"),
)

FINANCIAL_EXTRACTION_PROMPT = """
You are a financial analyst. Extract key financial data from the sources provided.

Return ONLY a JSON object. No explanation. No markdown. No extra text.

{
  "revenue": {
    "value": "e.g. $4.6B or unknown",
    "period": "e.g. FY2024 or Q3 2024 or unknown",
    "yoy_growth": "e.g. +22% year-over-year or unknown"
  },
  "profitability": {
    "status": "one of: profitable | loss-making | breakeven | unknown",
    "detail": "e.g. Net income $400M or Burning $200M per year or unknown"
  },
  "valuation": "e.g. $95B at last funding round or unknown",
  "key_risks": [
    "Risk 1 — be specific, cite the source",
    "Risk 2 — be specific, cite the source",
    "Risk 3 — be specific, cite the source"
  ],
  "key_metrics": [
    "Any other important number worth noting e.g. total users, market share, ARR"
  ],
  "source": "URL of the report or page this data came from",
  "confidence": "high | medium | low",
  "confidence_reason": "One sentence explaining the data quality and why you rated it this way."
}

Rules:
- If you cannot find a specific value, write unknown — never guess or fabricate numbers.
- Only extract numbers explicitly stated in the source text.
- key_risks must come directly from the report, not your own assumptions.
- If multiple sources conflict on a number, use the most recent one and note the conflict in confidence_reason.
"""



async def find_financial_sources(company: str) -> list[dict]:
    queries = [
        f"{company} annual report 2024 revenue",
        f"{company} 10-K SEC filing 2024",
        f"{company} financial results earnings 2024",
        f"{company} investor relations revenue growth"
    ]
    
    # run all 4 queries concurrently
    res = await asyncio.gather(*[tavily_search(q) for q in queries])
    all_res = [item for batch in res for item in batch]
    
    seen = set()
    unique = []
    for r in all_res:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)
            
    return unique


async def run_financial_agent(company: str) -> str:

    sources = await find_financial_sources(company)
    
    if not sources:
        fallback_data = {"error": True, "message": "No sources found", "results": []}
        return json.dumps(fallback_data)
    top_sources = sources[:4]
    
    scraped_texts = await asyncio.gather(
        *[scrape_article(s.get("url", ""), max_chars=4000) for s in top_sources]
    )
    
    formatted_pieces = []
    # zip() lets us loop through the source dictionaries and the scraped text at the same time
    for source_dict, scraped_text in zip(top_sources, scraped_texts):
        url = source_dict.get("url", "N/A")
        formatted_pieces.append(f"SOURCE: {url}\n\n{scraped_text}")
        
    context = "\n\n---\n\n".join(formatted_pieces)
    
    print(f"Extracting financial data for {company}...")
    
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",       
                contents=[f"Extract financial data for {company} from these sources:\n\n{context}"],   
                config=types.GenerateContentConfig(
                    system_instruction=FINANCIAL_EXTRACTION_PROMPT,
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