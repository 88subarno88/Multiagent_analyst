import asyncio
import os
from google import genai
import json
from google.genai import types
from tools.tavily_client import tavily_search, format_results
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(
      api_key=os.environ["GEMINI_API_KEY"],
      http_options=types.HttpOptions(api_version="v1beta"),
  )


SYSTEM_PROMPT=open("prompts/researcher.txt").read()


async def run_search_agent(queries: list[str]) -> str:
    results = await asyncio.gather(*[tavily_search(q) for q in queries])
    all_results = [item for batch in results for item in batch]
    if len(all_results)==0:
            fallback_data ={
            "results": [],
            "message": "No results found"
            }
            return json.dumps(fallback_data)
    read_txt=format_results(all_results)
    print("calling gemini api..")
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash" ,        
                contents= [read_txt],   
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,       
                    max_output_tokens=2000
                )
            )
            return response.text.replace("```json\n", "").replace("```json", "").replace("```", "").strip()
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

