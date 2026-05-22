import asyncio
import os
import streamlit as st
from google import genai
import json
from google.genai import types
from tools.tavily_client import tavily_search, format_results
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(
      api_key=st.environ["GEMINI_API_KEY"],
      http_options=types.HttpOptions(api_version="v1beta"),
  )


SYSTEM_PROMPT=open("prompts/researcher.txt").read()
async def run_search_agent(queries: list[str]) -> str:
    allthing= []
    
    # process one query at a time so Gemini never misses a fact!
    for q in queries:
        print(f"\n Searching and analyzing: '{q}'...")
        results=await tavily_search(q)
        if not results:
            continue
            
        read_txt =format_results(results)
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",       
                    contents=[read_txt],   
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,       
                        max_output_tokens=2000,
                        response_mime_type="application/json" 
                    )
                )
                data =json.loads(response.text)
                
                if "findings" in data:
                     allthing.extend(data["findings"])
                    
                break 
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    print(f"Servers busy. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                
                print(f"GEMINI API ERROR on '{q}':{e}")
                break
    if not  allthing:
        return json.dumps({"error": True, 
                           "message": "No results found.",
                           "findings": []})
    return json.dumps({"findings": allthing})