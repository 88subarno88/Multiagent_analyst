import json
import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types


client = genai.Client(
      api_key=os.environ["GEMINI_API_KEY"],
      http_options=types.HttpOptions(api_version="v1beta"),
)


SYSTEM_PROMPT = open("prompts/synthesiser.txt").read()


def parse_agent_output(raw: str, agent_name: str) -> str:
    try:
        data = json.loads(raw)
        return f"=== {agent_name.upper()} FINDINGS ===\n{json.dumps(data, indent=2)}"
        
    except json.JSONDecodeError:
        return f"=== {agent_name.upper()} FINDINGS ===\n{raw}"



async def run_synthesiser(
    company: str,
    search_findings: str,
    news_findings: str,
    financial_findings: str,
)-> str:

     context = "\n\n".join([
        parse_agent_output(search_findings,"web search"),
        parse_agent_output(news_findings,"news"),
        parse_agent_output(financial_findings,"financial"),
    ])
 
     user_message = (
        f"Company to analyse: {company}\n\n"
        f"Research findings from all agents:\n\n"
        f"{context}\n\n"
        f"Write the complete competitive intelligence brief now."
    )
 
     try:
       
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,     
                max_output_tokens=4000,
            ),
        )
        report = response.text
        print("  [synthesiser] Report written successfully.")
        return report
 
     except Exception as e:
        print(f"  [synthesiser] Error: {e}")
        return (
            f"## Executive Summary\n\n"
            f"Research was collected for **{company}** but the synthesis step failed: {e}\n\n"
            f"## Raw Findings\n\n"
            f"```\n{context[:2000]}\n```"
        )