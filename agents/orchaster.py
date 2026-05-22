import asyncio
import json
import os
import streamlit as st
from google import genai
from google.genai import types
from agents.searcher import run_search_agent
from agents.newsgiver import run_news_agent
from agents.Financial import run_financial_agent
from agents.synthesizer import run_synthesiser


client = genai.Client(
      api_key=st.secrets["GEMINI_API_KEY"],
      http_options=types.HttpOptions(api_version="v1beta"),
)


SYSTEM_PROMPT = open("prompts/orchestrator.txt").read()


TOOLS = [
    types.Tool(
        function_declarations=[
            # General Web Search
            types.FunctionDeclaration(
                name="web_search",
                description="Search the web for general information, competitors, market positioning, or product details. Pass multiple specific queries to gather broad context.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "queries": types.Schema(
                            type="ARRAY", 
                            items=types.Schema(type="STRING"),
                            description="A list of specific search queries to run."
                        )
                    },
                    required=["queries"]
                )
            ),
            
            #  News Search
            types.FunctionDeclaration(
                name="news_search",
                description="Search for and scrape recent news articles, leadership updates, product launches, controversies, or recent events. Always use this for 'recent' or 'latest' information.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "queries": types.Schema(
                            type="ARRAY", 
                            items=types.Schema(type="STRING"),
                            description="A list of specific news search queries to run."
                        )
                    },
                    required=["queries"]
                )
            ),
            
            # Financial Research
            types.FunctionDeclaration(
                name="financial_research",
                description="Retrieve in-depth financial data, revenue figures, earnings reports, SEC filings, and investor relations data for a specific company.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "company": types.Schema(
                            type="STRING",
                            description="The exact name of the company to analyze (e.g., 'Stripe')."
                        )
                    },
                    required=["company"]
                )
            )
        ]
    )
]


async def dispatch(tool_call, company: str) -> str:

    name = tool_call.function_call.name
    args = tool_call.function_call.args
    if name == "web_search":
        return await run_search_agent(args["queries"])
        
    elif name == "news_search":
        return await run_news_agent(args["queries"])
        
    elif name == "financial_research":
        return await run_financial_agent(args["company"])
        
    else:
        return json.dumps({"error": True, "message": f"Unknown tool called: {name}", "results": []})


async def run_orchestrator(company: str) -> str:
    messages = [
        types.Content(
            role="user", 
            parts=[types.Part(text=f"Research this company: {company}")]
        )
    ]
    findings = {
        "web_search": "",
        "news_search": "",
        "financial_research": ""
    }
    
    max_iterations = 5
    print(f"\n Launching Boss Agent for: {company}...")
    
    for iteration in range(max_iterations):
        print(f"\n Boss Agent Thinking (Iteration {iteration + 1}/{max_iterations})...")
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash", 
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=TOOLS,
                temperature=0.1
            )
        )
        
        parts = response.candidates[0].content.parts
        tool_calls = [p for p in parts if p.function_call]
        if not tool_calls:
            print("Boss Agent has finished gathering intelligence!")
            break
            
        tool_names = [tc.function_call.name for tc in tool_calls]
        print(f" Boss Agent is dispatching workers: {', '.join(tool_names)}")
        
        results = []
        for tc in tool_calls:
            print(f" Pausing for 5 seconds to prevent API overload before running {tc.function_call.name}...")
            await asyncio.sleep(5)
            
            res = await dispatch(tc, company)
            results.append(res)
        for tc, result in zip(tool_calls, results):
            name = tc.function_call.name
            if name in findings:
                findings[name] = result
                
        messages.append(types.Content(role="model", parts=parts))
        messages.append(
            types.Content(
                role="user", 
                parts=[
                    types.Part.from_function_response(
                        name=tc.function_call.name,
                        response={"result": result}
                    )
                    for tc, result in zip(tool_calls, results)
                ]
            )
        )
        
    final_report = await run_synthesiser(
        company=company,
        search_findings=findings["web_search"],
        news_findings=findings["news_search"],
        financial_findings=findings["financial_research"]
    )
    
    return final_report