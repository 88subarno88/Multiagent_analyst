import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from agents.orchaster import run_orchestrator 
from tools.pdfgenerate import md_to_pdf


load_dotenv()

st.set_page_config(
    page_title="Multiagent_research", 
    layout="centered"
)

st.title(" Multi-Agent Research Analyst")
st.caption("Made with Gemini 2.5 Flash, Tavily Search + AI agents. Enter a company below to generate a comprehensive competitive intelligence brief.")
st.divider()
company = st.text_input(
    "Company name", 
    placeholder="e.g. Stripe, Notion, Figma, OpenAI"
)
generate = st.button("Generate Brief", disabled=not company)


if generate and company:
    with st.status(f"Deploying AI Agents to research {company}...", expanded=True) as status:
        st.write("Main Agent is planning the research strategy...")
        st.write(" Web Scrapers are gathering market positioning...")
        st.write(" News Agents are pulling recent headlines...")
        st.write(" Financial Agents are analyzing revenue data...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                report_md = loop.run_until_complete(run_orchestrator(company))
            finally:
                loop.close()
            st.write(" Synthesising final executive report...")
            pdf_bytes = md_to_pdf(company, report_md)
            status.update(label="Executive Brief Ready!", state="complete", expanded=False)
            
        except Exception as e:
            status.update(label="Research Failed", state="error", expanded=True)
            st.error(f"An error occurred: {e}")
            st.stop()

    st.download_button(
        label=" Download Report as PDF",
        data=pdf_bytes,
        file_name=f"{company}_Competitive_Brief.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    st.divider()
    st.markdown(report_md)
st.divider()
st.caption("Built with <3 using Python, Streamlit, and Google Gemini.")
