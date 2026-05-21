import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


BLUE     = colors.HexColor("#1A73E8") 
DARK     = colors.HexColor("#202124") 
MID_GREY = colors.HexColor("#5F6368")
TITLE_STYLE  = ParagraphStyle("title", 
                              fontSize=20, 
                              textColor=BLUE, 
                              fontName="Helvetica-Bold", 
                              spaceAfter=15)
H2_STYLE     = ParagraphStyle("h2", 
                              fontSize=14, 
                              textColor=BLUE, 
                              fontName="Helvetica-Bold", 
                              spaceBefore=15, 
                              spaceAfter=6)
H3_STYLE     = ParagraphStyle("h3", 
                              fontSize=11, 
                              textColor=DARK, 
                              fontName="Helvetica-Bold", 
                              spaceBefore=10, 
                              spaceAfter=4)
BODY_STYLE   = ParagraphStyle("body", 
                              fontSize=9.5, 
                              textColor=DARK, 
                              fontName="Helvetica", 
                              leading=15,
                                spaceAfter=8)
BULLET_STYLE = ParagraphStyle("bullet", 
                              fontSize=9.5, 
                              textColor=DARK, 
                              fontName="Helvetica", 
                              leading=15, 
                              leftIndent=15, 
                              spaceAfter=4)



def md_to_story(company: str, markdown_text: str) -> list:
    story = []
    story.append(Paragraph(f"Competitive  Brief: {company}", TITLE_STYLE))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=15, spaceBefore=5))
    lines = markdown_text.split("\n")
    buffer = []

    def flush_buffer():
        if buffer:
            text = " ".join(buffer)
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
            story.append(Paragraph(text, BODY_STYLE))
            buffer.clear()

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            flush_buffer()
        elif line.startswith("# "):
            flush_buffer()
            clean_text = line.replace("# ", "").strip()
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean_text)
            story.append(Paragraph(clean_text, H2_STYLE))
            story.append(HRFlowable(width="100%", 
                                    thickness=1, 
                                    color=MID_GREY,
                                      spaceAfter=10,
                                     spaceBefore=5))
        elif line.startswith("## "):
            flush_buffer()
            clean_text = line.replace("## ", "").strip()
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean_text)
            story.append(Paragraph(clean_text, H2_STYLE))
            story.append(HRFlowable(width="100%", 
                                    thickness=0.5, 
                                    color=MID_GREY, 
                                    spaceAfter=10, 
                                    spaceBefore=5))
            
        elif line.startswith("### "):
            flush_buffer()
            clean_text = line.replace("### ", "").strip()
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean_text)
            story.append(Paragraph(clean_text
                                   , H3_STYLE))
            
        elif line.startswith("- ") or line.startswith("* "):
            flush_buffer()
            clean_text = line[2:].strip()
            clean_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", clean_text)
            story.append(Paragraph(f"•  {clean_text}", 
                                   BULLET_STYLE))
            
        else:
            buffer.append(stripped_line)
    flush_buffer()
    return story



def md_to_pdf(company: str, markdown_text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        leftMargin=20*mm, 
        rightMargin=20*mm, 
        topMargin=18*mm, 
        bottomMargin=18*mm
    )
    story = md_to_story(company, markdown_text)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()