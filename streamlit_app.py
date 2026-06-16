import streamlit as st
import sqlite3
import os
import re
import json
import uuid
import pdfplumber
import pandas as pd
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

# Set page config for beautiful layout
st.set_page_config(
    page_title="KCET Admission Predictor & RAG Analyst",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Matte Black Dashboard Aesthetic (Reference-Aligned)
st.markdown("""
<style>
    /* Import Google Fonts: Bebas Neue for headers, Plus Jakarta Sans for UI */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Global Canvas styling */
    .stApp {
        background-color: #0b0b0c !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        color: #ffffff !important;
    }
    
    /* Layout container centering & padding */
    .block-container {
        max-width: 1100px !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-top: 3.5rem !important;
        padding-bottom: 3.5rem !important;
        margin: auto !important;
    }
    
    /* Header typography */
    h1 {
        font-family: 'Bebas Neue', sans-serif !important;
        color: #ffffff !important;
        letter-spacing: 0.08em !important;
        font-size: 3.6rem !important;
    }
    
    h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    
    p, span, label, div {
        color: #d1d1d6;
    }

    /* Muted label styling */
    .stWidgetLabel p {
        color: #8e8e93 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Target input container wrappers to eliminate all white background leakage */
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background-color: #151518 !important;
        border: 1px solid #28282c !important;
        border-radius: 30px !important;
        color: #ffffff !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Highlight inputs on focus */
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border-color: #a5ff50 !important;
        box-shadow: 0 0 10px rgba(165, 255, 80, 0.15) !important;
    }

    /* Style inner text inputs */
    input, select, textarea {
        background-color: transparent !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
        border: none !important;
    }

    /* Specific overrides for Select input children to prevent light-mode background color inheritance */
    div[data-baseweb="select"] div {
        background-color: transparent !important;
        color: #ffffff !important;
    }

    /* Multiselect Tags (Pills) styling */
    div[data-baseweb="tag"] {
        background-color: #222226 !important;
        border: 1px solid #28282c !important;
        border-radius: 20px !important;
        color: #ffffff !important;
        padding: 4px 12px !important;
        margin: 3px !important;
        transition: all 0.25s ease !important;
    }
    
    div[data-baseweb="tag"]:hover {
        background-color: #2c2c32 !important;
        border-color: #ffa029 !important;
    }

    div[data-baseweb="tag"] span {
        color: #ffffff !important;
    }

    div[data-baseweb="tag"] svg {
        fill: #8e8e93 !important;
        transition: fill 0.2s ease !important;
    }

    div[data-baseweb="tag"] svg:hover {
        fill: #ff453a !important;
    }

    /* Dropdown option menu styling */
    div[role="listbox"], [data-baseweb="menu"] {
        background-color: #151518 !important;
        border: 1px solid #28282c !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    }

    [data-baseweb="menu"] [role="option"] {
        background-color: transparent !important;
        color: #ffffff !important;
        transition: all 0.15s ease !important;
        padding: 8px 16px !important;
    }

    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"][aria-selected="true"] {
        background-color: #222226 !important;
        color: #a5ff50 !important;
    }

    /* Number Input Buttons override */
    div[data-testid="stNumberInput"] button {
        background-color: #222226 !important;
        color: #ffffff !important;
        border: 1px solid #28282c !important;
        border-radius: 50% !important;
        width: 28px !important;
        height: 28px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 2px !important;
        padding: 0 !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background-color: #a5ff50 !important;
        color: #0b0b0c !important;
        border-color: #a5ff50 !important;
        transform: scale(1.1) !important;
    }

    /* Hide default borders & backgrounds of outer input boxes */
    .stTextInput > div, .stNumberInput > div, .stSelectbox > div, .stMultiSelect > div {
        background-color: transparent !important;
        border: none !important;
    }

    /* Metric Containers (Matched to Card style) */
    [data-testid="stMetric"] {
        background: #151518 !important;
        border: 1px solid #222226 !important;
        border-radius: 24px !important;
        padding: 24px !important;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: #ffa029 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.03em !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8e8e93 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Horizontal Segmented controls for Tab bars */
    [data-baseweb="tab-list"] {
        background-color: #151518 !important;
        border-radius: 30px !important;
        padding: 6px !important;
        border: none !important;
        gap: 6px !important;
    }
    
    [data-baseweb="tab"] {
        border-radius: 24px !important;
        padding: 10px 22px !important;
        color: #8e8e93 !important;
        font-weight: 600 !important;
        border: none !important;
        background: none !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #222226 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* Hide the red underline bar on selected tabs */
    div[data-baseweb="tab-highlight-id"] {
        background-color: transparent !important;
        display: none !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: transparent !important;
        display: none !important;
    }
    [data-baseweb="tab-list"]::after {
        background-color: transparent !important;
        display: none !important;
    }

    /* Radio buttons styled as segmented control pills */
    div[data-testid="stRadio"] > div {
        gap: 8px !important;
    }
    div[data-testid="stRadio"] label {
        background-color: #151518 !important;
        border: 1px solid #28282c !important;
        border-radius: 20px !important;
        padding: 8px 18px !important;
        margin-right: 4px !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }
    div[data-testid="stRadio"] label:hover {
        border-color: #a5ff50 !important;
        background-color: #1c1c21 !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #222226 !important;
        border-color: #a5ff50 !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) p {
        color: #a5ff50 !important;
        font-weight: 700 !important;
    }
    /* Hide default radio circle icon */
    div[data-testid="stRadio"] label span[role="presentation"],
    div[data-testid="stRadio"] label div[role="presentation"] {
        display: none !important;
    }

    /* Expandable Advanced configurations headers */
    [data-testid="stExpander"] {
        background-color: #151518 !important;
        border: 1px solid #28282c !important;
        border-radius: 20px !important;
        overflow: hidden !important;
    }

    /* Rounded Pill Action Button - Vibrant Gradient */
    div.stButton > button {
        background: linear-gradient(135deg, #ffa029 0%, #ff453a 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 20px rgba(255, 160, 41, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100% !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(255, 160, 41, 0.4) !important;
    }

    /* Matte Dark Cards for recommendations list */
    .prediction-card {
        background: #151518 !important;
        border: 1px solid #222226 !important;
        border-radius: 24px !important;
        padding: 26px !important;
        margin-bottom: 18px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    .prediction-card:hover {
        transform: translateY(-4px) !important;
        border-color: #a5ff50 !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 0 0 15px rgba(165, 255, 80, 0.1) !important;
    }

    /* High-contrast Pill Badges */
    .custom-badge {
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        text-align: center;
    }
    
    .safe-badge {
        background-color: rgba(165, 255, 80, 0.08) !important;
        color: #a5ff50 !important;
        border: 1px solid rgba(165, 255, 80, 0.2) !important;
    }
    
    .target-badge {
        background-color: rgba(255, 160, 41, 0.08) !important;
        color: #ffa029 !important;
        border: 1px solid rgba(255, 160, 41, 0.2) !important;
    }
    
    .dream-badge {
        background-color: rgba(255, 69, 58, 0.08) !important;
        color: #ff453a !important;
        border: 1px solid rgba(255, 69, 58, 0.2) !important;
    }

    /* File Uploader area styling */
    [data-testid="stFileUploader"] {
        background-color: #151518 !important;
        border: 1px dashed #28282c !important;
        border-radius: 24px !important;
        padding: 20px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #a5ff50 !important;
        background-color: #1c1c21 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #222226 !important;
        color: #ffffff !important;
        border: 1px solid #28282c !important;
        border-radius: 20px !important;
        padding: 6px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #28282c !important;
        color: #a5ff50 !important;
        border-color: #a5ff50 !important;
    }

    /* Chat bubble enhancements */
    [data-testid="stChatMessage"] {
        background-color: #151518 !important;
        border-radius: 20px !important;
        border: 1px solid #222226 !important;
        margin-bottom: 12px !important;
        padding: 16px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #1c1c21 !important;
        border-color: #2c2c32 !important;
    }

    /* Signal indicator */
    .pulse-indicator {
        width: 10px;
        height: 10px;
        background: #a5ff50;
        border-radius: 50%;
        display: inline-block;
        margin-right: 12px;
        box-shadow: 0 0 12px #a5ff50;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.6; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# File Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "kcet_data.db")

# Default API Keys (User's working credentials)
DEFAULT_MISTRAL_KEY = "IO4gZPwvajIzQGg0dUFxYJpkGSm4pxDc"
DEFAULT_PINECONE_KEY = "pcsk_6KkZ5p_QBPf37BqwgnCBo7WU4HPkuwnjiBsaAKeQgU5zxfGXQAbTzamjQE1EUXoZzRAqvu"
DEFAULT_PINECONE_INDEX = "raggg"

# Course Map for Parsing
COURSE_MAP = {
    "ARTIFICIAL INTELLIGENCE": "AI",
    "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE": "AI",
    "ARTIFICIAL INTELLIGENCE & MACHINE LEARNING": "AI",
    "BIO TECHNOLOGY": "BT",
    "CIVIL": "CE",
    "CIVIL ENGINEERING": "CE",
    "COMPUTER SCIENCE AND ENGINEERING": "CS",
    "COMPUTER SCIENCE & ENGINEERING": "CS",
    "COMPUTERS": "CS",
    "COMPUTER SCIENCE AND DESIGN": "CD",
    "COMPUTER SCIENCE & DESIGN": "CD",
    "CS- CYBER SECURITY": "CY",
    "CS - CYBER SECURITY": "CY",
    "COMP. SC. ENGG- DATA SC.": "DS",
    "COMPUTER SCIENCE AND BUSINESS SYSTEMS": "CB",
    "ELECTRICAL & ELECTRONICS ENGINEERING": "EE",
    "ELECTRICAL AND ELECTRONICS ENGINEERING": "EE",
    "ELECTRICAL": "EE",
    "ELECTRONICS AND COMMUNICATION ENGG": "EC",
    "ELECTRONICS & COMMUNICATION ENGINEERING": "EC",
    "ELECTRONICS": "EC",
    "ELECTRONICS AND INSTRUMENTATION ENGINEERING": "EI",
    "ELEC. INST. ENGG": "EI",
    "ELECTRONICS AND TELECOMMUNICATION ENGINEERING": "ET",
    "ELEC. TELECOMMN. ENGG.": "ET",
    "INFORMATION SCIENCE AND ENGINEERING": "IS",
    "INFORMATION SCIENCE & ENGINEERING": "IS",
    "INFO.SCIENCE": "IS",
    "INDUSTRIAL ENGINEERING & MANAGEMENT": "IM",
    "IND. ENGG. MGMT.": "IM",
    "MECHANICAL": "ME",
    "MECHANICAL ENGINEERING": "ME",
    "AERO SPACE ENGG.": "SE",
    "AEROSPACE ENGINEERING": "SE"
}

# Setup custom Pinecone embedding class to truncate dimension to 512
class TruncatedEmbeddings:
    def __init__(self, base_embeddings, target_dim=512):
        self.base_embeddings = base_embeddings
        self.target_dim = target_dim
        
    def embed_documents(self, texts):
        vectors = self.base_embeddings.embed_documents(texts)
        return [v[:self.target_dim] for v in vectors]
        
    def embed_query(self, text):
        vector = self.base_embeddings.embed_query(text)
        return vector[:self.target_dim]

# Initialize Session State
if "rank" not in st.session_state:
    st.session_state.rank = 5000
if "category" not in st.session_state:
    st.session_state.category = "GM"
if "selected_branches" not in st.session_state:
    st.session_state.selected_branches = ["CS", "IS", "EC", "AI"]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your **KCET Admission Assistant**. To help you find college options, please set your rank & category in the sidebar or ask me directly (e.g. *'I have rank 12000 in GM, what options do I have?'*)."}
    ]

# Parameter Extractor from Text
def extract_parameters(text):
    rank_match = re.search(r"\b([1-9]\d{2,5})\b", text)
    rank = int(rank_match.group(1)) if rank_match else None
    
    cat_match = re.search(r"\b(GM|GMR|GMK|1G|2AG|2AR|2BG|3AG|3BG|SCG|STG)\b", text, re.IGNORECASE)
    category = cat_match.group(1).upper() if cat_match else None
    
    courses = []
    if re.search(r"\b(CS|CSE|computer)\b", text, re.IGNORECASE):
        courses.append("CS")
    if re.search(r"\b(IS|ISE|information)\b", text, re.IGNORECASE):
        courses.append("IS")
    if re.search(r"\b(EC|ECE|electronics)\b", text, re.IGNORECASE):
        courses.append("EC")
    if re.search(r"\b(AI|ML|artificial|machine)\b", text, re.IGNORECASE):
        courses.append("AI")
        
    return rank, category, courses

# Robust course parsing using layout codes and keyword matching
def clean_and_extract_course(course_cell):
    text = course_cell.replace("\n", " ").strip()
    
    # 1. Regex to check if it starts with a 2-3 letter code followed by space or text
    # e.g. "CS Computers", "CY CS- Cyber Security", "AI Artificial Intelligence", "IE Info.Science"
    match = re.match(r"^([A-Z]{2,3})\b\s*(.*)$", text)
    if match:
        code = match.group(1)
        name = match.group(2).strip()
        if code == "IE":
            code = "IS"
        if code == "CA":
            code = "AI"
        if name == "":
            name = text
        return code, name.title()
        
    text_upper = text.upper()
    
    # 2. Keyword check mapping to normalise course codes
    if "COMPUTER" in text_upper or "COMPUTERS" in text_upper or "COMP.SC" in text_upper:
        return "CS", text.title()
    if "INFORMATION" in text_upper or "INFO.SCIENCE" in text_upper or "INFO SCIENCE" in text_upper:
        return "IS", text.title()
    if "ELECTRONICS" in text_upper or "ELECTRONIC" in text_upper or "ELECT. COMM" in text_upper or "COMMUNICATIO" in text_upper:
        return "EC", text.title()
    if "ARTIFICIAL" in text_upper or "MACHINE LEARNING" in text_upper or "AI & ML" in text_upper or "AI/ML" in text_upper:
        return "AI", text.title()
    if "ELECTRICAL" in text_upper:
        return "EE", text.title()
    if "CIVIL" in text_upper:
        return "CE", text.title()
    if "MECHANICAL" in text_upper:
        return "ME", text.title()
    if "BIOTECH" in text_upper or "BIO TECHNOLOGY" in text_upper or "BIO-TECHNOLOGY" in text_upper:
        return "BT", text.title()
    if "SILK" in text_upper:
        return "ST", text.title()
    if "TEXTILE" in text_upper:
        return "TX", text.title()
    if "INDUSTRIAL" in text_upper:
        return "IM", text.title()
    if "AERO" in text_upper:
        return "AE", text.title()
    if "CHEMICAL" in text_upper:
        return "CH", text.title()
    if "AUTOMOBILE" in text_upper:
        return "AU", text.title()
        
    first_word = text.split(" ")[0]
    if first_word.isupper() and 2 <= len(first_word) <= 3:
        return first_word, text.title()
        
    return "XX", text.title()

# Layout-aware PDF Cutoff Parsing Engine
def parse_and_store_pdf(uploaded_file, year, round_num, api_keys_dict, progress_bar, status_text):
    # Save file to temp disk
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    status_text.text("Initialising SQLite connection...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS college_cutoffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            round INTEGER NOT NULL,
            college_code TEXT NOT NULL,
            college_name TEXT NOT NULL,
            course_code TEXT NOT NULL,
            course_name TEXT NOT NULL,
            category TEXT NOT NULL,
            cutoff_rank INTEGER NOT NULL
        )
    """)
    conn.commit()
    
    records_added = 0
    cutoff_inserts = []
    documents = []
    
    # Determine header regex pattern based on year
    if year == 2024:
        college_header_pattern = re.compile(r"^(\d+)\s+([A-Z]\d{3})\s+(.+)$")
        allotment_str = "Mock 1 Allotment"
    else:
        college_header_pattern = re.compile(r"^College:\s+([A-Z]\d{3})\s+(.+)$")
        allotment_str = "Session 2 Allotment"
        
    try:
        with pdfplumber.open(temp_pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for p_idx, page in enumerate(pdf.pages):
                status_text.text(f"Extracting layout-aware tables from page {p_idx+1}/{total_pages}...")
                progress_bar.progress((p_idx + 1) / total_pages)
                
                # Extract words to reconstruct lines geometrically
                words = page.extract_words()
                lines = {}
                for w in words:
                    top = round(w["top"], 0)
                    if top not in lines:
                        lines[top] = []
                    lines[top].append(w)
                    
                sorted_lines = []
                for top in sorted(lines.keys()):
                    line_words = sorted(lines[top], key=lambda x: x["x0"])
                    line_text = " ".join([w["text"] for w in line_words])
                    sorted_lines.append((top, line_text))
                    
                # Cache headers on current page
                headers = []
                for top, text in sorted_lines:
                    match = college_header_pattern.match(text.strip())
                    if match:
                        if year == 2024:
                            headers.append({"top": top, "code": match.group(2), "name": match.group(3)})
                        else:
                            headers.append({"top": top, "code": match.group(1), "name": match.group(2)})
                            
                # Extract tables
                tables = page.find_tables()
                for table in tables:
                    x0, top, x1, bottom = table.bbox
                    
                    # Match to closest header above table
                    closest_header = None
                    for h in headers:
                        if h["top"] < top:
                            if closest_header is None or h["top"] > closest_header["top"]:
                                closest_header = h
                                
                    if not closest_header:
                        continue
                        
                    grid = table.extract()
                    categories = []
                    for row in grid:
                        row = [str(cell).strip() if cell else "" for cell in row]
                        if not row or not any(row):
                            continue
                            
                        if "1G" in row or "GM" in row:
                            categories = row[1:]
                            continue
                            
                        if categories and row[0] and row[0] not in ["Course Name", "CourseName"] and not row[0].startswith("KARNATAKA"):
                            course_cell = row[0]
                            
                            # Parse code & name using robust parser logic
                            course_code, course_name = clean_and_extract_course(course_cell)
                                
                            ranks = row[1:]
                            for cat_idx, rank_str in enumerate(ranks):
                                if cat_idx < len(categories):
                                    category = categories[cat_idx]
                                    if rank_str and rank_str != "--":
                                        clean_rank = re.sub(r"[^\d]", "", rank_str.split(".")[0])
                                        if clean_rank:
                                            val_rank = int(clean_rank)
                                            if val_rank <= 250000:
                                                cutoff_inserts.append((
                                                    year, round_num, closest_header["code"], closest_header["name"],
                                                    course_code, course_name.title(), category, val_rank
                                                ))
                                                records_added += 1
                                                
                                                # Create RAG semantic chunk document
                                                text_desc = f"In {year}, for {allotment_str}, {closest_header['name']} ({closest_header['code']}) for the course {course_name} ({course_code}) has a cutoff rank of {val_rank} for Category {category}."
                                                metadata = {
                                                    "college_code": closest_header["code"],
                                                    "college_name": closest_header["name"],
                                                    "course_code": course_code,
                                                    "course_name": course_name,
                                                    "category": category,
                                                    "cutoff_rank": val_rank,
                                                    "year": year,
                                                    "round": round_num
                                                }
                                                documents.append(Document(page_content=text_desc, metadata=metadata))
                                                
                # Write batches to SQLite to save memory
                if len(cutoff_inserts) >= 500:
                    cursor.executemany("""
                        INSERT INTO college_cutoffs (year, round, college_code, college_name, course_code, course_name, category, cutoff_rank)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, cutoff_inserts)
                    conn.commit()
                    cutoff_inserts = []
                    
        # Flush remaining SQL rows
        if cutoff_inserts:
            cursor.executemany("""
                INSERT INTO college_cutoffs (year, round, college_code, college_name, course_code, course_name, category, cutoff_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, cutoff_inserts)
            conn.commit()
            
        conn.close()
        
        # Optionally upload to Pinecone index if API credentials are functional
        m_key = api_keys_dict.get("mistral")
        p_key = api_keys_dict.get("pinecone")
        p_idx = api_keys_dict.get("index")
        
        if m_key and p_key and p_idx and documents:
            # Set environment variables dynamically for LangChain client detection
            os.environ["MISTRAL_API_KEY"] = m_key
            os.environ["PINECONE_API_KEY"] = p_key
            
            status_text.text("Connecting and uploading embedded chunks to Pinecone Vector Index...")
            try:
                base_embeddings = MistralAIEmbeddings(api_key=m_key)
                embeddings = TruncatedEmbeddings(base_embeddings, target_dim=512)
                
                # Upload in batches
                batch_size = 150
                for idx in range(0, len(documents), batch_size):
                    batch = documents[idx:idx+batch_size]
                    status_text.text(f"Uploading vectors: batch {idx // batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
                    PineconeVectorStore.from_documents(
                        batch,
                        embeddings,
                        index_name=p_idx,
                        pinecone_api_key=p_key
                    )
                status_text.text("SQLite database and Pinecone Vector Store successfully updated!")
            except Exception as ex:
                st.warning(f"Failed to ingest to Pinecone Vector DB, but data saved to SQLite. Error: {ex}")
        else:
            status_text.text("Finished! SQLite Database successfully updated with new cutoffs.")
            
    except Exception as e:
        st.error(f"Error during PDF processing: {e}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
    return records_added


# Setup DB query prediction logic
def calculate_recommendations(rank, category, preferred_courses, seat_matrix_toggle, max_cutoff_factor=None):
    if not os.path.exists(DB_PATH):
        return []
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query all matching records for the category and preferred course codes
    query = """
        SELECT year, round, college_code, college_name, course_code, course_name, cutoff_rank 
        FROM college_cutoffs 
        WHERE category = ? AND course_code IN ({})
    """.format(','.join(['?'] * len(preferred_courses)))
    
    cursor.execute(query, [category] + preferred_courses)
    rows = cursor.fetchall()
    conn.close()
    
    # Group results by college + course combination
    grouped = {}
    for row in rows:
        year, r_num, col_code, col_name, course_code, course_name, cutoff_val = row
        key = (col_code, col_name, course_code, course_name)
        if key not in grouped:
            grouped[key] = {"r1": None, "r2": None, "r3": None}
            
        if r_num == 1:
            grouped[key]["r1"] = cutoff_val
        elif r_num == 2:
            grouped[key]["r2"] = cutoff_val
        elif r_num == 3:
            grouped[key]["r3"] = cutoff_val
            
    predictions = []
    for key, rounds in grouped.items():
        col_code, col_name, course_code, course_name = key
        
        # Take latest available round cutoff
        r1 = rounds["r1"]
        r2 = rounds["r2"]
        r3 = rounds["r3"]
        
        target_raw = r3 if r3 else (r2 if r2 else r1)
        if not target_raw:
            continue
            
        # Dynamic Seat Matrix capacity buffer check
        # Add 10% cutoff allowance if CSE seat counts are expanded in RVCE (E005) or BMSCE (E003)
        cse_capacity_increased = (course_code == "CS" and col_code in ["E005", "E003"])
        if seat_matrix_toggle and cse_capacity_increased:
            adjusted_cutoff = int(target_raw * 1.10)
            adjustment_applied = True
        else:
            adjusted_cutoff = target_raw
            adjustment_applied = False
            
        # Exclude highly non-competitive options if max_cutoff_factor is specified
        if max_cutoff_factor is not None and adjusted_cutoff > rank * max_cutoff_factor:
            continue
            
        # Classify Option
        if rank <= adjusted_cutoff * 0.90:
            status = "Safe"
            explanation = f"Your rank {rank} is safely below the cutoff of {adjusted_cutoff}."
        elif rank <= adjusted_cutoff * 1.10:
            status = "Target"
            explanation = f"Your rank {rank} is within ±10% threshold of the cutoff {adjusted_cutoff}."
        else:
            status = "Dream"
            explanation = f"Your rank {rank} is higher than the cutoff of {adjusted_cutoff}."
            
        predictions.append({
            "college_code": col_code,
            "college_name": col_name,
            "course_code": course_code,
            "course_name": course_name,
            "round_1_cutoff": r1 if r1 else "N/A",
            "round_2_cutoff": r2 if r2 else "N/A",
            "round_3_cutoff": r3 if r3 else "N/A",
            "adjusted_cutoff": adjusted_cutoff,
            "status": status,
            "explanation": explanation,
            "adjustment_applied": adjustment_applied
        })
        
    return predictions


# Main Application Layout
st.markdown("<h1 style='text-align: center; margin-bottom: 0; font-family: \"Bebas Neue\", sans-serif !important; letter-spacing: 0.08em; font-size: 3.8rem; text-transform: uppercase;'>🎓 CET PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8e8e93; font-family: \"Plus Jakarta Sans\", sans-serif; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.82rem; font-weight: 700; margin-top: 5px; margin-bottom: 25px;'>ADMISSION & CUTOFF ANALYSIS DASHBOARD</p>", unsafe_allow_html=True)
st.divider()

# Parameter Input Panel (Horizontal layout at top of page)
st.markdown("### 🔍 Student Profile")
col1, col2, col3 = st.columns(3)
with col1:
    rank_input = st.number_input(
        "KCET Rank",
        min_value=1,
        max_value=250000,
        value=int(st.session_state.rank),
        step=500
    )
    st.session_state.rank = rank_input
with col2:
    categories = ["GM", "GMR", "GMK", "1G", "2AG", "2AR", "2BG", "3AG", "3BG", "SCG", "STG"]
    default_cat_idx = categories.index(st.session_state.category) if st.session_state.category in categories else 0
    cat_input = st.selectbox(
        "Category",
        options=categories,
        index=default_cat_idx
    )
    st.session_state.category = cat_input
with col3:
    branches_input = st.multiselect(
        "Preferred Branches (Order of Priority)",
        options=["CS", "IS", "EC", "AI"],
        default=st.session_state.selected_branches
    )
    if branches_input:
        st.session_state.selected_branches = branches_input
    else:
        st.session_state.selected_branches = ["CS", "IS", "EC", "AI"]

# Expandable Advanced Configurations
with st.expander("🛠️ Advanced Settings & Credentials (Optional)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        seat_matrix_toggle = st.toggle("Apply 2026 Seat-Matrix Buffers (+10%)", value=True)
    with c2:
        exclude_very_safe = st.checkbox("Filter out highly non-competitive options", value=True, help="Hides colleges whose cutoffs are extremely high compared to your rank (where admission is 100% guaranteed).")
        
    if exclude_very_safe:
        max_cutoff_factor = st.slider(
            "Only show options with cutoff ≤ X times your rank:",
            min_value=1.1,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="For example, if your rank is 18,000 and factor is 2.0x, only colleges with cutoffs up to 36,000 will be shown."
        )
    else:
        max_cutoff_factor = None
        
    st.divider()
    st.markdown("##### API Credentials")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        mistral_key = st.text_input("Mistral API Key", value=DEFAULT_MISTRAL_KEY, type="password")
    with ac2:
        pinecone_key = st.text_input("Pinecone API Key", value=DEFAULT_PINECONE_KEY, type="password")
    with ac3:
        pinecone_index = st.text_input("Pinecone Index Name", value=DEFAULT_PINECONE_INDEX)
        
    # Set environment variables dynamically for LangChain client detection
    if mistral_key:
        os.environ["MISTRAL_API_KEY"] = mistral_key
    if pinecone_key:
        os.environ["PINECONE_API_KEY"] = pinecone_key

st.divider()

# Tab Layout
predictor_tab, chatbot_tab, ingestor_tab = st.tabs([
    "📊 Predictor & Recommendations",
    "🤖 AI Admission Advisor",
    "📤 Cutoff PDF Uploader"
])

# Create the Predictions List
predictions = calculate_recommendations(
    st.session_state.rank,
    st.session_state.category,
    st.session_state.selected_branches,
    seat_matrix_toggle,
    max_cutoff_factor
)

# Sorting Function: Match Proximity-to-Rank Sorter
# Prioritizes options around the student's rank, grouped by dynamic proximity bins, and ordered by branch priority.
def sort_key(x):
    rank = st.session_state.rank
    # Dynamic bin size (10% of rank, minimum 1000 ranks)
    bin_size = max(1000, int(rank * 0.1))
    
    # Absolute difference between cutoff and user's rank
    diff = abs(x["adjusted_cutoff"] - rank)
    proximity_bin = int(diff / bin_size)
    
    # Sort branch priority
    branch = x["course_code"]
    b_val = st.session_state.selected_branches.index(branch) if branch in st.session_state.selected_branches else 99
    
    # Secondary sorting within bins: target matches (where cutoff >= rank) slightly ahead of dreams
    fit_val = 0 if x["adjusted_cutoff"] >= rank else 1
    
    return (proximity_bin, b_val, fit_val, diff)

predictions.sort(key=sort_key)

# TAB 1: Predictor Dashboard
with predictor_tab:
    st.markdown("### 📊 Recommendation Dashboard")
    st.write(
        f"Displaying college recommendations for **Rank {st.session_state.rank}** and **Category {st.session_state.category}**, "
        f"sorted by your branch priorities: **{', '.join(st.session_state.selected_branches)}**."
    )
    
    if not predictions:
        st.info("No matching records found. Please adjust your criteria, selected branches, or upload some cutoffs.")
    else:
        # Counters
        safes = [p for p in predictions if p["status"] == "Safe"]
        targets = [p for p in predictions if p["status"] == "Target"]
        dreams = [p for p in predictions if p["status"] == "Dream"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Safe Choices (Rank <= 90% Cutoff)", value=len(safes))
        with col2:
            st.metric(label="Target Choices (Within +-10%)", value=len(targets))
        with col3:
            st.metric(label="Dream Choices (Rank > 110% Cutoff)", value=len(dreams))
            
        st.markdown("#### Filtered Results")
        
        status_filter = st.radio(
            "Filter by Category Status:",
            options=["All", "Safe Options", "Target Options", "Dream Options"],
            horizontal=True
        )
        
        filtered_preds = []
        if status_filter == "All":
            filtered_preds = predictions
        elif status_filter == "Safe Options":
            filtered_preds = safes
        elif status_filter == "Target Options":
            filtered_preds = targets
        else:
            filtered_preds = dreams
            
        if not filtered_preds:
            st.warning("No choices match this filter status.")
        else:
            for p in filtered_preds:
                # Set dynamic badges & titles
                status = p["status"]
                if status == "Safe":
                    badge_html = "<span class='custom-badge safe-badge'>Safe Option</span>"
                elif status == "Target":
                    badge_html = "<span class='custom-badge target-badge'>Target Option</span>"
                else:
                    badge_html = "<span class='custom-badge dream-badge'>Dream Option</span>"
                    
                # Seat adjustment warning text
                adj_text = ""
                if p["adjustment_applied"]:
                    adj_text = " <span style='color: #00ff88; font-size: 0.85rem;'>(+10% Seat capacity buffer applied)</span>"
                    
                card_title = f"**{p['college_name']}** (`{p['college_code']}`)"
                card_body = f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #fff;">{card_title}</h4>
                        {badge_html}
                    </div>
                    <p style="margin: 0 0 8px 0; font-size: 0.95rem; color: #bbb;">
                        Course: <strong>{p['course_name']} ({p['course_code']})</strong> {adj_text}
                    </p>
                    <div style="display: flex; gap: 20px; font-size: 0.9rem; color: #aaa; margin-bottom: 8px;">
                        <span>Round 1 Cutoff: <strong>{p['round_1_cutoff']}</strong></span>
                        <span>Round 2 Cutoff: <strong>{p['round_2_cutoff']}</strong></span>
                        <span>Round 3 Cutoff: <strong>{p['round_3_cutoff']}</strong></span>
                        <span>Adjusted Cutoff: <strong style="color: #fff;">{p['adjusted_cutoff']}</strong></span>
                    </div>
                    <p style="margin: 0; font-size: 0.9rem; color: #888; font-style: italic;">
                        {p['explanation']}
                    </p>
                </div>
                """
                st.markdown(card_body, unsafe_allow_html=True)

# TAB 2: AI Advisor Chatbot
with chatbot_tab:
    st.markdown("### 🤖 Agentic Admission Assistant")
    st.caption("The chatbot automatically extracts ranks/categories from your messages to sync the sidebar filters dynamically!")
    
    # Display Chat Messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # User Input
    if user_prompt := st.chat_input("Ask a question (e.g. 'My rank is 6500 in 3BG. What options do I have in CS?')"):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        # Extract parameters and update state
        new_rank, new_cat, new_courses = extract_parameters(user_prompt)
        state_updated = False
        if new_rank:
            st.session_state.rank = new_rank
            state_updated = True
        if new_cat:
            st.session_state.category = new_cat
            state_updated = True
        if new_courses:
            st.session_state.selected_branches = new_courses
            state_updated = True
            
        # Re-query options based on current session state parameters
        current_predictions = calculate_recommendations(
            st.session_state.rank,
            st.session_state.category,
            st.session_state.selected_branches,
            seat_matrix_toggle,
            max_cutoff_factor
        )
        current_predictions.sort(key=sort_key)
        
        # Build context prompt text from matching SQLite rows
        matched_details = []
        for p in current_predictions[:10]:
            matched_details.append(
                f"- {p['college_name']} ({p['college_code']}), Course: {p['course_name']} ({p['course_code']}). "
                f"Round 1: {p['round_1_cutoff']}, Round 2: {p['round_2_cutoff']}, Round 3: {p['round_3_cutoff']}, "
                f"Adjusted: {p['adjusted_cutoff']}, Status: {p['status']}."
            )
        context_str = "\n".join(matched_details)
        
        # Call RAG LLM or fallback to local SQLite summarizer if keys are absent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing recommendations..."):
                if mistral_key and context_str:
                    try:
                        # Full Vector-RAG chatbot response using Mistral AI
                        prompt_template = ChatPromptTemplate.from_template("""
                        You are a premium, expert KCET Admission Recommendation assistant.
                        Answer the student's question concisely using only the matching historical data (context) below.
                        
                        Details on student matches:
                        {context}
                        
                        Question: {question}
                        Current Rank: {rank}
                        Current Category: {category}
                        
                        Address the student directly. Highlight which options are Safe, Target, or Dream, and suggest priorities. Keep your answer professional, clean, and concise.
                        """)
                        llm = ChatMistralAI(model="mistral-large-latest", api_key=mistral_key)
                        chain = prompt_template | llm | StrOutputParser()
                        response = chain.invoke({
                            "context": context_str,
                            "question": user_prompt,
                            "rank": st.session_state.rank,
                            "category": st.session_state.category
                        })
                    except Exception as e:
                        # Fallback response in case of API exception
                        response = f"I retrieved the cutoffs from SQLite, but encountered an API issue calling Mistral AI ({e}).\n\nHere are the top matches from database: \n\n" + context_str
                else:
                    # SQLite Rule-based precise fallback generator
                    response = f"Here is what I found in the database for **Rank {st.session_state.rank}** (Category **{st.session_state.category}**):\n\n"
                    
                    safes = [p for p in current_predictions if p["status"] == "Safe"]
                    targets = [p for p in current_predictions if p["status"] == "Target"]
                    dreams = [p for p in current_predictions if p["status"] == "Dream"]
                    
                    if safes:
                        response += "🟢 **Safe Options**:\n"
                        for s in safes[:3]:
                            response += f"- {s['college_name']} - {s['course_name']} (Adj. Cutoff: {s['adjusted_cutoff']})\n"
                    if targets:
                        response += "\n🟡 **Target Options**:\n"
                        for t in targets[:3]:
                            response += f"- {t['college_name']} - {t['course_name']} (Adj. Cutoff: {t['adjusted_cutoff']})\n"
                    if dreams:
                        response += "\n🔴 **Dream Options** (Reach):\n"
                        for d in dreams[:3]:
                            response += f"- {d['college_name']} - {d['course_name']} (Adj. Cutoff: {d['adjusted_cutoff']})\n"
                            
                    if not current_predictions:
                        response = f"I couldn't find any historical cutoffs matching Category **{st.session_state.category}** and courses **{st.session_state.selected_branches}** in my database."
                    else:
                        response += "\n*Note: Set your API key in the sidebar to unlock conversational LLM explanations.*"
                        
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
        # Rerun to sync sidebar parameters instantly if changed in chat
        if state_updated:
            st.rerun()

# TAB 3: PDF Uploader & Ingestor
with ingestor_tab:
    st.markdown("### 📤 Upload New Cutoff Sheets")
    st.write("Upload official government KCET PDF cutoff sheets to expand the recommendation database dynamically.")
    
    uploaded_pdfs = st.file_uploader("Choose KCET Cutoff PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs:
        st.markdown("#### Configure Uploaded Files")
        st.write("Review or adjust the Year and Round for each file before parsing:")
        
        configs = []
        for idx, pdf in enumerate(uploaded_pdfs):
            filename_lower = pdf.name.lower()
            default_round = 1
            if "round2" in filename_lower or "round_2" in filename_lower or "r2" in filename_lower or "session2" in filename_lower:
                default_round = 2
            elif "round3" in filename_lower or "round_3" in filename_lower or "r3" in filename_lower:
                default_round = 3
            elif "mock" in filename_lower:
                default_round = 1
                
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"📄 **{pdf.name}**")
            with col2:
                year = st.number_input("Year", min_value=2020, max_value=2030, value=2025, key=f"year_{pdf.name}_{idx}")
            with col3:
                round_choice = st.selectbox("Round", options=[1, 2, 3], index=default_round-1, key=f"round_{pdf.name}_{idx}")
                
            configs.append({
                "file": pdf,
                "year": year,
                "round": round_choice
            })
            st.divider()
            
        if st.button("Parse and Ingest All PDFs"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            total_files = len(configs)
            total_records = 0
            
            keys_dict = {
                "mistral": mistral_key,
                "pinecone": pinecone_key,
                "index": pinecone_index
            }
            
            for file_idx, config in enumerate(configs):
                pdf_file = config["file"]
                y_val = config["year"]
                r_val = config["round"]
                
                status_text.text(f"Processing file {file_idx+1}/{total_files}: {pdf_file.name}...")
                
                records = parse_and_store_pdf(pdf_file, y_val, r_val, keys_dict, progress_bar, status_text)
                total_records += records
                
            if total_records > 0:
                st.success(f"Success! Ingested a total of {total_records} cutoff entries across {total_files} file(s).")
                # Refresh recommendations
                st.rerun()
            else:
                st.error("No valid cutoff entries could be parsed from any of the uploaded PDFs. Verify the PDF format conforms to KEA layout standards.")

# Premium Project Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #8e8e93; font-size: 0.85rem; margin-top: 40px; margin-bottom: 20px;">
    <h4 style="color: #8e8e93 !important; font-weight: 600; margin-bottom: 8px;">About KCET Predictor</h4>
    <p style="margin: 0; color: #8e8e93 !important;">This admission assistant analyzes historical cutoffs using a layout-aware geometric PDF parser. All predictions are generated in real-time using local SQLite analytics combined with optional hybrid vector-RAG logic powered by Mistral & Pinecone.</p>
    <p style="margin: 8px 0 0 0; color: #636366 !important; font-size: 0.8rem;">© 2026 KCET Predictor System. Designed with Apple Dark Aesthetics.</p>
</div>
""", unsafe_allow_html=True)
