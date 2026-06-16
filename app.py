from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import json
import re
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

app = Flask(__name__, static_folder="static", template_folder="templates")
DB_PATH = os.path.join(os.path.dirname(__file__), "kcet_data.db")

# Fallback Credentials
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "IO4gZPwvajIzQGg0dUFxYJpkGSm4pxDc")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_6KkZ5p_QBPf37BqwgnCBo7WU4HPkuwnjiBsaAKeQgU5zxfGXQAbTzamjQE1EUXoZzRAqvu")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "raggg")

# Set environment variables for LangChain client detection
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Custom Embedding Wrapper to handle Pinecone index dimension constraints (512 dimensions)
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

# Setup LLM based on available keys
def get_llm():
    print("Using Mistral AI Large Model...")
    return ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)

# Query seats for adjustment
def get_seat_change_pct(college_code, course_code, category):
    if not os.path.exists(DB_PATH):
        return 0.0
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT year, total_seats FROM seat_matrix 
            WHERE college_code = ? AND course_code = ? AND category = ? AND year IN (2025, 2026)
        """, (college_code, course_code, category))
        rows = cursor.fetchall()
        conn.close()
        
        seats = {row[0]: row[1] for row in rows}
        if 2025 in seats and 2026 in seats:
            s25 = seats[2025]
            s26 = seats[2026]
            if s26 > s25:
                return (s26 - s25) / s25
    except Exception as e:
        print(f"Seat matrix lookup error: {e}")
    return 0.0

# Retrieve predictions from Vector DB via RAG pipeline or SQLite fallback
def run_rag_prediction(rank, category, courses):
    # Check if we have API credentials. If not, use local SQLite lookup for 100% accurate cutoff matching
    has_credentials = MISTRAL_API_KEY and PINECONE_API_KEY
    
    if not has_credentials:
        print("Running in SQLite Database Mode (No API keys present)...")
        if not os.path.exists(DB_PATH):
            return []
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Select cutoffs matching category and preferred branches
        query = """
            SELECT year, round, college_code, college_name, course_code, course_name, cutoff_rank 
            FROM college_cutoffs 
            WHERE category = ? AND course_code IN ({})
        """.format(','.join(['?'] * len(courses)))
        
        cursor.execute(query, [category] + courses)
        rows = cursor.fetchall()
        conn.close()
        
        # Group duplicates by College + Course (taking values across years and rounds)
        grouped = {}
        for row in rows:
            year, r_num, col_code, col_name, course_code, course_name, cutoff_val = row
            key = (col_code, col_name, course_code, course_name)
            if key not in grouped:
                grouped[key] = {"r1": 0, "r2": 0, "r3": 0}
            
            # Map rounds (Round 1, 2, 3)
            if r_num == 1:
                grouped[key]["r1"] = cutoff_val
            elif r_num == 2:
                grouped[key]["r2"] = cutoff_val
            elif r_num == 3:
                grouped[key]["r3"] = cutoff_val
                
        predictions = []
        for key, rounds in grouped.items():
            col_code, col_name, course_code, course_name = key
            
            # We look for the latest round cutoff
            r1_raw = rounds["r1"]
            r2_raw = rounds["r2"]
            r3_raw = rounds["r3"]
            
            target_raw = r3_raw if r3_raw else (r2_raw if r2_raw else r1_raw)
            if target_raw == 0:
                continue
                
            # Apply dynamic seat adjustments
            is_cse_increased = (course_code == "CS" and col_code in ["E005", "E003"])
            adjusted_cutoff = int(target_raw * 1.10) if is_cse_increased else target_raw
            
            if rank <= adjusted_cutoff * 0.90:
                status = "Safe"
                explanation = f"Your rank {rank} is safely below the cutoff of {adjusted_cutoff}."
                badge_color = "safe-badge"
            elif rank <= adjusted_cutoff * 1.10:
                status = "Target"
                explanation = f"Your rank {rank} is within ±10% threshold of the cutoff {adjusted_cutoff}."
                badge_color = "target-badge"
            else:
                status = "Dream"
                explanation = f"Your rank {rank} is higher than the cutoff of {adjusted_cutoff}."
                badge_color = "dream-badge"
                
            predictions.append({
                "college_code": col_code,
                "college_name": col_name,
                "course_code": course_code,
                "course_name": course_name,
                "round_1_cutoff": r1_raw if r1_raw else "N/A",
                "round_2_cutoff": r2_raw if r2_raw else "N/A",
                "round_3_cutoff": r3_raw if r3_raw else "N/A",
                "adjusted_cutoff": adjusted_cutoff,
                "status": status,
                "explanation": explanation,
                "badge_color": badge_color
            })
            
        status_order = {"Safe": 0, "Target": 1, "Dream": 2}
        predictions.sort(key=lambda x: (status_order.get(x.get("status", "Dream"), 2), x.get("college_name", "")))
        return predictions

    # --- REAL MISTRAL + PINECONE RAG PIPELINE ---
    print(f"Connecting to Pinecone Index '{PINECONE_INDEX}' (512-dim truncated)...")
    base_embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
    embeddings = TruncatedEmbeddings(base_embeddings, target_dim=512)
    
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY
    )
    
    query_str = f"Cutoff ranks for Category {category} in courses {', '.join(courses)}"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    docs = retriever.get_relevant_documents(query_str)
    
    relevant_docs = []
    for doc in docs:
        course_code = doc.metadata.get("course_code")
        doc_category = doc.metadata.get("category")
        if course_code in courses and doc_category == category:
            relevant_docs.append(doc)
            
    if not relevant_docs:
        relevant_docs = docs[:15]

    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = ChatPromptTemplate.from_template("""
You are a expert KCET Admission Recommendation agent.
Based on the student's rank of {rank}, category of {category}, and the following historical college cutoff details (context):
{context}

Classify each option into a status:
- "Safe": If the student's rank is at or below 90% of the Round 3 cutoff rank.
- "Target": If the student's rank is between 90% and 110% of the Round 3 cutoff rank.
- "Dream": If the student's rank is above 110% of the Round 3 cutoff rank.

Adjust the Round 3 cutoff (Seat Matrix shift adjustment) by adding 10% to the cutoff if it is a Computer Science (CS) branch at RVCE (E005) or BMSCE (E003) due to increased seat capacity.

Format your output STRICTLY as a raw JSON array of objects. Do not include markdown code block backticks (like ```json). Return only the JSON string.
Schema:
[
  {{
    "college_code": "College code (e.g. E005)",
    "college_name": "College name",
    "course_code": "Branch code (e.g. CS)",
    "course_name": "Branch name",
    "round_1_cutoff": 123,
    "round_2_cutoff": 145,
    "round_3_cutoff": 180,
    "adjusted_cutoff": 198,
    "status": "Safe / Target / Dream",
    "explanation": "Brief explanation describing how the rank compares to the cutoff.",
    "badge_color": "safe-badge / target-badge / dream-badge"
  }}
]
""")

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    try:
        response_str = chain.invoke({
            "context": context,
            "rank": rank,
            "category": category
        })
        
        response_str = response_str.strip()
        if response_str.startswith("```"):
            response_str = re.sub(r"^```(json)?\n", "", response_str)
            response_str = re.sub(r"\n```$", "", response_str)
            
        predictions = json.loads(response_str)
        status_order = {"Safe": 0, "Target": 1, "Dream": 2}
        predictions.sort(key=lambda x: (status_order.get(x.get("status", "Dream"), 2), x.get("college_name", "")))
        return predictions
    except Exception as e:
        print(f"RAG Chain Error: {e}")
        return []

# Helper to extract parameters from text messages
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
        
    if not courses:
        courses = ["CS", "IS", "EC", "AI"]
        
    return rank, category, courses

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    rank = data.get("rank")
    category = data.get("category")
    courses = data.get("courses", ["CS", "IS", "EC", "AI"])
    
    if not rank or not category:
        return jsonify({"error": "Rank and Category are required fields."}), 400
        
    predictions = run_rag_prediction(int(rank), category.upper(), courses)
    return jsonify({"predictions": predictions})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "")
    state = data.get("state", {})
    
    rank = state.get("rank")
    category = state.get("category")
    courses = state.get("courses", [])
    
    new_rank, new_category, new_courses = extract_parameters(message)
    
    if new_rank:
        rank = new_rank
    if new_category:
        category = new_category
    if new_courses and len(new_courses) < 4:
        courses = new_courses
        
    state["rank"] = rank
    state["category"] = category
    if courses:
        state["courses"] = courses
        
    if not rank:
        reply = "Hello! I am your **KCET Admission Assistant**. To help you find college cutoffs, please tell me your **KCET Rank** (e.g., *'My rank is 4500'*)."
        return jsonify({"reply": reply, "state": state, "predictions": []})
        
    if not category:
        reply = f"Got it! Rank **{rank}**. What is your **Category** (e.g., *GM, 2AG, 3BG, SCG, STG*)?"
        return jsonify({"reply": reply, "state": state, "predictions": []})
        
    if not courses:
        courses = ["CS", "IS", "EC", "AI"]
        state["courses"] = courses
        
    predictions = run_rag_prediction(rank, category, courses)
    
    course_str = ", ".join(courses)
    reply = f"Perfect! Here are your college recommendations based on RAG similarity retrieval for **Rank {rank}**, Category **{category}**, and Courses **[{course_str}]**. See the analysis cards below!"
    
    return jsonify({
        "reply": reply,
        "state": state,
        "predictions": predictions
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
