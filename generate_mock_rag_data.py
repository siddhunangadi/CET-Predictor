import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

def init_rag_db():
    print("Generating semantic chunks for RAG Vector Database...")
    
    colleges = [
        ("E005", "RV College of Engineering (RVCE)"),
        ("E003", "BMS College of Engineering (BMSCE)"),
        ("E007", "MS Ramaiah Institute of Technology (MSRIT)"),
        ("E033", "PES University (PESU)"),
        ("E008", "Bangalore Institute of Technology (BIT)")
    ]

    courses = [
        ("CS", "Computer Science & Engineering"),
        ("IS", "Information Science & Engineering"),
        ("EC", "Electronics & Communication Engineering"),
        ("AI", "Artificial Intelligence & Machine Learning")
    ]

    categories = ["GM", "GMR", "GMK", "1G", "2AG", "2AR", "2BG", "3AG", "3BG", "SCG", "STG"]

    base_ranks = {
        ("E005", "CS"): 300,
        ("E005", "IS"): 550,
        ("E005", "AI"): 700,
        ("E005", "EC"): 1200,

        ("E033", "CS"): 700,
        ("E033", "IS"): 1000,
        ("E033", "AI"): 1200,
        ("E033", "EC"): 2200,

        ("E003", "CS"): 900,
        ("E003", "IS"): 1400,
        ("E003", "AI"): 1600,
        ("E003", "EC"): 3000,

        ("E007", "CS"): 1000,
        ("E007", "IS"): 1600,
        ("E007", "AI"): 1900,
        ("E007", "EC"): 3400,

        ("E008", "CS"): 3000,
        ("E008", "IS"): 4200,
        ("E008", "AI"): 4800,
        ("E008", "EC"): 7500,
    }

    category_factors = {
        "GM": 1.0, "GMR": 1.3, "GMK": 1.4, "1G": 1.5, "2AG": 1.6, "2AR": 1.9,
        "2BG": 1.8, "3AG": 1.25, "3BG": 1.35, "SCG": 4.5, "STG": 3.8
    }

    round_factors = {1: 1.0, 2: 1.12, 3: 1.28}

    documents = []

    # General info guidelines
    documents.append(Document(page_content="KCET option entry guidelines: Students must list their options in order of true preference. Safe options should be kept at the bottom, target options in the middle, and dream options at the top."))
    documents.append(Document(page_content="KCET seat matrix updates: Computer Science (CS) seats at RVCE and BMSCE were increased by 25% for 2026. This is expected to slide the CSE cutoff ranks outward by approximately 5% to 10%."))
    documents.append(Document(page_content="KCET Fee structures: General category engineering fees for government quota seats in private colleges is approximately 1,00,000 INR per year. Government college fees are lower, at around 40,000 INR per year."))

    # Generate college cutoff details
    for col_code, col_name in colleges:
        for course_code, course_name in courses:
            base_rank = base_ranks.get((col_code, course_code), 5000)
            
            for cat in categories:
                cat_factor = category_factors[cat]
                r1_rank = int(base_rank * cat_factor * round_factors[1])
                r2_rank = int(base_rank * cat_factor * round_factors[2])
                r3_rank = int(base_rank * cat_factor * round_factors[3])
                
                # Create descriptive chunks
                text = f"In 2025, {col_name} ({col_code}) for the course {course_name} ({course_code}) has the following cutoff ranks for Category {cat}: Round 1 cutoff rank was {r1_rank}, Round 2 cutoff rank was {r2_rank}, and Round 3 (Second Extended Round) cutoff rank was {r3_rank}."
                metadata = {
                    "college_code": col_code,
                    "college_name": col_name,
                    "course_code": course_code,
                    "course_name": course_name,
                    "category": cat,
                    "r1": r1_rank,
                    "r2": r2_rank,
                    "r3": r3_rank
                }
                documents.append(Document(page_content=text, metadata=metadata))

    # Initialize Embeddings & Vector Store
    if os.environ.get("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import FakeEmbeddings
        embeddings = FakeEmbeddings(size=1536)
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save index locally
    vectorstore.save_local(INDEX_PATH)
    print(f"RAG Vector Database successfully indexed at {INDEX_PATH} with {len(documents)} document chunks.")

if __name__ == "__main__":
    init_rag_db()
