import pdfplumber
import os
import re
from langchain.schema import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Configurations with user's keys as fallbacks
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "IO4gZPwvajIzQGg0dUFxYJpkGSm4pxDc")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_6KkZ5p_QBPf37BqwgnCBo7WU4HPkuwnjiBsaAKeQgU5zxfGXQAbTzamjQE1EUXoZzRAqvu")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX_NAME", "raggg")

# Set environment variables for LangChain client detection
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Custom Embedding Wrapper to handle Pinecone index dimension constraints
class TruncatedEmbeddings:
    def __init__(self, base_embeddings, target_dim=512):
        self.base_embeddings = base_embeddings
        self.target_dim = target_dim
        
    def embed_documents(self, texts):
        vectors = self.base_embeddings.embed_documents(texts)
        # Normalize and truncate to target dimension
        return [v[:self.target_dim] for v in vectors]
        
    def embed_query(self, text):
        vector = self.base_embeddings.embed_query(text)
        return vector[:self.target_dim]

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

def ingest_pdf_to_rag(pdf_path, year):
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}")
        return

    print(f"Ingesting PDF {pdf_path} into Pinecone Index '{PINECONE_INDEX}' (512-dim truncated) using Mistral...")
    
    if year == 2024:
        round_num = 1
        college_header_pattern = re.compile(r"^(\d+)\s+([A-Z]\d{3})\s+(.+)$")
        allotment_str = "Mock 1 Allotment"
    else:
        round_num = 2
        college_header_pattern = re.compile(r"^College:\s+([A-Z]\d{3})\s+(.+)$")
        allotment_str = "Session 2 Allotment"

    documents = []
    records_added = 0

    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages):
            # 1. Extract words and reconstruct lines
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
                
            # Parse college headers
            headers = []
            for top, text in sorted_lines:
                match = college_header_pattern.match(text.strip())
                if match:
                    if year == 2024:
                        headers.append({
                            "top": top,
                            "code": match.group(2),
                            "name": match.group(3)
                        })
                    else:
                        headers.append({
                            "top": top,
                            "code": match.group(1),
                            "name": match.group(2)
                        })

            # 2. Extract tables on this page
            tables = page.find_tables()
            for table in tables:
                x0, top, x1, bottom = table.bbox
                
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
                                            # Create semantic chunk
                                            text = f"In {year}, for {allotment_str}, {closest_header['name']} ({closest_header['code']}) for the course {course_name} ({course_code}) has a cutoff rank of {val_rank} for Category {category}."
                                            
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
                                            documents.append(Document(page_content=text, metadata=metadata))
                                            records_added += 1
            print(f"Page {p_idx+1}/{len(pdf.pages)}: parsed. Total chunks cached: {records_added}")

    if not documents:
        print("No cutoff entries extracted from PDF.")
        return

    # Add to Pinecone Vector Store in batches
    print("Uploading 512-truncated embedded documents to Pinecone Index. This can take a few minutes...")
    base_embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
    embeddings = TruncatedEmbeddings(base_embeddings, target_dim=512)
    
    batch_size = 150
    for idx in range(0, len(documents), batch_size):
        batch = documents[idx:idx+batch_size]
        print(f"Uploading batch {idx // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} ({len(batch)} documents)...")
        PineconeVectorStore.from_documents(
            batch,
            embeddings,
            index_name=PINECONE_INDEX,
            pinecone_api_key=PINECONE_API_KEY
        )
    
    print(f"Successfully uploaded {len(documents)} descriptive chunks to Pinecone Index '{PINECONE_INDEX}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 ingest_pdf.py <pdf_path> <year>")
    else:
        ingest_pdf_to_rag(sys.argv[1], int(sys.argv[2]))
