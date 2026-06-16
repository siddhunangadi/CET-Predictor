import pdfplumber
import sqlite3
import os
import re

DB_PATH = os.path.join(os.path.dirname(__file__), "kcet_data.db")

# Course Name to Code mappings for 2025
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

def parse_kcet_pdf(pdf_path, year):
    if not os.path.exists(pdf_path):
        print(f"PDF file not found at: {pdf_path}")
        return

    print(f"Parsing PDF {pdf_path} for Year {year} using layout-aware spatial geometry...")
    
    # Determine header regex and round number based on year
    if year == 2024:
        round_num = 1
        college_header_pattern = re.compile(r"^(\d+)\s+([A-Z]\d{3})\s+(.+)$")
    else:
        round_num = 2
        college_header_pattern = re.compile(r"^College:\s+([A-Z]\d{3})\s+(.+)$")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables if not exist
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

    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages):
            # 1. Extract words to reconstruct lines and get their coordinates
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
                
            # Filter and cache college headers on this page
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
                
                # Match table to closest college header above it
                closest_header = None
                for h in headers:
                    if h["top"] < top:
                        if closest_header is None or h["top"] > closest_header["top"]:
                            closest_header = h
                
                # If no header is found on this page, it might be a continuation of the last college from the previous page.
                # However, since KEA repeats college headers at the top of pages, closest_header is normally found.
                if not closest_header:
                    continue
                
                # Extract grid cells
                grid = table.extract()
                categories = []
                for row in grid:
                    row = [str(cell).strip() if cell else "" for cell in row]
                    if not row or not any(row):
                        continue
                        
                    # Check for header category row
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

            if len(cutoff_inserts) >= 500:
                cursor.executemany("""
                    INSERT INTO college_cutoffs (year, round, college_code, college_name, course_code, course_name, category, cutoff_rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, cutoff_inserts)
                conn.commit()
                cutoff_inserts = []
            
            print(f"Page {p_idx+1}/{len(pdf.pages)}: processed. Total ranks cached: {records_added}")

    if cutoff_inserts:
        cursor.executemany("""
            INSERT INTO college_cutoffs (year, round, college_code, college_name, course_code, course_name, category, cutoff_rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, cutoff_inserts)
        conn.commit()

    conn.close()
    print(f"Finished parsing. Saved {records_added} records to database {DB_PATH}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 extract_cutoffs.py <pdf_path> <year>")
    else:
        parse_kcet_pdf(sys.argv[1], int(sys.argv[2]))
