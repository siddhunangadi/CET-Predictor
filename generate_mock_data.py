import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "kcet_data.db")

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Removed existing mock database.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create cutoff tables
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

    # Create seat matrix table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seat_matrix (
            year INTEGER NOT NULL,
            college_code TEXT NOT NULL,
            course_code TEXT NOT NULL,
            category TEXT NOT NULL,
            total_seats INTEGER NOT NULL
        )
    """)

    # Mock Data Lists
    colleges = [
        ("E005", "RV College of Engineering (RVCE)"),
        ("E003", "BMS College of Engineering (BMSCE)"),
        ("E007", "MS Ramaiah Institute of Technology (MSRIT)"),
        ("E033", "PES University (PESU) - Ring Road Campus"),
        ("E008", "Bangalore Institute of Technology (BIT)")
    ]

    courses = [
        ("CS", "Computer Science & Engineering"),
        ("IS", "Information Science & Engineering"),
        ("EC", "Electronics & Communication Engineering"),
        ("AI", "Artificial Intelligence & Machine Learning")
    ]

    categories = ["GM", "GMR", "GMK", "1G", "2AG", "2AR", "2BG", "3AG", "3BG", "SCG", "STG"]

    # Base ranks for GM Round 1 (highest demand / lowest rank numbers)
    # We will derive other rounds and categories dynamically to maintain realistic consistency
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

    # Factors to derive ranks for other categories (GM is base = 1.0)
    category_factors = {
        "GM": 1.0,
        "GMR": 1.3,
        "GMK": 1.4,
        "1G": 1.5,
        "2AG": 1.6,
        "2AR": 1.9,
        "2BG": 1.8,
        "3AG": 1.25,
        "3BG": 1.35,
        "SCG": 4.5,
        "STG": 3.8
    }

    # Round factors (Round 1 is base = 1.0)
    round_factors = {
        1: 1.0,
        2: 1.12,  # Ranks slide outwards by ~12%
        3: 1.28   # Ranks slide outwards by ~28%
    }

    cutoff_inserts = []
    seat_inserts = []

    # Populate cutoffs for 2025 (last year data)
    for col_code, col_name in colleges:
        for course_code, course_name in courses:
            base_rank = base_ranks.get((col_code, course_code), 5000)
            
            # Insert cutoffs
            for r_num in [1, 2, 3]:
                r_factor = round_factors[r_num]
                for cat in categories:
                    cat_factor = category_factors[cat]
                    
                    # Compute cutoff
                    rank = int(base_rank * cat_factor * r_factor)
                    cutoff_inserts.append((
                        2025, r_num, col_code, col_name, course_code, course_name, cat, rank
                    ))
            
            # Insert seat matrix for 2025
            for cat in categories:
                # baseline seats per course per category
                base_seats = 10 if cat in ["GM", "2AG", "3BG", "SCG"] else 3
                seat_inserts.append((
                    2025, col_code, course_code, cat, base_seats
                ))
                
                # Insert seat matrix for current year 2026 (some CS seats increased by 20%)
                is_increased = (course_code == "CS" and col_code in ["E005", "E003"])
                curr_seats = int(base_seats * 1.25) if is_increased else base_seats
                seat_inserts.append((
                    2026, col_code, course_code, cat, curr_seats
                ))

    # Insert batch data
    cursor.executemany("""
        INSERT INTO college_cutoffs (year, round, college_code, college_name, course_code, course_name, category, cutoff_rank)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, cutoff_inserts)

    cursor.executemany("""
        INSERT INTO seat_matrix (year, college_code, course_code, category, total_seats)
        VALUES (?, ?, ?, ?, ?)
    """, seat_inserts)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH} with {len(cutoff_inserts)} cutoff records.")

if __name__ == "__main__":
    init_db()
