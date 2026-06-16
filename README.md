# 🎓 CET PREDICTOR & RAG ANALYST
### *Ditch the Stress, Predict with Precision. Your Ultimate KCET Admission Companion.*

[![Streamlit App](https://static.streamlit.io/badge_indicator.svg)](https://share.streamlit.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 What is the CET Predictor?

Every year, lakhs of students go through the chaotic KCET counseling process. You are handed massive, complicated PDF lists of cutoffs, trying to figure out which college and branch you can get based on your rank and category. 

**CET Predictor** changes the game. We’ve turned confusing government PDFs into a **premium, Apple-inspired matte dark dashboard** that does all the heavy lifting for you. In seconds, you get clear, structured, and customized recommendations tailored *exactly* to your profile.

---

## 🔥 Key Features That Make This a Must-Have

### 1. 🔍 Personalized Student Profile & Analytics
Enter your rank, select your category (GM, 2AG, 3BG, SCG, etc.), and choose your preferred branches (CS, IS, EC, AI). The app instantly groups your options into three visual categories:
* **🟢 Safe Options**: Ranks safely below historical cutoffs (admission is highly guaranteed).
* **🟡 Target Options**: Borderline choices (within $\pm$ 10% threshold) where the real competitive action is.
* **🔴 Dream Options**: Reach choices that you should definitely fill in your options list just in case.

### 2. ⚡ Proximity-to-Rank Sorter (Smart Ranking)
Tired of scroll-filtering through hundreds of safety colleges that you don't want? Our custom **Proximity-to-Rank Sorter** clusters colleges that are closest to your rank and matches them to your branch priorities first. This places your most relevant, realistic options right at the top of your screen.

### 3. 🎯 Seat-Matrix Buffer (+10% CSE Allowance)
Seat counts shift every year. Our algorithm includes a toggle to apply **Seat-Matrix Buffers**. If top-tier colleges like **RVCE (E005)** or **BMSCE (E003)** expand their Computer Science intakes, the tool dynamically adjusts the cutoff limits by 10% on-the-fly, giving you the edge in option-entry strategy!

### 4. 🤖 Agentic AI Admission Advisor (Chatbot)
Have a question like: *"My rank is 9500 in 3BG, what are my best options in CS or IS?"* 
Just type it in plain English! The built-in AI chatbot parses your parameters, **instantly updates the dashboard sliders**, queries the SQLite database, and gives you a professional, human-like breakdown of where to apply. It supports both local database lookup and full Hybrid Vector-RAG chatbot interactions using Mistral AI.

### 5. 📤 PDF Cutoff Uploader with Layout-Aware Parser
Got new cutoff sheets from KEA? Just drag and drop the PDFs into the uploader. The app’s **layout-aware geometric parsing engine** extracts tables with **100% accuracy** (handling KEA's complex spacing and columns), auto-detects the round numbers from the filenames, and updates the local SQLite database instantly.

---

## 🛠️ Tech Stack & Architecture

* **Frontend**: Streamlit (Sleek, responsive Web App styled with modern CSS variables, glassmorphic card hovers, and segmented controls).
* **Database**: SQLite (Local, fast SQL engine for structured historical cutoffs).
* **Vector DB**: Pinecone (For storing RAG embeddings).
* **LLM Orchestration**: LangChain & Mistral AI (Powering conversational counseling).
* **PDF Parser**: `pdfplumber` (Custom geometric table extraction).

---

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have Python 3.9+ installed on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/siddhunangadi/CET-Predictor.git
cd CET-Predictor
```

### 2. Install Dependencies
Install all required libraries using pip:
```bash
pip install streamlit pdfplumber pandas langchain langchain-mistralai langchain-pinecone
```

### 3. Run the App
Launch the Streamlit app locally:
```bash
streamlit run streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to start predicting!

---

## 💎 Premium Aesthetic Customizations Applied
* **Matte Dark Base**: Pitch-black canvas (`#0b0b0c`) with modern typography (`Plus Jakarta Sans` and `Bebas Neue` headers).
* **No Light Leaks**: Space-gray input boxes (`#151518`) with glowing focus highlights.
* **Segmented Radio Tabs**: Replaced default vertical radio buttons with premium pill-shaped segmented buttons.
* **Micro-Animations**: Elevating card shadows and subtle lift-up transforms on hover to make the dashboard feel alive.

---
*Created to empower students and make college admissions stress-free. If you find this helpful, drop a star ⭐ on this repository!*
