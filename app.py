import streamlit as st
import google.generativeai as genai
import sqlite3
import os
import tempfile
import time
import warnings
import logging
import requests
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from datetime import datetime
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. OPTIMIZATIONS & SETUP ---
# Silence all the annoying warnings to keep the app clean
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)
logging.getLogger("google_genai").setLevel(logging.ERROR)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ API Key missing! Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --- 2. DATABASE & MEMORY MANAGEMENT ---
def init_db():
    conn = sqlite3.connect('study_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (timestamp TEXT, question TEXT, answer TEXT, source TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(question, answer, source):
    conn = sqlite3.connect('study_history.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?)", (timestamp, question, answer, source))
    conn.commit()
    conn.close()

def fetch_history():
    conn = sqlite3.connect('study_history.db')
    c = conn.cursor()
    # Get all history for the summary
    c.execute("SELECT question, answer FROM history")
    data = c.fetchall()
    conn.close()
    return data

def clear_memory():
    """Wipes the DB and refreshes the page immediately."""
    conn = sqlite3.connect('study_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    
    # Clear session state
    st.session_state.messages = []
    st.session_state.gemini_files = []
    st.rerun() # Instant refresh

# --- 3. FAST MODEL SELECTOR ---
@st.cache_resource
def get_fast_model():
    """Prioritizes the fastest stable models."""
    try:
        priority = [
            "models/gemini-2.5-flash-lite", # Fastest if available
            "models/gemini-1.5-flash",      # Standard fast
            "models/gemini-1.5-flash-001"
        ]
        my_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for p in priority:
            if p in my_models:
                return p.replace("models/", "")
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

if "model_name" not in st.session_state:
    st.session_state.model_name = get_fast_model()

# --- 4. TOOLS (Web & Search) ---
def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3) # 3s timeout for speed
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove junk
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        text = soup.get_text()
        clean_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        return clean_text[:6000] # Limit char count for speed
    except Exception:
        return ""

def google_search_fallback(query):
    try:
        # Limit to top 2 results for maximum speed
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            if results:
                return "\n".join([f"{r['body']}" for r in results])
    except: return ""
    return ""

# --- 5. FILE UPLOAD (Fast) ---
def upload_to_gemini(uploaded_files):
    uploaded_content = []
    st.toast(f"🚀 Uploading {len(uploaded_files)} files...", icon="⚡")
    
    for up_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up_file.getvalue())
                tmp_path = tmp.name
            
            gemini_file = genai.upload_file(tmp_path, mime_type="application/pdf")
            
            # Optimized wait loop
            start = time.time()
            while gemini_file.state.name == "PROCESSING":
                if time.time() - start > 10: break # Lower timeout
                time.sleep(0.5)
                gemini_file = genai.get_file(gemini_file.name)
            
            if gemini_file.state.name == "ACTIVE":
                uploaded_content.append(gemini_file)
        except: pass
        finally:
            try: os.remove(tmp_path)
            except: pass
    return uploaded_content

# --- 6. THE FRIENDLY BRAIN (Streaming) ---
def stream_answer(question, context_files=None, google_context=None, website_context=None, mode="Answer"):
    model = genai.GenerativeModel(st.session_state.model_name)
    
    # Allow the AI to answer freely without getting blocked easily
    safety = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE}

    prompt_parts = []
    
    if mode == "Recall":
        prompt_parts.append(f"""
        Here is the chat history:
        {google_context}
        
        TASK:
        You are a helpful Study Assistant. 📝
        Create a bullet-point summary of all the topics discussed above. 
        Make it easy to review!
        """)
    else:
        # Add Contexts
        if context_files:
            prompt_parts.extend(context_files)
            prompt_parts.append("Source: PDF Notes.")
        if website_context:
            prompt_parts.append(f"Source: Website Content:\n{website_context}")
        if google_context:
            prompt_parts.append(f"Source: Google Search:\n{google_context}")
        
        # THE PERSONA
        prompt_parts.append(f"""
        User Question: {question}
        
        Your Personality: 
        You are a friendly, energetic, and super-fast Study Buddy! ⚡😊
        
        Instructions:
        1. Answer in SIMPLE English.
        2. Keep it SHORT (2-3 sentences) for speed.
        3. Only give long answers if the user asks to "elaborate" or "explain in detail".
        4. Use emojis occasionally to be friendly.
        5. If using a source, mention it briefly.
        """)

    try:
        # stream=True makes it feel much faster
        response_stream = model.generate_content(prompt_parts, stream=True, safety_settings=safety)
        for chunk in response_stream:
            try:
                if chunk.text: yield chunk.text
            except ValueError: pass
    except Exception:
        yield "⚠️ My brain is buffering... try asking again!"

# --- 7. UI LAYOUT ---
def main():
    st.set_page_config(page_title="Study Bot", page_icon="⚡", layout="wide")
    init_db()
    
    st.title("⚡Study Buddy")
    
    if "gemini_files" not in st.session_state:
        st.session_state.gemini_files = []
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! 👋 I'm ready. Upload notes, paste a link, or just ask away!"}]

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🧰 Control Panel")
        
        st.subheader(" Add Files")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        if st.button("⚡ Process PDFs"):
            if pdf_docs:
                st.session_state.gemini_files = upload_to_gemini(pdf_docs)
                if st.session_state.gemini_files: st.success(f"Memorized {len(pdf_docs)} files!")
        
        st.divider()
        
        st.subheader("Settings")
        # THE CLEAR BUTTON (Red for visibility)
        if st.button("🗑️ Clear Memory", type="primary"):
            clear_memory()

    # --- CHAT WINDOW ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask me anything...")

    if user_question:
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        source_labels = []
        google_info = ""
        website_info = ""
        mode = "Answer"
        
        # LOGIC: RECALL
        if "recall" in user_question.lower():
            history = fetch_history()
            if not history:
                google_info = "No history yet."
            else:
                google_info = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
            mode = "Recall"
            source_labels.append("🧠 Memory")
            
        # LOGIC: ANSWER
        else:
            # 1. Check for URL
            urls = extract_urls(user_question)
            if urls:
                with st.spinner("🔗 Reading link..."):
                    website_info = scrape_website(urls[0])
                    if website_info: source_labels.append("🌐 Website")
            
            # 2. Google Search (Only if we have no other context)
            if not st.session_state.gemini_files and not website_info:
                 with st.spinner("🔎 Googling..."):
                    google_info = google_search_fallback(user_question)
                    if google_info: source_labels.append("🔎 Internet")
            
            if st.session_state.gemini_files:
                source_labels.append("📚 PDFs")

        source_text = " + ".join(source_labels) if source_labels else "🤖 AI Brain"

        with st.chat_message("assistant"):
            st.caption(f"Source: {source_text}")
            full_response = st.write_stream(stream_answer(
                user_question, 
                context_files=st.session_state.gemini_files, 
                google_context=google_info, 
                website_context=website_info,
                mode=mode
            ))
        
        save_to_db(user_question, full_response, source_text)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
