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

# --- 1. SETUP & WARNING FIXES ---
# These lines strictly block the warnings you saw
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ API Key missing! Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

# --- 2. MODEL SELECTOR (Fast & Stable) ---
@st.cache_resource
def get_fast_model():
    try:
        # Priority list
        priority = [
            "models/gemini-2.5-flash-lite",
            "models/gemini-1.5-flash",
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

# --- 3. DATABASE ---
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
    c.execute("SELECT question, answer FROM history ORDER BY rowid DESC LIMIT 5")
    data = c.fetchall()
    conn.close()
    return data

# --- 4. WEBSITE READER (New Feature!) ---
def extract_urls(text):
    """Finds all URLs in the user's text."""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

def scrape_website(url):
    """Visits a website and grabs the text."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Kill script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return clean_text[:5000] # Limit to first 5000 chars to save tokens
    except Exception as e:
        return f"Error reading website: {e}"

# --- 5. FILE UPLOAD ---
def upload_to_gemini(uploaded_files):
    uploaded_content = []
    st.toast(f"🚀 Processing {len(uploaded_files)} files...", icon="⏳")
    
    for up_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up_file.getvalue())
                tmp_path = tmp.name
            
            gemini_file = genai.upload_file(tmp_path, mime_type="application/pdf")
            
            start = time.time()
            while gemini_file.state.name == "PROCESSING":
                if time.time() - start > 15: break
                time.sleep(0.5)
                gemini_file = genai.get_file(gemini_file.name)
                
            if gemini_file.state.name == "ACTIVE":
                uploaded_content.append(gemini_file)
            else:
                st.error(f"Skipped {up_file.name} (Processing failed)")
        except Exception:
            st.error(f"Error reading {up_file.name}")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except: pass
            
    return uploaded_content

# --- 6. SEARCH FALLBACK ---
def google_search_fallback(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            if results:
                return "\n".join([f"{r['body']}" for r in results])
    except: return ""
    return ""

# --- 7. STREAMING GENERATOR (Friendly & Simple) ---
def stream_answer(question, context_files=None, google_context=None, website_context=None, mode="Answer"):
    model = genai.GenerativeModel(st.session_state.model_name)
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Build Prompt
    prompt_parts = []
    
    if mode == "Recall":
        prompt_parts.append(f"Summarize this conversation history simply in bullet points:\n{google_context}")
    else:
        # Add Files
        if context_files:
            prompt_parts.extend(context_files)
            prompt_parts.append("Context: Use the PDF notes provided above.")
        
        # Add Website Content
        if website_context:
            prompt_parts.append(f"Context: Content from the website link provided by user:\n{website_context}")
            
        # Add Google Search
        if google_context:
            prompt_parts.append(f"Context: Internet Search Results:\n{google_context}")
        
        # --- THE "FRIENDLY & SIMPLE" BRAIN ---
        prompt_parts.append(f"""
        User Question: {question}
        
        Your Persona: You are a super friendly, easy-going Study Buddy. 🤝
        
        Instructions:
        1. ANSWER SIMPLY: Use plain English. Avoid big jargon.
        2. KEEP IT SHORT: Aim for 2-3 sentences unless asked to "elaborate" or "explain in detail".
        3. BE HELPFUL: If the user asks to "elaborate", then give a simple detailed explanation with analogies.
        4. If using a website link, mention "According to the link...".
        """)

    try:
        response_stream = model.generate_content(prompt_parts, stream=True, safety_settings=safety_settings)
        for chunk in response_stream:
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                pass
    except Exception as e:
        yield f"⚠️ Oops, network glitch! Try again."

# --- 8. UI ---
def main():
    st.set_page_config(page_title="Study Bot", page_icon="🤖", layout="wide")
    init_db()
    
    st.title("🤖 Study Buddy")
    st.caption(f"No worries about study ")

    if "gemini_files" not in st.session_state:
        st.session_state.gemini_files = []

    with st.sidebar:
        st.header("📂 Upload Notes")
        pdf_docs = st.file_uploader("Drag & Drop PDFs", accept_multiple_files=True, type=['pdf'])
        if st.button("⚡ Read"):
            if pdf_docs:
                st.session_state.gemini_files = upload_to_gemini(pdf_docs)
                if st.session_state.gemini_files:
                    st.success(f"Got it! I read {len(pdf_docs)} files.")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! 👋 Paste a link, upload a PDF, or just ask me anything. I'll keep it simple!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Paste a link or ask a question...")

    if user_question:
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        source_labels = []
        google_info = ""
        website_info = ""
        mode = "Answer"
        
        # A. Handle "Recall"
        if "recall" in user_question.lower():
            history = fetch_history()
            google_info = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
            mode = "Recall"
            source_labels.append("🧠 Memory")
            
        else:
            # B. Handle URL in Chat (New Feature!)
            urls = extract_urls(user_question)
            if urls:
                with st.spinner(f"🔗 Reading website: {urls[0]}..."):
                    website_info = scrape_website(urls[0])
                    if website_info:
                        source_labels.append("🌐 Website Link")
            
            # C. Handle Google Search (Only if no files and no URL provided)
            if not st.session_state.gemini_files and not website_info:
                 with st.spinner("🔎 Searching Google..."):
                    google_info = google_search_fallback(user_question)
                    if google_info: source_labels.append("🔎 Google")
            
            if st.session_state.gemini_files:
                source_labels.append("📚 PDF Notes")

        source_text = " + ".join(source_labels) if source_labels else "🤖 AI Knowledge"

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