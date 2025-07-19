# main.py
import os
import requests
import google.generativeai as genai
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from the .env file
load_dotenv()

# --- Pydantic Model for incoming data ---
class RefineRequest(BaseModel):
    original_proposal: str
    user_feedback: str

# --- FastAPI App Setup ---
app = FastAPI()

# Add CORS middleware to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI and External API Configuration ---
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- Helper Functions ---
def search_semantic_scholar(topic: str, limit: int = 5):
    """Searches Semantic Scholar for papers on a given topic."""
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {'x-api-key': api_key} if api_key else {}
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {'query': topic, 'limit': limit, 'fields': 'title,abstract'}
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Semantic Scholar request failed: {e}")
    results = response.json()
    return [paper for paper in results.get('data', []) if paper and paper.get('abstract')]

def search_arxiv(topic: str, limit: int = 5):
    """Searches the arXiv API for papers on a given topic."""
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'all:{topic}'
    params = {'search_query': search_query, 'start': 0, 'max_results': limit}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"arXiv request failed: {e}")
    
    root = ET.fromstring(response.content)
    papers = []
    namespace = '{http://www.w3.org/2005/Atom}'
    for entry in root.findall(f'{namespace}entry'):
        title = entry.find(f'{namespace}title').text.strip()
        abstract = entry.find(f'{namespace}summary').text.strip()
        papers.append({'title': title, 'abstract': abstract})
    return papers

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "AI Chemist API is running."}

@app.get("/api/summarize")
def get_summary(topic: str, source: str = 'semantic'):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    
    print(f"Searching for papers on: {topic} using source: {source}")
    try:
        if source == 'arxiv':
            papers = search_arxiv(topic)
        else: # Default to Semantic Scholar
            papers = search_semantic_scholar(topic)

        if not papers:
            return {"topic": topic, "summary": "Could not find any relevant papers on this topic.", "proposal": ""}

        abstracts_text = "\n\n---\n\n".join(f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers)
        
        summary_prompt = f"Summarize these abstracts for a chemist:\n{abstracts_text}"
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text

        proposal_prompt = f"Based on this summary, propose a novel research direction:\n{summary_text}"
        proposal_response = model.generate_content(proposal_prompt)
        proposal_text = proposal_response.text

        return {"topic": topic, "summary": summary_text, "proposal": proposal_text}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine_proposal")
def refine_proposal(request: RefineRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    
    print("Refining proposal based on user feedback...")
    
    refine_prompt = f"""
    A user disliked a research proposal for the following reason:
    '{request.user_feedback}'

    Here was the original proposal they disliked:
    '{request.original_proposal}'

    Please generate a new, different research proposal that addresses the user's concern.
    """
    
    try:
        response = model.generate_content(refine_prompt)
        return {"new_proposal": response.text}
    except Exception as e:
        print(f"An error occurred during refinement: {e}")
        raise HTTPException(status_code=500, detail=str(e))