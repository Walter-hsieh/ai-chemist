# main.py
import os
import requests
import google.generativeai as genai
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from rdkit import Chem # RDKit imports
from rdkit.Chem import Draw
import io
import base64
from fastapi.responses import PlainTextResponse

# Load environment variables
load_dotenv()

# --- Pydantic Models ---
class RefineRequest(BaseModel):
    original_proposal: str
    user_feedback: str

class StructureRequest(BaseModel):
    proposal_text: str

# --- FastAPI App Setup ---
app = FastAPI()

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

# --- Helper Functions (search_semantic_scholar, search_arxiv) ---
# (These functions remain the same as before)
def search_semantic_scholar(topic: str, limit: int = 5):
    # ... code from previous step
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
    # ... code from previous step
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
    # ... code from previous step
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    print(f"Searching for papers on: {topic} using source: {source}")
    try:
        if source == 'arxiv':
            papers = search_arxiv(topic)
        else:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine_proposal")
def refine_proposal(request: RefineRequest):
    # ... code from previous step
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    print("Refining proposal based on user feedback...")
    refine_prompt = f"A user disliked a research proposal for this reason: '{request.user_feedback}'. Original proposal: '{request.original_proposal}'. Generate a new proposal addressing the concern."
    try:
        response = model.generate_content(refine_prompt)
        return {"new_proposal": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW ENDPOINT FOR GENERATING MOLECULAR STRUCTURES ---
@app.post("/api/generate_structure", response_class=PlainTextResponse)
def generate_structure(request: StructureRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured.")

    print("Generating molecular structure...")
    # Ask the AI for a SMILES string representation of a molecule
    smiles_prompt = f"""
    Based on the following research proposal, generate a single, novel chemical structure that would be a good candidate.
    
    IMPORTANT: Respond with ONLY the SMILES string for the molecule and nothing else. For example: CCO for ethanol.

    Proposal:
    {request.proposal_text}
    """
    
    try:
        response = model.generate_content(smiles_prompt)
        smiles_string = response.text.strip()
        print(f"Generated SMILES: {smiles_string}")

        # Use RDKit to convert the SMILES string to a molecule object
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise HTTPException(status_code=400, detail="AI returned an invalid SMILES string.")

        # Generate a PNG image of the molecule in memory
        img = Draw.MolToImage(mol, size=(300, 300))
        
        # Save the image to a byte buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        # Encode the image in base64 to send it over JSON
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return img_base64 # Return the raw base64 string

    except Exception as e:
        print(f"An error occurred during structure generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))