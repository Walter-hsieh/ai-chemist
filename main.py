# main.py
import os
import requests
import google.generativeai as genai
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64
from fastapi.responses import JSONResponse
import openpyxl
from docx import Document

# Load environment variables from the .env file
load_dotenv()

# --- Pydantic Models for API data structures ---
class RefineRequest(BaseModel):
    original_proposal: str
    user_feedback: str

class StructureRequest(BaseModel):
    proposal_text: str

class FinalProposalRequest(BaseModel):
    summary_text: str
    proposal_text: str
    smiles_string: str

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

# --- AI Configuration ---
try:
    # Use the recommended 'gemini-1.5-pro-latest' for the best quality
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- Helper Functions for Data Retrieval ---
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
    if not model: raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    try:
        papers = search_arxiv(topic) if source == 'arxiv' else search_semantic_scholar(topic)
        if not papers: return {"topic": topic, "summary": "Could not find any relevant papers on this topic.", "proposal": ""}
        abstracts_text = "\n\n---\n\n".join(f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers)
        summary_prompt = f"Summarize these abstracts for a chemist:\n{abstracts_text}"
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text
        proposal_prompt = f"Based on this summary, propose a novel research direction:\n{summary_text}"
        proposal_response = model.generate_content(proposal_prompt)
        proposal_text = proposal_response.text
        return {"topic": topic, "summary": summary_text, "proposal": proposal_text}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine_proposal")
def refine_proposal(request: RefineRequest):
    if not model: raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    refine_prompt = f"A user disliked a proposal for this reason: '{request.user_feedback}'. Original proposal: '{request.original_proposal}'. Generate a new proposal addressing the concern."
    try:
        response = model.generate_content(refine_prompt)
        return {"new_proposal": response.text}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_structure")
def generate_structure(request: StructureRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured.")

    print("Generating molecular structure...")
    
    # Retry loop to handle occasional invalid SMILES from the AI
    for attempt in range(3):
        print(f"Attempt {attempt + 1} to generate a valid SMILES string...")
        
        smiles_prompt = f"""
        Based on the following research proposal, generate a single, novel chemical structure that would be a good candidate.
        IMPORTANT: Respond with ONLY the SMILES string for the molecule and nothing else. For example: CCO

        Proposal:
        {request.proposal_text}
        """
        
        try:
            response = model.generate_content(smiles_prompt)
            # Clean up potential markdown or other text from the AI response
            smiles_string = response.text.strip().replace("`", "").replace("python", "")
            print(f"Generated SMILES: {smiles_string}")

            mol = Chem.MolFromSmiles(smiles_string)
            
            if mol is not None:
                # Success! Generate the image and return the data.
                img = Draw.MolToImage(mol, size=(300, 300))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return {"image": img_base64, "smiles": smiles_string}
            else:
                # If mol is None, the SMILES is invalid. The loop will try again.
                print(f"Invalid SMILES received on attempt {attempt + 1}. Retrying...")

        except Exception as e:
            print(f"An error occurred during attempt {attempt + 1}: {e}")

    # If the loop completes without a valid SMILES, raise an error.
    raise HTTPException(status_code=500, detail="The AI failed to generate a valid chemical structure after multiple attempts.")


@app.post("/api/generate_proposal")
def generate_final_proposal(request: FinalProposalRequest):
    if not model: raise HTTPException(status_code=500, detail="Gemini API is not configured.")
    print("Generating final research proposal documents...")
    try:
        full_proposal_prompt = f"""
        You are a PhD chemist writing a research proposal. Based on the following information, write a complete proposal document.
        The document must have these H2 sections, in this order: ## Project Framework, ## Literature Review, ## Experimental Details.
        Under ## Project Framework, create these H3 sections: ### Need, ### Solution, ### Differentiation, ### Benefits.
        Under ## Experimental Details, create these H3 sections: ### Chemicals and Reagents, ### Synthesis Pathway.
        Flesh out each section thoroughly.
        CONTEXT:
        - Initial Literature Summary: {request.summary_text}
        - Approved Research Proposal: {request.proposal_text}
        - Target Molecule SMILES String: {request.smiles_string}
        """
        full_proposal_text = model.generate_content(full_proposal_prompt).text
        
        recipe_prompt = f"""
        Based on the approved research proposal and target molecule, create a synthesis recipe.
        IMPORTANT: Respond with ONLY a comma-separated list, with each item on a new line. Do not use headers.
        Format: Chemical Name,Molar Mass (g/mol),Amount (mg or µL),Equivalents
        PROPOSAL: {request.proposal_text}
        TARGET SMILES: {request.smiles_string}
        """
        recipe_data_str = model.generate_content(recipe_prompt).text
        
        wb_recipe = openpyxl.Workbook()
        ws_recipe = wb_recipe.active
        ws_recipe.title = "Synthesis Recipe"
        ws_recipe.append(["Chemical Name", "Molar Mass (g/mol)", "Amount (mg or µL)", "Equivalents"])
        for row in recipe_data_str.strip().split('\n'):
            # Ensure row has the correct number of columns to avoid errors
            split_row = [item.strip() for item in row.split(',')]
            if len(split_row) == 4:
                ws_recipe.append(split_row)
        recipe_buffer = io.BytesIO()
        wb_recipe.save(recipe_buffer)
        recipe_base64 = base64.b64encode(recipe_buffer.getvalue()).decode('utf-8')
        
        wb_data = openpyxl.Workbook()
        ws_data = wb_data.active
        ws_data.title = "Experimental Data"
        ws_data.append(["Experiment ID", "Date", "Reactant 1 (mg)", "Reactant 2 (mg)", "Solvent (mL)", "Reaction Time (h)", "Temperature (°C)", "Yield (%)", "Notes"])
        data_buffer = io.BytesIO()
        wb_data.save(data_buffer)
        data_base64 = base64.b64encode(data_buffer.getvalue()).decode('utf-8')

        doc = Document()
        doc.add_heading('AI-Generated Research Proposal', 0)
        for line in full_proposal_text.split('\n'):
            if line.startswith('### '):
                doc.add_heading(line.replace('### ', ''), level=3)
            elif line.startswith('## '):
                doc.add_heading(line.replace('## ', ''), level=2)
            elif line.strip():
                doc.add_paragraph(line)
        doc_buffer = io.BytesIO()
        doc.save(doc_buffer)
        doc_base64 = base64.b64encode(doc_buffer.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "full_proposal_text": full_proposal_text,
            "recipe_file": recipe_base64,
            "data_template_file": data_base64,
            "proposal_docx_file": doc_base64
        })
    except Exception as e:
        print(f"An error occurred during final proposal generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))