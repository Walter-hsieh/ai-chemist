# main.py
import os
import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- FastAPI App Setup ---
app = FastAPI()

# Add CORS middleware to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- AI and External API Configuration ---

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # You might want to handle this more gracefully
    model = None

# --- Helper Functions ---

def search_semantic_scholar(topic: str, limit: int = 5):
    """Searches Semantic Scholar for papers on a given topic."""
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': topic,
        'limit': limit,
        'fields': 'title,abstract'
    }
    try:
        response = requests.get(search_url, params=params, timeout=10) # Added a timeout
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Semantic Scholar request failed: {e}")
        raise HTTPException(status_code=503, detail="Failed to fetch from Semantic Scholar")

    results = response.json()
    return [paper for paper in results.get('data', []) if paper and paper.get('abstract')]


# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "AI Chemist API is running. Open the index.html file to use the app."}


@app.get("/api/summarize")
def get_summary(topic: str):
    """
    Searches for papers, generates a summary, and then proposes a research direction.
    """
    if not model:
        raise HTTPException(status_code=500, detail="Gemini API is not configured correctly.")

    print(f"Searching for papers on: {topic}")
    try:
        papers = search_semantic_scholar(topic)
        if not papers:
            return {"topic": topic, "summary": "Could not find any relevant papers with abstracts on this topic. Please try a different query.", "proposal": ""}

        # Combine abstracts into one block of text
        abstracts_text = "\n\n---\n\n".join(
            f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers
        )

        # Create the prompt for the summary
        summary_prompt = f"""
        You are a research assistant for a professional chemist.
        Based on the following research paper abstracts, provide a comprehensive summary that covers:
        1. The current state of the field.
        2. Established methods and key findings.
        3. Existing challenges or unanswered questions.

        Abstracts:
        {abstracts_text}
        """

        print("Generating summary with Gemini...")
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text

        # Create the prompt for the proposal
        print("Generating research proposal...")
        proposal_prompt = f"""
        Based on the following research summary, propose one single, innovative research direction or a novel hypothesis to test.
        Make it concise, actionable, and state the potential impact.

        Summary:
        {summary_text}
        """
        proposal_response = model.generate_content(proposal_prompt)
        proposal_text = proposal_response.text

        # Return all three pieces of information
        return {
            "topic": topic,
            "summary": summary_text,
            "proposal": proposal_text
        }

    except HTTPException as e:
        # Re-raise HTTPExceptions from helper functions
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Return a generic server error for other exceptions
        raise HTTPException(status_code=500, detail=str(e))