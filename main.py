# main.py
import os
import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ... (FastAPI app and CORS middleware setup) ...
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# ... (keep the search_semantic_scholar function) ...
def search_semantic_scholar(topic: str, limit: int = 5):
    """Searches Semantic Scholar for papers on a given topic."""
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': topic,
        'limit': limit,
        'fields': 'title,abstract'
    }
    response = requests.get(search_url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch from Semantic Scholar")

    results = response.json()
    return [paper for paper in results.get('data', []) if paper.get('abstract')]


@app.get("/api/summarize")
def get_summary(topic: str):
    print(f"Searching for papers on: {topic}")
    try:
        papers = search_semantic_scholar(topic)
        if not papers:
            return {"topic": topic, "summary": "Could not find any papers on this topic."}

        # Combine abstracts into one block of text
        abstracts_text = "\n\n---\n\n".join(
            f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers
        )

        # Create the prompt for the AI
        prompt = f"""
        You are a research assistant for a professional chemist.
        Based on the following research paper abstracts, provide a comprehensive summary that covers:
        1. The current state of the field.
        2. Established methods and key findings.
        3. Existing challenges or unanswered questions.

        Abstracts:
        {abstracts_text}
        """

        print("Generating summary with Gemini...")
        response = model.generate_content(prompt)

        return {"topic": topic, "summary": response.text}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))