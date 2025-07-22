# AI Chemist: Local Setup and User Guide

This guide provides step-by-step instructions on how to download, set up, and run the AI Chemist application on your local machine.

## 1. Prerequisites

Before you begin, ensure you have the following software installed on your computer:

* **Python (version 3.8 or newer):** The application's backend is built with Python. You can download it from the [official Python website](https://www.python.org/downloads/). During installation, make sure to check the box that says **"Add Python to PATH"**.
* **Git:** This is required to download the application files from GitHub. You can download it from the [Git website](https://git-scm.com/downloads).

## 2. Download the Application

First, you need to download the application files from its GitHub repository.

1.  **Open a Terminal or Command Prompt:**
    * On **Windows**, you can search for "Command Prompt" or "PowerShell".
    * On **macOS** or **Linux**, you can search for "Terminal".

2.  **Clone the Repository:**
    Run the following command to download the project files to your computer. This will create a new folder named `ai-chemist-app`.

    ```bash
    git clone <repository_url>
    ```
    *(Replace `<repository_url>` with the actual URL of the GitHub repository.)*

3.  **Navigate into the Project Directory:**
    Once the download is complete, move into the newly created folder:
    ```bash
    cd ai-chemist-app
    ```

## 3. Set Up the Backend

The backend server powers all the AI functionality. Follow these steps to get it running.

1.  **Create a Virtual Environment:**
    It's a best practice to create a virtual environment to manage the project's dependencies separately.

    ```bash
    # For macOS/Linux
    python3 -m venv venv

    # For Windows
    python -m venv venv
    ```

2.  **Activate the Virtual Environment:**
    You must activate the environment before installing packages.

    ```bash
    # For macOS/Linux
    source venv/bin/activate

    # For Windows
    .\venv\Scripts\activate
    ```
    Your terminal prompt should now show `(venv)` at the beginning.

3.  **Install Required Packages:**
    The project comes with a `requirements.txt` file that lists all the necessary Python libraries. Install them with a single command:

    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing the libraries from `main.py`: `fastapi`, `uvicorn`, `python-dotenv`, `requests`, `google-generativeai`, `openai`, `rdkit-pypi`, `python-docx`, `openpyxl`, `pypdf`)*

4.  **Start the Backend Server:**
    Now you can run the backend server using Uvicorn, which is a fast ASGI server.

    ```bash
    uvicorn main:app --reload
    ```
    If everything is successful, you will see a message indicating that the server is running, typically on `http://127.0.0.1:8000`. Keep this terminal window open.

## 4. Launch the Frontend

The frontend is the user interface you interact with in your browser.

1.  **Open `index.html`:**
    Navigate to the `ai-chemist-app` folder on your computer using your file explorer.
2.  **Open in Browser:**
    Simply double-click the `index.html` file. It should open in your default web browser (like Chrome, Firefox, or Edge).

## 5. How to Use the Application

You should now have the AI Chemist interface open in your browser and the backend server running in your terminal.

1.  **Select AI Provider:**
    Use the first dropdown to choose between **Google Gemini** and **OpenAI**.

2.  **Enter Your API Key:**
    * If you chose **Google Gemini**, paste your Gemini API key into the input field.
    * If you chose **OpenAI**, paste your OpenAI API key.

3.  **Choose a Model (Optional):**
    You can specify a particular model (e.g., `gemini-1.5-pro` or `gpt-4-turbo`). If you leave this blank, the application will use a default model (`gemini-1.5-flash` or `gpt-3.5-turbo`).

4.  **Select a Data Source:**
    * **Semantic Scholar / arXiv:** To search online databases for literature.
    * **Local Knowledge:** To use your own documents. If you select this, the file upload section will appear. Click "Choose Files" to select your PDFs, DOCX, or TXT files, then click "Upload Files".

5.  **Enter a Research Topic:**
    Type in the area of research you want to investigate (e.g., "MOFs for carbon capture").

6.  **Generate Proposal:**
    Click the **"Search & Generate"** button. The application will fetch the literature, generate a summary, and propose an initial research direction.

7.  **Follow the Workflow:**
    From here, you can approve the proposal to generate a molecular structure, and finally, approve the structure to generate all the downloadable documents (`.docx` and `.xlsx` files).

You are now all set to use the AI Chemist application!


### Issues
- [ ] The recipe and data recording format is not good (provide examples for ai to learn)
- [ ] Add chemical name aside the image of proposed chemical structure
- [ ] The proposal doesn't follow NSDB structure
- [ ] try grok's api
- [ ] prosoal.docx shoud include the name of the llm, and also user's prompt as the appedix in the end.
- [ ] add perplexity as one of choice.
