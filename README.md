# Sarcasm Detection System

This project is a sarcasm detection system that uses a machine learning model and a generative AI layer for validation. It uses the `google.genai` package to interact with the Gemini API.

## How to Run

### Command-Line Interface (for quick testing)

This uses the `main.py` file to get predictions directly in your terminal.

To run it, execute:
```bash
python main.py
```

### Web Interface (full application)

This provides a user-friendly interface in your browser. It requires two steps:

**Step 1: Start the API server.** This runs in the background to handle requests. It must be run with `uvicorn`.
```bash
uvicorn api.main:app --reload
```

**Step 2: Start the Streamlit UI.** This opens the web application.
```bash
streamlit run ui/app.py
```

You can then access the application in your browser at the address provided by Streamlit.
