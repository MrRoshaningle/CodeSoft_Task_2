# Movie Recommendation System

A beginner-friendly recommendation system built with Python, Streamlit, Pandas, and scikit-learn. It uses content-based filtering with TF-IDF and cosine similarity.

## Requirements
- Python 3.10 or newer
- Internet connection for the first-time package installation

## Run on Windows
1. Extract the ZIP.
2. Open PowerShell or Command Prompt in the extracted `recommendation_system` folder.
3. Create a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the app:
   ```bash
   python -m streamlit run app.py
   ```

6. Open the local URL printed in the terminal, usually http://localhost:8501.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features
- 20 sample movies
- Select a movie you like
- Optional genre filters
- Adjustable number of recommendations
- Similarity scores
- Optional dataset preview

## How it works
The app transforms the genre text into TF-IDF feature vectors and computes cosine similarity between movies. The displayed score is a feature-similarity score, not a guarantee that a person will like a movie.

## Project structure
```text
recommendation_system/
├── app.py
├── requirements.txt
└── README.md
```

This is a local educational prototype. User accounts, persistent ratings, a database, and collaborative filtering are not included.
