$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python -m streamlit run src/frontend/app.py --server.headless true
