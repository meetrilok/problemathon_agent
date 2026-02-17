# Idea Validation Bot (Flask + LangGraph)

A Python + HTML project that captures idea submissions, checks semantic similarity against existing ideas, stores all submissions, auto-categorizes them, and provides a manager dashboard.

## Stack
- Flask API + Jinja HTML templates
- LangGraph workflow orchestration
- Azure OpenAI embeddings + chat model (with local fallback if env vars are missing)
- FAISS vector index for semantic search persistence
- SQLite for exact submission records

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables
```bash
export FLASK_SECRET_KEY="replace-me"
export MATCH_THRESHOLD="0.30"
export MANAGER_USERNAME="manager"
export MANAGER_PASSWORD="manager123"

# Optional Azure OpenAI settings
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini"
```

## Run
```bash
python app.py
```

Visit:
- Idea form: `http://localhost:5000/`
- Manager login: `http://localhost:5000/manager/login`

## Notes
- Semantic match threshold is configurable via `MATCH_THRESHOLD` (default `0.30`).
- If match score exceeds threshold, the bot asks the submitter to contact the original idea owner.
- Every submission is stored regardless of match status.
