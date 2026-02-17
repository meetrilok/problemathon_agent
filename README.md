# Idea Validation Platform (Flask + LangGraph + Azure OpenAI + FAISS)

Production-style project structure for idea intake, semantic duplicate detection, categorization, persistence, and manager analytics.

## Architecture

- **Factory Pattern**: `create_app()` composes dependencies and routes.
- **Repository Pattern**: `IdeaRepository` isolates SQLite operations.
- **Strategy Pattern**:
  - Embeddings provider (Azure or deterministic fallback)
  - Categorizer (Azure LLM or rule-based fallback)
- **Workflow Orchestration**: LangGraph pipeline (match -> categorize -> persist)
- **Single Responsibility** modules under `src/idea_bot/...`

## Project Structure

```text
run.py
src/idea_bot/
  app_factory.py
  config.py
  domain/models.py
  repositories/idea_repository.py
  services/
    embedding_service.py
    categorization_service.py
    similarity_service.py
  workflows/idea_workflow.py
  web/
    routes.py
    templates/
```

## Tech stack
- Flask API + Jinja HTML
- LangGraph
- Azure OpenAI embeddings + chat model
- FAISS vector DB
- SQLite for transactional records

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python run.py
```

## Environment Variables

```bash
export FLASK_SECRET_KEY="replace-me"
export MATCH_THRESHOLD="0.30"
export MANAGER_USERNAME="manager"
export MANAGER_PASSWORD="manager123"

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-02-01"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding"
export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4o-mini"
```

## Features
- Submit ideas with submitter details (name, email)
- Semantic match check with configurable threshold (`MATCH_THRESHOLD`)
- Bot response asks submitter to contact original owner if match > threshold
- Stores every submission regardless of match outcome
- Auto-categorization
- Manager login and dashboard insights:
  - total ideas
  - ideas today
  - ideas this week
  - category distribution
