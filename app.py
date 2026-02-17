import hashlib
import json
import os
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import END, StateGraph

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "ideas.db"
FAISS_PATH = DATA_DIR / "faiss_index"


class DeterministicEmbeddings(Embeddings):
    """Fallback embedding implementation when Azure config is unavailable."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return vec.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]


@dataclass
class Services:
    embeddings: Embeddings
    llm: Optional[AzureChatOpenAI]


class IdeaState(TypedDict):
    name: str
    email: str
    title: str
    description: str
    category: Optional[str]
    match_found: bool
    match_score: float
    matched_idea: Optional[Dict[str, Any]]
    response_message: str
    created_at: str


def get_services() -> Services:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    embed_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    llm_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if endpoint and api_key and embed_deployment:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=embed_deployment,
        )
        llm = None
        if llm_deployment:
            llm = AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                azure_deployment=llm_deployment,
                temperature=0,
            )
        return Services(embeddings=embeddings, llm=llm)

    return Services(embeddings=DeterministicEmbeddings(), llm=None)


services = get_services()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def index_storage() -> Dict[str, Any]:
    if FAISS_PATH.exists():
        with open(FAISS_PATH, "rb") as f:
            return pickle.load(f)
    return {"index": None, "docstore": {}, "mapping": {}}


def persist_index(data: Dict[str, Any]) -> None:
    with open(FAISS_PATH, "wb") as f:
        pickle.dump(data, f)


def get_faiss_store() -> Optional[FAISS]:
    data = index_storage()
    if data["index"] is None:
        return None
    return FAISS(
        embedding_function=services.embeddings,
        index=data["index"],
        docstore=InMemoryDocstore(data["docstore"]),
        index_to_docstore_id=data["mapping"],
    )


def save_faiss_store(store: FAISS) -> None:
    data = {
        "index": store.index,
        "docstore": store.docstore._dict,
        "mapping": store.index_to_docstore_id,
    }
    persist_index(data)


def infer_category(title: str, description: str) -> str:
    text = f"{title}\n{description}"
    if services.llm:
        prompt = (
            "Categorize the following idea into one of: Product, Process, Customer, Operations, "
            "Technology, Finance, HR, Sustainability, Other. Return only the category name.\n\n"
            f"Idea:\n{text}"
        )
        try:
            response = services.llm.invoke([HumanMessage(content=prompt)])
            category = response.content.strip().splitlines()[0]
            if category:
                return category
        except Exception:
            pass

    lowered = text.lower()
    if any(k in lowered for k in ["automation", "workflow", "efficiency"]):
        return "Process"
    if any(k in lowered for k in ["customer", "client", "support", "user"]):
        return "Customer"
    if any(k in lowered for k in ["revenue", "cost", "budget", "pricing"]):
        return "Finance"
    if any(k in lowered for k in ["hiring", "employee", "training"]):
        return "HR"
    if any(k in lowered for k in ["ai", "ml", "model", "data", "software", "app"]):
        return "Technology"
    return "Other"


def load_idea(idea_id: int) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    conn.close()
    return row


def add_idea_to_db(state: IdeaState, embedding: List[float]) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO ideas (name, email, title, description, category, created_at, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            state["name"],
            state["email"],
            state["title"],
            state["description"],
            state["category"] or "Other",
            state["created_at"],
            json.dumps(embedding),
        ),
    )
    conn.commit()
    idea_id = cursor.lastrowid
    conn.close()
    return int(idea_id)


def find_semantic_match(state: IdeaState) -> IdeaState:
    query_text = f"{state['title']}\n{state['description']}"
    query_embedding = services.embeddings.embed_query(query_text)
    threshold = float(os.getenv("MATCH_THRESHOLD", "0.30"))

    conn = get_connection()
    rows = conn.execute("SELECT id, name, email, title, description, embedding FROM ideas").fetchall()
    conn.close()

    best_score = 0.0
    best_row = None
    for row in rows:
        existing_embedding = json.loads(row["embedding"])
        score = cosine_similarity(query_embedding, existing_embedding)
        if score > best_score:
            best_score = score
            best_row = row

    match_found = best_row is not None and best_score >= threshold
    state["match_found"] = match_found
    state["match_score"] = round(best_score, 4)
    if match_found and best_row is not None:
        state["matched_idea"] = {
            "id": best_row["id"],
            "name": best_row["name"],
            "email": best_row["email"],
            "title": best_row["title"],
        }
        state[
            "response_message"
        ] = f"We found a similar idea ({best_score:.0%} match). Please reach out to {best_row['name']} at {best_row['email']}."
    else:
        state["matched_idea"] = None
        state["response_message"] = (
            "Thanks! Your idea is unique enough based on the current submissions and has been recorded."
        )

    state["_embedding"] = query_embedding  # internal field
    return state


def categorize_idea(state: IdeaState) -> IdeaState:
    state["category"] = infer_category(state["title"], state["description"])
    return state


def persist_idea(state: IdeaState) -> IdeaState:
    embedding = state.pop("_embedding")
    idea_id = add_idea_to_db(state, embedding)

    text = f"{state['title']}\n{state['description']}"
    metadata = {
        "idea_id": idea_id,
        "name": state["name"],
        "email": state["email"],
        "category": state["category"],
    }

    store = get_faiss_store()
    if store is None:
        store = FAISS.from_texts([text], services.embeddings, metadatas=[metadata])
    else:
        store.add_texts([text], metadatas=[metadata])
    save_faiss_store(store)
    return state


def build_graph():
    graph = StateGraph(IdeaState)
    graph.add_node("match", find_semantic_match)
    graph.add_node("categorize", categorize_idea)
    graph.add_node("persist", persist_idea)

    graph.set_entry_point("match")
    graph.add_edge("match", "categorize")
    graph.add_edge("categorize", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


workflow = build_graph()
init_db()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")


@app.route("/", methods=["GET", "POST"])
def submit_idea():
    if request.method == "POST":
        state: IdeaState = {
            "name": request.form["name"].strip(),
            "email": request.form["email"].strip(),
            "title": request.form["title"].strip(),
            "description": request.form["description"].strip(),
            "category": None,
            "match_found": False,
            "match_score": 0.0,
            "matched_idea": None,
            "response_message": "",
            "created_at": datetime.utcnow().isoformat(),
        }
        result = workflow.invoke(state)
        return render_template("submit.html", result=result)

    return render_template("submit.html", result=None)


@app.route("/manager/login", methods=["GET", "POST"])
def manager_login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        expected_user = os.getenv("MANAGER_USERNAME", "manager")
        expected_pass = os.getenv("MANAGER_PASSWORD", "manager123")

        if username == expected_user and password == expected_pass:
            session["manager_authenticated"] = True
            return redirect(url_for("manager_dashboard"))

        flash("Invalid credentials", "error")
    return render_template("manager_login.html")


@app.route("/manager/logout")
def manager_logout():
    session.pop("manager_authenticated", None)
    return redirect(url_for("manager_login"))


@app.route("/manager/dashboard")
def manager_dashboard():
    if not session.get("manager_authenticated"):
        return redirect(url_for("manager_login"))

    conn = get_connection()
    ideas = conn.execute("SELECT * FROM ideas ORDER BY created_at DESC").fetchall()
    conn.close()

    now = datetime.utcnow()
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_week = now - timedelta(days=7)

    ideas_today = 0
    ideas_week = 0
    categories: Dict[str, int] = {}

    for idea in ideas:
        created = datetime.fromisoformat(idea["created_at"])
        if created >= start_today:
            ideas_today += 1
        if created >= start_week:
            ideas_week += 1
        categories[idea["category"]] = categories.get(idea["category"], 0) + 1

    return render_template(
        "dashboard.html",
        ideas=ideas,
        total_ideas=len(ideas),
        ideas_today=ideas_today,
        ideas_week=ideas_week,
        categories=categories,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
