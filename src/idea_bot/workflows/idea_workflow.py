from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from idea_bot.config import AppConfig
from idea_bot.domain.models import IdeaWorkflowResult
from idea_bot.repositories.idea_repository import IdeaRepository
from idea_bot.services.categorization_service import Categorizer
from idea_bot.services.embedding_service import EmbeddingProvider, FaissIndexStore
from idea_bot.services.similarity_service import SimilarityService


class WorkflowState(TypedDict):
    name: str
    email: str
    title: str
    description: str
    created_at: str
    category: str | None
    match_score: float
    match_found: bool
    matched_idea: dict[str, Any] | None
    response_message: str
    embedding: list[float]


class IdeaWorkflow:
    """Facade over LangGraph workflow orchestration."""

    def __init__(
        self,
        *,
        config: AppConfig,
        repository: IdeaRepository,
        embedding_provider: EmbeddingProvider,
        categorizer: Categorizer,
        similarity_service: SimilarityService,
        faiss_store: FaissIndexStore,
    ) -> None:
        self._config = config
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._categorizer = categorizer
        self._similarity = similarity_service
        self._faiss_store = faiss_store
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("match", self._match)
        graph.add_node("categorize", self._categorize)
        graph.add_node("persist", self._persist)
        graph.set_entry_point("match")
        graph.add_edge("match", "categorize")
        graph.add_edge("categorize", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def _match(self, state: WorkflowState) -> WorkflowState:
        text = f"{state['title']}\n{state['description']}"
        embedding = self._embedding_provider.embed_query(text)
        ideas = self._repository.list_ideas()
        match = self._similarity.best_match(
            candidate_embedding=embedding,
            existing=ideas,
            threshold=self._config.match_threshold,
        )
        state["embedding"] = embedding
        state["match_score"] = match.score
        state["match_found"] = match.is_match
        state["matched_idea"] = match.matched_idea
        if match.is_match and match.matched_idea:
            state["response_message"] = (
                f"We found a similar idea ({match.score:.0%} match). "
                f"Please reach out to {match.matched_idea['name']} at {match.matched_idea['email']}."
            )
        else:
            state["response_message"] = "Thanks! Your idea has been recorded."
        return state

    def _categorize(self, state: WorkflowState) -> WorkflowState:
        state["category"] = self._categorizer.categorize(state["title"], state["description"])
        return state

    def _persist(self, state: WorkflowState) -> WorkflowState:
        new_id = self._repository.add_idea(
            name=state["name"],
            email=state["email"],
            title=state["title"],
            description=state["description"],
            category=state["category"] or "Other",
            created_at=state["created_at"],
            embedding=state["embedding"],
        )
        self._faiss_store.add_text(
            f"{state['title']}\n{state['description']}",
            {
                "idea_id": new_id,
                "name": state["name"],
                "email": state["email"],
                "category": state["category"],
            },
        )
        return state

    def run(self, *, name: str, email: str, title: str, description: str) -> IdeaWorkflowResult:
        state: WorkflowState = {
            "name": name,
            "email": email,
            "title": title,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "category": None,
            "match_score": 0.0,
            "match_found": False,
            "matched_idea": None,
            "response_message": "",
            "embedding": [],
        }
        result = self._graph.invoke(state)
        return IdeaWorkflowResult(
            response_message=result["response_message"],
            category=result["category"] or "Other",
            match_score=result["match_score"],
            created_at=datetime.fromisoformat(result["created_at"]),
        )
