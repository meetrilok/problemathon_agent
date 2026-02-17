from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class IdeaSubmission:
    name: str
    email: str
    title: str
    description: str


@dataclass
class IdeaRecord:
    id: int
    name: str
    email: str
    title: str
    description: str
    category: str
    created_at: str
    embedding: list[float]


@dataclass
class MatchResult:
    is_match: bool
    score: float
    matched_idea: dict[str, Any] | None


@dataclass
class IdeaWorkflowResult:
    response_message: str
    category: str
    match_score: float
    created_at: datetime
