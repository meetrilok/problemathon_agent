import math

from idea_bot.domain.models import IdeaRecord, MatchResult


class SimilarityService:
    def cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_mag = math.sqrt(sum(a * a for a in left))
        right_mag = math.sqrt(sum(b * b for b in right))
        if left_mag == 0 or right_mag == 0:
            return 0.0
        return dot / (left_mag * right_mag)

    def best_match(
        self,
        *,
        candidate_embedding: list[float],
        existing: list[IdeaRecord],
        threshold: float,
    ) -> MatchResult:
        best_score = 0.0
        best_idea: IdeaRecord | None = None
        for idea in existing:
            score = self.cosine_similarity(candidate_embedding, idea.embedding)
            if score > best_score:
                best_score = score
                best_idea = idea

        is_match = best_idea is not None and best_score >= threshold
        return MatchResult(
            is_match=is_match,
            score=round(best_score, 4),
            matched_idea=(
                {
                    "id": best_idea.id,
                    "name": best_idea.name,
                    "email": best_idea.email,
                    "title": best_idea.title,
                }
                if is_match and best_idea
                else None
            ),
        )
