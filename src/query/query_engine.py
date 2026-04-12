"""
flux-knowledge-federation: Query engine.

Takes natural language questions and matches them against registered
knowledge entries using keyword matching and domain filtering.
Results are ranked by a combined relevance + confidence score.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from ..registry.knowledge_registry import KnowledgeEntry, KnowledgeRegistry


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class Query:
    """
    A knowledge federation query.

    Attributes:
        question:       Natural language question to answer.
        domains:        Restrict results to these domains (empty = any).
        max_results:    Maximum number of results to return.
        min_confidence: Minimum confidence threshold (0.0–1.0).
    """
    question: str
    domains: list[str] = field(default_factory=list)
    max_results: int = 10
    min_confidence: float = 0.0


@dataclass
class QueryResult:
    """
    A single result from a federation query.

    Attributes:
        entry:           The matched knowledge entry.
        relevance_score: 0.0–1.0 computed relevance to the question.
        source_agent:    Name of the agent that contributed the entry.
    """
    entry: KnowledgeEntry
    relevance_score: float
    source_agent: str


# ── Helpers ───────────────────────────────────────────────────────

_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "it", "its",
    "this", "that", "these", "those", "and", "or", "but", "not", "no",
    "what", "who", "how", "why", "when", "where", "which", "whom",
}


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, and remove non-alphanumeric chars."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize and remove stop words."""
    return [
        tok for tok in _normalize(text).split()
        if tok and tok not in _STOP_WORDS
    ]


# ── Query engine ──────────────────────────────────────────────────

class QueryEngine:
    """
    Matches natural language questions against a KnowledgeRegistry.

    Uses keyword matching over entry domain + topics combined with
    domain filtering and confidence thresholds. Results are ranked
    by a weighted combination of relevance hit-rate and confidence.
    """

    # Weights for the final ranking score.
    WEIGHT_RELEVANCE: float = 0.6
    WEIGHT_CONFIDENCE: float = 0.4

    def __init__(self, registry: KnowledgeRegistry) -> None:
        self._registry = registry

    def ask(self, query: Query) -> list[QueryResult]:
        """
        Execute a query and return ranked results.

        Algorithm:
        1. Filter entries by domain and min_confidence.
        2. Tokenize the question.
        3. Score each candidate entry by keyword hit rate.
        4. Compute final score = relevance * W_R + confidence * W_C.
        5. Sort descending, cap at max_results.
        """
        question_tokens = _tokenize(query.question)
        if not question_tokens:
            return []

        # Step 1 — candidate entries
        candidates: list[KnowledgeEntry] = []
        if query.domains:
            for domain in query.domains:
                candidates.extend(
                    self._registry.query(
                        by_domain=domain,
                        by_min_confidence=query.min_confidence,
                    )
                )
        else:
            candidates = self._registry.query(
                by_min_confidence=query.min_confidence,
            )

        # Deduplicate by id
        seen: set[str] = set()
        unique: list[KnowledgeEntry] = []
        for entry in candidates:
            if entry.id not in seen:
                seen.add(entry.id)
                unique.append(entry)
        candidates = unique

        # Steps 2–4 — score and rank
        scored: list[QueryResult] = []
        for entry in candidates:
            relevance = self._score(entry, question_tokens)
            if relevance <= 0.0:
                continue
            final = (
                relevance * self.WEIGHT_RELEVANCE
                + entry.confidence * self.WEIGHT_CONFIDENCE
            )
            scored.append(QueryResult(
                entry=entry,
                relevance_score=round(final, 4),
                source_agent=entry.agent_name,
            ))

        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored[: query.max_results]

    def _score(self, entry: KnowledgeEntry, tokens: list[str]) -> float:
        """
        Compute relevance as fraction of question tokens found in
        the entry's searchable text (domain + topics).
        """
        searchable = _normalize(" ".join([entry.domain] + entry.topics))
        searchable_tokens = set(searchable.split())
        if not searchable_tokens:
            return 0.0
        hits = sum(1 for tok in tokens if tok in searchable_tokens)
        return hits / len(tokens)
