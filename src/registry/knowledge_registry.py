"""
flux-knowledge-federation: Central knowledge registry.

Agents register their expertise as KnowledgeEntry objects. The registry
supports query, search, persistence, and federation (merging multiple
agent registries into one unified index).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class KnowledgeEntry:
    """A single registered piece of knowledge from an agent."""

    agent_name: str
    domain: str
    topics: list[str] = field(default_factory=list)
    confidence: float = 0.5
    evidence_urls: list[str] = field(default_factory=list)
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeEntry:
        return cls(**data)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KnowledgeEntry):
            return NotImplemented
        return self.id == other.id


class KnowledgeRegistry:
    """
    Central registry for federated knowledge entries.

    Each agent registers its expertise here. The registry supports
    filtering, full-text search on topics, JSON persistence, and
    merging with other registries for federation.
    """

    def __init__(self, entries: Optional[list[KnowledgeEntry]] = None) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}
        if entries:
            for entry in entries:
                self._entries[entry.id] = entry

    # ── Registration ──────────────────────────────────────────────

    def register(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """
        Register (or update) a knowledge entry. If an entry with the
        same id already exists it is replaced entirely.
        """
        if not isinstance(entry, KnowledgeEntry):
            raise TypeError(f"Expected KnowledgeEntry, got {type(entry)}")
        entry.last_updated = datetime.now(timezone.utc).isoformat()
        self._entries[entry.id] = entry
        return entry

    def remove(self, agent_name: Optional[str] = None,
               domain: Optional[str] = None,
               entry_id: Optional[str] = None) -> int:
        """
        Remove entries matching the given filters. Returns the number
        of entries removed. If *entry_id* is given, removes that single
        entry regardless of other filters.
        """
        if entry_id:
            removed = entry_id in self._entries
            self._entries.pop(entry_id, None)
            return int(removed)

        to_remove = [
            eid for eid, e in self._entries.items()
            if (agent_name is None or e.agent_name == agent_name)
            and (domain is None or e.domain == domain)
        ]
        for eid in to_remove:
            del self._entries[eid]
        return len(to_remove)

    # ── Querying ──────────────────────────────────────────────────

    def query(
        self,
        *,
        by_domain: Optional[str] = None,
        by_topic: Optional[str] = None,
        by_agent: Optional[str] = None,
        by_min_confidence: Optional[float] = None,
    ) -> list[KnowledgeEntry]:
        """
        Filter entries by domain, topic (substring match), agent name,
        or minimum confidence. All filters are AND-combined; omit a
        filter to skip it.
        """
        results: list[KnowledgeEntry] = []

        for entry in self._entries.values():
            if by_domain is not None and entry.domain != by_domain:
                continue
            if by_agent is not None and entry.agent_name != by_agent:
                continue
            if by_min_confidence is not None and entry.confidence < by_min_confidence:
                continue
            if by_topic is not None:
                topic_lower = by_topic.lower()
                if not any(topic_lower in t.lower() for t in entry.topics):
                    continue
            results.append(entry)

        return sorted(results, key=lambda e: e.confidence, reverse=True)

    def search(self, query_str: str, limit: int = 20) -> list[KnowledgeEntry]:
        """
        Full-text search across domain and topics using simple keyword
        matching. Returns entries sorted by number of keyword hits then
        confidence.
        """
        keywords = [k.lower() for k in query_str.split() if k.strip()]
        if not keywords:
            return []

        scored: list[tuple[int, float, KnowledgeEntry]] = []
        for entry in self._entries.values():
            searchable = " ".join([entry.domain] + entry.topics).lower()
            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                scored.append((hits, entry.confidence, entry))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [entry for _, _, entry in scored[:limit]]

    @property
    def entries(self) -> list[KnowledgeEntry]:
        return list(self._entries.values())

    @property
    def agents(self) -> list[str]:
        return sorted({e.agent_name for e in self._entries.values()})

    @property
    def domains(self) -> list[str]:
        return sorted({e.domain for e in self._entries.values()})

    def __len__(self) -> int:
        return len(self._entries)

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the registry to a JSON file."""
        data = {
            "version": 1,
            "exported": datetime.now(timezone.utc).isoformat(),
            "entries": [e.to_dict() for e in self._entries.values()],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> KnowledgeRegistry:
        """Load a registry from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entries = [KnowledgeEntry.from_dict(d) for d in data.get("entries", [])]
        return cls(entries)

    # ── Federation ────────────────────────────────────────────────

    def merge(self, other: KnowledgeRegistry) -> list[KnowledgeEntry]:
        """
        Merge another registry into this one. When both registries
        contain entries with the same id the entry with the higher
        confidence wins. Returns a list of entries that were added
        or replaced.
        """
        changed: list[KnowledgeEntry] = []
        for entry in other.entries:
            existing = self._entries.get(entry.id)
            if existing is None or entry.confidence > existing.confidence:
                self._entries[entry.id] = entry
                changed.append(entry)
        return changed
