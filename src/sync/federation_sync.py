"""
flux-knowledge-federation: Federation sync protocol.

Keeps a local KnowledgeRegistry up to date by pulling knowledge
entries from other agents' repos. Supports delta sync (only entries
changed since last pull) and conflict resolution (higher confidence
wins).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..registry.knowledge_registry import KnowledgeEntry, KnowledgeRegistry


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class SyncStatus:
    """Tracks the last successful sync time per agent."""
    agent_name: str
    last_sync: str = ""
    entries_pulled: int = 0
    entries_updated: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    """Summary of a single sync operation."""
    agent_name: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_seen: int = 0
    new_entries: int = 0
    updated_entries: int = 0
    skipped_unchanged: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ── Federation sync ───────────────────────────────────────────────

# Default path relative to an agent repo where published knowledge
# entries are stored for fleet consumption.
_FLEET_KNOWLEDGE_REL = "message-in-a-bottle/for-fleet/knowledge.json"


class FederationSync:
    """
    Syncs knowledge from other agents' repos into a local registry.

    Transport model:
    - Reads JSON files from other agents' ``for-fleet/`` directories
      on the local filesystem (git-based; assumes repos are cloned).
    - Delta sync: only entries whose ``last_updated`` is newer than
      the previously recorded sync timestamp are applied.
    - Conflict resolution: when two entries share the same id the
      one with higher confidence wins.
    """

    def __init__(
        self,
        local_registry: KnowledgeRegistry,
        agent_repos: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Args:
            local_registry: The registry to sync entries into.
            agent_repos:    Mapping of agent_name → absolute path to
                            that agent's repo root on disk.
        """
        self._registry = local_registry
        self._agent_repos: dict[str, str] = agent_repos or {}
        self._sync_status: dict[str, SyncStatus] = {}

    # ── Public API ────────────────────────────────────────────────

    def add_agent_repo(self, agent_name: str, repo_path: str) -> None:
        """Register an agent's repo path for future syncs."""
        self._agent_repos[agent_name] = repo_path

    def pull(self, agent_name: str) -> SyncResult:
        """
        Pull all knowledge from a single agent's repo. If no
        ``for-fleet/knowledge.json`` exists, returns an empty result.
        """
        repo_path = self._agent_repos.get(agent_name)
        if not repo_path:
            return SyncResult(
                agent_name=agent_name,
                errors=[f"No repo path registered for agent '{agent_name}'"],
            )

        knowledge_path = os.path.join(repo_path, _FLEET_KNOWLEDGE_REL)
        if not os.path.isfile(knowledge_path):
            return SyncResult(
                agent_name=agent_name,
                errors=[f"Knowledge file not found: {knowledge_path}"],
            )

        return self._sync_from_file(agent_name, knowledge_path)

    def pull_all(self) -> dict[str, SyncResult]:
        """Pull knowledge from every registered agent repo."""
        return {name: self.pull(name) for name in self._agent_repos}

    def delta_sync(self, agent_name: str, since: Optional[str] = None) -> SyncResult:
        """
        Pull only entries that changed since *since* (ISO timestamp).
        If *since* is None, uses the last recorded sync time for the
        agent (full pull if no prior sync).
        """
        if since is None:
            status = self._sync_status.get(agent_name)
            since = status.last_sync if status else ""

        repo_path = self._agent_repos.get(agent_name)
        if not repo_path:
            return SyncResult(
                agent_name=agent_name,
                errors=[f"No repo path registered for agent '{agent_name}'"],
            )

        knowledge_path = os.path.join(repo_path, _FLEET_KNOWLEDGE_REL)
        if not os.path.isfile(knowledge_path):
            return SyncResult(
                agent_name=agent_name,
                errors=[f"Knowledge file not found: {knowledge_path}"],
            )

        return self._sync_from_file(agent_name, knowledge_path, since=since)

    @property
    def sync_status(self) -> dict[str, SyncStatus]:
        return dict(self._sync_status)

    # ── Internal ──────────────────────────────────────────────────

    def _sync_from_file(
        self,
        agent_name: str,
        file_path: str,
        since: Optional[str] = None,
    ) -> SyncResult:
        """Read a knowledge JSON file and merge entries into local registry."""
        result = SyncResult(agent_name=agent_name)

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            result.errors.append(f"Failed to read {file_path}: {exc}")
            self._record_status(agent_name, result)
            return result

        raw_entries = data if isinstance(data, list) else data.get("entries", [])
        result.total_seen = len(raw_entries)

        for raw in raw_entries:
            try:
                entry = KnowledgeEntry.from_dict(raw)
            except (TypeError, KeyError) as exc:
                result.errors.append(f"Invalid entry in {file_path}: {exc}")
                continue

            # Delta filter
            if since and entry.last_updated <= since:
                result.skipped_unchanged += 1
                continue

            # Merge with conflict resolution
            merged = self._merge_entry(entry)
            if merged:
                result.new_entries += 1
            else:
                result.updated_entries += 1

        self._record_status(agent_name, result)
        return result

    def _merge_entry(self, incoming: KnowledgeEntry) -> bool:
        """
        Merge an incoming entry into the registry.

        Returns True if the entry was newly added, False if it
        replaced an existing entry (conflict resolution applied).
        """
        existing = self._registry._entries.get(incoming.id)

        if existing is None:
            self._registry.register(incoming)
            return True

        # Conflict resolution: higher confidence wins; ties go to
        # the more recently updated entry.
        if (
            incoming.confidence > existing.confidence
            or (
                incoming.confidence == existing.confidence
                and incoming.last_updated > existing.last_updated
            )
        ):
            self._registry.register(incoming)
            return False

        return False

    def _record_status(self, agent_name: str, result: SyncResult) -> None:
        self._sync_status[agent_name] = SyncStatus(
            agent_name=agent_name,
            last_sync=result.timestamp,
            entries_pulled=result.new_entries + result.updated_entries,
            entries_updated=result.updated_entries,
            errors=list(result.errors),
        )

    # ── Static helpers ────────────────────────────────────────────

    @staticmethod
    def read_fleet_knowledge(repo_path: str) -> list[KnowledgeEntry]:
        """
        Convenience: read knowledge entries from a single agent's
        ``for-fleet/knowledge.json`` without syncing.
        """
        knowledge_path = os.path.join(repo_path, _FLEET_KNOWLEDGE_REL)
        if not os.path.isfile(knowledge_path):
            return []

        with open(knowledge_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        raw_entries = data if isinstance(data, list) else data.get("entries", [])
        entries: list[KnowledgeEntry] = []
        for raw in raw_entries:
            try:
                entries.append(KnowledgeEntry.from_dict(raw))
            except (TypeError, KeyError):
                continue
        return entries
