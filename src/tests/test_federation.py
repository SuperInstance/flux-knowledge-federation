"""
Tests for the flux-knowledge-federation layer.

Covers:
  - KnowledgeRegistry: register, query, search, remove, persistence, merge
  - QueryEngine: natural language questions, ranking, domain filtering
  - FederationSync: delta updates, conflict resolution, git-based transport
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from ..registry.knowledge_registry import KnowledgeEntry, KnowledgeRegistry
from ..query.query_engine import Query, QueryEngine, QueryResult
from ..sync.federation_sync import FederationSync, SyncResult


# ── Helpers ───────────────────────────────────────────────────────

def _make_entry(
    agent_name: str = "Quill",
    domain: str = "test-domain",
    topics: list[str] | None = None,
    confidence: float = 0.8,
    entry_id: str = "entry-001",
    last_updated: str = "2026-04-12T08:00:00Z",
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        agent_name=agent_name,
        domain=domain,
        topics=topics or ["topic-a", "topic-b"],
        confidence=confidence,
        evidence_urls=["http://example.com/evidence"],
        last_updated=last_updated,
    )


# ══════════════════════════════════════════════════════════════════
# KnowledgeRegistry tests
# ══════════════════════════════════════════════════════════════════

class TestRegisterAndQuery(unittest.TestCase):
    """Test basic register and query operations."""

    def test_register_single_entry(self) -> None:
        reg = KnowledgeRegistry()
        entry = _make_entry()
        reg.register(entry)
        self.assertEqual(len(reg), 1)

    def test_register_multiple_entries(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(entry_id="e1"))
        reg.register(_make_entry(entry_id="e2", domain="other"))
        self.assertEqual(len(reg), 2)

    def test_register_replaces_same_id(self) -> None:
        reg = KnowledgeRegistry()
        e1 = _make_entry(confidence=0.5, entry_id="dup")
        e2 = _make_entry(confidence=0.9, entry_id="dup")
        reg.register(e1)
        reg.register(e2)
        self.assertEqual(len(reg), 1)
        self.assertEqual(reg.entries[0].confidence, 0.9)

    def test_register_invalid_type_raises(self) -> None:
        reg = KnowledgeRegistry()
        with self.assertRaises(TypeError):
            reg.register("not an entry")  # type: ignore[arg-type]

    def test_query_by_domain(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(domain="isa-convergence", entry_id="e1"))
        reg.register(_make_entry(domain="vm-design", entry_id="e2"))
        reg.register(_make_entry(domain="isa-convergence", entry_id="e3"))

        results = reg.query(by_domain="isa-convergence")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.domain, "isa-convergence")

    def test_query_by_agent(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(agent_name="Quill", entry_id="e1"))
        reg.register(_make_entry(agent_name="Oracle1", entry_id="e2"))

        results = reg.query(by_agent="Quill")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].agent_name, "Quill")

    def test_query_by_min_confidence(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(confidence=0.4, entry_id="low"))
        reg.register(_make_entry(confidence=0.7, entry_id="mid"))
        reg.register(_make_entry(confidence=0.95, entry_id="high"))

        results = reg.query(by_min_confidence=0.7)
        self.assertEqual(len(results), 2)

    def test_query_by_topic(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(topics=["isa", "convergence"], entry_id="e1"))
        reg.register(_make_entry(topics=["vm", "design"], entry_id="e2"))

        results = reg.query(by_topic="convergence")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "e1")

    def test_query_combined_filters(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(domain="d1", confidence=0.9, topics=["alpha"], entry_id="e1"))
        reg.register(_make_entry(domain="d1", confidence=0.3, topics=["alpha"], entry_id="e2"))
        reg.register(_make_entry(domain="d2", confidence=0.9, topics=["alpha"], entry_id="e3"))

        results = reg.query(by_domain="d1", by_min_confidence=0.5, by_topic="alpha")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "e1")

    def test_query_returns_sorted_by_confidence(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(confidence=0.5, entry_id="low"))
        reg.register(_make_entry(confidence=0.9, entry_id="high"))
        reg.register(_make_entry(confidence=0.7, entry_id="mid"))

        results = reg.query()
        confs = [r.confidence for r in results]
        self.assertEqual(confs, [0.9, 0.7, 0.5])


class TestSearch(unittest.TestCase):
    """Test full-text search across domains and topics."""

    def test_search_finds_matching_entries(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(
            domain="isa-convergence",
            topics=["halt-opcode", "instruction-set"],
            entry_id="e1",
        ))
        reg.register(_make_entry(
            domain="vm-design",
            topics=["virtual-machine"],
            entry_id="e2",
        ))

        results = reg.search("halt opcode instruction")
        ids = [r.id for r in results]
        self.assertIn("e1", ids)
        self.assertNotIn("e2", ids)

    def test_search_empty_query(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry())
        self.assertEqual(reg.search(""), [])
        self.assertEqual(reg.search("   "), [])

    def test_search_limit(self) -> None:
        reg = KnowledgeRegistry()
        for i in range(10):
            reg.register(_make_entry(topics=["keyword"], entry_id=f"e{i}"))
        results = reg.search("keyword", limit=3)
        self.assertEqual(len(results), 3)


class TestRemove(unittest.TestCase):
    """Test entry removal."""

    def test_remove_by_entry_id(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(entry_id="e1"))
        reg.register(_make_entry(entry_id="e2"))
        removed = reg.remove(entry_id="e1")
        self.assertEqual(removed, 1)
        self.assertEqual(len(reg), 1)

    def test_remove_by_agent(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(agent_name="Quill", entry_id="e1"))
        reg.register(_make_entry(agent_name="Oracle1", entry_id="e2"))
        reg.register(_make_entry(agent_name="Quill", entry_id="e3"))
        removed = reg.remove(agent_name="Quill")
        self.assertEqual(removed, 2)
        self.assertEqual(len(reg), 1)

    def test_remove_by_domain(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(domain="d1", entry_id="e1"))
        reg.register(_make_entry(domain="d2", entry_id="e2"))
        removed = reg.remove(domain="d1")
        self.assertEqual(removed, 1)


class TestPersistence(unittest.TestCase):
    """Test JSON save/load round-trip."""

    def test_save_and_load(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(entry_id="e1"))
        reg.register(_make_entry(entry_id="e2", domain="other"))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            reg.save(tmp_path)
            loaded = KnowledgeRegistry.load(tmp_path)
            self.assertEqual(len(loaded), 2)
            domains = sorted(e.domain for e in loaded.entries)
            self.assertEqual(domains, ["other", "test-domain"])
        finally:
            os.unlink(tmp_path)


class TestMerge(unittest.TestCase):
    """Test merging registries from multiple agents."""

    def test_merge_disjoint_registries(self) -> None:
        reg_a = KnowledgeRegistry([
            _make_entry(agent_name="Quill", entry_id="e1"),
        ])
        reg_b = KnowledgeRegistry([
            _make_entry(agent_name="Oracle1", entry_id="e2"),
        ])

        changed = reg_a.merge(reg_b)
        self.assertEqual(len(reg_a), 2)
        self.assertEqual(len(changed), 1)

    def test_merge_conflict_higher_confidence_wins(self) -> None:
        e_low = _make_entry(confidence=0.4, entry_id="conflict")
        e_high = _make_entry(confidence=0.9, entry_id="conflict")

        reg_a = KnowledgeRegistry([e_low])
        reg_b = KnowledgeRegistry([e_high])

        changed = reg_a.merge(reg_b)
        self.assertEqual(len(reg_a), 1)
        self.assertEqual(reg_a.entries[0].confidence, 0.9)
        self.assertEqual(len(changed), 1)

    def test_merge_conflict_lower_confidence_ignored(self) -> None:
        e_high = _make_entry(confidence=0.9, entry_id="conflict")
        e_low = _make_entry(confidence=0.4, entry_id="conflict")

        reg_a = KnowledgeRegistry([e_high])
        reg_b = KnowledgeRegistry([e_low])

        changed = reg_a.merge(reg_b)
        self.assertEqual(len(reg_a), 1)
        self.assertEqual(reg_a.entries[0].confidence, 0.9)
        self.assertEqual(len(changed), 0)

    def test_merge_from_multiple_agents(self) -> None:
        reg = KnowledgeRegistry()
        quill = KnowledgeRegistry([
            _make_entry(agent_name="Quill", domain="isa-convergence", entry_id="q1"),
            _make_entry(agent_name="Quill", domain="protocol-design", entry_id="q2"),
        ])
        superz = KnowledgeRegistry([
            _make_entry(agent_name="Super Z", domain="fleet-auditing", entry_id="sz1"),
        ])
        oracle = KnowledgeRegistry([
            _make_entry(agent_name="Oracle1", domain="vm-design", entry_id="o1"),
        ])

        reg.merge(quill)
        reg.merge(superz)
        reg.merge(oracle)

        self.assertEqual(len(reg), 4)
        self.assertEqual(sorted(reg.agents), ["Oracle1", "Quill", "Super Z"])

    def test_agents_and_domains_properties(self) -> None:
        reg = KnowledgeRegistry()
        reg.register(_make_entry(agent_name="Quill", domain="d1", entry_id="e1"))
        reg.register(_make_entry(agent_name="Quill", domain="d2", entry_id="e2"))
        reg.register(_make_entry(agent_name="Oracle1", domain="d1", entry_id="e3"))

        self.assertEqual(sorted(reg.agents), ["Oracle1", "Quill"])
        self.assertEqual(sorted(reg.domains), ["d1", "d2"])


# ══════════════════════════════════════════════════════════════════
# QueryEngine tests
# ══════════════════════════════════════════════════════════════════

class TestQueryEngineRanking(unittest.TestCase):
    """Test query engine relevance scoring and ranking."""

    def _build_registry(self) -> KnowledgeRegistry:
        reg = KnowledgeRegistry([
            KnowledgeEntry(
                id="q1", agent_name="Quill", domain="isa-convergence",
                topics=["isa", "unified", "convergence", "halt-opcode"],
                confidence=0.92, last_updated="2026-04-12T08:00:00Z",
            ),
            KnowledgeEntry(
                id="q2", agent_name="Quill", domain="protocol-design",
                topics=["protocol", "message-format", "handshake"],
                confidence=0.88, last_updated="2026-04-12T08:00:00Z",
            ),
            KnowledgeEntry(
                id="o1", agent_name="Oracle1", domain="opcode-specification",
                topics=["opcode", "halt", "encoding", "instruction"],
                confidence=0.94, last_updated="2026-04-12T08:00:00Z",
            ),
            KnowledgeEntry(
                id="sz1", agent_name="Super Z", domain="fleet-auditing",
                topics=["fleet", "auditing", "compliance"],
                confidence=0.90, last_updated="2026-04-12T08:00:00Z",
            ),
        ])
        return reg

    def test_basic_question_returns_results(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(question="Who knows about halt opcodes?"))
        self.assertGreater(len(results), 0)

    def test_results_sorted_by_relevance(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(question="halt opcode instruction"))
        scores = [r.relevance_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_domain_filter(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(
            question="isa convergence halt",
            domains=["isa-convergence"],
        ))
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.entry.domain, "isa-convergence")

    def test_min_confidence_filter(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(
            question="protocol",
            min_confidence=0.9,
        ))
        for r in results:
            self.assertGreaterEqual(r.entry.confidence, 0.9)

    def test_max_results_limit(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(question="isa halt opcode", max_results=2))
        self.assertLessEqual(len(results), 2)

    def test_empty_question_returns_empty(self) -> None:
        engine = QueryEngine(self._build_registry())
        self.assertEqual(engine.ask(Query(question="")), [])
        self.assertEqual(engine.ask(Query(question="the a an")), [])

    def test_no_matching_entries(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(question="quantum entanglement"))
        self.assertEqual(len(results), 0)

    def test_source_agent_populated(self) -> None:
        engine = QueryEngine(self._build_registry())
        results = engine.ask(Query(question="opcode"))
        for r in results:
            self.assertEqual(r.source_agent, r.entry.agent_name)

    def test_fleet_knowledge_json_loads(self) -> None:
        """Verify the shipped fleet-knowledge.json loads correctly."""
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
        )
        path = os.path.join(data_dir, "fleet-knowledge.json")
        if not os.path.isfile(path):
            self.skipTest("fleet-knowledge.json not found")
        reg = KnowledgeRegistry.load(path)
        self.assertGreater(len(reg), 0)
        self.assertIn("Quill", reg.agents)
        self.assertIn("Super Z", reg.agents)
        self.assertIn("Oracle1", reg.agents)

    def test_query_engine_with_fleet_data(self) -> None:
        """End-to-end: load fleet data, ask a question, get ranked results."""
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
        )
        path = os.path.join(data_dir, "fleet-knowledge.json")
        if not os.path.isfile(path):
            self.skipTest("fleet-knowledge.json not found")
        reg = KnowledgeRegistry.load(path)
        engine = QueryEngine(reg)
        results = engine.ask(Query(question="Who knows about VM opcode encoding?"))
        self.assertGreater(len(results), 0)
        # Oracle1 should be top result for opcode question
        self.assertEqual(results[0].source_agent, "Oracle1")


# ══════════════════════════════════════════════════════════════════
# FederationSync tests
# ══════════════════════════════════════════════════════════════════

class TestFederationSync(unittest.TestCase):
    """Test sync protocol with simulated agent repos."""

    def _make_agent_repo(
        self,
        entries: list[dict],
    ) -> str:
        """Create a temp directory mimicking an agent repo with for-fleet/."""
        repo_dir = tempfile.mkdtemp()
        fleet_dir = os.path.join(
            repo_dir, "message-in-a-bottle", "for-fleet"
        )
        os.makedirs(fleet_dir, exist_ok=True)
        knowledge_path = os.path.join(fleet_dir, "knowledge.json")
        with open(knowledge_path, "w", encoding="utf-8") as fh:
            json.dump({"entries": entries}, fh, indent=2)
        return repo_dir

    def test_pull_from_agent_repo(self) -> None:
        entries = [
            {
                "id": "q1", "agent_name": "Quill",
                "domain": "isa-convergence",
                "topics": ["isa", "convergence"],
                "confidence": 0.9,
                "evidence_urls": [],
                "last_updated": "2026-04-12T08:00:00Z",
            }
        ]
        repo_path = self._make_agent_repo(entries)
        try:
            reg = KnowledgeRegistry()
            sync = FederationSync(reg, {"Quill": repo_path})
            result = sync.pull("Quill")
            self.assertTrue(result.success)
            self.assertEqual(result.new_entries, 1)
            self.assertEqual(len(reg), 1)
        finally:
            import shutil
            shutil.rmtree(repo_path)

    def test_pull_missing_repo(self) -> None:
        reg = KnowledgeRegistry()
        sync = FederationSync(reg, {})
        result = sync.pull("Nobody")
        self.assertFalse(result.success)

    def test_pull_missing_knowledge_file(self) -> None:
        repo_dir = tempfile.mkdtemp()
        try:
            reg = KnowledgeRegistry()
            sync = FederationSync(reg, {"Quill": repo_dir})
            result = sync.pull("Quill")
            self.assertFalse(result.success)
        finally:
            import shutil
            shutil.rmtree(repo_dir)

    def test_delta_sync_skips_unchanged(self) -> None:
        entries = [
            {
                "id": "d1", "agent_name": "Quill",
                "domain": "test", "topics": ["a"],
                "confidence": 0.5,
                "evidence_urls": [],
                "last_updated": "2026-04-10T08:00:00Z",
            },
            {
                "id": "d2", "agent_name": "Quill",
                "domain": "test", "topics": ["b"],
                "confidence": 0.8,
                "evidence_urls": [],
                "last_updated": "2026-04-14T08:00:00Z",
            },
        ]
        repo_path = self._make_agent_repo(entries)
        try:
            reg = KnowledgeRegistry()
            sync = FederationSync(reg, {"Quill": repo_path})

            # Delta sync since April 12 — only d2 should come through
            result = sync.delta_sync("Quill", since="2026-04-12T00:00:00Z")
            self.assertTrue(result.success)
            self.assertEqual(result.new_entries, 1)
            self.assertEqual(result.skipped_unchanged, 1)
            self.assertEqual(len(reg), 1)
        finally:
            import shutil
            shutil.rmtree(repo_path)

    def test_conflict_resolution_higher_confidence_wins(self) -> None:
        # Existing entry with lower confidence
        existing = _make_entry(
            entry_id="conflict", confidence=0.4,
            last_updated="2026-04-12T08:00:00Z",
        )
        reg = KnowledgeRegistry([existing])

        # Incoming from another agent with higher confidence
        incoming_entries = [
            {
                "id": "conflict", "agent_name": "Quill",
                "domain": "test", "topics": ["updated"],
                "confidence": 0.95,
                "evidence_urls": [],
                "last_updated": "2026-04-13T08:00:00Z",
            }
        ]
        repo_path = self._make_agent_repo(incoming_entries)
        try:
            sync = FederationSync(reg, {"Quill": repo_path})
            result = sync.pull("Quill")
            self.assertTrue(result.success)
            self.assertEqual(result.updated_entries, 1)
            self.assertEqual(reg.entries[0].confidence, 0.95)
            self.assertEqual(reg.entries[0].topics, ["updated"])
        finally:
            import shutil
            shutil.rmtree(repo_path)

    def test_conflict_resolution_lower_confidence_kept(self) -> None:
        existing = _make_entry(
            entry_id="conflict", confidence=0.95,
            last_updated="2026-04-12T08:00:00Z",
        )
        reg = KnowledgeRegistry([existing])

        incoming_entries = [
            {
                "id": "conflict", "agent_name": "Quill",
                "domain": "test", "topics": ["stale"],
                "confidence": 0.3,
                "evidence_urls": [],
                "last_updated": "2026-04-13T08:00:00Z",
            }
        ]
        repo_path = self._make_agent_repo(incoming_entries)
        try:
            sync = FederationSync(reg, {"Quill": repo_path})
            result = sync.pull("Quill")
            # The entry is seen but skipped because existing has higher confidence
            self.assertEqual(result.new_entries, 0)
            self.assertEqual(reg.entries[0].confidence, 0.95)
            self.assertEqual(reg.entries[0].topics, ["topic-a", "topic-b"])
        finally:
            import shutil
            shutil.rmtree(repo_path)

    def test_pull_all_across_agents(self) -> None:
        repo_a = self._make_agent_repo([
            {
                "id": "a1", "agent_name": "AgentA",
                "domain": "d1", "topics": ["t1"],
                "confidence": 0.8, "evidence_urls": [],
                "last_updated": "2026-04-12T08:00:00Z",
            }
        ])
        repo_b = self._make_agent_repo([
            {
                "id": "b1", "agent_name": "AgentB",
                "domain": "d2", "topics": ["t2"],
                "confidence": 0.9, "evidence_urls": [],
                "last_updated": "2026-04-12T08:00:00Z",
            }
        ])
        try:
            reg = KnowledgeRegistry()
            sync = FederationSync(reg, {"AgentA": repo_a, "AgentB": repo_b})
            results = sync.pull_all()
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.success for r in results.values()))
            self.assertEqual(len(reg), 2)
        finally:
            import shutil
            shutil.rmtree(repo_a)
            shutil.rmtree(repo_b)

    def test_sync_status_tracking(self) -> None:
        repo_path = self._make_agent_repo([
            {
                "id": "s1", "agent_name": "Quill",
                "domain": "d", "topics": ["t"],
                "confidence": 0.8, "evidence_urls": [],
                "last_updated": "2026-04-12T08:00:00Z",
            }
        ])
        try:
            reg = KnowledgeRegistry()
            sync = FederationSync(reg, {"Quill": repo_path})
            sync.pull("Quill")
            status = sync.sync_status["Quill"]
            self.assertEqual(status.entries_pulled, 1)
            self.assertEqual(status.entries_updated, 0)
            self.assertNotEqual(status.last_sync, "")
        finally:
            import shutil
            shutil.rmtree(repo_path)


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
