"""flux-knowledge-federation: Federated knowledge layer."""

from .registry.knowledge_registry import KnowledgeEntry, KnowledgeRegistry
from .query.query_engine import Query, QueryEngine, QueryResult
from .sync.federation_sync import FederationSync, SyncResult

__all__ = [
    "KnowledgeEntry",
    "KnowledgeRegistry",
    "Query",
    "QueryEngine",
    "QueryResult",
    "FederationSync",
    "SyncResult",
]
