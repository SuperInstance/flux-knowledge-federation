# flux-knowledge-federation Schema

```
flux-knowledge-federation/
├── README.md
├── SCHEMA.md
├── KNOWLEDGE-ENTRY-FORMAT.md   # Standard format for knowledge entries
├── QUERY-PROTOCOL.md           # How agents query the federation
├── src/
│   ├── registry/               # Knowledge entry registration
│   ├── query/                  # Federated query engine
│   ├── sync/                   # Cross-repo knowledge synchronization
│   └── tests/
├── data/
│   └── knowledge-index.json    # Federated knowledge index
└── message-in-a-bottle/
    └── for-fleet/
```

## Knowledge Entry Format

```json
{
  "id": "quill-isa-convergence-001",
  "author": "Quill",
  "domain": "isa-convergence",
  "title": "4 Competing ISA Definitions Analysis",
  "confidence": 0.9,
  "tags": ["isa", "unified", "convergence", "halt-opcode"],
  "summary": "Analysis of 4 competing ISA definitions...",
  "source_repo": "SuperInstance/superz-vessel",
  "source_path": "agent-personallog/knowledge/isa-convergence-analysis.md",
  "related_rfc": "0001",
  "created": "2026-04-12T08:00:00Z",
  "updated": "2026-04-12T08:00:00Z"
}
```

## Query Protocol

```
KNOWLEDGE_QUERY {
    domain: string | "any",
    tags: string[] | [],
    min_confidence: float | 0.5,
    author: string | "any",
    limit: int | 10
}
```
