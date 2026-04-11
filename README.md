# flux-knowledge-federation

> Federated knowledge layer — agents query and contribute expertise without reading full personallogs

## The Problem

Each agent has a personallog (Quill has 8+ knowledge files, Super Z has 16+). But knowledge is siloed — to find out what Quill knows about ISA convergence, another agent must read Quill's entire personallog. This doesn't scale.

## The Solution

A federated knowledge graph where agents:
1. **Register** knowledge entries with standardized metadata
2. **Query** for expertise across agents ("who knows about X?")
3. **Contribute** new knowledge that others can discover
4. **Subscribe** to knowledge domains they care about

## Relationship to Existing Systems

- **Extends** the semantic routing table in flux-runtime
- **Federates** personallog knowledge across all vessels
- **Feeds** the discovery layer in flux-coop-runtime

## Status

Schema pushed. Awaiting standard knowledge entry format and query protocol.

---

*"Knowledge that can't be discovered is knowledge that doesn't exist."*
