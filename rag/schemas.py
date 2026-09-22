from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievedDocument:
    id: str
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float | None = None
    similarity_score: float | None = None
    rank: int = 0
    rerank_score: float | None = None


@dataclass
class QueryPlan:
    query: str
    retrieval_mode: str = "dense"
    top_k: int = 5
    filters: Dict[str, Any] | None = None
    reason: str = "default semantic retrieval"


@dataclass
class RAGResponse:
    answer: str
    sources: List[RetrievedDocument]
    plan: QueryPlan
