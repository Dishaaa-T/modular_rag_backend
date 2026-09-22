from .schemas import RAGResponse


class ModularRAG:
    """
    Orchestrates independent modules.

    Router -> Retriever -> Reranker -> Generator
    """

    def __init__(self, router, retrievers, rerankers, generators):
        self.router = router
        self.retrievers = retrievers
        self.rerankers = rerankers
        self.generators = generators

    def query(self, query: str, top_k: int):
        plan = self.router.plan(query, top_k)

        retriever = self.retrievers.get(plan.retrieval_mode)
        documents = retriever.retrieve(
            query=query,
            top_k=plan.top_k,
            filters=plan.filters,
        )

        reranker = self.rerankers.get("cross_encoder" if "cross_encoder" in self.rerankers.rerankers else "none")
        documents = reranker.rerank(
            query=query,
            documents=documents,
            top_k=min(plan.top_k, 5),
        )

        generator = self.generators.get("groq")
        answer = generator.generate(query, documents)

        return RAGResponse(
            answer=answer,
            sources=documents,
            plan=plan,
        )
