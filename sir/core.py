import uuid
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

console = Console()


class RetrievedDocument(BaseModel):
    """Model for representing a retrieved document with its ID and content."""

    doc_id: str
    document: str
    score: Optional[float] = None

    def __str__(self):
        """Pretty string representation of the document."""
        return f"{self.document}" + (
            f" (Score: {self.score:.4f})" if self.score is not None else ""
        )


class DocStorage:
    """Document storage class with vector embedding capabilities."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        self._ids: List[str] = []
        self.docs: List[str] = []
        self.embeddings: List[torch.Tensor] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_docs(
        self, docs: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the storage with optional metadata."""
        if metadata is None:
            metadata = [{} for _ in docs]
        self._ids.extend([str(uuid.uuid4()) for _ in docs])
        self.docs.extend(docs)
        self.embeddings.extend(self.model.encode(docs, convert_to_tensor=True))
        self.metadata.extend(metadata)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        ignore_ids: Optional[List[str]] = None,
        use_hybrid: bool = False,
        bm25_weight: float = 0.5,
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents based on query similarity, with optional hybrid search.

        Args:
            query: The search query
            top_k: Number of documents to retrieve
            ignore_ids: List of document IDs to exclude from results
            use_hybrid: Whether to use hybrid search (combining dense and sparse retrieval)
            bm25_weight: Weight for BM25 in hybrid search (between 0.0 and 1.0)

        Returns:
            List of retrieved documents
        """
        # Get query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate similarities using cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), torch.stack(self.embeddings), dim=1
        )

        # Hybrid search implementation
        if use_hybrid:
            # Simple BM25-like implementation (term frequency based)
            bm25_scores = torch.zeros_like(similarities)
            query_terms = set(query.lower().split())

            for i, doc in enumerate(self.docs):
                doc_terms = set(doc.lower().split())
                term_overlap = len(query_terms.intersection(doc_terms))
                bm25_scores[i] = term_overlap / max(
                    1, len(doc_terms)
                )  # Normalize by doc length

            # Combine scores (linear interpolation)
            combined_scores = (
                1 - bm25_weight
            ) * similarities + bm25_weight * bm25_scores
        else:
            combined_scores = similarities

        # Set scores of ignored documents to -infinity
        if ignore_ids:
            for i, doc_id in enumerate(self._ids):
                if doc_id in ignore_ids:
                    combined_scores[i] = -float("inf")

        # Get top_k indices
        top_k_indices = torch.argsort(combined_scores, descending=True)[:top_k]

        return [
            RetrievedDocument(
                doc_id=self._ids[i],
                document=self.docs[i],
                score=combined_scores[i].item(),
            )
            for i in top_k_indices
        ]


class SIR:
    """Sequential Information Retrieval pipeline."""

    def __init__(
        self,
        model: SentenceTransformer,
        docs: DocStorage,
        use_cross_encoder: bool = False,
        cross_encoder_model: str = "Alibaba-NLP/gte-reranker-modernbert-base",
    ):
        """
        Initialize the SIR pipeline.

        Args:
            model: The sentence transformer model for embeddings
            docs: The document storage
            use_cross_encoder: Whether to use cross-encoder for reranking
            cross_encoder_model: Model name for the cross-encoder
        """
        self.model = model
        self.docs = docs
        self.use_cross_encoder = use_cross_encoder

        # Only load cross-encoder if enabled
        if use_cross_encoder:
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        else:
            self.cross_encoder = None

    def _hop(self, query: str, memory_documents: Optional[List[str]] = None) -> str:
        """Create a hopped query by combining the original query with memory documents."""
        if not memory_documents:
            return query

        # Format context from memory documents
        context = "".join(f"\n{j + 1}. {s}" for j, s in enumerate(memory_documents))
        return f"Query: {query}\n\nContext:\n{context}"

    def run(
        self,
        query: str,
        top_k: int = 3,
        max_hops: int = 3,
        use_hybrid: bool = False,
        bm25_weight: float = 0.5,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run the sequential information retrieval process.

        Args:
            query: The input query
            top_k: Number of documents to retrieve per hop
            max_hops: Maximum number of retrieval hops
            use_hybrid: Whether to use hybrid search
            bm25_weight: Weight for BM25 in hybrid search

        Returns:
            Dictionary with hop information
        """
        # Display initial query
        console.print(f"\n[bold cyan]Initial Query:[/] {query}\n")

        hops = {}
        memory_documents = []
        ignore_ids = []
        all_retrieved_docs = []  # Track all retrieved documents

        for i in range(max_hops):
            # Create query with context from previous hops
            query_context = self._hop(query, memory_documents)

            # Display concise hop info
            console.print(
                f"[bold magenta]Hop {i + 1}[/] - {'Using context' if memory_documents else 'Base query'}"
            )

            # Retrieve documents
            retrieved_docs = self.docs.retrieve(
                query_context,
                top_k,
                ignore_ids,
                use_hybrid=use_hybrid,
                bm25_weight=bm25_weight,
            )

            # Apply cross-encoder reranking if enabled
            if self.use_cross_encoder and retrieved_docs:
                pairs = [(query, doc.document) for doc in retrieved_docs]
                scores = self.cross_encoder.predict(pairs)

                for doc, score in zip(retrieved_docs, scores):
                    doc.score = score

                retrieved_docs.sort(
                    key=lambda x: x.score if x.score is not None else 0, reverse=True
                )
                retrieved_docs = retrieved_docs[:top_k]

            # Display concise results for this hop
            for j, doc in enumerate(retrieved_docs):
                score_str = f"{doc.score:.4f}" if doc.score is not None else "N/A"
                console.print(
                    f"  [green]{j + 1}.[/green] {doc.document[:60]}... [cyan]({score_str})[/cyan]"
                )

            # Add to memory and ignore list for next hop
            memory_documents.extend([doc.document for doc in retrieved_docs])
            ignore_ids.extend([doc.doc_id for doc in retrieved_docs])
            all_retrieved_docs.extend(
                [
                    {
                        "hop": i + 1,
                        "doc_id": doc.doc_id,
                        "document": doc.document,
                        "score": doc.score,
                    }
                    for doc in retrieved_docs
                ]
            )

            # Save hop data
            hops[i] = {
                "query": query_context,
                "retrieved_docs": [
                    {"doc_id": doc.doc_id, "document": doc.document, "score": doc.score}
                    for doc in retrieved_docs
                ],
            }

            console.print("[dim]" + "-" * 50 + "[/dim]\n")

        # Print summary of all retrieved documents
        console.print("\n[bold yellow]All Retrieved Documents Summary:[/bold yellow]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Hop", style="magenta", width=5)
        table.add_column("#", style="dim", width=3)
        table.add_column("Document", style="green")
        table.add_column("Score", style="cyan", width=10)

        # Group by hop for the summary table
        for i in range(max_hops):
            hop_docs = [doc for doc in all_retrieved_docs if doc["hop"] == i + 1]
            if not hop_docs:
                continue

            for j, doc in enumerate(hop_docs):
                score_str = f"{doc['score']:.4f}" if doc["score"] is not None else "N/A"
                table.add_row(f"{i + 1}", f"{j + 1}", doc["document"], score_str)

            # Add a separator row between hops, except after the last hop
            if i < max_hops - 1 and i < len(
                [d for d in all_retrieved_docs if d["hop"] == i + 2]
            ):
                table.add_row("", "", "[dim]" + "-" * 30 + "[/dim]", "")

        console.print(table)

        return hops
