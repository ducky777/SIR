import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .embeddings import Embedder


@dataclass
class RetrievalResult:
    """Container for document and similarity score"""

    document: str
    similarity: float
    doc_id: str  # Added document ID field
    metadata: Optional[Dict[str, Any]] = None

    def __lt__(self, other):
        # Enable sorting by similarity in descending order
        if not isinstance(other, RetrievalResult):
            return NotImplemented
        return self.similarity > other.similarity


@dataclass
class HopResult:
    """Container for tracking information about each retrieval hop"""

    hop_number: int
    query: str
    result: RetrievalResult
    elapsed_time: float


class SIR:
    """Self-Iterating Retriever for multi-hop question answering"""

    def __init__(
        self,
        retrieval_fn: Callable,
        reformulation_fn: Optional[Callable] = None,
        similarity_threshold: float = 0.7,
        max_hops: int = 5,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        embedder: Optional[Union["Embedder", Dict[str, Any]]] = None,
    ):
        # Initialize basic parameters
        self.retrieval_fn = retrieval_fn
        self.reformulation_fn = reformulation_fn or self._default_reformulation
        self.similarity_threshold = similarity_threshold
        self.max_hops = max_hops
        self.verbose = verbose
        self.logger = logger or self._setup_default_logger()

        # Setup embedder if provided
        self.embedder = None
        if embedder is not None:
            if isinstance(embedder, "Embedder"):  # Replace with actual import
                self.embedder = embedder
            elif isinstance(embedder, dict):
                # Import Embedder here to avoid circular import
                from .embeddings import Embedder

                self.embedder = Embedder(**embedder)
            else:
                raise TypeError("embedder must be an Embedder instance or dict")

    def retrieve(
        self, query: str, top_k: int = 1, max_sim: Optional[float] = None
    ) -> Tuple[List[List[RetrievalResult]], List[HopResult]]:
        """Execute multi-hop retrieval process on a query"""
        self.logger.info(f"Starting retrieval for query: {query} with top_k={top_k}")

        # Use provided max_sim or fall back to similarity_threshold
        max_similarity = max_sim if max_sim is not None else self.similarity_threshold

        retrieved_results_by_hop = []
        hop_results = []
        current_query = query
        # Track document IDs that have already been retrieved
        seen_doc_ids: Set[str] = set()

        for hop in range(self.max_hops):
            start_time = time.time()

            # Get results from retrieval function
            try:
                raw_results = self.retrieval_fn(current_query)
                elapsed = time.time() - start_time

                # Process the retrieval results (might be single item or list)
                processed_results = self._process_retrieval_results(raw_results, top_k)

                # Filter out documents we've already seen using doc_ids
                new_results = []
                for result in processed_results:
                    if result.doc_id not in seen_doc_ids:
                        new_results.append(result)
                        seen_doc_ids.add(result.doc_id)

                processed_results = new_results

            except Exception as e:
                self.logger.error(f"Error in retrieval function: {str(e)}")
                raise RuntimeError(f"Retrieval function failed: {str(e)}") from e

            # Store results for this hop
            retrieved_results_by_hop.append(processed_results)

            # Get highest similarity score from this hop
            best_result = processed_results[0] if processed_results else None

            # Create hop result for tracking (using best result for metadata)
            hop_result = HopResult(
                hop_number=hop + 1,
                query=current_query,
                result=best_result,
                elapsed_time=elapsed,
            )
            hop_results.append(hop_result)

            # Log progress if verbose
            if self.verbose:
                print(f"Hop {hop + 1}: Retrieved {len(processed_results)} documents")
                for i, result in enumerate(processed_results[:3]):  # Show top 3 at most
                    print(
                        f"  {i + 1}. Similarity: {result.similarity:.4f}, Doc ID: {result.doc_id}, Doc: {result.document[:80]}..."
                    )
                if len(processed_results) > 3:
                    print(f"  ... and {len(processed_results) - 3} more")

            # Log debug info
            self.logger.debug(
                f"Hop {hop + 1}: retrieved {len(processed_results)} docs, "
                f"best_sim={best_result.similarity if best_result else 'N/A'}, "
                f"time={elapsed:.3f}s"
            )

            # Check stopping conditions - if any document exceeds the threshold
            if any(r.similarity >= max_similarity for r in processed_results):
                if self.verbose:
                    print(f"Stopping: Similarity threshold {max_similarity} reached")
                self.logger.info(
                    f"Stopping at hop {hop + 1}: Similarity threshold reached"
                )
                break

            # Check if we've reached max hops
            if hop + 1 >= self.max_hops:
                if self.verbose:
                    print(f"Stopping: Maximum hops ({self.max_hops}) reached")
                self.logger.info(f"Stopping: Maximum hops ({self.max_hops}) reached")
                break

            # Reformulate query for next hop
            try:
                # Collect all documents retrieved so far (flattened)
                all_docs = [
                    r.document
                    for hop_results in retrieved_results_by_hop
                    for r in hop_results
                ]

                current_query = self.reformulation_fn(query, all_docs)
                if not current_query:
                    raise ValueError("Reformulation function returned empty query")
            except Exception as e:
                self.logger.error(f"Error in reformulation function: {str(e)}")
                raise RuntimeError(f"Reformulation function failed: {str(e)}") from e

        return retrieved_results_by_hop, hop_results

    def _process_retrieval_results(
        self, results: Any, top_k: int = 1
    ) -> List[RetrievalResult]:
        """Convert various return types to a standardized list of RetrievalResults"""
        processed_results = []

        # Handle case where results is already a list
        if isinstance(results, list):
            for item in results:
                processed_results.append(self._process_single_result(item))
        else:
            # Handle single result
            processed_results.append(self._process_single_result(results))

        # Sort by similarity (descending) and limit to top_k
        processed_results.sort(key=lambda x: x.similarity, reverse=True)
        return processed_results[:top_k]

    def _process_single_result(self, result: Any) -> RetrievalResult:
        """Convert a single result to a standardized RetrievalResult"""
        if isinstance(result, RetrievalResult):
            return result

        if isinstance(result, tuple):
            if (
                len(result) >= 3
            ):  # Now expecting (document, similarity, doc_id, *optional_other_stuff)
                # Assume (document, similarity, doc_id, *optional_other_stuff)
                document, similarity, doc_id = result[0], result[1], result[2]
                metadata = {}
                # If there are more items in the tuple, add them to metadata
                if len(result) > 3:
                    metadata["additional_data"] = result[3:]
                return RetrievalResult(
                    document=str(document),
                    similarity=float(similarity),
                    doc_id=str(doc_id),
                    metadata=metadata,
                )
            else:
                raise ValueError(
                    f"Tuple result must have at least 3 elements (document, similarity, doc_id), got {len(result)}"
                )

        # If just a document is returned, generate a doc_id from the document content
        # This is a fallback for backward compatibility
        doc_id = str(hash(str(result)))
        return RetrievalResult(document=str(result), similarity=1.0, doc_id=doc_id)

    def _default_reformulation(
        self, original_query: str, retrieved_docs: List[str]
    ) -> str:
        """Simple default reformulation if none provided"""
        if not retrieved_docs:
            return original_query

        context = "\n".join([f"{i + 1}. {doc}" for i, doc in enumerate(retrieved_docs)])
        return f"{original_query}\n\nContext:\n{context}"

    def _setup_default_logger(self) -> logging.Logger:
        """Set up a default logger if none is provided"""
        logger = logging.getLogger("SIR")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)  # Default to warnings only
        return logger

    def get_answer(
        self, query: str, top_k: int = 1, max_sim: Optional[float] = None
    ) -> Tuple[str, List[str]]:
        """Convenience method to get final answer and supporting documents"""
        results_by_hop, _ = self.retrieve(query, top_k=top_k, max_sim=max_sim)

        # Flatten results from all hops
        all_results = [r for hop_results in results_by_hop for r in hop_results]

        if not all_results:
            return "", []

        # Find document with highest similarity across all hops
        best_result = max(all_results, key=lambda x: x.similarity)

        # Return all documents as supporting context
        all_docs = [r.document for r in all_results]

        return best_result.document, all_docs

    def get_reasoning_chain(
        self, query: str, top_k: int = 1, max_sim: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get detailed reasoning chain with all intermediate steps"""
        start_time = time.time()
        results_by_hop, hop_results = self.retrieve(query, top_k=top_k, max_sim=max_sim)
        total_time = time.time() - start_time

        return {
            "query": query,
            "total_hops": len(hop_results),
            "total_time": total_time,
            "hops": [
                {
                    "hop": hop.hop_number,
                    "query": hop.query,
                    "top_documents": [
                        {
                            "document": result.document,
                            "similarity": result.similarity,
                            "doc_id": result.doc_id,
                        }
                        for result in results_by_hop[i]
                        if i < len(results_by_hop)
                    ],
                    "time": hop.elapsed_time,
                }
                for i, hop in enumerate(hop_results)
            ],
            "final_similarity": hop_results[-1].result.similarity
            if hop_results
            else None,
        }
