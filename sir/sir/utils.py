from typing import Any, Dict, List


def create_simple_tfidf_retriever(documents: List[str], return_top_k: int = 1):
    """Create a simple TF-IDF based retriever function"""
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError("sklearn is required for this retriever")

    tfidf = TfidfVectorizer().fit(documents)
    doc_vectors = tfidf.transform(documents)

    def retriever(query: str):
        query_vec = tfidf.transform([query])
        similarities = cosine_similarity(query_vec, doc_vectors)[0]

        if return_top_k == 1:
            best_idx = similarities.argmax()
            # Generate a simple doc_id based on index
            doc_id = f"doc_{best_idx}"
            return documents[best_idx], float(similarities[best_idx]), doc_id
        else:
            # Get indices of top_k highest similarities
            top_indices = np.argsort(similarities)[-return_top_k:][::-1]

            # Return list of (document, similarity, doc_id) tuples
            return [
                (documents[idx], float(similarities[idx]), f"doc_{idx}")
                for idx in top_indices
            ]

    return retriever


def create_embedding_retriever(
    documents: List[str], embedder_config: Dict[str, Any] = None, return_top_k: int = 1
):
    """Create a retriever function using embeddings"""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required for this retriever")

    # Import here to avoid circular imports
    from .embeddings import Embedder

    # Create embedder with provided config or defaults
    emb = Embedder(**(embedder_config or {}))

    # Create document embeddings
    doc_embeddings = np.array([emb.embed(doc) for doc in documents])

    def cosine_similarity(a, b):
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retriever(query: str):
        # Get query embedding
        query_embedding = np.array(emb.embed(query))

        # Calculate similarities
        similarities = [
            cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings
        ]

        # Get indices of top_k highest similarities
        top_indices = np.argsort(similarities)[-return_top_k:][::-1]

        # Return list of (document, similarity, doc_id) tuples
        return [
            (documents[idx], float(similarities[idx]), f"doc_{idx}")
            for idx in top_indices
        ]

    return retriever
