"""
Semantic Scorer

Encodes job description and candidate texts using sentence-transformers,
then computes cosine similarity via FAISS for fast retrieval.
"""

import os

# Prevent macOS multiprocessing fork issues — must be set before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from pathlib import Path

from config.settings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDINGS_CACHE,
    PROCESSED_DIR,
)


class SemanticScorer:
    """
    Compute semantic similarity between a job description and candidate texts
    using sentence-transformers + FAISS.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the semantic scorer.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        if model_name is None:
            model_name = EMBEDDING_MODEL_NAME

        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"  Embedding dimension: {self.dimension}")

    def score(
        self,
        jd_text: str,
        candidate_texts: List[str],
        candidate_ids: List[str],
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """
        Compute semantic similarity scores between JD and all candidates.

        Args:
            jd_text: The job description text (semantic version).
            candidate_texts: List of enriched candidate texts.
            candidate_ids: List of candidate IDs (same order as texts).
            use_cache: Whether to cache/load embeddings from disk.

        Returns:
            Dict mapping candidate_id → semantic_score (0 to 1).
        """
        assert len(candidate_texts) == len(candidate_ids), \
            "candidate_texts and candidate_ids must have same length"

        # Encode candidates
        candidate_embeddings = self._encode_candidates(candidate_texts, use_cache)

        # Encode job description
        jd_embedding = self.model.encode([jd_text], convert_to_numpy=True)
        faiss.normalize_L2(jd_embedding)

        # Build FAISS index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine after normalization
        index.add(candidate_embeddings)

        # Search for all candidates (k = total candidates)
        k = len(candidate_ids)
        similarities, indices = index.search(jd_embedding, k)

        # Build score dict — collect raw similarities first
        raw_scores = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if 0 <= idx < len(candidate_ids):
                raw_scores[candidate_ids[idx]] = float(sim)

        # Min-max normalize across the candidate pool for better spread
        if raw_scores:
            min_sim = min(raw_scores.values())
            max_sim = max(raw_scores.values())
            spread = max_sim - min_sim

            if spread > 1e-6:
                scores = {
                    cid: (sim - min_sim) / spread
                    for cid, sim in raw_scores.items()
                }
            else:
                # All scores identical — everyone gets 0.5
                scores = {cid: 0.5 for cid in raw_scores}
        else:
            scores = {}

        return scores

    def _encode_candidates(
        self, texts: List[str], use_cache: bool
    ) -> np.ndarray:
        """Encode candidate texts, using disk cache if available."""
        cache_path = EMBEDDINGS_CACHE

        if use_cache and cache_path.exists():
            print(f"  Loading cached embeddings from {cache_path.name}")
            embeddings = np.load(str(cache_path))
            if embeddings.shape[0] == len(texts):
                return embeddings
            print(f"  Cache size mismatch ({embeddings.shape[0]} vs {len(texts)}), re-encoding")

        print(f"  Encoding {len(texts)} candidate texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        print(f"  Encoding complete. Shape: {embeddings.shape}")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Cache to disk
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), embeddings)
        print(f"  Cached embeddings to {cache_path.name}")

        return embeddings
