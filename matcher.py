"""
Candidate Matcher Module

This module provides functionality to match job descriptions to candidates
using semantic similarity with a pre-built FAISS index.
"""

import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class CandidateMatcher:
    """
    A class to match job descriptions to candidates using semantic similarity.

    Attributes:
        model: SentenceTransformer model for encoding text
        index: FAISS index for fast similarity search
        candidate_ids: List of candidate IDs corresponding to index entries
    """

    def __init__(self,
                 index_path: str = '/Users/tanayjalan/candidate_matcher/data/candidate_index.faiss',
                 ids_path: str = '/Users/tanayjalan/candidate_matcher/data/candidate_ids.pkl',
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the CandidateMatcher.

        Args:
            index_path: Path to the FAISS index file
            ids_path: Path to the candidate IDs pickle file
            model_name: Name of the SentenceTransformer model to use
        """
        print(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

        print(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        print(f"Index loaded with {self.index.ntotal} vectors")

        print(f"Loading candidate IDs from: {ids_path}")
        with open(ids_path, 'rb') as f:
            self.candidate_ids = pickle.load(f)
        print(f"Loaded {len(self.candidate_ids)} candidate IDs")

        # Verify that index size matches number of IDs
        assert self.index.ntotal == len(self.candidate_ids), \
            f"Index size ({self.index.ntotal}) does not match number of IDs ({len(self.candidate_ids)})"

    def match_candidates(self,
                        job_description: str,
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find top-k candidates similar to the given job description.

        Args:
            job_description: The job description text to match against
            top_k: Number of top candidates to return

        Returns:
            List of tuples (candidate_id, similarity_score) sorted by score descending
        """
        # Encode the job description
        query_embedding = self.model.encode([job_description], convert_to_numpy=True)

        # Normalize for cosine similarity (inner product with normalized vectors = cosine similarity)
        faiss.normalize_L2(query_embedding)

        # Search the index
        similarities, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.candidate_ids):  # Safety check
                candidate_id = self.candidate_ids[idx]
                results.append((candidate_id, float(similarity)))

        return results

    def get_candidate_count(self) -> int:
        """Get the total number of candidates in the index."""
        return len(self.candidate_ids)


def main():
    """Demo function to test the matcher."""
    # Initialize matcher
    matcher = CandidateMatcher()

    # Example job descriptions to test
    test_jobs = [
        "Senior Python developer with experience in machine learning and AWS",
        "Data scientist specializing in deep learning and natural language processing",
        "Frontend engineer with React and TypeScript expertise",
        "DevOps engineer knowledgeable in Kubernetes and Docker",
        "Product manager with background in agile methodologies and user experience design"
    ]

    print("\n" + "="*80)
    print("CANDIDATE MATCHER DEMO")
    print("="*80)

    for job_desc in test_jobs:
        print(f"\nJob Description: {job_desc}")
        print("-" * 80)

        # Get top 5 matches
        matches = matcher.match_candidates(job_desc, top_k=5)

        for i, (candidate_id, score) in enumerate(matches, 1):
            print(f"{i:2d}. Candidate ID: {candidate_id:<15} Similarity: {score:.4f}")

    print("\n" + "="*80)
    print(f"Total candidates in index: {matcher.get_candidate_count()}")
    print("="*80)


if __name__ == "__main__":
    main()