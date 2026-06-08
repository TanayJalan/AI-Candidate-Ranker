#!/usr/bin/env python3
"""
AI Candidate Ranking System — Main Entry Point

Usage:
    python rank.py
    python rank.py --candidates ./data/raw/sample_candidates.json --out ./data/output/submission.csv
    python rank.py --help

This script orchestrates the full ranking pipeline:
1. Parse job description → extract requirements
2. Load & normalize candidate data
3. Build rich text representations for embedding
4. Compute semantic similarity scores (sentence-transformers + FAISS)
5. Compute structured scores (title, skills, experience, industry, education)
6. Compute behavioral scores (redrob platform signals)
7. Combine into hybrid ranking with bonus modifiers
8. Generate reasoning strings
9. Output submission.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    CANDIDATES_JSON,
    SUBMISSION_CSV,
    OUTPUT_DIR,
    TOP_K,
)
from src.job_parser import parse_job_description, get_jd_summary
from src.candidate_loader import load_candidates
from src.text_enricher import enrich_all_candidates
from src.semantic_scorer import SemanticScorer
from src.structured_scorer import score_all_candidates as score_structured
from src.behavioral_scorer import score_all_candidates as score_behavioral
from src.hybrid_ranker import rank_candidates, print_ranking_summary
from src.reasoning_generator import generate_all_reasoning
from src.honeypot_detector import detect_all_honeypots


def main():
    parser = argparse.ArgumentParser(
        description="AI Candidate Ranking System — Rank candidates against a job description"
    )
    parser.add_argument(
        "--candidates", "-c",
        type=str,
        default=str(CANDIDATES_JSON),
        help=f"Path to candidates JSON/JSONL file (default: {CANDIDATES_JSON})",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default=str(SUBMISSION_CSV),
        help=f"Output path for submission CSV (default: {SUBMISSION_CSV})",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K,
        help=f"Number of candidates to rank (default: {TOP_K})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache (re-encode all candidates)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed scoring breakdown",
    )
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  AI CANDIDATE RANKING SYSTEM")
    print("=" * 70)

    # ── Step 1: Parse Job Description ────────────────────────────────────
    print("\n[1/7] Parsing job description...")
    jd = parse_job_description()
    print(f"  JD: {get_jd_summary()}")
    print(f"  Required skills: {len(jd['requirements']['required_skills'])}")
    print(f"  Semantic text length: {len(jd['semantic_text'])} chars")

    # ── Step 2: Load Candidates ──────────────────────────────────────────
    print("\n[2/7] Loading candidates...")
    candidates = load_candidates(Path(args.candidates))
    actual_k = min(args.top_k, len(candidates))
    print(f"  Will rank top {actual_k} of {len(candidates)} candidates")

    # ── Step 2.5: Honeypot Detection ─────────────────────────────────────
    print("\n[2.5/7] Running honeypot detection...")
    honeypot_results = detect_all_honeypots(candidates, verbose=args.verbose)
    honeypot_ids = {cid for cid, r in honeypot_results.items() if r["is_honeypot"]}
    print(f"  Flagged {len(honeypot_ids)} potential honeypots")

    # ── Step 3: Enrich Candidate Texts ───────────────────────────────────
    print("\n[3/7] Building rich text representations...")
    enriched = enrich_all_candidates(candidates)
    candidate_ids = [cid for cid, _ in enriched]
    candidate_texts = [text for _, text in enriched]
    print(f"  Enriched {len(enriched)} candidate profiles")
    if args.verbose and enriched:
        print(f"  Sample text ({enriched[0][0]}): {enriched[0][1][:200]}...")

    # ── Step 4: Semantic Scoring ─────────────────────────────────────────
    print("\n[4/7] Computing semantic similarity scores...")
    semantic_scorer = SemanticScorer()
    semantic_scores = semantic_scorer.score(
        jd_text=jd["semantic_text"],
        candidate_texts=candidate_texts,
        candidate_ids=candidate_ids,
        use_cache=not args.no_cache,
    )
    avg_sem = sum(semantic_scores.values()) / len(semantic_scores) if semantic_scores else 0
    print(f"  Average semantic score: {avg_sem:.4f}")

    # ── Step 5: Structured Scoring ───────────────────────────────────────
    print("\n[5/7] Computing structured match scores...")
    structured_scores = score_structured(candidates)
    avg_str = sum(structured_scores.values()) / len(structured_scores) if structured_scores else 0
    print(f"  Average structured score: {avg_str:.4f}")

    # Zero out honeypot structured scores
    for cid in honeypot_ids:
        if cid in structured_scores:
            structured_scores[cid] *= 0.1  # Severely penalize, don't fully zero)

    # ── Step 6: Behavioral Scoring ───────────────────────────────────────
    print("\n[6/7] Computing behavioral scores...")
    behavioral_scores = score_behavioral(candidates)
    avg_beh = sum(behavioral_scores.values()) / len(behavioral_scores) if behavioral_scores else 0
    print(f"  Average behavioral score: {avg_beh:.4f}")

    # ── Step 7: Combine & Rank ───────────────────────────────────────────
    print("\n[7/7] Combining scores and generating final ranking...")
    ranked = rank_candidates(
        candidates=candidates,
        semantic_scores=semantic_scores,
        structured_scores=structured_scores,
        behavioral_scores=behavioral_scores,
        top_k=actual_k,
    )

    # Generate reasoning strings (pass honeypot info)
    generate_all_reasoning(ranked, honeypot_results=honeypot_results)

    # Print summary
    print_ranking_summary(ranked, top_n=min(15, actual_k))

    # ── Write Submission CSV ─────────────────────────────────────────────
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for entry in ranked:
            writer.writerow([
                entry["candidate_id"],
                entry["rank"],
                f"{entry['final_score']:.4f}",
                entry["reasoning"],
            ])

    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"  DONE — Submission written to: {output_path}")
    print(f"  Candidates ranked: {len(ranked)}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")

    # Verbose mode: print detailed breakdown
    if args.verbose:
        print("\n  DETAILED SCORE BREAKDOWN:")
        print(f"  {'ID':<16} {'Sem':>7} {'Str':>7} {'Beh':>7} {'Bon':>7} {'Final':>7}")
        print(f"  {'-'*52}")
        for entry in ranked:
            print(
                f"  {entry['candidate_id']:<16} "
                f"{entry['semantic_score']:>7.4f} "
                f"{entry['structured_score']:>7.4f} "
                f"{entry['behavioral_score']:>7.4f} "
                f"{entry['bonus_score']:>7.4f} "
                f"{entry['final_score']:>7.4f}"
            )


if __name__ == "__main__":
    main()
