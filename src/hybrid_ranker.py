"""
Hybrid Ranker

Combines semantic, structured, and behavioral scores into a final
composite ranking. Applies bonus modifiers for availability signals
and produces the final sorted candidate list.
"""

from typing import Dict, List, Any, Tuple

from config.settings import (
    WEIGHT_SEMANTIC,
    WEIGHT_STRUCTURED,
    WEIGHT_BEHAVIORAL,
    WEIGHT_BONUS,
    BONUS_OPEN_TO_WORK,
    BONUS_VERIFIED,
    BONUS_RELOCATE,
    BONUS_LINKEDIN,
    BONUS_LOW_NOTICE,
    BONUS_SAVED_BY_RECRUITERS,
    JD_REQUIREMENTS,
    TOP_K,
)


def compute_bonus(candidate: Dict[str, Any]) -> float:
    """
    Compute bonus modifier score (0 to 1) based on availability signals.

    These are additive bonuses that reward candidates who are actively
    available and engaged.
    """
    signals = candidate.get("redrob_signals", {})
    bonus = 0.0

    # Open to work flag
    if signals.get("open_to_work_flag", False):
        bonus += BONUS_OPEN_TO_WORK

    # Verified email + phone
    verified_count = 0
    if signals.get("verified_email", False):
        verified_count += 1
    if signals.get("verified_phone", False):
        verified_count += 1
    bonus += BONUS_VERIFIED * (verified_count / 2.0)

    # Willing to relocate
    if signals.get("willing_to_relocate", False):
        bonus += BONUS_RELOCATE

    # LinkedIn connected
    if signals.get("linkedin_connected", False):
        bonus += BONUS_LINKEDIN

    # Low notice period (≤30 days)
    notice = signals.get("notice_period_days", 90)
    if notice <= JD_REQUIREMENTS.get("max_notice_period_days", 30):
        bonus += BONUS_LOW_NOTICE

    # Saved by recruiters (social proof)
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved > 0:
        bonus += BONUS_SAVED_BY_RECRUITERS * min(1.0, saved / 10.0)

    # Normalize to 0-1
    return min(1.0, bonus)


def rank_candidates(
    candidates: List[Dict[str, Any]],
    semantic_scores: Dict[str, float],
    structured_scores: Dict[str, float],
    behavioral_scores: Dict[str, float],
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Combine all scores and produce a final ranked list.

    Args:
        candidates: List of normalized candidate dicts.
        semantic_scores: candidate_id → semantic similarity score (0-1).
        structured_scores: candidate_id → structured match score (0-1).
        behavioral_scores: candidate_id → behavioral score (0-1).
        top_k: Number of candidates to return. Defaults to TOP_K from config.

    Returns:
        List of dicts, sorted by final_score descending:
        [
            {
                "candidate_id": str,
                "rank": int,
                "final_score": float,
                "semantic_score": float,
                "structured_score": float,
                "behavioral_score": float,
                "bonus_score": float,
                "candidate": dict,  # Original candidate data
            },
            ...
        ]
    """
    if top_k is None:
        top_k = TOP_K

    ranked = []

    for candidate in candidates:
        cid = candidate["candidate_id"]

        sem = semantic_scores.get(cid, 0.0)
        struct = structured_scores.get(cid, 0.0)
        behav = behavioral_scores.get(cid, 0.0)
        bonus = compute_bonus(candidate)

        # Weighted composite score
        final_score = (
            WEIGHT_SEMANTIC * sem
            + WEIGHT_STRUCTURED * struct
            + WEIGHT_BEHAVIORAL * behav
            + WEIGHT_BONUS * bonus
        )

        ranked.append({
            "candidate_id": cid,
            "final_score": final_score,
            "semantic_score": sem,
            "structured_score": struct,
            "behavioral_score": behav,
            "bonus_score": bonus,
            "candidate": candidate,
        })

    # Sort by final_score (rounded to 4 decimal places to match CSV output) descending,
    # then by candidate_id ascending for ties
    ranked.sort(key=lambda x: (-round(x["final_score"], 4), x["candidate_id"]))

    # Assign ranks and truncate
    for i, entry in enumerate(ranked[:top_k], 1):
        entry["rank"] = i

    return ranked[:top_k]


def print_ranking_summary(ranked: List[Dict], top_n: int = 10):
    """Print a summary of the top-N ranked candidates."""
    print(f"\n{'='*90}")
    print(f"  TOP {top_n} CANDIDATES")
    print(f"{'='*90}")
    print(f"  {'Rank':<6} {'ID':<16} {'Title':<30} {'Final':>7} {'Sem':>6} {'Str':>6} {'Beh':>6} {'Bon':>6}")
    print(f"  {'-'*84}")

    for entry in ranked[:top_n]:
        c = entry["candidate"]
        title = c["profile"]["current_title"][:28]
        print(
            f"  {entry['rank']:<6} "
            f"{entry['candidate_id']:<16} "
            f"{title:<30} "
            f"{entry['final_score']:>6.4f} "
            f"{entry['semantic_score']:>6.3f} "
            f"{entry['structured_score']:>6.3f} "
            f"{entry['behavioral_score']:>6.3f} "
            f"{entry['bonus_score']:>6.3f}"
        )

    print(f"{'='*90}")
