from typing import Dict, List, Any, Tuple, Optional
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

    signals = candidate.get("redrob_signals", {})
    bonus = 0.0

    if signals.get("open_to_work_flag", False):
        bonus += BONUS_OPEN_TO_WORK

    verified_count = 0
    if signals.get("verified_email", False):
        verified_count += 1
    if signals.get("verified_phone", False):
        verified_count += 1
    bonus += BONUS_VERIFIED * (verified_count / 2.0)

    if signals.get("willing_to_relocate", False):
        bonus += BONUS_RELOCATE

    if signals.get("linkedin_connected", False):
        bonus += BONUS_LINKEDIN

    notice = signals.get("notice_period_days", 90)
    if notice <= JD_REQUIREMENTS.get("max_notice_period_days", 30):
        bonus += BONUS_LOW_NOTICE

    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved > 0:
        bonus += BONUS_SAVED_BY_RECRUITERS * min(1.0, saved / 10.0)

    return min(1.0, bonus)


def rank_candidates(
    candidates: List[Dict[str, Any]],
    semantic_scores: Dict[str, float],
    structured_scores: Dict[str, float],
    behavioral_scores: Dict[str, float],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if top_k is None:
        top_k = TOP_K

    ranked = []

    for candidate in candidates:
        cid = candidate["candidate_id"]

        sem = semantic_scores.get(cid, 0.0)
        struct = structured_scores.get(cid, 0.0)
        behav = behavioral_scores.get(cid, 0.0)
        bonus = compute_bonus(candidate)

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

    ranked.sort(key=lambda x: (-round(x["final_score"], 4), x["candidate_id"]))

    for i, entry in enumerate(ranked[:top_k], 1):
        entry["rank"] = i

    return ranked[:top_k]


def print_ranking_summary(ranked: List[Dict], top_n: int = 10):

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
