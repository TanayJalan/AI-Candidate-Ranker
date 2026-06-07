"""
Behavioral Scorer

Scores candidates based on Redrob platform behavioral signals.
These signals capture "recruit-ability" — whether a candidate is
actually reachable, engaged, and likely to respond.

The JD explicitly says:
"A perfect-on-paper candidate who hasn't logged in for 6 months and
has a 5% recruiter response rate is, for hiring purposes, not actually
available. Down-weight them appropriately."
"""

import math
from datetime import datetime, date
from typing import Dict, Any, List

from config.settings import BEHAVIORAL_WEIGHTS, JD_REQUIREMENTS


def score_candidate(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute behavioral scores for a single candidate.

    Args:
        candidate: Normalized candidate dict.

    Returns:
        Dict with individual signal scores and weighted total.
    """
    signals = candidate.get("redrob_signals", {})

    scores = {
        "recruiter_response_rate": _score_response_rate(signals),
        "interview_completion_rate": _score_interview_rate(signals),
        "profile_completeness": _score_completeness(signals),
        "recency": _score_recency(signals),
        "github_activity": _score_github(signals),
        "offer_acceptance_rate": _score_offer_acceptance(signals),
        "response_time": _score_response_time(signals),
        "notice_period": _score_notice_period(signals),
        "search_appearance": _score_search_appearance(signals),
    }

    # Weighted total
    total = sum(
        scores[key] * BEHAVIORAL_WEIGHTS[key]
        for key in scores
    )
    scores["total"] = min(1.0, max(0.0, total))

    return scores


def score_all_candidates(candidates: List[Dict]) -> Dict[str, float]:
    """Score all candidates → dict of candidate_id → total behavioral score."""
    results = {}
    for c in candidates:
        scores = score_candidate(c)
        results[c["candidate_id"]] = scores["total"]
    return results


# ─── Individual signal scorers ───────────────────────────────────────────────

def _score_response_rate(signals: Dict) -> float:
    """
    Recruiter response rate: 0 to 1 directly.
    This is the strongest behavioral signal per the JD.
    """
    rate = signals.get("recruiter_response_rate", 0.0)
    # Apply slight sigmoid to amplify differences in the 0.3-0.8 range
    return min(1.0, max(0.0, rate))


def _score_interview_rate(signals: Dict) -> float:
    """Interview completion rate: 0 to 1 directly."""
    rate = signals.get("interview_completion_rate", 0.0)
    return min(1.0, max(0.0, rate))


def _score_completeness(signals: Dict) -> float:
    """Profile completeness: 0-100 → 0-1."""
    score = signals.get("profile_completeness_score", 0)
    return min(1.0, score / 100.0)


def _score_recency(signals: Dict) -> float:
    """
    How recently the candidate was active.
    JD says candidates who haven't logged in for 6 months are not available.
    """
    last_active = signals.get("last_active_date", "2020-01-01")

    try:
        if isinstance(last_active, str):
            last_date = datetime.strptime(last_active, "%Y-%m-%d").date()
        elif isinstance(last_active, date):
            last_date = last_active
        else:
            return 0.1
    except (ValueError, TypeError):
        return 0.1

    today = date.today()
    days_ago = (today - last_date).days

    if days_ago <= 7:
        return 1.0
    elif days_ago <= 30:
        return 0.9
    elif days_ago <= 90:
        return 0.7
    elif days_ago <= 180:
        return 0.4
    else:
        # Inactive for 6+ months — heavily penalize
        return 0.1


def _score_github(signals: Dict) -> float:
    """
    GitHub activity score: -1 to 100.
    -1 means no GitHub linked (neutral, not penalty for non-tech roles,
    but for AI Engineer it's a mild negative).
    """
    score = signals.get("github_activity_score", -1)

    if score < 0:
        # No GitHub linked — mild penalty for an AI Engineer role
        return 0.3

    return min(1.0, score / 100.0)


def _score_offer_acceptance(signals: Dict) -> float:
    """
    Historical offer acceptance rate.
    -1 means no offer history (neutral).
    """
    rate = signals.get("offer_acceptance_rate", -1)

    if rate < 0:
        return 0.5  # No history — neutral

    return min(1.0, max(0.0, rate))


def _score_response_time(signals: Dict) -> float:
    """
    Average response time in hours. Lower is better.
    """
    hours = signals.get("avg_response_time_hours", 999)

    if hours <= 2:
        return 1.0
    elif hours <= 6:
        return 0.9
    elif hours <= 12:
        return 0.8
    elif hours <= 24:
        return 0.6
    elif hours <= 48:
        return 0.4
    elif hours <= 72:
        return 0.3
    else:
        return 0.1


def _score_notice_period(signals: Dict) -> float:
    """
    Notice period in days. JD prefers sub-30 day.
    """
    days = signals.get("notice_period_days", 90)
    max_preferred = JD_REQUIREMENTS.get("max_notice_period_days", 30)

    if days <= max_preferred:
        return 1.0
    elif days <= 60:
        return 0.6
    elif days <= 90:
        return 0.4
    else:
        return 0.2


def _score_search_appearance(signals: Dict) -> float:
    """
    Search appearances in last 30 days.
    Higher = more visible/active on platform.
    """
    appearances = signals.get("search_appearance_30d", 0)

    if appearances == 0:
        return 0.1
    elif appearances <= 5:
        return 0.4
    elif appearances <= 15:
        return 0.6
    elif appearances <= 30:
        return 0.8
    else:
        return 1.0
