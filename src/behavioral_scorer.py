import math
import os
import sys
from datetime import datetime, date
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BEHAVIORAL_WEIGHTS, JD_REQUIREMENTS


def score_candidate(candidate: Dict[str, Any]) -> Dict[str, float]:
    
    signals = candidate.get("redrob_signals", {})

    scores = {
        "recruiter_response_rate": _score_response_rate(signals),
        "interview_completion_rate": _score_interview_rate(signals),
        "profile_completeness": _score_completeness(signals),
        "recency": _score_recency(signals),
        "github_activity": _score_github(signals, candidate.get("profile", {})),
        "offer_acceptance_rate": _score_offer_acceptance(signals),
        "response_time": _score_response_time(signals),
        "notice_period": _score_notice_period(signals),
        "search_appearance": _score_search_appearance(signals),
    }

    total = sum(
        scores[key] * BEHAVIORAL_WEIGHTS[key]
        for key in scores
    )
    scores["total"] = min(1.0, max(0.0, total))

    return scores


def score_all_candidates(candidates: List[Dict]) -> Dict[str, float]:
    
    results = {}
    for c in candidates:
        scores = score_candidate(c)
        results[c["candidate_id"]] = scores["total"]
    return results



def _score_response_rate(signals: Dict) -> float:

    rate = signals.get("recruiter_response_rate", 0.0)
    return min(1.0, max(0.0, rate))


def _score_interview_rate(signals: Dict) -> float:

    rate = signals.get("interview_completion_rate", 0.0)
    return min(1.0, max(0.0, rate))


def _score_completeness(signals: Dict) -> float:

    score = signals.get("profile_completeness_score", 0)
    return min(1.0, score / 100.0)


def _score_recency(signals: Dict) -> float:

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
        return 0.1


def _score_github(signals: Dict, profile: Dict) -> float:

    # 1. First try to scrape actual contributions if a URL is provided
    github_url = profile.get("github_url") or signals.get("github_url")
    if github_url:
        from src.github_scraper import extract_github_username, get_github_contributions
        username = extract_github_username(github_url)
        if username:
            contributions = get_github_contributions(username)
            if contributions is not None:
                # Map contributions to a score (e.g., > 500 is 1.0)
                if contributions == 0:
                    return 0.1
                elif contributions < 50:
                    return 0.4
                elif contributions < 200:
                    return 0.7
                elif contributions < 500:
                    return 0.9
                else:
                    return 1.0

    # 2. Fall back to the synthetic redrob_signal score
    score = signals.get("github_activity_score", -1)

    if score < 0:
        return 0.3

    return min(1.0, score / 100.0)


def _score_offer_acceptance(signals: Dict) -> float:

    rate = signals.get("offer_acceptance_rate", -1)

    if rate < 0:
        return 0.5

    return min(1.0, max(0.0, rate))


def _score_response_time(signals: Dict) -> float:

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
