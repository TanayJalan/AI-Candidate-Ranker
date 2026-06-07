"""
Structured Scorer

Rule-based scoring that evaluates hard, measurable signals:
- Title match (current title vs JD role)
- Skills overlap (weighted by proficiency & endorsements)
- Experience fit (Gaussian around ideal range)
- Industry relevance (product vs consulting background)
- Education tier bonus

This is the layer that catches the "keyword stuffer trap" the JD
explicitly warns about — a Marketing Manager with 9 AI skills listed
should NOT rank above an ML Engineer with 3 proven skills.
"""

import math
from typing import Dict, Any, List
from difflib import SequenceMatcher

from config.settings import JD_REQUIREMENTS, STRUCTURED_WEIGHTS


def score_candidate(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute structured scores for a single candidate.

    Args:
        candidate: Normalized candidate dict.

    Returns:
        Dict with individual component scores and the weighted total:
        {
            "title_match": float,
            "skills_match": float,
            "experience_fit": float,
            "industry_relevance": float,
            "education_tier": float,
            "total": float,
        }
    """
    req = JD_REQUIREMENTS

    scores = {
        "title_match": _score_title(candidate, req),
        "skills_match": _score_skills(candidate, req),
        "experience_fit": _score_experience(candidate, req),
        "industry_relevance": _score_industry(candidate, req),
        "education_tier": _score_education(candidate, req),
    }

    # Weighted total
    total = sum(
        scores[key] * STRUCTURED_WEIGHTS[key]
        for key in scores
    )

    # ── Keyword-stuffer penalty ──────────────────────────────────────
    # The JD explicitly warns: a candidate with lots of AI keywords but
    # a clearly non-tech title (Marketing Manager, HR Manager, etc.)
    # is a trap. Apply a multiplicative penalty when title is a weak fit
    # but other scores are suspiciously high.
    if scores["title_match"] <= 0.15:
        # Title is a clear mismatch — apply damper
        total *= 0.55  # Significant penalty
    elif scores["title_match"] <= 0.3:
        # Title is a weak match
        total *= 0.75

    scores["total"] = min(1.0, max(0.0, total))

    return scores


def score_all_candidates(candidates: List[Dict]) -> Dict[str, float]:
    """
    Score all candidates and return mapping of candidate_id → total score.
    """
    results = {}
    for c in candidates:
        scores = score_candidate(c)
        results[c["candidate_id"]] = scores["total"]
    return results


# ─── Component scorers ───────────────────────────────────────────────────────

def _score_title(candidate: Dict, req: Dict) -> float:
    """
    Score how well the candidate's title matches the JD role.

    This is THE decisive signal against keyword-stuffer traps per the JD.
    A candidate whose title is "Marketing Manager" but has lots of AI skills
    listed should score LOW here, which drags down their total.

    CRITICAL: We check weak-fit titles FIRST to prevent false positives
    from fuzzy matching (e.g., "civil engineer" matching "ai engineer"
    because SequenceMatcher sees them as similar due to shared "engineer").
    """
    current_title = candidate["profile"]["current_title"].lower().strip()

    if not current_title:
        return 0.0

    # Check career history for ANY strong-fit title (current or past)
    career_bonus = 0.0
    for job in candidate.get("career_history", []):
        job_title = job.get("title", "").lower().strip()
        for strong_title in req["strong_fit_titles"]:
            if _title_match(job_title, strong_title):
                career_bonus = max(career_bonus, 0.6)

    # ── FIRST: Check weak fit titles (trap detection) ────────────────
    # These are explicitly non-technical roles that the JD warns about
    for weak_title in req["weak_fit_titles"]:
        if _title_match(current_title, weak_title):
            # Title is a poor fit — this is a trap candidate
            return max(0.05, career_bonus * 0.4)

    # ── THEN: Check strong fit titles ────────────────────────────────
    for strong_title in req["strong_fit_titles"]:
        if _title_match(current_title, strong_title):
            return 1.0

    # Title doesn't clearly match either list — check for tech relevance
    tech_keywords = ["engineer", "developer", "scientist", "analyst", "architect", "lead"]
    for kw in tech_keywords:
        if kw in current_title:
            return max(0.4, career_bonus)

    return max(0.15, career_bonus)


def _score_skills(candidate: Dict, req: Dict) -> float:
    """
    Score skill overlap between candidate and JD requirements.

    Weighted by proficiency level and endorsements.
    Penalizes candidates who list skills without proficiency depth
    (the "endorsement-and-duration trust multiplier" the JD mentions).
    """
    candidate_skills = candidate.get("skills", [])
    if not candidate_skills:
        return 0.0

    required = set(req["required_skills"])
    preferred = set(req["preferred_skills"])

    # Also check assessment scores for quantitative proof
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    total_score = 0.0
    max_possible = len(required) * 1.0  # Max if all required skills matched at expert level

    for skill in candidate_skills:
        skill_name = skill.get("name", "").lower().strip()
        if not skill_name:
            continue

        # Check if this skill matches any required skill
        is_required = any(_skill_match(skill_name, r) for r in required)
        is_preferred = any(_skill_match(skill_name, r) for r in preferred)

        if not is_required and not is_preferred:
            continue

        # Base match value
        base = 1.0 if is_required else 0.4

        # Proficiency multiplier
        proficiency = skill.get("proficiency", "beginner")
        prof_mult = {
            "expert": 1.0,
            "advanced": 0.8,
            "intermediate": 0.5,
            "beginner": 0.2,
        }.get(proficiency, 0.3)

        # Duration trust multiplier (longer = more credible)
        duration = skill.get("duration_months", 0)
        duration_mult = min(1.0, duration / 36)  # Maxes out at 3 years

        # Endorsement trust multiplier
        endorsements = skill.get("endorsements", 0)
        endorse_mult = min(1.0, 0.3 + endorsements / 20)

        # Assessment score bonus (strongest proof)
        assessment_bonus = 0.0
        if skill_name in {k.lower(): k for k in assessments}:
            for k, v in assessments.items():
                if k.lower() == skill_name:
                    assessment_bonus = v / 100.0 * 0.3
                    break

        skill_score = base * (0.4 * prof_mult + 0.3 * duration_mult + 0.3 * endorse_mult) + assessment_bonus
        total_score += skill_score

    # Normalize: compare to max possible score
    if max_possible == 0:
        return 0.0

    normalized = total_score / max_possible
    return min(1.0, normalized)


def _score_experience(candidate: Dict, req: Dict) -> float:
    """
    Score how well candidate's experience fits the JD range.

    Uses a Gaussian penalty centered on the ideal experience.
    Within the range (5-9): high score.
    Outside: gradual penalty.
    """
    years = candidate["profile"].get("years_of_experience", 0)
    if years == 0:
        return 0.1

    min_exp, max_exp = req["experience_range"]
    ideal = req["ideal_experience"]

    # If within the range, high score with slight preference for ideal
    if min_exp <= years <= max_exp:
        # Gaussian centered at ideal within the range
        distance = abs(years - ideal)
        max_distance = max(ideal - min_exp, max_exp - ideal)
        if max_distance == 0:
            return 1.0
        return 0.8 + 0.2 * math.exp(-0.5 * (distance / max_distance) ** 2)

    # Outside range: penalize based on distance
    if years < min_exp:
        # Too junior
        distance = min_exp - years
        return max(0.1, 0.7 * math.exp(-0.5 * (distance / 2) ** 2))
    else:
        # Too senior (less penalty — the JD says it's a range, not a hard limit)
        distance = years - max_exp
        return max(0.2, 0.75 * math.exp(-0.3 * (distance / 3) ** 2))


def _score_industry(candidate: Dict, req: Dict) -> float:
    """
    Score industry relevance.

    Penalizes consulting-heavy backgrounds (per JD explicit instruction).
    Rewards product company experience.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    current_industry = profile.get("current_industry", "").lower()
    current_company = profile.get("current_company", "").lower()

    # Check if current company is in the consulting blacklist
    is_consulting = any(
        c in current_company
        for c in req["consulting_companies"]
    )

    # Check if candidate has product company experience in career history
    has_product_exp = False
    consulting_only = True
    for job in career:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()

        job_is_consulting = any(c in company for c in req["consulting_companies"])
        if not job_is_consulting:
            consulting_only = False

        # Check for preferred industry match (but NOT at consulting companies)
        if not job_is_consulting and any(ind in industry for ind in req["preferred_industries"]):
            has_product_exp = True

    # Score based on findings
    if consulting_only and len(career) > 1:
        # JD explicitly says "entire career at consulting" is a disqualifier
        return 0.1

    if is_consulting and has_product_exp:
        # Currently at consulting but has prior product experience — OK per JD
        return 0.5

    if is_consulting and not has_product_exp:
        return 0.15

    # Check industry match
    industry_match = any(
        ind in current_industry
        for ind in req["preferred_industries"]
    )

    if industry_match:
        return 0.9

    # Neutral industry
    return 0.5


def _score_education(candidate: Dict, req: Dict) -> float:
    """
    Score education quality based on institution tier and field relevance.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # No education data — neutral

    best_score = 0.0

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()

        # Tier score
        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.7,
            "tier_3": 0.4,
            "tier_4": 0.2,
            "unknown": 0.3,
        }
        tier_score = tier_scores.get(tier, 0.3)

        # Field relevance bonus
        relevant_fields = [
            "computer science", "artificial intelligence", "machine learning",
            "data science", "information technology", "software engineering",
            "mathematics", "statistics", "electrical engineering",
            "electronics", "information systems"
        ]
        field_bonus = 0.0
        for rf in relevant_fields:
            if rf in field or field in rf:
                field_bonus = 0.2
                break

        edu_score = min(1.0, tier_score + field_bonus)
        best_score = max(best_score, edu_score)

    return best_score


# ─── Utility functions ───────────────────────────────────────────────────────

def _title_match(candidate_title: str, reference_title: str) -> bool:
    """
    Check if a candidate title matches a reference title.

    Uses word-set overlap instead of character-level fuzzy matching
    to avoid false positives like 'civil engineer' matching 'ai engineer'.
    """
    if not candidate_title or not reference_title:
        return False

    ct = candidate_title.lower().strip()
    rt = reference_title.lower().strip()

    # Exact match
    if ct == rt:
        return True

    # Word-set comparison: check if all significant words of the reference
    # title appear in the candidate title (or vice versa)
    ct_words = set(ct.split())
    rt_words = set(rt.split())

    # Remove common filler words
    fillers = {"a", "an", "the", "and", "or", "of", "in", "at", "for", "with", "-", "|", "/"}
    ct_words -= fillers
    rt_words -= fillers

    if not ct_words or not rt_words:
        return False

    # Check if one is a subset of the other (handles "senior ai engineer" matching "ai engineer")
    if rt_words.issubset(ct_words) or ct_words.issubset(rt_words):
        return True

    # Check word overlap ratio
    overlap = len(ct_words & rt_words)
    total = max(len(ct_words), len(rt_words))
    if total > 0 and overlap / total >= 0.7:
        return True

    return False


def _fuzzy_match(s1: str, s2: str) -> float:
    """Compute fuzzy similarity between two strings (0-1)."""
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _skill_match(candidate_skill: str, required_skill: str) -> bool:
    """
    Check if a candidate skill matches a required skill.
    Uses word-boundary aware matching to avoid false positives
    like 'ml' matching 'html'.
    """
    cs = candidate_skill.lower().strip()
    rs = required_skill.lower().strip()

    # Exact match
    if cs == rs:
        return True

    # For short skill names (≤3 chars like 'ml', 'nlp', 'ai'),
    # require exact match only — no substring matching
    if len(rs) <= 3 or len(cs) <= 3:
        return cs == rs

    # One contains the other (only for longer skill names)
    if len(rs) >= 5 and rs in cs:
        return True
    if len(cs) >= 5 and cs in rs:
        return True

    # Fuzzy match for close variations
    if _fuzzy_match(cs, rs) > 0.8:
        return True

    return False
