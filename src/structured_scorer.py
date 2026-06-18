import math
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher
from config.settings import JD_REQUIREMENTS, STRUCTURED_WEIGHTS

def score_candidate(candidate: Dict[str, Any], req: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    if req is None:
        req = JD_REQUIREMENTS

    scores = {
        "title_match": _score_title(candidate, req),
        "skills_match": _score_skills(candidate, req),
        "experience_fit": _score_experience(candidate, req),
        "industry_relevance": _score_industry(candidate, req),
        "education_tier": _score_education(candidate, req),
    }

    total = sum(
        scores[key] * STRUCTURED_WEIGHTS[key]
        for key in scores
    )

    if scores["title_match"] <= 0.15:
        total *= 0.55
    elif scores["title_match"] <= 0.3:
        total *= 0.75

    # Keyword stuffing penalty
    stuffing_penalty = _detect_keyword_stuffing(candidate, req)
    if stuffing_penalty < 1.0:
        total *= stuffing_penalty

    scores["keyword_stuffing_penalty"] = stuffing_penalty
    scores["total"] = min(1.0, max(0.0, total))

    return scores

def score_all_candidates(candidates: List[Dict[str, Any]], req: Optional[Dict[str, Any]] = None) -> Dict[str, float]:

    results = {}
    for c in candidates:
        scores = score_candidate(c, req=req)
        results[c["candidate_id"]] = scores["total"]
    return results


def _score_title(candidate: Dict[str, Any], req: Dict[str, Any]) -> float:

    current_title = candidate["profile"]["current_title"].lower().strip()

    if not current_title:
        return 0.0

    career_bonus = 0.0
    for job in candidate.get("career_history", []):
        job_title = job.get("title", "").lower().strip()
        for strong_title in req["strong_fit_titles"]:
            if _title_match(job_title, strong_title):
                career_bonus = max(career_bonus, 0.6)

    for weak_title in req["weak_fit_titles"]:
        if _title_match(current_title, weak_title):
            return max(0.05, career_bonus * 0.4)

    for strong_title in req["strong_fit_titles"]:
        if _title_match(current_title, strong_title):
            return 1.0

    tech_keywords = ["engineer", "developer", "scientist", "analyst", "architect", "lead"]
    for kw in tech_keywords:
        if kw in current_title:
            return max(0.4, career_bonus)

    return max(0.15, career_bonus)


def _score_skills(candidate: Dict[str, Any], req: Dict[str, Any]) -> float:

    candidate_skills = candidate.get("skills", [])
    if not candidate_skills:
        return 0.0

    required = set(req["required_skills"])
    preferred = set(req["preferred_skills"])

    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    total_score = 0.0
    max_possible = len(required) * 1.0

    for skill in candidate_skills:
        skill_name = skill.get("name", "").lower().strip()
        if not skill_name:
            continue

        is_required = any(_skill_match(skill_name, r) for r in required)
        is_preferred = any(_skill_match(skill_name, r) for r in preferred)

        if not is_required and not is_preferred:
            continue

        base = 1.0 if is_required else 0.4

        proficiency = skill.get("proficiency", "beginner")
        prof_mult = {
            "expert": 1.0,
            "advanced": 0.8,
            "intermediate": 0.5,
            "beginner": 0.2,
        }.get(proficiency, 0.3)

        duration = skill.get("duration_months", 0)
        duration_mult = min(1.0, duration / 36)

        endorsements = skill.get("endorsements", 0)
        endorse_mult = min(1.0, 0.3 + endorsements / 20)

        assessment_bonus = 0.0
        if skill_name in {k.lower(): k for k in assessments}:
            for k, v in assessments.items():
                if k.lower() == skill_name:
                    assessment_bonus = v / 100.0 * 0.3
                    break

        skill_score = base * (0.4 * prof_mult + 0.3 * duration_mult + 0.3 * endorse_mult) + assessment_bonus
        total_score += skill_score

    if max_possible == 0:
        return 0.0

    normalized = total_score / max_possible
    return min(1.0, normalized)


def _score_experience(candidate: Dict[str, Any], req: Dict[str, Any]) -> float:

    years = candidate["profile"].get("years_of_experience", 0)
    if years == 0:
        return 0.1

    min_exp, max_exp = req["experience_range"]
    ideal = req["ideal_experience"]

    if min_exp <= years <= max_exp:
        distance = abs(years - ideal)
        max_distance = max(ideal - min_exp, max_exp - ideal)
        if max_distance == 0:
            return 1.0
        return 0.8 + 0.2 * math.exp(-0.5 * (distance / max_distance) ** 2)

    if years < min_exp:
        distance = min_exp - years
        return max(0.1, 0.7 * math.exp(-0.5 * (distance / 2) ** 2))
    else:
        distance = years - max_exp
        return max(0.2, 0.75 * math.exp(-0.3 * (distance / 3) ** 2))


def _score_industry(candidate: Dict[str, Any], req: Dict[str, Any]) -> float:

    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    current_industry = profile.get("current_industry", "").lower()
    current_company = profile.get("current_company", "").lower()

    is_consulting = any(
        c in current_company
        for c in req["consulting_companies"]
    )

    has_product_exp = False
    consulting_only = True
    for job in career:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()

        job_is_consulting = any(c in company for c in req["consulting_companies"])
        if not job_is_consulting:
            consulting_only = False

        if not job_is_consulting and any(ind in industry for ind in req["preferred_industries"]):
            has_product_exp = True

    if consulting_only and len(career) > 1:
        return 0.1

    if is_consulting and has_product_exp:
        return 0.5

    if is_consulting and not has_product_exp:
        return 0.15

    industry_match = any(
        ind in current_industry
        for ind in req["preferred_industries"]
    )

    if industry_match:
        return 0.9

    return 0.5


def _score_education(candidate: Dict[str, Any], req: Dict[str, Any]) -> float:

    education = candidate.get("education", [])
    if not education:
        return 0.3

    best_score = 0.0

    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()

        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.7,
            "tier_3": 0.4,
            "tier_4": 0.2,
            "unknown": 0.3,
        }
        tier_score = tier_scores.get(tier, 0.3)

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

def _title_match(candidate_title: str, reference_title: str) -> bool:

    if not candidate_title or not reference_title:
        return False

    ct = candidate_title.lower().strip()
    rt = reference_title.lower().strip()

    if ct == rt:
        return True

    ct_words = set(ct.split())
    rt_words = set(rt.split())

    fillers = {"a", "an", "the", "and", "or", "of", "in", "at", "for", "with", "-", "|", "/"}
    ct_words -= fillers
    rt_words -= fillers

    if not ct_words or not rt_words:
        return False

    if rt_words.issubset(ct_words) or ct_words.issubset(rt_words):
        return True

    overlap = len(ct_words & rt_words)
    total = max(len(ct_words), len(rt_words))
    if total > 0 and overlap / total >= 0.7:
        return True

    return False


def _fuzzy_match(s1: str, s2: str) -> float:

    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def _skill_match(candidate_skill: str, required_skill: str) -> bool:

    cs = candidate_skill.lower().strip()
    rs = required_skill.lower().strip()

    if cs == rs:
        return True

    if len(rs) <= 3 or len(cs) <= 3:
        return cs == rs

    if len(rs) >= 5 and rs in cs:
        return True
    if len(cs) >= 5 and cs in rs:
        return True

    if _fuzzy_match(cs, rs) > 0.8:
        return True

    return False


def _detect_keyword_stuffing(candidate: Dict[str, Any], req: Dict[str, Any]) -> float:
    """
    Detect keyword stuffing: candidates who list many JD keywords
    but with shallow proficiency and short duration.

    Returns a penalty multiplier (1.0 = no penalty, < 1.0 = penalized).
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 1.0

    required = set(req.get("required_skills", []))
    preferred = set(req.get("preferred_skills", []))
    all_jd_keywords = required | preferred

    if not all_jd_keywords:
        return 1.0

    matched_count = 0
    beginner_count = 0
    short_duration_count = 0

    for skill in skills:
        skill_name = skill.get("name", "").lower().strip()
        if not skill_name:
            continue

        is_match = any(_skill_match(skill_name, kw) for kw in all_jd_keywords)
        if is_match:
            matched_count += 1
            if skill.get("proficiency", "beginner") == "beginner":
                beginner_count += 1
            if skill.get("duration_months", 0) <= 3:
                short_duration_count += 1

    total_skills = len(skills)
    if total_skills == 0:
        return 1.0

    # Keyword density: what fraction of their skills are JD keywords?
    keyword_density = matched_count / total_skills

    # Shallow ratio: what fraction of matched skills are beginner-level?
    shallow_ratio = beginner_count / matched_count if matched_count > 0 else 0

    # Short duration ratio: what fraction of matched skills have ≤ 3 months?
    short_ratio = short_duration_count / matched_count if matched_count > 0 else 0

    # Flag: high keyword density + mostly shallow proficiency + short durations
    if keyword_density > 0.8 and shallow_ratio > 0.6 and short_ratio > 0.5:
        return 0.3  # Heavy penalty
    elif keyword_density > 0.7 and shallow_ratio > 0.5:
        return 0.6  # Moderate penalty

    return 1.0
