"""
Reasoning Generator

Generates concise, human-readable reasoning strings for each ranked
candidate explaining WHY they were ranked at their position.

The reasoning should help a recruiter understand the ranking at a glance.
"""

from typing import Dict, Any, List

from config.settings import REASONING_MAX_LENGTH, JD_REQUIREMENTS


def generate_reasoning(ranked_entry: Dict[str, Any]) -> str:
    """
    Generate a reasoning string for a single ranked candidate.

    The reasoning captures:
    1. Role/title fit
    2. Experience relevance
    3. Key differentiating strengths
    4. Behavioral signal highlights
    5. Any notable red flags or concerns

    Args:
        ranked_entry: Dict from hybrid_ranker with scores and candidate data.

    Returns:
        Concise reasoning string (max ~200 chars).
    """
    c = ranked_entry["candidate"]
    profile = c["profile"]
    signals = c.get("redrob_signals", {})

    parts = []

    # ── Title + experience ───────────────────────────────────────────────
    title = profile.get("current_title", "Unknown")
    years = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")

    title_str = f"{title}"
    if company:
        title_str += f" at {company}"
    title_str += f" ({years:.1f} yrs)"
    parts.append(title_str)

    # ── Score breakdown highlights ───────────────────────────────────────
    sem = ranked_entry.get("semantic_score", 0)
    struct = ranked_entry.get("structured_score", 0)
    behav = ranked_entry.get("behavioral_score", 0)

    # Semantic fit
    if sem > 0.75:
        parts.append("strong semantic fit")
    elif sem > 0.6:
        parts.append("moderate semantic fit")

    # Skills match assessment
    skills = c.get("skills", [])
    required_skills = set(JD_REQUIREMENTS["required_skills"])
    matched_skills = []
    for s in skills:
        name = s.get("name", "").lower()
        if any(r in name or name in r for r in required_skills):
            matched_skills.append(s.get("name", ""))

    if matched_skills:
        top_skills = matched_skills[:4]
        parts.append(f"skills: {', '.join(top_skills)}")

    # ── Behavioral highlights ────────────────────────────────────────────
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= 0.7:
        parts.append(f"responsive ({response_rate:.0%})")
    elif response_rate <= 0.2:
        parts.append(f"low responsiveness ({response_rate:.0%})")

    github = signals.get("github_activity_score", -1)
    if github >= 60:
        parts.append(f"active GitHub ({github}/100)")

    if signals.get("open_to_work_flag", False):
        parts.append("open to work")

    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        parts.append(f"notice: {notice}d")

    # ── Industry/title fit notes ─────────────────────────────────────────
    current_title_lower = profile.get("current_title", "").lower()
    is_weak_title = any(
        wt in current_title_lower
        for wt in JD_REQUIREMENTS["weak_fit_titles"]
    )

    if is_weak_title:
        parts.append("title mismatch with role")

    # Check consulting background
    current_company_lower = profile.get("current_company", "").lower()
    is_consulting = any(
        cc in current_company_lower
        for cc in JD_REQUIREMENTS["consulting_companies"]
    )
    if is_consulting:
        parts.append("consulting background")

    # ── Location ─────────────────────────────────────────────────────────
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    in_preferred_location = any(
        loc in location or loc in country
        for loc in JD_REQUIREMENTS["location_preferences"]
    )
    if in_preferred_location:
        parts.append(f"location: {profile.get('location', '')}")

    # Build final reasoning string
    reasoning = "; ".join(parts)

    # Truncate if needed
    if len(reasoning) > REASONING_MAX_LENGTH:
        reasoning = reasoning[:REASONING_MAX_LENGTH - 3] + "..."

    return reasoning


def generate_all_reasoning(ranked_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate reasoning for all ranked candidates.

    Modifies ranked_list in-place by adding 'reasoning' key.
    Also returns the list for convenience.
    """
    for entry in ranked_list:
        entry["reasoning"] = generate_reasoning(entry)
    return ranked_list
