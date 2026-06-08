"""
Reasoning Generator (v2)

Generates concise, specific, recruiter-facing reasoning strings for each
ranked candidate. Designed to pass the Stage 4 manual review checks:

1. Specific facts: References exact years, companies, skills from profile
2. JD connection: Explains WHY this candidate fits specific JD requirements
3. Honest concerns: Acknowledges gaps (location, notice, consulting)
4. No hallucination: Every claim backed by actual profile data
5. Variation: Each candidate gets genuinely different reasoning
6. Rank consistency: Tone matches rank position (top = positive, bottom = honest)
"""

import random
from typing import Dict, Any, List, Optional

from config.settings import REASONING_MAX_LENGTH, JD_REQUIREMENTS


# ─── Tone templates based on rank tier ────────────────────────────────────────

_TOP_TIER_STARTERS = [
    "Strong fit:",
    "Excellent match:",
    "Top candidate:",
    "Highly relevant:",
]

_MID_TIER_STARTERS = [
    "Moderate fit:",
    "Partial match:",
    "Relevant background:",
    "Decent alignment:",
]

_LOW_TIER_STARTERS = [
    "Weak fit:",
    "Limited alignment:",
    "Peripheral match:",
    "Below threshold:",
]


def _get_rank_tier(rank: int, total: int) -> str:
    """Determine rank tier for tone adjustment."""
    pct = rank / max(total, 1)
    if pct <= 0.15:
        return "top"
    elif pct <= 0.50:
        return "mid"
    else:
        return "low"


def _get_matched_skills(candidate: Dict) -> List[str]:
    """Find skills that match JD requirements (using exact name from profile)."""
    skills = candidate.get("skills", [])
    required = set(JD_REQUIREMENTS["required_skills"])
    preferred = set(JD_REQUIREMENTS["preferred_skills"])

    matched_required = []
    matched_preferred = []

    for s in skills:
        name = s.get("name", "")
        name_lower = name.lower()

        # Check required
        for r in required:
            if name_lower == r or (len(r) > 3 and r in name_lower) or (len(name_lower) > 3 and name_lower in r):
                matched_required.append(name)
                break
        else:
            # Check preferred
            for p in preferred:
                if name_lower == p or (len(p) > 3 and p in name_lower) or (len(name_lower) > 3 and name_lower in p):
                    matched_preferred.append(name)
                    break

    return matched_required, matched_preferred


def _get_career_highlights(candidate: Dict) -> str:
    """Build a career trajectory summary."""
    career = candidate.get("career_history", [])
    if not career:
        return ""

    titles = [j.get("title", "") for j in career if j.get("title")]
    companies = [j.get("company", "") for j in career if j.get("company")]

    if len(career) == 1:
        return f"{titles[0]} at {companies[0]}"
    else:
        # Show progression
        latest = f"{titles[0]} at {companies[0]}"
        prev_companies = [c for c in companies[1:] if c]
        if prev_companies:
            return f"{latest}, previously at {prev_companies[0]}"
        return latest


def _get_concerns(candidate: Dict, rank: int, total: int) -> List[str]:
    """Identify honest concerns about the candidate."""
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    concerns = []

    # Title mismatch
    title_lower = profile.get("current_title", "").lower()
    if any(wt in title_lower for wt in JD_REQUIREMENTS["weak_fit_titles"]):
        concerns.append(f"current title ({profile['current_title']}) not aligned with AI/ML role")

    # Consulting background
    company_lower = profile.get("current_company", "").lower()
    if any(cc in company_lower for cc in JD_REQUIREMENTS["consulting_companies"]):
        concerns.append(f"consulting background ({profile['current_company']})")

    # Experience out of range
    years = profile.get("years_of_experience", 0)
    min_exp, max_exp = JD_REQUIREMENTS["experience_range"]
    if years < min_exp:
        concerns.append(f"underexperienced ({years:.1f}yr vs {min_exp}-{max_exp}yr required)")
    elif years > max_exp + 2:
        concerns.append(f"overexperienced ({years:.1f}yr vs {min_exp}-{max_exp}yr target)")

    # Location
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    in_preferred = any(
        loc in location or loc in country
        for loc in JD_REQUIREMENTS["location_preferences"]
    )
    if not in_preferred and location:
        concerns.append(f"location ({profile.get('location', 'unknown')}) outside preferred geographies")

    # Notice period
    notice = signals.get("notice_period_days", 90)
    if notice > 60:
        concerns.append(f"high notice period ({notice} days)")

    # Low responsiveness
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate < 0.2:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    return concerns


def generate_reasoning(
    ranked_entry: Dict[str, Any],
    total_ranked: int,
    honeypot_results: Optional[Dict] = None,
) -> str:
    """
    Generate a specific, fact-based reasoning string for a single candidate.

    Each reasoning is unique because it pulls directly from the candidate's
    actual profile data rather than using generic templates.
    """
    c = ranked_entry["candidate"]
    profile = c["profile"]
    signals = c.get("redrob_signals", {})
    rank = ranked_entry.get("rank", 50)

    tier = _get_rank_tier(rank, total_ranked)
    parts = []

    # ── Starter (varied by tier) ─────────────────────────────────────────
    if tier == "top":
        starter = _TOP_TIER_STARTERS[rank % len(_TOP_TIER_STARTERS)]
    elif tier == "mid":
        starter = _MID_TIER_STARTERS[rank % len(_MID_TIER_STARTERS)]
    else:
        starter = _LOW_TIER_STARTERS[rank % len(_LOW_TIER_STARTERS)]

    # ── Core identity ────────────────────────────────────────────────────
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    years = profile.get("years_of_experience", 0)
    industry = profile.get("current_industry", "")

    identity = f"{title}"
    if company:
        identity += f" at {company}"
    if industry:
        identity += f" ({industry})"
    identity += f" with {years:.1f} years experience"
    parts.append(identity)

    # ── Skills match ─────────────────────────────────────────────────────
    matched_req, matched_pref = _get_matched_skills(c)

    if matched_req:
        skill_names = [s for s in matched_req[:4]]
        parts.append(f"relevant skills: {', '.join(skill_names)}")
    elif matched_pref:
        parts.append(f"adjacent skills: {', '.join(matched_pref[:3])}")
    else:
        parts.append("no directly matching AI/ML skills found")

    # ── Career trajectory (for top/mid tier) ─────────────────────────────
    if tier in ("top", "mid"):
        career_highlight = _get_career_highlights(c)
        if career_highlight and len(career_highlight) < 60:
            parts.append(f"career: {career_highlight}")

    # ── Positive signals ─────────────────────────────────────────────────
    positives = []

    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= 0.7:
        positives.append(f"responsive to recruiters ({response_rate:.0%})")

    if signals.get("open_to_work_flag", False):
        positives.append("actively open to work")

    github = signals.get("github_activity_score", -1)
    if github >= 50:
        positives.append(f"active GitHub contributor ({github}/100)")

    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        positives.append(f"available quickly ({notice}-day notice)")

    # Location match
    location = profile.get("location", "").lower()
    if any(loc in location for loc in JD_REQUIREMENTS["location_preferences"]):
        positives.append(f"based in {profile.get('location', '')}")

    if positives:
        parts.append("; ".join(positives[:2]))

    # ── Concerns (honest acknowledgment) ─────────────────────────────────
    concerns = _get_concerns(c, rank, total_ranked)
    if concerns:
        # Top candidates: mention 1 concern gently
        # Low candidates: lead with concerns
        if tier == "top" and concerns:
            parts.append(f"minor concern: {concerns[0]}")
        elif tier == "mid" and concerns:
            parts.append(f"concern: {concerns[0]}")
        else:
            parts.append(f"concerns: {'; '.join(concerns[:2])}")

    # ── Honeypot flag ────────────────────────────────────────────────────
    if honeypot_results:
        cid = c["candidate_id"]
        hp = honeypot_results.get(cid, {})
        if hp.get("is_honeypot"):
            parts.append(f"profile inconsistencies detected ({hp['flags'][0]})")

    # ── Assemble ─────────────────────────────────────────────────────────
    reasoning = f"{starter} {'; '.join(parts)}"

    # Truncate cleanly
    if len(reasoning) > REASONING_MAX_LENGTH:
        reasoning = reasoning[:REASONING_MAX_LENGTH - 3].rsplit(";", 1)[0] + "..."

    return reasoning


def generate_all_reasoning(
    ranked_list: List[Dict[str, Any]],
    honeypot_results: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Generate reasoning for all ranked candidates.
    Modifies ranked_list in-place by adding 'reasoning' key.
    """
    total = len(ranked_list)
    for entry in ranked_list:
        entry["reasoning"] = generate_reasoning(
            entry,
            total_ranked=total,
            honeypot_results=honeypot_results,
        )
    return ranked_list
