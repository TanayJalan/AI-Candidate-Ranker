"""
Honeypot Detector

Identifies candidates with subtly impossible profiles that signal
they are synthetic "trap" entries in the dataset.

Per the submission spec (~80 honeypots in the full dataset):
- Impossible tenure: 8+ years at a company founded 3 years ago
- Impossible skill proficiency: "expert" in 10+ skills with 0 months duration
- Experience/career inconsistencies
- Impossible endorsement/duration ratios

Honeypot candidates get their scores zeroed to push them to the bottom.
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import math


# ─── Honeypot checks ─────────────────────────────────────────────────────────

def _check_career_duration_mismatch(candidate: Dict) -> Tuple[bool, str]:
    """
    Check if total career history duration doesn't match claimed experience.

    A real candidate's career history should roughly add up to their
    claimed years of experience. Large mismatches (>3 years off) are suspicious.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    if not career:
        return False, ""

    claimed_months = profile.get("years_of_experience", 0) * 12
    total_career_months = sum(
        job.get("duration_months", 0) for job in career
    )

    # Allow some slack (job overlaps, rounding)
    if total_career_months > 0 and abs(total_career_months - claimed_months) > 36:
        return True, (
            f"career duration mismatch: {total_career_months/12:.1f}yr in history "
            f"vs {profile['years_of_experience']}yr claimed"
        )

    return False, ""


def _check_impossible_skill_proficiency(candidate: Dict) -> Tuple[bool, str]:
    """
    Check for 'expert' proficiency claims with zero or near-zero duration.

    A genuine expert in a skill should have significant time invested.
    Having 3+ expert skills with 0 months is a strong honeypot signal.
    """
    skills = candidate.get("skills", [])
    expert_zero = []

    for s in skills:
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)

        if prof == "expert" and duration <= 3:
            expert_zero.append(s.get("name", "unknown"))

    if len(expert_zero) >= 3:
        return True, (
            f"expert proficiency with ≤3 months duration in {len(expert_zero)} skills: "
            f"{', '.join(expert_zero[:5])}"
        )

    return False, ""


def _check_excessive_expert_skills(candidate: Dict) -> Tuple[bool, str]:
    """
    Check for unrealistic number of expert-level skills.

    Having 8+ expert skills is extremely rare in genuine candidates,
    especially if combined with other suspicious signals.
    """
    skills = candidate.get("skills", [])
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")

    if expert_count >= 8:
        return True, f"{expert_count} skills at expert level (unrealistic)"

    return False, ""


def _check_endorsement_anomalies(candidate: Dict) -> Tuple[bool, str]:
    """
    Check for impossibly high endorsements on skills with zero usage duration.

    Getting 30+ endorsements on a skill you've used for 0 months is suspicious.
    """
    skills = candidate.get("skills", [])
    suspicious = []

    for s in skills:
        endorsements = s.get("endorsements", 0)
        duration = s.get("duration_months", 0)

        # Very high endorsements relative to very low duration
        if endorsements > 30 and duration <= 3:
            suspicious.append(
                f"{s.get('name', 'unknown')} ({endorsements} endorsements, {duration}mo)"
            )

    if len(suspicious) >= 2:
        return True, f"anomalous endorsements: {', '.join(suspicious[:3])}"

    return False, ""


def _check_career_timeline_gaps(candidate: Dict) -> Tuple[bool, str]:
    """
    Check for impossible career timelines.

    E.g., overlapping jobs that sum to more than the person's total experience,
    or start_date after end_date.
    """
    career = candidate.get("career_history", [])
    if len(career) < 2:
        return False, ""

    total_duration = sum(j.get("duration_months", 0) for j in career)
    claimed = candidate["profile"].get("years_of_experience", 0) * 12

    # If career durations sum to >2x claimed experience, suspicious overlaps
    if claimed > 0 and total_duration > claimed * 2:
        return True, (
            f"career durations sum to {total_duration/12:.1f}yr "
            f"but only {claimed/12:.1f}yr experience claimed"
        )

    return False, ""


def _check_assessment_vs_proficiency(candidate: Dict) -> Tuple[bool, str]:
    """
    Check for contradictions between assessment scores and proficiency claims.

    E.g., 'expert' proficiency but assessment score of 10/100.
    """
    skills = candidate.get("skills", [])
    assessments = candidate.get("redrob_signals", {}).get(
        "skill_assessment_scores", {}
    )

    if not assessments:
        return False, ""

    contradictions = []
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")

        # Check if this skill has an assessment
        for assess_name, score in assessments.items():
            if assess_name.lower() == name.lower():
                # Expert claiming but low assessment
                if prof == "expert" and score < 30:
                    contradictions.append(f"{name}: expert but scored {score}/100")
                # Beginner claiming but perfect assessment
                elif prof == "beginner" and score > 90:
                    contradictions.append(f"{name}: beginner but scored {score}/100")

    if len(contradictions) >= 2:
        return True, f"assessment contradictions: {'; '.join(contradictions[:3])}"

    return False, ""


def _check_impossible_experience_for_career(candidate: Dict) -> Tuple[bool, str]:
    """
    Check for very short total experience but many different career entries.

    E.g., 1.5 years experience but 5 different jobs is suspicious.
    """
    profile = candidate["profile"]
    career = candidate.get("career_history", [])

    years = profile.get("years_of_experience", 0)
    if years < 2 and len(career) >= 4:
        return True, f"{years}yr experience but {len(career)} career entries"

    return False, ""


# ─── Main detection function ─────────────────────────────────────────────────

def detect_honeypot(candidate: Dict) -> Dict[str, Any]:
    """
    Run all honeypot checks on a single candidate.

    Returns:
        {
            "is_honeypot": bool,
            "confidence": float (0-1),
            "flags": list of str,
            "flag_count": int,
        }
    """
    checks = [
        _check_career_duration_mismatch,
        _check_impossible_skill_proficiency,
        _check_excessive_expert_skills,
        _check_endorsement_anomalies,
        _check_career_timeline_gaps,
        _check_assessment_vs_proficiency,
        _check_impossible_experience_for_career,
    ]

    flags = []
    for check in checks:
        triggered, reason = check(candidate)
        if triggered:
            flags.append(reason)

    flag_count = len(flags)

    # Confidence scoring:
    # 1 flag = 0.3 (could be edge case)
    # 2 flags = 0.7 (likely honeypot)
    # 3+ flags = 0.95 (almost certainly honeypot)
    if flag_count == 0:
        confidence = 0.0
    elif flag_count == 1:
        confidence = 0.3
    elif flag_count == 2:
        confidence = 0.7
    else:
        confidence = min(0.95, 0.5 + flag_count * 0.15)

    return {
        "is_honeypot": confidence >= 0.6,
        "confidence": confidence,
        "flags": flags,
        "flag_count": flag_count,
    }


def detect_all_honeypots(
    candidates: List[Dict],
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Run honeypot detection on all candidates.

    Returns:
        candidate_id → honeypot result dict
    """
    results = {}
    flagged_count = 0

    for c in candidates:
        cid = c["candidate_id"]
        result = detect_honeypot(c)
        results[cid] = result

        if result["is_honeypot"]:
            flagged_count += 1

    if verbose:
        print(f"  Honeypot detection: {flagged_count}/{len(candidates)} flagged")
        for cid, r in results.items():
            if r["is_honeypot"]:
                print(f"    🍯 {cid} (confidence: {r['confidence']:.0%})")
                for flag in r["flags"]:
                    print(f"       → {flag}")

    return results
