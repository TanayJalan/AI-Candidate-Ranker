from typing import Dict, Any

from src.bias_mitigator import sanitize_text


def enrich_candidate_text(candidate: Dict[str, Any]) -> str:
    parts = []
    profile = candidate["profile"]

    if profile["headline"]:
        parts.append(f"Professional headline: {profile['headline']}")

    if profile["current_title"]:
        title_str = f"Current role: {profile['current_title']}"
        if profile["current_company"]:
            title_str += f" at {profile['current_company']}"
        if profile["current_industry"]:
            title_str += f" in {profile['current_industry']}"
        parts.append(title_str)

    if profile["years_of_experience"]:
        parts.append(f"Total experience: {profile['years_of_experience']} years")

    if profile["summary"]:
        parts.append(f"Summary: {profile['summary']}")

    skills = candidate.get("skills", [])
    if skills:
        skill_strings = []
        for s in skills:
            name = s.get("name", "")
            if not name:
                continue
            proficiency = s.get("proficiency", "")
            endorsements = s.get("endorsements", 0)
            duration = s.get("duration_months", 0)

            detail = name
            if proficiency and proficiency != "beginner":
                detail += f" ({proficiency})"
            if duration and duration > 12:
                detail += f" [{duration // 12}+ yrs]"
            if endorsements and endorsements > 5:
                detail += f" [{endorsements} endorsements]"
            skill_strings.append(detail)

        if skill_strings:
            parts.append(f"Skills: {', '.join(skill_strings)}")

    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        high_scores = [
            f"{skill}: {score}/100"
            for skill, score in sorted(assessments.items(), key=lambda x: -x[1])
            if score >= 60
        ]
        if high_scores:
            parts.append(f"Skill assessments: {', '.join(high_scores[:10])}")

    career = candidate.get("career_history", [])
    if career:
        career_parts = []
        for job in career:
            job_str = ""
            title = job.get("title", "")
            company = job.get("company", "")
            industry = job.get("industry", "")
            duration = job.get("duration_months", 0)
            desc = job.get("description", "")

            if title:
                job_str = title
                if company:
                    job_str += f" at {company}"
                if industry:
                    job_str += f" ({industry})"
                if duration:
                    years = duration / 12
                    job_str += f" for {years:.1f} years"
                if desc:
                    desc_short = desc[:300] + "..." if len(desc) > 300 else desc
                    job_str += f". {desc_short}"
                career_parts.append(job_str)

        if career_parts:
            parts.append("Career history: " + " | ".join(career_parts))

    education = candidate.get("education", [])
    if education:
        edu_parts = []
        for edu in education:
            edu_str = ""
            degree = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            institution = edu.get("institution", "")
            tier = edu.get("tier", "")

            if degree or field:
                edu_str = f"{degree} in {field}" if degree and field else (degree or field)
                if institution:
                    edu_str += f" from {institution}"
                if tier and tier in ("tier_1", "tier_2"):
                    edu_str += f" ({tier.replace('_', ' ')})"
                edu_parts.append(edu_str)

        if edu_parts:
            parts.append("Education: " + "; ".join(edu_parts))

    certs = candidate.get("certifications", [])
    if certs:
        cert_names = [c.get("name", "") for c in certs if c.get("name")]
        if cert_names:
            parts.append(f"Certifications: {', '.join(cert_names)}")

    location_parts = []
    if profile.get("location"):
        location_parts.append(profile["location"])
    if profile.get("country"):
        location_parts.append(profile["country"])
    if location_parts:
        parts.append(f"Location: {', '.join(location_parts)}")

    return sanitize_text(" | ".join(parts))


def enrich_all_candidates(candidates: list) -> list:

    results = []
    for c in candidates:
        text = enrich_candidate_text(c)
        results.append((c["candidate_id"], text))
    return results
