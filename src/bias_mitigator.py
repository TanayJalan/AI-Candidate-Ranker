"""
Bias Mitigator

Strips demographic proxies from candidate text before embedding
to ensure merit-based evaluation. Removes names, gendered pronouns,
graduation years, and age-indicating phrases.
"""

import re
from typing import List


# Gendered pronouns and honorifics
_GENDERED_PATTERNS = [
    r"\b(he|she|him|her|his|hers|himself|herself)\b",
    r"\b(mr|mrs|ms|miss|sir|madam)\b\.?",
]

# Graduation years (e.g., "class of 2015", "graduated 2018", "batch of 2012")
_YEAR_PATTERNS = [
    r"\b(class|batch|graduated|graduation|passing)\s+(of\s+)?\d{4}\b",
    r"\b(19|20)\d{2}\s*[-–]\s*(19|20)?\d{2}\b",  # Year ranges like 2015-2019
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
]

# Age-indicating phrases
_AGE_PATTERNS = [
    r"\b\d{1,2}\s*years?\s*old\b",
    r"\bage[d]?\s*\d{1,2}\b",
    r"\bborn\s+in\s+\d{4}\b",
    r"\bdate\s+of\s+birth\b",
    r"\bdob\b",
]

# Common name patterns (e.g., "Name: John Smith" or "Candidate: Jane Doe")
_NAME_PATTERNS = [
    r"(name|candidate)\s*:\s*[A-Z][a-z]+(\s+[A-Z][a-z]+){0,2}",
]

# Compile all patterns
_ALL_PATTERNS: List[re.Pattern] = []
for pattern_list in [_GENDERED_PATTERNS, _YEAR_PATTERNS, _AGE_PATTERNS, _NAME_PATTERNS]:
    for p in pattern_list:
        _ALL_PATTERNS.append(re.compile(p, re.IGNORECASE))


def sanitize_text(text: str) -> str:
    """
    Remove demographic proxy signals from text before embedding.

    Strips:
    - Gendered pronouns and honorifics
    - Graduation years and date ranges
    - Age-indicating phrases
    - Explicit name labels

    Args:
        text: The enriched candidate text.

    Returns:
        Sanitized text with demographic proxies removed.
    """
    sanitized = text

    for pattern in _ALL_PATTERNS:
        sanitized = pattern.sub("", sanitized)

    # Collapse multiple spaces
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()

    return sanitized
