"""
GitHub Scraper

Retrieves GitHub contribution data without requiring an API key
by directly requesting the public contributions calendar HTML.
"""

import re
import requests
from typing import Optional


def extract_github_username(url: str) -> Optional[str]:
    """
    Extract the GitHub username from a profile URL.
    Returns None if not a valid GitHub URL.
    """
    if not url:
        return None

    url = url.strip().lower()
    
    # Matches https://github.com/torvalds, github.com/torvalds/, www.github.com/torvalds
    match = re.search(r'(?:https?://)?(?:www\.)?github\.com/([^/]+)/?', url)
    if match:
        username = match.group(1).strip()
        if username:
            return username

    return None


def get_github_contributions(username: str) -> Optional[int]:
    """
    Fetch the total number of GitHub contributions in the last year
    for a given username. Returns None if the profile doesn't exist
    or data cannot be parsed.
    """
    if not username:
        return None

    url = f"https://github.com/users/{username}/contributions"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None

        # Look for phrases like "1,234 contributions in the last year"
        # in the returned HTML fragment.
        matches = re.findall(r'([\d,]+)\s+contributions', resp.text, re.IGNORECASE)
        if matches:
            # The first match in the contributions fragment is typically the total
            count_str = matches[0].replace(',', '')
            return int(count_str)

    except Exception:
        return None

    return None
