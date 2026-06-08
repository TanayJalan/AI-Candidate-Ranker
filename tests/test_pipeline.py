import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.structured_scorer import _score_experience
from src.honeypot_detector import _check_career_duration_mismatch
from src.job_parser import parse_job_description

def test_experience_scoring():
    """Ensure Gaussian experience scoring accurately penalizes out-of-range exp."""
    req = {"experience_range": [5.0, 9.0], "ideal_experience": 7.0}
    
    def mock_cand(yoe):
        return {"profile": {"years_of_experience": yoe}}
    
    perfect_score = _score_experience(mock_cand(7.0), req)
    assert perfect_score > 0.95  # Should be near 1.0

    low_score = _score_experience(mock_cand(2.0), req)
    assert low_score < 0.5  # 2 years is too junior

    high_score = _score_experience(mock_cand(15.0), req)
    assert high_score < 0.5  # 15 years is too senior

    # 5 and 9 should be respectable but not perfect
    assert 0.7 < _score_experience(mock_cand(5.0), req) < 0.95
    assert 0.7 < _score_experience(mock_cand(9.0), req) < 0.95


def test_honeypot_timeline_logic():
    """Ensure impossible timelines are flagged."""
    # Impossible: 10 years experience but career only spans 24 months
    bad_profile = {
        "profile": {"years_of_experience": 10.0},
        "career_history": [{"duration_months": 24}]
    }
    flagged, msg = _check_career_duration_mismatch(bad_profile)
    assert flagged is True
    assert "career duration mismatch" in msg

    # Valid: 2 years exp, 24 months duration
    good_profile = {
        "profile": {"years_of_experience": 2.0},
        "career_history": [{"duration_months": 24}]
    }
    flagged, _ = _check_career_duration_mismatch(good_profile)
    assert flagged is False


def test_jd_parsing_fallback():
    """Ensure fallback JD dict is loaded if DOCX parsing fails."""
    jd = parse_job_description()
    # Let's check for some general keys that the JD provides
    assert "semantic_text" in jd
    assert len(jd["semantic_text"]) > 100
