import streamlit as st
import pandas as pd
import json
import sys
import time
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    CANDIDATES_JSON,
    TOP_K,
    WEIGHT_SEMANTIC,
    WEIGHT_STRUCTURED,
    WEIGHT_BEHAVIORAL,
    WEIGHT_BONUS,
    JD_REQUIREMENTS,
    EMBEDDING_MODEL_NAME,
)
from src.job_parser import parse_job_description, get_jd_summary
from src.candidate_loader import load_candidates
# Heavy ML imports (semantic_scorer, structured_scorer, behavioral_scorer,
# text_enricher, honeypot_detector, reasoning_generator) are lazy-loaded
# inside run_pipeline() so the app boots fast on HF Spaces.


st.set_page_config(
    page_title="AI Candidate Ranker — Redrob Hackathon",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.85;
    }

    .score-bar {
        height: 8px;
        border-radius: 4px;
        background: #e0e0e0;
        margin: 2px 0;
    }
    .score-fill {
        height: 100%;
        border-radius: 4px;
    }

    .rank-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .rank-top { background: #d4edda; color: #155724; }
    .rank-mid { background: #fff3cd; color: #856404; }
    .rank-low { background: #f8d7da; color: #721c24; }

    .honeypot-flag {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Candidate Ranking System")
st.caption("Redrob Hackathon — Semantic + Structured + Behavioral Hybrid Ranker")

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Scoring Weights")
    st.caption("Adjust the importance of each scoring dimension. Weights are auto-normalized to sum to 100%.")

    raw_semantic = st.slider("🧠 Semantic", 0, 100, int(WEIGHT_SEMANTIC * 100), step=5, key="w_sem")
    raw_structured = st.slider("📋 Structured", 0, 100, int(WEIGHT_STRUCTURED * 100), step=5, key="w_str")
    raw_behavioral = st.slider("📊 Behavioral", 0, 100, int(WEIGHT_BEHAVIORAL * 100), step=5, key="w_beh")
    raw_bonus = st.slider("⭐ Bonus", 0, 100, int(WEIGHT_BONUS * 100), step=5, key="w_bon")

    # Normalize to sum to 1.0
    raw_total = raw_semantic + raw_structured + raw_behavioral + raw_bonus
    if raw_total > 0:
        user_weights = {
            "semantic": raw_semantic / raw_total,
            "structured": raw_structured / raw_total,
            "behavioral": raw_behavioral / raw_total,
            "bonus": raw_bonus / raw_total,
        }
    else:
        user_weights = {
            "semantic": WEIGHT_SEMANTIC,
            "structured": WEIGHT_STRUCTURED,
            "behavioral": WEIGHT_BEHAVIORAL,
            "bonus": WEIGHT_BONUS,
        }

    st.info(f"""
    **Normalized:**
    - Semantic: {user_weights['semantic']:.0%}
    - Structured: {user_weights['structured']:.0%}
    - Behavioral: {user_weights['behavioral']:.0%}
    - Bonus: {user_weights['bonus']:.0%}
    """)

    st.subheader("Model")
    st.code(EMBEDDING_MODEL_NAME)

    st.subheader("JD Requirements")
    st.write(f"**Role**: {JD_REQUIREMENTS['title']}")
    st.write(f"**Experience**: {JD_REQUIREMENTS['experience_range'][0]}-{JD_REQUIREMENTS['experience_range'][1]} years")
    st.write(f"**Required Skills**: {len(JD_REQUIREMENTS['required_skills'])}")
    st.write(f"**Preferred Skills**: {len(JD_REQUIREMENTS['preferred_skills'])}")

    st.divider()
    st.subheader("📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload candidates JSON or a single PDF/DOCX resume",
        type=["json", "pdf", "docx"],
        help="Upload a JSON file (batch) or a single PDF/DOCX resume."
    )

    use_sample = st.checkbox("Use sample data (50 candidates)", value=True)

def run_pipeline(candidates_path_or_data, weights=None):
    # Lazy imports — keeps app startup fast for HF Spaces health checks
    from src.text_enricher import enrich_all_candidates
    from src.semantic_scorer import SemanticScorer
    from src.structured_scorer import score_all_candidates as score_structured
    from src.behavioral_scorer import score_all_candidates as score_behavioral
    from src.hybrid_ranker import rank_candidates
    from src.reasoning_generator import generate_all_reasoning
    from src.honeypot_detector import detect_all_honeypots

    progress = st.progress(0, text="Initializing...")

    progress.progress(5, text="Parsing job description...")
    jd = parse_job_description()

    progress.progress(10, text="Loading candidates...")
    if isinstance(candidates_path_or_data, (str, Path)):
        candidates = load_candidates(Path(candidates_path_or_data))
    else:
        candidates = candidates_path_or_data

    actual_k = min(TOP_K, len(candidates))

    progress.progress(15, text="Running honeypot detection...")
    honeypot_results = detect_all_honeypots(candidates)
    honeypot_ids = {cid for cid, r in honeypot_results.items() if r["is_honeypot"]}

    progress.progress(20, text="Building rich text representations...")
    enriched = enrich_all_candidates(candidates)
    candidate_ids = [cid for cid, _ in enriched]
    candidate_texts = [text for _, text in enriched]

    progress.progress(30, text="Computing semantic similarity (this may take a moment)...")
    semantic_scorer = SemanticScorer()
    semantic_scores = semantic_scorer.score(
        jd_text=jd["semantic_text"],
        candidate_texts=candidate_texts,
        candidate_ids=candidate_ids,
        use_cache=True,
    )

    progress.progress(60, text="Computing structured match scores...")
    structured_scores = score_structured(candidates)

    for cid in honeypot_ids:
        if cid in structured_scores:
            structured_scores[cid] *= 0.1

    progress.progress(70, text="Computing behavioral scores...")
    behavioral_scores = score_behavioral(candidates)

    progress.progress(85, text="Combining scores and ranking...")
    ranked = rank_candidates(
        candidates=candidates,
        semantic_scores=semantic_scores,
        structured_scores=structured_scores,
        behavioral_scores=behavioral_scores,
        top_k=actual_k,
        weights=weights,
    )

    progress.progress(95, text="Generating reasoning strings...")
    generate_all_reasoning(ranked, honeypot_results=honeypot_results)

    progress.progress(100, text="Done!")
    time.sleep(0.5)
    progress.empty()

    return ranked, honeypot_results, len(candidates)



if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.honeypots = None
    st.session_state.total_candidates = 0

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 🚀 Run the Ranking Pipeline")
    st.markdown("Click below to rank candidates against the Senior AI Engineer JD.")

with col2:
    run_button = st.button("▶️ Run Pipeline", use_container_width=True, type="primary")

if run_button:
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == "json":
            data = json.loads(uploaded_file.read())
            if not isinstance(data, list):
                st.error("❌ Invalid file format: Expected a JSON list of candidates.")
                st.stop()
                
            if len(data) > 100:
                st.warning(f"Uploaded file has {len(data)} candidates. Truncating to 100 for demo.")
                data = data[:100]
            from src.candidate_loader import _normalize_candidate
            candidates = [_normalize_candidate(c) for c in data]
        elif file_ext in ["pdf", "docx"]:
            from src.resume_parser import extract_text_from_pdf, extract_text_from_docx, heuristic_parse_resume
            file_bytes = uploaded_file.read()
            if file_ext == "pdf":
                raw_text = extract_text_from_pdf(file_bytes)
            else:
                raw_text = extract_text_from_docx(file_bytes)
            
            mock_candidate = heuristic_parse_resume(raw_text, JD_REQUIREMENTS)
            candidates = [mock_candidate]
        else:
            st.error("Unsupported file type.")
            st.stop()

        ranked, honeypots, total = run_pipeline(candidates, weights=user_weights)
    elif use_sample:
        ranked, honeypots, total = run_pipeline(CANDIDATES_JSON, weights=user_weights)
    else:
        st.error("Please upload a candidates JSON file or check 'Use sample data'.")
        st.stop()

    st.session_state.results = ranked
    st.session_state.honeypots = honeypots
    st.session_state.total_candidates = total



if st.session_state.results:
    ranked = st.session_state.results
    honeypots = st.session_state.honeypots
    total = st.session_state.total_candidates

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Candidates", total)
    with col2:
        st.metric("Ranked", len(ranked))
    with col3:
        honeypot_count = sum(1 for r in honeypots.values() if r["is_honeypot"])
        st.metric("Honeypots Flagged", f"🍯 {honeypot_count}")
    with col4:
        top_score = ranked[0]["final_score"] if ranked else 0
        st.metric("Top Score", f"{top_score:.4f}")

    st.divider()

    st.subheader("📊 Ranked Candidates")

    rows = []
    for entry in ranked:
        c = entry["candidate"]
        p = c["profile"]
        hp = honeypots.get(entry["candidate_id"], {})

        rows.append({
            "Rank": entry["rank"],
            "Candidate ID": entry["candidate_id"],
            "Title": p["current_title"],
            "Company": p["current_company"],
            "Experience": f"{p['years_of_experience']:.1f} yrs",
            "Location": p.get("location", ""),
            "Final Score": round(entry["final_score"], 4),
            "Semantic": round(entry["semantic_score"], 3),
            "Structured": round(entry["structured_score"], 3),
            "Behavioral": round(entry["behavioral_score"], 3),
            "Bonus": round(entry["bonus_score"], 3),
            "🍯": "⚠️" if hp.get("is_honeypot") else "",
            "Reasoning": entry.get("reasoning", ""),
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        height=600,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Final Score": st.column_config.ProgressColumn(
                "Final Score",
                min_value=0,
                max_value=1,
                format="%.4f",
            ),
            "Semantic": st.column_config.ProgressColumn(
                "Semantic", min_value=0, max_value=1, format="%.3f",
            ),
            "Structured": st.column_config.ProgressColumn(
                "Structured", min_value=0, max_value=1, format="%.3f",
            ),
            "Behavioral": st.column_config.ProgressColumn(
                "Behavioral", min_value=0, max_value=1, format="%.3f",
            ),
            "Bonus": st.column_config.ProgressColumn(
                "Bonus", min_value=0, max_value=1, format="%.3f",
            ),
        },
        hide_index=True,
    )

    st.divider()
    st.subheader("🔍 Candidate Deep Dive")

    selected_rank = st.selectbox(
        "Select a candidate by rank:",
        options=list(range(1, len(ranked) + 1)),
        format_func=lambda r: f"Rank #{r} — {ranked[r-1]['candidate']['profile']['current_title']} ({ranked[r-1]['candidate_id']})"
    )

    entry = ranked[selected_rank - 1]
    c = entry["candidate"]
    p = c["profile"]
    signals = c.get("redrob_signals", {})

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {p['current_title']}")
        st.markdown(f"**{p['current_company']}** • {p.get('current_industry', '')} • {p.get('location', '')}, {p.get('country', '')}")
        st.markdown(f"*{p.get('headline', '')}*")

        st.markdown("#### 📝 Reasoning")
        st.info(entry.get("reasoning", ""))

        st.markdown("#### 🛠️ Skills")
        skills_data = []
        for s in c.get("skills", []):
            skills_data.append({
                "Skill": s.get("name", ""),
                "Proficiency": s.get("proficiency", ""),
                "Duration (months)": s.get("duration_months", 0),
                "Endorsements": s.get("endorsements", 0),
            })
        if skills_data:
            st.dataframe(pd.DataFrame(skills_data), hide_index=True, use_container_width=True)

        st.markdown("#### 💼 Career History")
        for job in c.get("career_history", []):
            duration_yrs = job.get("duration_months", 0) / 12
            current_badge = " 🟢" if job.get("is_current") else ""
            st.markdown(
                f"- **{job.get('title', '')}** at {job.get('company', '')} "
                f"({job.get('industry', '')}) — {duration_yrs:.1f} yrs{current_badge}"
            )

    with col2:
        st.markdown("#### 📊 Score Breakdown")

        scores = {
            "Semantic": entry["semantic_score"],
            "Structured": entry["structured_score"],
            "Behavioral": entry["behavioral_score"],
            "Bonus": entry["bonus_score"],
        }

        for name, score in scores.items():
            st.markdown(f"**{name}**: `{score:.4f}`")
            st.progress(min(1.0, score))

        st.markdown(f"**Final Score**: `{entry['final_score']:.4f}`")

        st.divider()
        st.markdown("#### 📡 Platform Signals")
        st.markdown(f"- Response Rate: **{signals.get('recruiter_response_rate', 0):.0%}**")
        st.markdown(f"- Open to Work: **{'Yes' if signals.get('open_to_work_flag') else 'No'}**")
        st.markdown(f"- Notice Period: **{signals.get('notice_period_days', 'N/A')}d**")
        github_url = p.get("github_url") or signals.get("github_url")
        if github_url:
            st.markdown(f"- GitHub: [Profile]({github_url})")
            from src.github_scraper import extract_github_username, get_github_contributions
            username = extract_github_username(github_url)
            if username:
                contributions = get_github_contributions(username)
                if contributions is not None:
                    st.markdown(f"- GitHub Contributions (1yr): **{contributions}**")
        else:
            st.markdown(f"- GitHub Score: **{signals.get('github_activity_score', 'N/A')}**")
        st.markdown(f"- Profile Complete: **{signals.get('profile_completeness_pct', 0):.0%}**")

        hp = honeypots.get(entry["candidate_id"], {})
        if hp.get("is_honeypot"):
            st.warning(f"🍯 **Honeypot detected** (confidence: {hp['confidence']:.0%})")
            for flag in hp["flags"]:
                st.caption(f"→ {flag}")

    st.divider()
    csv_data = df[["Candidate ID", "Rank", "Final Score", "Reasoning"]].rename(
        columns={"Candidate ID": "candidate_id", "Rank": "rank", "Final Score": "score", "Reasoning": "reasoning"}
    )
    csv_string = csv_data.to_csv(index=False)

    st.download_button(
        label="📥 Download submission.csv",
        data=csv_string,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.markdown("---")
    st.markdown("""

    This system ranks candidates using a **4-layer hybrid scoring** approach:

    1. **🧠 Semantic Scoring** — Sentence-transformer embeddings + FAISS cosine similarity
    2. **📋 Structured Scoring** — Title match, skill overlap, experience fit, industry relevance
    3. **📊 Behavioral Scoring** — Recruiter response rates, GitHub activity, platform engagement
    4. **⭐ Bonus Modifiers** — Open to work, verified contacts, low notice period

    Plus **🍯 Honeypot Detection** to flag candidates with impossible profiles.

    Click **Run Pipeline** above to see results!
    """)
