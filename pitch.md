# Redrob Hackathon Pitch / Defense

*Use this quick 2-minute outline to defend your architecture to the judges.*

### 1. The Hook (The Problem)
"We all know the standard AI recruiting approach: throw the Job Description and the Resumes into an LLM, or use standard vector search. The problem? That doesn't work. Vector search thinks a 'Civil Engineer' is an 80% match for an 'AI Engineer'. And simple keyword matching is easily gamed by candidates stuffing 'Machine Learning' into a resume where they worked as a Customer Support agent."

### 2. Our Solution (The Architecture)
"Our system doesn't just read words, it ranks candidates exactly how a senior recruiter does. We built a 4-Layer Hybrid Engine:
1. **Semantic Layer (25%)**: We use `all-mpnet-base-v2` and FAISS for deep contextual understanding of skills.
2. **Structured Layer (40%)**: This is our anchor. It verifies hard constraints: Title relevance, actual years of experience (using a Gaussian curve around the ideal range), and skill proficiency scaled by duration.
3. **Behavioral Layer (25%)**: We integrate Redrob's platform signals. We don't just want a good resume, we want someone who responds to messages, has an active GitHub, and is open to work.
4. **Honeypot Detector**: We built an automated fraud detector. It checks 7 different heuristic rules to instantly flag synthetic or physically impossible profiles, docking their scores by 90%."

### 3. Why we win (The Engineering)
"We built this for scale and realism. 
- **Fast & Cheap**: We process 100 candidates in under 50 seconds on pure CPU. Zero LLM API calls, zero network lag.
- **Explainable AI**: Our system dynamically generates specific, fact-based reasoning strings for every candidate. No hallucinated LLM summaries — it cites exact years and skills.
- **Transparent**: We built a Streamlit dashboard that breaks down exactly *why* a candidate scored the way they did across all four layers."
