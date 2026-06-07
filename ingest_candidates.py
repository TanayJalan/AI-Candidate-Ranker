import json
import pandas as pd
from tqdm import tqdm
import os

def extract_candidate_text(candidate):
    """
    Combine various fields into a single text string for embedding.
    """
    parts = []
    # profile summary
    if 'profile' in candidate and 'summary' in candidate['profile']:
        parts.append(candidate['profile']['summary'])
    # headline
    if 'profile' in candidate and 'headline' in candidate['profile']:
        parts.append(candidate['profile']['headline'])
    # skills
    if 'skills' in candidate:
        skill_names = [s.get('name', '') for s in candidate['skills'] if isinstance(s, dict)]
        if skill_names:
            parts.append('Skills: ' + ', '.join(skill_names))
    # career history descriptions
    if 'career_history' in candidate:
        for job in candidate['career_history']:
            if isinstance(job, dict) and 'description' in job:
                parts.append(job['description'])
    # education maybe not needed but add field of study
    if 'education' in candidate:
        for edu in candidate['education']:
            if isinstance(edu, dict) and 'field_of_study' in edu:
                parts.append('Studied: ' + edu['field_of_study'])
    # Join with spaces
    return ' '.join(parts)

def main():
    input_path = '/Users/tanayjalan/Downloads/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl'
    output_dir = '/Users/tanayjalan/candidate_matcher/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'candidates_processed.parquet')
    
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc='Processing candidates')):
            try:
                cand = json.loads(line.strip())
                cand_id = cand.get('candidate_id')
                text = extract_candidate_text(cand)
                # Keep some metadata for later scoring: years_of_experience, redrob_signals, etc.
                years = cand.get('profile', {}).get('years_of_experience')
                redrob = cand.get('redrob_signals', {})
                # extract some behavioral signals
                signals = {
                    'profile_completeness': redrob.get('profile_completeness_score'),
                    'last_active_date': redrob.get('last_active_date'),
                    'open_to_work': redrob.get('open_to_work_flag'),
                    'recruiter_response_rate': redrob.get('recruiter_response_rate'),
                    'github_activity_score': redrob.get('github_activity_score'),
                    'search_appearance_30d': redrob.get('search_appearance_30d'),
                    'saved_by_recruiters_30d': redrob.get('saved_by_recruiters_30d'),
                    'interview_completion_rate': redrob.get('interview_completion_rate'),
                    'offer_acceptance_rate': redrob.get('offer_acceptance_rate'),
                }
                records.append({
                    'candidate_id': cand_id,
                    'text': text,
                    'years_of_experience': years,
                    **signals
                })
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    df = pd.DataFrame.from_records(records)
    print(f"Processed {len(df)} candidates")
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
