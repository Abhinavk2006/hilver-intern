import pandas as pd
from datasets import load_dataset
import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.config import RESULTS_DIR

def detailed_candidate_analysis():
    ds = load_dataset('SunidhiSriram/twcs', split='train')
    df = ds.to_pandas()

    candidates = ['AppleSupport', 'SpotifyCares', 'AmazonHelp', 'Uber_Support']
    
    report = {}

    for handle in candidates:
        company_df = df[df['author_id'] == handle]
        # Sample 500 company responses
        sample_responses = company_df['text'].sample(n=min(500, len(company_df)), random_state=42)
        
        # Check link/DM redirection rate (e.g. "DM us", "Direct Message", "link")
        dm_redirects = sample_responses.str.contains('DM|direct message|link|contact us', case=False).mean()
        
        # Check average character length
        avg_len = company_responses_len = sample_responses.str.len().mean()

        # Find customer inbound messages handled by this company
        company_tweet_ids = set(company_df['tweet_id'])
        company_replied_ids = set(company_df['in_response_to_tweet_id'].dropna().astype(int))
        
        customer_df = df[(df['inbound'] == True) & (df['tweet_id'].isin(company_replied_ids))]
        customer_sample = customer_df['text'].sample(n=min(500, len(customer_df)), random_state=42)

        report[handle] = {
            "company_responses_count": len(company_df),
            "customer_inbound_count": len(customer_df),
            "dm_redirect_rate_sample": round(float(dm_redirects), 4),
            "avg_response_len": round(float(avg_len), 2),
            "sample_customer_messages": customer_sample.head(5).tolist(),
            "sample_brand_responses": sample_responses.head(5).tolist()
        }

    out_file = RESULTS_DIR / "candidate_deep_dive.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print("Detailed analysis complete. Output written to candidate_deep_dive.json")

if __name__ == "__main__":
    detailed_candidate_analysis()
