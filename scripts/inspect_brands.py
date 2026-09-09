import pandas as pd
from datasets import load_dataset
import sys
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.config import RAW_DATA_DIR, RESULTS_DIR

def inspect_dataset_and_brands():
    print("Loading SunidhiSriram/twcs dataset...")
    ds = load_dataset('SunidhiSriram/twcs', split='train')
    df = ds.to_pandas()

    print("\n--- DATASET OVERVIEW ---")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    # Non-inbound tweets authored by support handle (usually end with 'Support' or 'Care' or brand handles)
    # Support handles have inbound == False
    support_tweets = df[df['inbound'] == False]
    author_counts = support_tweets['author_id'].value_counts()

    print(f"Total unique support handles: {len(author_counts):,}")
    print("\nTop 20 Support Brands by Message Volume:")
    top_20 = author_counts.head(20)
    for handle, count in top_20.items():
        print(f"  - {handle}: {count:,} messages")

    # Let's inspect conversation reconstruction stats for top 10 brands
    print("\nAnalyzing conversation thread quality for top candidate brands...")
    
    # Fast map of tweet_id to row
    # In TWCS, inbound=True means customer tweet, inbound=False means company tweet
    # Let's count how many customer inbound tweets explicitly mention or reply to top company handles
    
    brand_stats = []
    top_candidates = author_counts.head(10).index.tolist()

    for handle in top_candidates:
        company_msgs = df[df['author_id'] == handle]
        nb_company_msgs = len(company_msgs)
        
        # Inbound tweets replying to this company's tweets or where response_tweet_id leads to company
        # We can also check tweets in response to company tweets
        in_reply_to_company = df[df['in_response_to_tweet_id'].isin(company_msgs['tweet_id'])]
        
        # Inbound tweets where company replied (company's in_response_to_tweet_id)
        company_replied_to_tweet_ids = company_msgs['in_response_to_tweet_id'].dropna().astype(int)
        customer_inbound_replied = df[df['tweet_id'].isin(company_replied_to_tweet_ids)]
        
        stat = {
            "handle": handle,
            "company_responses": nb_company_msgs,
            "customer_inbound_tweets": len(customer_inbound_replied),
            "total_associated_turns": nb_company_msgs + len(customer_inbound_replied)
        }
        brand_stats.append(stat)

    stats_df = pd.DataFrame(brand_stats)
    print("\n--- BRAND SELECTION METRICS ---")
    print(stats_df.to_string(index=False))

    # Save summary report
    out_file = RESULTS_DIR / "brand_inspection_summary.json"
    with open(out_file, "w") as f:
        json.dump({
            "total_rows": len(df),
            "columns": df.columns.tolist(),
            "top_20_brands": author_counts.head(20).to_dict(),
            "candidate_brand_stats": brand_stats
        }, f, indent=2)
    print(f"\nSaved brand inspection results to {out_file}")

if __name__ == "__main__":
    inspect_dataset_and_brands()
