import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from src.preprocessing import clean_text

class ConversationReconstructor:
    """
    Reconstructs customer-brand conversation threads from Twitter dataset graph.
    """
    def __init__(self, brand_handle: str = "AppleSupport"):
        self.brand_handle = brand_handle

    def reconstruct_brand_pairs(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Reconstructs customer query -> brand response pairs.
        Each pair consists of:
        - customer_tweet_id
        - customer_author_id
        - customer_text_raw
        - customer_text_cleaned
        - customer_created_at
        - brand_tweet_id
        - brand_text_raw
        - brand_text_cleaned
        - brand_created_at
        """
        print(f"Reconstructing conversation threads for brand: {self.brand_handle}...")

        # Filter brand tweets
        brand_df = df[df['author_id'] == self.brand_handle].copy()
        
        # Fast dictionary lookup for tweets by tweet_id
        # Note: tweet_id in TWCS is numeric (int) or float if missing
        tweet_map = df.set_index('tweet_id').to_dict(orient='index')

        pairs = []
        skipped_no_customer = 0

        for _, brand_row in brand_df.iterrows():
            brand_tweet_id = brand_row['tweet_id']
            in_reply_to_id = brand_row.get('in_response_to_tweet_id')

            if pd.isna(in_reply_to_id):
                continue
            
            try:
                in_reply_to_id = int(in_reply_to_id)
            except (ValueError, TypeError):
                continue

            # Look up customer parent tweet
            customer_tweet = tweet_map.get(in_reply_to_id)
            if not customer_tweet or not customer_tweet.get('inbound'):
                skipped_no_customer += 1
                continue

            c_text_raw = customer_tweet['text']
            c_text_cleaned = clean_text(c_text_raw)

            b_text_raw = brand_row['text']
            b_text_cleaned = clean_text(b_text_raw)

            # Ignore empty or ultra-short messages (< 5 chars)
            if len(c_text_cleaned) < 5 or len(b_text_cleaned) < 5:
                continue

            pair = {
                "conversation_id": f"{in_reply_to_id}_{brand_tweet_id}",
                "customer_tweet_id": int(in_reply_to_id),
                "customer_author_id": str(customer_tweet['author_id']),
                "customer_created_at": str(customer_tweet['created_at']),
                "customer_text_raw": c_text_raw,
                "customer_text_cleaned": c_text_cleaned,
                "brand_tweet_id": int(brand_tweet_id),
                "brand_author_id": self.brand_handle,
                "brand_created_at": str(brand_row['created_at']),
                "brand_text_raw": b_text_raw,
                "brand_text_cleaned": b_text_cleaned
            }
            pairs.append(pair)

        print(f"Successfully reconstructed {len(pairs):,} customer-brand pairs for {self.brand_handle} (skipped {skipped_no_customer:,} orphan pairs).")
        return pairs
