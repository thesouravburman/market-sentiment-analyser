"""
Run entry point for Market Sentiment Analyser.
"""

import argparse
import pandas as pd
from pipeline.analyser import SentimentAnalyser

def parse_args():
    parser = argparse.ArgumentParser(description="Market Sentiment Analyser")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file with reviews")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.input)
    analyser = SentimentAnalyser()
    results = analyser.analyse_batch(df["review"].tolist())
    for r in results:
        print(f"[{r['label'].upper():8s}] ({r['confidence']:.2f}) {r['text'][:60]}...")
