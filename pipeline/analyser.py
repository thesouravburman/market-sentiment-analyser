"""
Core sentiment analysis pipeline using HuggingFace Transformers.
"""

from transformers import pipeline

class SentimentAnalyser:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipe = pipeline("sentiment-analysis", model=model_name)

    def analyse(self, text: str) -> dict:
        result = self.pipe(text[:512])[0]
        return {
            "text": text,
            "label": result["label"].lower(),
            "confidence": round(result["score"], 4)
        }

    def analyse_batch(self, texts: list) -> list:
        return [self.analyse(t) for t in texts]
