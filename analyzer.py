
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from collections import Counter
import nltk

# Download required NLTK data silently
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except:
    STOPWORDS = {"the","a","an","is","it","in","on","at","to","for","of","and","or","but","i","we","you"}

VADER = SentimentIntensityAnalyzer()

def analyse_text(text: str) -> dict:
    """Analyse a single piece of text. Returns full sentiment breakdown."""
    text = str(text).strip()
    if not text:
        return None

    # VADER scores
    vs = VADER.polarity_scores(text)
    compound = vs["compound"]

    # Label + confidence
    if compound >= 0.05:
        label, color, emoji = "POSITIVE", "#10B981", "😊"
    elif compound <= -0.05:
        label, color, emoji = "NEGATIVE", "#F43F5E", "😠"
    else:
        label, color, emoji = "NEUTRAL",  "#F59E0B", "😐"

    confidence = abs(compound)  # 0–1

    # TextBlob subjectivity
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity  # 0=objective, 1=subjective

    # Word count & top words
    words = re.findall(r"[a-z]+", text.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    top_words = Counter(filtered).most_common(10)

    return {
        "text": text,
        "label": label,
        "color": color,
        "emoji": emoji,
        "compound": round(compound, 4),
        "confidence": round(confidence * 100, 1),
        "positive": round(vs["pos"] * 100, 1),
        "negative": round(vs["neg"] * 100, 1),
        "neutral":  round(vs["neu"] * 100, 1),
        "subjectivity": round(subjectivity * 100, 1),
        "word_count": len(words),
        "top_words": top_words,
    }

def analyse_batch(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Analyse a full dataframe. Returns df with sentiment columns added."""
    results = []
    for text in df[text_col]:
        r = analyse_text(str(text))
        if r:
            results.append({
                "Sentiment": r["label"],
                "Score": r["compound"],
                "Confidence (%)": r["confidence"],
                "Positive (%)": r["positive"],
                "Negative (%)": r["negative"],
                "Neutral (%)": r["neutral"],
                "Subjectivity (%)": r["subjectivity"],
            })
        else:
            results.append({"Sentiment":"NEUTRAL","Score":0,"Confidence (%)":0,
                            "Positive (%)":0,"Negative (%)":0,"Neutral (%)":100,"Subjectivity (%)":50})
    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)

def get_sample_reviews() -> pd.DataFrame:
    """Returns 40 sample product reviews for demo."""
    reviews = [
        ("Absolutely love this product! Best purchase I've made all year.", "Skincare", 5),
        ("Terrible quality, broke after one use. Complete waste of money.", "Electronics", 1),
        ("It's okay I guess. Nothing special but does the job.", "Apparel", 3),
        ("Outstanding customer service! They resolved my issue instantly.", "Service", 5),
        ("Product arrived damaged and the packaging was awful.", "Electronics", 1),
        ("Pretty good for the price. Would recommend to friends.", "Skincare", 4),
        ("Not bad, not great. Somewhere in the middle honestly.", "Food", 3),
        ("Exceeded all expectations! The quality is phenomenal.", "Skincare", 5),
        ("Horrible experience from start to finish. Never again.", "Service", 1),
        ("Decent product but shipping took way too long.", "Apparel", 2),
        ("This moisturiser is absolutely incredible. My skin glows!", "Skincare", 5),
        ("Waste of money. Does not work as advertised at all.", "Haircare", 1),
        ("Average product. Nothing to write home about.", "Wellness", 3),
        ("Fast delivery, great packaging, product is perfect!", "Electronics", 5),
        ("Very disappointed. Expected much better for the price.", "Apparel", 2),
        ("Good quality and fair pricing. Happy with my purchase.", "Food", 4),
        ("The fragrance is divine! I get compliments everywhere I go.", "Fragrance", 5),
        ("Stopped working after a week. Extremely frustrating.", "Electronics", 1),
        ("It is fine. Does what it says, no more no less.", "Wellness", 3),
        ("Brilliant product! Changed my skincare routine completely.", "Skincare", 5),
        ("Returned immediately. Completely different from the pictures.", "Apparel", 1),
        ("Good but not great. Slight improvement in my hair texture.", "Haircare", 3),
        ("Best lipstick I've ever used. The colour stays all day.", "Makeup", 5),
        ("The smell is awful and it caused a rash on my skin.", "Skincare", 1),
        ("Neutral experience. Product is functional and reliable.", "Electronics", 3),
        ("Super fast shipping and the product is exactly as described!", "Food", 5),
        ("Cheap material, bad finish. Deeply regret buying this.", "Apparel", 1),
        ("Alright product, does its job without any fuss.", "Wellness", 3),
        ("Remarkable results in just two weeks! Highly recommend.", "Skincare", 5),
        ("Worst purchase ever. Stopped working on day one.", "Electronics", 1),
        ("The serum is good but takes time to show results.", "Skincare", 3),
        ("Packaging is beautiful and the product smells amazing!", "Fragrance", 5),
        ("Very oily texture and too heavy for daily use.", "Skincare", 2),
        ("Solid performance. Happy to buy again in the future.", "Electronics", 4),
        ("Awful taste, awful smell. Could not finish even one serving.", "Food", 1),
        ("Simple and effective. Does exactly what it claims to do.", "Wellness", 4),
        ("Love the new formula! Much better than the previous version.", "Haircare", 5),
        ("Misleading description. Product is completely different.", "Apparel", 1),
        ("Good enough for occasional use but not for daily wear.", "Makeup", 3),
        ("Sensational results! My hair has never looked this healthy.", "Haircare", 5),
    ]
    return pd.DataFrame(reviews, columns=["review_text", "category", "rating"])

def get_word_frequencies(texts: list, top_n: int = 20) -> pd.DataFrame:
    """Get top N word frequencies from a list of texts."""
    all_words = []
    for text in texts:
        words = re.findall(r"[a-z]+", str(text).lower())
        all_words.extend([w for w in words if w not in STOPWORDS and len(w) > 2])
    freq = Counter(all_words).most_common(top_n)
    return pd.DataFrame(freq, columns=["Word", "Frequency"])
