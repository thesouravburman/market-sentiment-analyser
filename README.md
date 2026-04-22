# 🧠 Market Sentiment Analyser

![SAMSUNG](https://img.shields.io/badge/SAMSUNG-R%26D%20PROJECT-1428A0?style=flat-square&logo=samsung&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-VADER%20%2B%20TEXTBLOB-6366F1?style=flat-square)
![Python](https://img.shields.io/badge/PYTHON-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/STREAMLIT-DEPLOYED-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
[![Live Demo](https://img.shields.io/badge/🌐%20LIVE%20DEMO-CLICK%20HERE-10B981?style=flat-square)](https://market-sentiment-analyser.streamlit.app)

**Professional NLP intelligence dashboard for decoding product review sentiment**
Built as part of the AI/ML portfolio by Sourav Burman · Samsung R&D Institute, India

---

## 🌐 Live Demo

## 👉 [https://market-sentiment-analyser.streamlit.app](https://market-sentiment-analyser.streamlit.app)

Paste any product review and instantly see:

- ✅ Positive / Negative / Neutral classification
- ✅ VADER compound score (−1.0 to +1.0)
- ✅ Confidence & subjectivity breakdown
- ✅ Top keyword extraction
- ✅ Batch CSV upload with downloadable results

---

## 🔬 How It Works

| Engine | Role | Output |
|--------|------|--------|
| **VADER** | Polarity scoring | Compound score + pos/neg/neu % |
| **TextBlob** | Subjectivity analysis | 0 (objective) → 1 (subjective) |
| **NLTK** | Stopword filtering | Clean keyword extraction |

### Sentiment Thresholds

| Label | Compound Score | Badge |
|-------|---------------|-------|
| 😊 POSITIVE | ≥ +0.05 | Emerald |
| 😐 NEUTRAL  | −0.05 to +0.05 | Amber |
| 😠 NEGATIVE | ≤ −0.05 | Rose |

---

## 📊 App Tabs

| Tab | What It Does |
|-----|-------------|
| 🧠 Analyse Text | Real-time scoring · score bars · keyword tags |
| 📂 Batch Analysis | CSV upload · donut chart · results table · download |
| 📊 Insights | Category breakdown · score distribution · word frequency |
| ℹ️ About | Project details · tech stack · builder info |

---

## 🚀 Run Locally

```bash
git clone https://github.com/thesouravburman/market-sentiment-analyser.git
cd market-sentiment-analyser
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

`Python 3.11` · `Streamlit` · `VADER` · `TextBlob` · `NLTK` · `Plotly` · `Pandas` · `NumPy`

---

**Built by [Sourav Burman](https://github.com/thesouravburman) · Samsung R&D Institute · India**
