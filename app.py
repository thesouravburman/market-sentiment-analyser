
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analyzer import analyse_text, analyse_batch, get_sample_reviews, get_word_frequencies

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyser · Sourav Burman",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Fonts + Global CSS ─────────────────────────────────────────────────────────
st.markdown('''<div style="position:fixed;top:14px;left:16px;z-index:9999;display:flex;gap:8px;align-items:center;"><a href="https://github.com/thesouravburman" target="_blank" style="text-decoration:none;"><img src="https://img.shields.io/badge/GitHub-thesouravburman-181717?style=flat-square&logo=github&logoColor=white" style="height:22px;border-radius:4px;"/></a><a href="https://linkedin.com/in/sourav-burman" target="_blank" style="text-decoration:none;"><img src="https://img.shields.io/badge/LinkedIn-sourav--burman-0A66C2?style=flat-square&logo=linkedin&logoColor=white" style="height:22px;border-radius:4px;"/></a><a href="mailto:thesouravburman@gmail.com" style="text-decoration:none;"><img src="https://img.shields.io/badge/Email-Gmail-D14836?style=flat-square&logo=gmail&logoColor=white" style="height:22px;border-radius:4px;"/></a></div>''', unsafe_allow_html=True)

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;">
  <div style="font-family:'Montserrat',sans-serif;font-size:0.7rem;letter-spacing:0.35em;
              color:#6366F1;text-transform:uppercase;margin-bottom:8px;">
    NLP INTELLIGENCE · VADER · TEXTBLOB · REAL-TIME
  </div>
  <h1 style="font-family:'Montserrat',sans-serif;font-size:2.6rem;font-weight:900;
             background:linear-gradient(135deg,#6366F1,#10B981,#F43F5E);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin:0;letter-spacing:0.04em;">
    MARKET SENTIMENT ANALYSER
  </h1>
  <div style="color:#64748B;font-size:0.85rem;margin-top:8px;font-family:'Poppins',sans-serif;">
    Decode product review sentiment · Analyse pipelines · Extract signal from noise
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="indigo-divider"></div>', unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🧠  ANALYSE TEXT", "📂  BATCH ANALYSIS", "📊  INSIGHTS", "ℹ️  ABOUT"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSE TEXT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_output = st.columns([1.05, 0.95], gap="large")

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#6366F1;margin-bottom:14px;">
                    ● TEXT INPUT</div>""", unsafe_allow_html=True)

        user_text = st.text_area(
            "Enter product review or any text",
            placeholder="e.g. This product is absolutely amazing! Best purchase I have made all year...",
            height=160,
            label_visibility="collapsed"
        )

        # Quick sample buttons
        st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.65rem;
                    letter-spacing:0.15em;color:#475569;margin:12px 0 8px;">
                    ⚡ QUICK SAMPLES</div>""", unsafe_allow_html=True)
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("😊 Positive"):
                st.session_state["sample"] = "Absolutely love this product! The quality is phenomenal and delivery was super fast. Highly recommend to everyone!"
        with qc2:
            if st.button("😠 Negative"):
                st.session_state["sample"] = "Terrible quality. Broke after one use and customer service was completely unhelpful. Total waste of money."
        with qc3:
            if st.button("😐 Neutral"):
                st.session_state["sample"] = "The product is okay. It does what it claims to do. Nothing exceptional but gets the job done."

        if "sample" in st.session_state and not user_text:
            user_text = st.session_state["sample"]

        analyse_btn = st.button("🧠  ANALYSE SENTIMENT")
        st.markdown("</div>", unsafe_allow_html=True)

        # How it works
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#6366F1;margin-bottom:12px;">⚙ HOW IT WORKS</div>
        <div style="font-size:0.8rem;color:#94A3B8;line-height:1.9;">
          <b style="color:#10B981;">VADER</b> — Valence Aware Dictionary & sEntiment Reasoner.
          Compound score ranges from <b style="color:#F43F5E;">−1.0 (most negative)</b> to
          <b style="color:#10B981;">+1.0 (most positive)</b>.<br>
          <b style="color:#F59E0B;">TextBlob</b> — measures subjectivity from
          <b style="color:#E2E8F0;">0 (objective)</b> to <b style="color:#E2E8F0;">1 (subjective)</b>.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        if analyse_btn and user_text.strip():
            r = analyse_text(user_text)
            if r:
                color_map = {"POSITIVE":"#10B981","NEGATIVE":"#F43F5E","NEUTRAL":"#F59E0B"}
                card_map  = {"POSITIVE":"pos-card","NEGATIVE":"neg-card","NEUTRAL":"neu-card"}
                clr = color_map[r["label"]]

                # Main result badge
                st.markdown(f"""
                <div class="{card_map[r["label"]]}" style="text-align:center;padding:28px 20px;">
                  <div style="font-size:3.5rem;margin-bottom:8px;">{r["emoji"]}</div>
                  <div style="font-family:'Montserrat',sans-serif;font-size:1.8rem;
                              font-weight:900;color:{clr};letter-spacing:0.1em;">
                    {r["label"]}
                  </div>
                  <div style="font-family:'Montserrat',sans-serif;font-size:2.8rem;
                              font-weight:900;color:{clr};margin:4px 0;">
                    {r["compound"]:+.4f}
                  </div>
                  <div style="color:#64748B;font-size:0.75rem;letter-spacing:0.12em;">
                    VADER COMPOUND SCORE
                  </div>
                </div>""", unsafe_allow_html=True)

                # Score breakdown bars
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                            letter-spacing:0.2em;color:#6366F1;margin-bottom:14px;">
                            ▲ SCORE BREAKDOWN</div>""", unsafe_allow_html=True)

                for label, val, bar_color in [
                    ("Positive",    r["positive"],    "#10B981"),
                    ("Negative",    r["negative"],    "#F43F5E"),
                    ("Neutral",     r["neutral"],     "#F59E0B"),
                    ("Confidence",  r["confidence"],  "#6366F1"),
                    ("Subjectivity",r["subjectivity"],"#8B5CF6"),
                ]:
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex;justify-content:space-between;
                                  font-size:0.76rem;color:#CBD5E1;margin-bottom:3px;">
                        <span>{label}</span>
                        <span style="color:{bar_color};">{val}%</span>
                      </div>
                      <div style="background:#1E293B;border-radius:4px;height:6px;">
                        <div style="width:{min(val,100)}%;height:6px;border-radius:4px;
                                    background:{bar_color};opacity:0.85;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Top words
                if r["top_words"]:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                                letter-spacing:0.2em;color:#F59E0B;margin-bottom:10px;">
                                ▲ KEY WORDS</div>""", unsafe_allow_html=True)
                    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:6px;">', unsafe_allow_html=True)
                    for word, freq in r["top_words"][:8]:
                        st.markdown(f'<span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:3px 12px;font-size:0.73rem;color:#A5B4FC;">{word} <b style="color:#6366F1;">{freq}</b></span>', unsafe_allow_html=True)
                    st.markdown("</div></div>", unsafe_allow_html=True)

        elif analyse_btn and not user_text.strip():
            st.warning("Please enter some text to analyse.")

        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:70px 20px;">
              <div style="font-size:3.5rem;margin-bottom:16px;">🧠</div>
              <div style="font-family:'Montserrat',sans-serif;font-size:1rem;
                          font-weight:700;color:#6366F1;letter-spacing:0.1em;">
                READY TO ANALYSE
              </div>
              <div style="color:#475569;font-size:0.82rem;margin-top:8px;">
                Type or paste any product review<br>
                then click <b style="color:#10B981;">🧠 ANALYSE SENTIMENT</b>
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    bc1, bc2 = st.columns([1, 1.1], gap="large")

    with bc1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#6366F1;margin-bottom:14px;">
                    📂 UPLOAD CSV</div>""", unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.78rem;color:#64748B;margin-bottom:12px;">
                    CSV must have a text column containing reviews. Optional columns: category, rating.</div>""",
                    unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        text_col_input = st.text_input("Text column name", value="review_text",
                                       help="Column containing the review text")

        use_sample = st.button("📋  USE SAMPLE DATASET (40 reviews)")
        st.markdown("</div>", unsafe_allow_html=True)

        # Format guide
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#F59E0B;margin-bottom:10px;">📋 CSV FORMAT</div>
        <div style="font-size:0.78rem;color:#94A3B8;line-height:2;">
          <b style="color:#E2E8F0;">Required:</b> review_text (or any text column)<br>
          <b style="color:#E2E8F0;">Optional:</b> category, rating, date, product_id<br>
          <b style="color:#E2E8F0;">Encoding:</b> UTF-8 recommended<br>
          <b style="color:#E2E8F0;">Max rows:</b> 5,000 for best performance
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bc2:
        batch_df = None
        result_df = None

        if use_sample:
            batch_df = get_sample_reviews()
            text_col_input = "review_text"
            st.success("✅ Sample dataset loaded — 40 MyGlamm-inspired product reviews")

        elif uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(batch_df):,} rows from {uploaded.name}")
            except Exception as e:
                st.error(f"CSV error: {e}")

        if batch_df is not None and text_col_input in batch_df.columns:
            with st.spinner("🧠 Analysing sentiment..."):
                result_df = analyse_batch(batch_df, text_col_input)

            # Summary KPIs
            total = len(result_df)
            pos_n = (result_df["Sentiment"]=="POSITIVE").sum()
            neg_n = (result_df["Sentiment"]=="NEGATIVE").sum()
            neu_n = (result_df["Sentiment"]=="NEUTRAL").sum()
            avg_score = result_df["Score"].mean()

            k1,k2,k3,k4 = st.columns(4)
            for col, val, lbl in [
                (k1, total,                   "TOTAL REVIEWS"),
                (k2, f"{pos_n/total*100:.0f}%", "POSITIVE"),
                (k3, f"{neg_n/total*100:.0f}%", "NEGATIVE"),
                (k4, f"{avg_score:+.3f}",        "AVG SCORE"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-tile">
                      <div class="metric-val">{val}</div>
                      <div class="metric-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Donut chart
            donut = go.Figure(go.Pie(
                labels=["Positive","Negative","Neutral"],
                values=[pos_n, neg_n, neu_n],
                hole=0.62,
                marker=dict(colors=["#10B981","#F43F5E","#F59E0B"],
                            line=dict(color="#05080F", width=2)),
                textinfo="label+percent",
                textfont=dict(family="Montserrat",size=11,color="#E2E8F0")
            ))
            donut.add_annotation(text=f"<b>{total}</b><br><span style='font-size:9'>REVIEWS</span>",
                                 x=0.5, y=0.5, showarrow=False,
                                 font=dict(size=16, color="#E2E8F0", family="Montserrat"))
            donut.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                showlegend=False,
                                margin=dict(t=10,b=10,l=10,r=10), height=220,
                                title="SENTIMENT SPLIT")
            st.plotly_chart(donut, width='stretch')

            # Results table
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                        letter-spacing:0.2em;color:#6366F1;margin-bottom:10px;">
                        📋 RESULTS TABLE</div>""", unsafe_allow_html=True)
            show_cols = ["Sentiment","Score","Confidence (%)","Positive (%)","Negative (%)","Subjectivity (%)", text_col_input]
            show_cols = [c for c in show_cols if c in result_df.columns]
            st.dataframe(result_df[show_cols].sort_values("Score", ascending=False),
                         width='stretch', height=260)

            # Download
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  DOWNLOAD RESULTS CSV", csv_out,
                               "sentiment_results.csv", "text/csv", width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        elif batch_df is not None:
            st.error(f"Column '{text_col_input}' not found. Available: {list(batch_df.columns)}")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:60px 20px;">
              <div style="font-size:3rem;margin-bottom:14px;">📂</div>
              <div style="font-family:'Montserrat',sans-serif;font-size:0.95rem;
                          font-weight:700;color:#6366F1;letter-spacing:0.1em;">
                UPLOAD OR USE SAMPLE
              </div>
              <div style="color:#475569;font-size:0.82rem;margin-top:8px;">
                Upload your CSV or click<br>
                <b style="color:#F59E0B;">📋 USE SAMPLE DATASET</b>
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                letter-spacing:0.2em;color:#6366F1;margin-bottom:4px;">
                ● INSIGHTS ARE GENERATED FROM THE SAMPLE DATASET</div>
                <div style="font-size:0.78rem;color:#475569;">
                Load the sample dataset in Batch Analysis tab to explore live insights from your own CSV.</div>""",
                unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Always compute from sample
    sample_df = get_sample_reviews()
    insights_df = analyse_batch(sample_df, "review_text")

    # KPI row
    total_i = len(insights_df)
    pos_i   = (insights_df["Sentiment"]=="POSITIVE").sum()
    neg_i   = (insights_df["Sentiment"]=="NEGATIVE").sum()
    avg_i   = insights_df["Score"].mean()
    avg_sub = insights_df["Subjectivity (%)"].mean()

    i1,i2,i3,i4,i5 = st.columns(5)
    for col, val, lbl in [
        (i1, total_i,                      "REVIEWS"),
        (i2, f"{pos_i/total_i*100:.0f}%",  "POSITIVE"),
        (i3, f"{neg_i/total_i*100:.0f}%",  "NEGATIVE"),
        (i4, f"{avg_i:+.3f}",              "AVG VADER"),
        (i5, f"{avg_sub:.1f}%",            "AVG SUBJECTIVITY"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-tile">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)

    # Sentiment by category
    with ic1:
        cat_sent = insights_df.groupby(["category","Sentiment"]).size().reset_index(name="count")
        fig_cat = px.bar(cat_sent, x="category", y="count", color="Sentiment",
                         color_discrete_map={"POSITIVE":"#10B981","NEGATIVE":"#F43F5E","NEUTRAL":"#F59E0B"},
                         barmode="group",
                         title="SENTIMENT BY PRODUCT CATEGORY")
        fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#94A3B8",family="Poppins"),
                              legend=dict(bgcolor="rgba(0,0,0,0)",title_text=""),
                              xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                              yaxis=dict(gridcolor="#1E293B"),
                              margin=dict(l=10,r=10,t=42,b=10), height=300)
        st.plotly_chart(fig_cat, width='stretch')

    # Score distribution
    with ic2:
        fig_hist = go.Figure()
        for sent, clr in [("POSITIVE","#10B981"),("NEGATIVE","#F43F5E"),("NEUTRAL","#F59E0B")]:
            sub = insights_df[insights_df["Sentiment"]==sent]["Score"]
            fig_hist.add_trace(go.Histogram(x=sub, name=sent, marker_color=clr,
                                            opacity=0.75, xbins=dict(size=0.1)))
        fig_hist.update_layout(barmode="overlay",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#94A3B8",family="Poppins"),
                               title="VADER SCORE DISTRIBUTION",
                               legend=dict(bgcolor="rgba(0,0,0,0)"),
                               xaxis=dict(gridcolor="#1E293B",title="Compound Score"),
                               yaxis=dict(gridcolor="#1E293B",title="Count"),
                               margin=dict(l=10,r=10,t=42,b=10), height=300)
        st.plotly_chart(fig_hist, width='stretch')

    ic3, ic4 = st.columns(2)

    # Rating vs sentiment score scatter
    with ic3:
        fig_scat = px.scatter(insights_df, x="rating", y="Score",
                              color="Sentiment",
                              color_discrete_map={"POSITIVE":"#10B981","NEGATIVE":"#F43F5E","NEUTRAL":"#F59E0B"},
                              size="Confidence (%)",
                              title="STAR RATING vs VADER SCORE",
                              labels={"rating":"Star Rating","Score":"VADER Score"})
        fig_scat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#94A3B8",family="Poppins"),
                               legend=dict(bgcolor="rgba(0,0,0,0)",title_text=""),
                               xaxis=dict(gridcolor="#1E293B"),
                               yaxis=dict(gridcolor="#1E293B"),
                               margin=dict(l=10,r=10,t=42,b=10), height=300)
        st.plotly_chart(fig_scat, width='stretch')

    # Subjectivity vs confidence
    with ic4:
        fig_sub = px.scatter(insights_df, x="Subjectivity (%)", y="Confidence (%)",
                             color="Sentiment",
                             color_discrete_map={"POSITIVE":"#10B981","NEGATIVE":"#F43F5E","NEUTRAL":"#F59E0B"},
                             title="SUBJECTIVITY vs CONFIDENCE",
                             trendline="ols")
        fig_sub.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#94A3B8",family="Poppins"),
                              legend=dict(bgcolor="rgba(0,0,0,0)",title_text=""),
                              xaxis=dict(gridcolor="#1E293B"),
                              yaxis=dict(gridcolor="#1E293B"),
                              margin=dict(l=10,r=10,t=42,b=10), height=300)
        st.plotly_chart(fig_sub, width='stretch')

    # Top words bar chart
    pos_texts = insights_df[insights_df["Sentiment"]=="POSITIVE"]["review_text"].tolist()
    neg_texts = insights_df[insights_df["Sentiment"]=="NEGATIVE"]["review_text"].tolist()
    wf_pos = get_word_frequencies(pos_texts, 12)
    wf_neg = get_word_frequencies(neg_texts, 12)

    wc1, wc2 = st.columns(2)
    with wc1:
        fig_wp = px.bar(wf_pos, x="Frequency", y="Word", orientation="h",
                        color="Frequency", color_continuous_scale=["#064e3b","#10B981","#d1fae5"],
                        title="TOP WORDS — POSITIVE REVIEWS")
        fig_wp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#94A3B8",family="Poppins"),
                             coloraxis_showscale=False,
                             yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                             xaxis=dict(gridcolor="#1E293B"),
                             margin=dict(l=10,r=10,t=42,b=10), height=320)
        st.plotly_chart(fig_wp, width='stretch')

    with wc2:
        fig_wn = px.bar(wf_neg, x="Frequency", y="Word", orientation="h",
                        color="Frequency", color_continuous_scale=["#4c0519","#F43F5E","#fecdd3"],
                        title="TOP WORDS — NEGATIVE REVIEWS")
        fig_wn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#94A3B8",family="Poppins"),
                             coloraxis_showscale=False,
                             yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                             xaxis=dict(gridcolor="#1E293B"),
                             margin=dict(l=10,r=10,t=42,b=10), height=320)
        st.plotly_chart(fig_wn, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    ab1, ab2 = st.columns([1.6, 1])

    with ab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#6366F1;margin-bottom:16px;">● ABOUT THIS PROJECT</div>
        <h2 style="font-family:'Montserrat',sans-serif;font-weight:900;font-size:1.5rem;
                   margin:0 0 12px;background:linear-gradient(135deg,#6366F1,#10B981,#F43F5E);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          MARKET SENTIMENT ANALYSER
        </h2>
        <p style="color:#94A3B8;line-height:1.8;font-size:0.88rem;">
          A professional NLP intelligence dashboard for decoding sentiment from product reviews.
          Combines <b style="color:#6366F1;">VADER</b> (Valence Aware Dictionary and sEntiment Reasoner)
          for polarity scoring and <b style="color:#F59E0B;">TextBlob</b> for subjectivity analysis —
          delivering enterprise-grade signal extraction without heavy model dependencies.
        </p>
        <p style="color:#94A3B8;line-height:1.8;font-size:0.88rem;">
          Built for e-commerce and D2C brands to understand customer sentiment at scale — from
          single review analysis to batch processing thousands of reviews via CSV upload.
        </p>
        <div style="margin-top:20px;">
          <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                      letter-spacing:0.2em;color:#10B981;margin-bottom:10px;">▲ KEY FEATURES</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82rem;color:#CBD5E1;">
            <div>🧠 Real-time sentiment scoring</div>
            <div>📂 Batch CSV processing</div>
            <div>📊 Polarity score breakdown</div>
            <div>📈 Subjectivity measurement</div>
            <div>🔍 Keyword extraction</div>
            <div>📉 Category-level insights</div>
            <div>⬇️ Downloadable results CSV</div>
            <div>🎯 Hot / Warm / Cold tiering</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ab2:
        st.markdown('<div class="neu-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#F59E0B;margin-bottom:16px;">● BUILDER</div>
        <div style="text-align:center;padding:10px 0;">
          <div style="font-size:3rem;margin-bottom:10px;">👨‍💻</div>
          <div style="font-family:'Montserrat',sans-serif;font-size:1.3rem;
                      font-weight:900;color:#E2E8F0;letter-spacing:0.06em;">SOURAV BURMAN</div>
          <div style="color:#64748B;font-size:0.78rem;letter-spacing:0.12em;margin-top:4px;">
            CS ENGINEER · AI/ML · NLP
          </div>
          <div style="margin-top:16px;font-size:0.8rem;color:#94A3B8;line-height:1.9;">
            🐙 github.com/thesouravburman<br>
            📧 thesouravburman@gmail.com
          </div>
          <div style="margin-top:20px;padding:10px 18px;
                      background:rgba(99,102,241,0.1);border-radius:8px;
                      border:1px solid rgba(99,102,241,0.25);
                      font-size:0.72rem;color:#6366F1;letter-spacing:0.1em;
                      font-family:'Montserrat',sans-serif;">
            🏢 SAMSUNG R&D INSTITUTE · INDIA
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top:0;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                    letter-spacing:0.2em;color:#6366F1;margin-bottom:12px;">⚡ TECH STACK</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">""", unsafe_allow_html=True)
        for t in ["Python 3.11","Streamlit","VADER","TextBlob","Plotly","NLTK","Pandas","NumPy"]:
            st.markdown(f'<span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:6px;padding:3px 10px;font-size:0.73rem;color:#A5B4FC;">{t}</span>', unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="indigo-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:12px 0 4px;font-family:'Montserrat',sans-serif;
            font-size:0.68rem;letter-spacing:0.2em;color:#334155;">
  BUILT BY <span style="color:#6366F1;">SOURAV BURMAN</span> ·
  <span style="color:#10B981;">MARKET SENTIMENT ANALYSER</span> ·
  <span style="color:#F43F5E;">VADER · TEXTBLOB · STREAMLIT · PLOTLY</span>
</div>
""", unsafe_allow_html=True)
