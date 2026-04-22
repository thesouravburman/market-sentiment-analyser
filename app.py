import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from analyzer import analyse_text, analyse_batch, get_sample_reviews, get_word_frequencies

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Sentiment Analyser · Sourav Burman",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global stylesheet ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&family=Poppins:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #05080F;
    color: #E2E8F0;
    font-family: 'Poppins', sans-serif;
}

/* FIX 1: Give the main content layer a stacking context so it sits ABOVE the z-index:0 canvas */
[data-testid="stAppViewContainer"] > .main {
    background: transparent;
    position: relative;
    z-index: 1;
}

.glass-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 14px;
    padding: 22px 22px 16px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}

.pos-card {
    background: rgba(16, 185, 129, 0.07);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}

.neg-card {
    background: rgba(244, 63, 94, 0.07);
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}

.neu-card {
    background: rgba(245, 158, 11, 0.07);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 16px;
}

.metric-tile {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-val {
    font-family: 'Montserrat', sans-serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #6366F1;
}

.metric-lbl {
    font-family: 'Montserrat', sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: #475569;
    margin-top: 4px;
}

.indigo-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #6366F1, transparent);
    margin: 18px 0;
    opacity: 0.4;
}

[data-testid="stTextArea"] textarea {
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 10px 20px !important;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    transform: translateY(-1px);
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    color: #475569 !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #6366F1 !important;
    border-bottom-color: #6366F1 !important;
}

.stDataFrame {
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Animated background ────────────────────────────────────────────────────────
# FIX 2: opacity:0.45 on the canvas keeps it subtle; z-index:0 keeps it behind
# the app content (which now has z-index:1 via CSS above).
components.html("""
<script>
(function () {
  const parentDoc = window.parent.document;

  const existing = parentDoc.getElementById('bg3d');
  if (existing) existing.remove();

  const canvas = parentDoc.createElement('canvas');
  canvas.id = 'bg3d';
  /* opacity:0.45 = subtle background; z-index:0 = behind the z-index:1 app layer */
  canvas.style.cssText =
    'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;opacity:0.45;';
  parentDoc.body.appendChild(canvas);

  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = window.parent.innerWidth;
    canvas.height = window.parent.innerHeight;
  }
  resize();
  window.parent.addEventListener('resize', resize);

  const COLORS = ['#6366F1','#10B981','#F43F5E','#F59E0B','#8B5CF6','#06B6D4'];

  /* ── Sphere points ── */
  const pts = [];
  for (let i = 0; i < 130; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const rad   = 90 + Math.random() * 210;
    pts.push({
      ox: rad * Math.sin(phi) * Math.cos(theta),
      oy: rad * Math.sin(phi) * Math.sin(theta),
      oz: rad * Math.cos(phi),
      r : Math.random() * 2.8 + 1.0,
      color: COLORS[Math.floor(Math.random() * COLORS.length)]
    });
  }

  /* ── Floating beads ── */
  const beads = [];
  for (let i = 0; i < 28; i++) {
    beads.push({
      x    : Math.random() * window.parent.innerWidth,
      y    : Math.random() * window.parent.innerHeight,
      r    : Math.random() * 5 + 3,
      vx   : (Math.random() - 0.5) * 0.45,
      vy   : (Math.random() - 0.5) * 0.45,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      phase: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.025 + 0.012
    });
  }

  let ang  = 0;
  let time = 0;

  function project(x, y, z) {
    const fov = 360, sc = fov / (fov + z + 180);
    return { sx: canvas.width / 2 + x * sc, sy: canvas.height / 2 + y * sc, sc };
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ang  += 0.0022;
    time += 0.016;

    const cA = Math.cos(ang),        sA = Math.sin(ang);
    const cB = Math.cos(ang * 0.37), sB = Math.sin(ang * 0.37);

    const projected = pts.map(p => {
      const rx     = p.ox * cA + p.oz * sA;
      const ry_raw = -p.ox * sA + p.oz * cA;
      const ry     = p.oy * cB - ry_raw * sB;
      const rz     = p.oy * sB + ry_raw * cB;
      const pr     = project(rx, ry, rz);
      return { ...pr, color: p.color, r: p.r, rz };
    }).sort((a, b) => b.rz - a.rz);

    /* Connection lines */
    for (let i = 0; i < projected.length; i++) {
      for (let j = i + 1; j < projected.length; j++) {
        const dx = projected[i].sx - projected[j].sx;
        const dy = projected[i].sy - projected[j].sy;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < 90) {
          ctx.beginPath();
          ctx.moveTo(projected[i].sx, projected[i].sy);
          ctx.lineTo(projected[j].sx, projected[j].sy);
          ctx.strokeStyle = 'rgba(99,102,241,' + (0.38 * (1 - d / 90)).toFixed(3) + ')';
          ctx.lineWidth   = 0.85;
          ctx.stroke();
        }
      }
    }

    /* Sphere dots */
    projected.forEach(p => {
      const size  = Math.max(1.2, p.r * p.sc * 5.5);
      const alpha = Math.min(1.0, Math.max(0.35, p.sc * 2.0));
      const hex   = Math.round(alpha * 255).toString(16).padStart(2, '0');

      const g = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, size * 5);
      g.addColorStop(0, p.color + 'aa');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, size * 5, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(p.sx, p.sy, size, 0, Math.PI * 2);
      ctx.fillStyle = p.color + hex;
      ctx.fill();
    });

    /* Floating beads */
    beads.forEach(b => {
      b.x += b.vx;
      b.y += b.vy;
      if (b.x < -25)                b.x = canvas.width  + 25;
      if (b.x > canvas.width  + 25) b.x = -25;
      if (b.y < -25)                b.y = canvas.height + 25;
      if (b.y > canvas.height + 25) b.y = -25;

      const pulse = b.r + Math.sin(time * b.speed * 60 + b.phase) * 2.2;
      const alpha = 0.55 + Math.sin(time * b.speed * 40 + b.phase) * 0.22;
      const hexA  = Math.round(alpha * 255).toString(16).padStart(2, '0');

      const glow = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, pulse * 7);
      glow.addColorStop(0, b.color + '55');
      glow.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(b.x, b.y, pulse * 7, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      const core = ctx.createRadialGradient(
        b.x - pulse * 0.3, b.y - pulse * 0.3, 0,
        b.x, b.y, pulse
      );
      core.addColorStop(0, '#ffffff' + hexA);
      core.addColorStop(0.4, b.color + hexA);
      core.addColorStop(1,   b.color + '99');
      ctx.beginPath();
      ctx.arc(b.x, b.y, pulse, 0, Math.PI * 2);
      ctx.fillStyle = core;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", height=0, scrolling=False)

# ── Social badge bar ───────────────────────────────────────────────────────────
# FIX 3: setTimeout(300ms) waits for parent DOM to be ready before injecting.
# setInterval(2000ms) re-injects every 2 s so badges survive Streamlit re-renders
# that wipe externally injected DOM nodes. The guard `getElementById` prevents
# duplicate injections on each interval tick.
components.html("""
<script>
(function () {
  function injectBadges() {
    const parentDoc = window.parent.document;
    if (!parentDoc || !parentDoc.body) return;

    /* Already injected — do nothing */
    if (parentDoc.getElementById('social-badges-bar')) return;

    const div = parentDoc.createElement('div');
    div.id = 'social-badges-bar';
    div.style.cssText =
      'position:fixed;top:11px;left:16px;z-index:9999;display:flex;gap:7px;align-items:center;';
    div.innerHTML = `
      <a href="https://github.com/thesouravburman" target="_blank" style="text-decoration:none;">
        <img src="https://img.shields.io/badge/GitHub-thesouravburman-181717?style=flat-square&logo=github&logoColor=white"
             style="height:21px;border-radius:4px;" />
      </a>
      <a href="https://linkedin.com/in/souravburman" target="_blank" style="text-decoration:none;">
        <img src="https://img.shields.io/badge/LinkedIn-Sourav%20Burman-0A66C2?style=flat-square&logo=linkedin&logoColor=white"
             style="height:21px;border-radius:4px;" />
      </a>
      <a href="mailto:thesouravburman@gmail.com" style="text-decoration:none;">
        <img src="https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail&logoColor=white"
             style="height:21px;border-radius:4px;" />
      </a>
    `;
    parentDoc.body.appendChild(div);
  }

  /* First attempt after a short delay so parent DOM is ready */
  setTimeout(injectBadges, 300);
  /* Keep re-checking to survive Streamlit hot-reloads */
  setInterval(injectBadges, 2000);
})();
</script>
""", height=0, scrolling=False)

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

# ── Tabs ───────────────────────────────────────────────────────────────────────
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

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("😊 Positive"):
                st.session_state["sample"] = "Absolutely love this product! The quality is phenomenal and delivery was super fast. Highly recommend to everyone!"
                st.rerun()
        with qc2:
            if st.button("😠 Negative"):
                st.session_state["sample"] = "Terrible quality. Broke after one use and customer service was completely unhelpful. Total waste of money."
                st.rerun()
        with qc3:
            if st.button("😐 Neutral"):
                st.session_state["sample"] = "The product is okay. It does what it claims to do. Nothing exceptional but gets the job done."
                st.rerun()

        st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.65rem;
                    letter-spacing:0.15em;color:#475569;margin:4px 0 8px;">
                    ⚡ QUICK SAMPLES ABOVE · OR TYPE BELOW</div>""", unsafe_allow_html=True)

        user_text = st.text_area(
            "Enter product review or any text",
            value=st.session_state.get("sample", ""),
            placeholder="e.g. This product is absolutely amazing! Best purchase I have made all year...",
            height=160,
            label_visibility="collapsed"
        )

        analyse_btn = st.button("🧠  ANALYSE SENTIMENT")
        st.markdown("</div>", unsafe_allow_html=True)

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
                color_map = {"POSITIVE": "#10B981", "NEGATIVE": "#F43F5E", "NEUTRAL": "#F59E0B"}
                card_map  = {"POSITIVE": "pos-card", "NEGATIVE": "neg-card", "NEUTRAL": "neu-card"}
                clr = color_map[r["label"]]

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

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                            letter-spacing:0.2em;color:#6366F1;margin-bottom:14px;">
                            ▲ SCORE BREAKDOWN</div>""", unsafe_allow_html=True)

                for label, val, bar_color in [
                    ("Positive",     r["positive"],    "#10B981"),
                    ("Negative",     r["negative"],    "#F43F5E"),
                    ("Neutral",      r["neutral"],     "#F59E0B"),
                    ("Confidence",   r["confidence"],  "#6366F1"),
                    ("Subjectivity", r["subjectivity"],"#8B5CF6"),
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

                if r["top_words"]:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                                letter-spacing:0.2em;color:#F59E0B;margin-bottom:10px;">
                                ▲ KEY WORDS</div>""", unsafe_allow_html=True)
                    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:6px;">', unsafe_allow_html=True)
                    for word, freq in r["top_words"][:8]:
                        st.markdown(
                            f'<span style="background:rgba(99,102,241,0.15);border:1px solid '
                            f'rgba(99,102,241,0.3);border-radius:20px;padding:3px 12px;'
                            f'font-size:0.73rem;color:#A5B4FC;">'
                            f'{word} <b style="color:#6366F1;">{freq}</b></span>',
                            unsafe_allow_html=True
                        )
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
                    CSV must have a text column containing reviews.
                    Optional columns: category, rating.</div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        text_col_input = st.text_input("Text column name", value="review_text",
                                       help="Column containing the review text")
        use_sample = st.button("📋  USE SAMPLE DATASET (40 reviews)")
        st.markdown("</div>", unsafe_allow_html=True)

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
        batch_df  = None
        result_df = None

        if use_sample:
            batch_df       = get_sample_reviews()
            text_col_input = "review_text"
            st.success("✅ Sample dataset loaded — 40 product reviews")
        elif uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(batch_df):,} rows from {uploaded.name}")
            except Exception as e:
                st.error(f"CSV error: {e}")

        if batch_df is not None and text_col_input in batch_df.columns:
            with st.spinner("🧠 Analysing sentiment..."):
                result_df = analyse_batch(batch_df, text_col_input)

            total     = len(result_df)
            pos_n     = (result_df["Sentiment"] == "POSITIVE").sum()
            neg_n     = (result_df["Sentiment"] == "NEGATIVE").sum()
            neu_n     = (result_df["Sentiment"] == "NEUTRAL").sum()
            avg_score = result_df["Score"].mean()

            k1, k2, k3, k4 = st.columns(4)
            for col, val, lbl in [
                (k1, total,                      "TOTAL REVIEWS"),
                (k2, f"{pos_n/total*100:.0f}%",  "POSITIVE"),
                (k3, f"{neg_n/total*100:.0f}%",  "NEGATIVE"),
                (k4, f"{avg_score:+.3f}",         "AVG SCORE"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-tile">
                      <div class="metric-val">{val}</div>
                      <div class="metric-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Donut chart
            donut = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[pos_n, neg_n, neu_n],
                hole=0.62,
                marker=dict(colors=["#10B981", "#F43F5E", "#F59E0B"],
                            line=dict(color="#05080F", width=2)),
                textinfo="label+percent",
                textfont=dict(family="Montserrat", size=11, color="#E2E8F0")
            ))
            donut.add_annotation(
                text=f"<b>{total}</b><br><span style='font-size:9'>REVIEWS</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#E2E8F0", family="Montserrat")
            )
            donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(t=34, b=10, l=10, r=10),
                height=220,
                title="SENTIMENT SPLIT",
                title_font=dict(family="Montserrat", size=12, color="#6366F1")
            )
            st.plotly_chart(donut, use_container_width=True)

            # Results table
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("""<div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                        letter-spacing:0.2em;color:#6366F1;margin-bottom:10px;">
                        📋 RESULTS TABLE</div>""", unsafe_allow_html=True)
            show_cols = [
                "Sentiment", "Score", "Confidence (%)",
                "Positive (%)", "Negative (%)", "Subjectivity (%)", text_col_input
            ]
            show_cols = [c for c in show_cols if c in result_df.columns]
            st.dataframe(
                result_df[show_cols].sort_values("Score", ascending=False),
                use_container_width=True,
                height=260
            )

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️  DOWNLOAD RESULTS CSV",
                csv_out,
                "sentiment_results.csv",
                "text/csv",
                use_container_width=True
            )
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
    st.markdown("""
    <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                letter-spacing:0.2em;color:#6366F1;margin-bottom:4px;">
                ● INSIGHTS ARE GENERATED FROM THE SAMPLE DATASET</div>
    <div style="font-size:0.78rem;color:#475569;">
      Load the sample dataset in Batch Analysis tab to explore live insights from your own CSV.
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    sample_df   = get_sample_reviews()
    insights_df = analyse_batch(sample_df, "review_text")

    total_i = len(insights_df)
    pos_i   = (insights_df["Sentiment"] == "POSITIVE").sum()
    neg_i   = (insights_df["Sentiment"] == "NEGATIVE").sum()
    avg_i   = insights_df["Score"].mean()
    avg_sub = insights_df["Subjectivity (%)"].mean()

    i1, i2, i3, i4, i5 = st.columns(5)
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

    # ── Sentiment by category ──────────────────────────────────────────────────
    with ic1:
        cat_sent = insights_df.groupby(["category", "Sentiment"]).size().reset_index(name="count")
        fig_cat  = px.bar(
            cat_sent, x="category", y="count", color="Sentiment",
            color_discrete_map={"POSITIVE": "#10B981", "NEGATIVE": "#F43F5E", "NEUTRAL": "#F59E0B"},
            barmode="group",
            title="SENTIMENT BY PRODUCT CATEGORY"
        )
        fig_cat.update_layout(
            title_font=dict(family="Montserrat", size=13, color="#6366F1"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Poppins"),
            legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=10, r=10, t=42, b=10),
            height=300
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Score distribution ─────────────────────────────────────────────────────
    with ic2:
        fig_hist = go.Figure()
        for sent, clr in [("POSITIVE","#10B981"), ("NEGATIVE","#F43F5E"), ("NEUTRAL","#F59E0B")]:
            sub = insights_df[insights_df["Sentiment"] == sent]["Score"]
            fig_hist.add_trace(go.Histogram(
                x=sub, name=sent, marker_color=clr,
                opacity=0.75, xbins=dict(size=0.1)
            ))
        fig_hist.update_layout(
            barmode="overlay",
            title="VADER SCORE DISTRIBUTION",
            title_font=dict(family="Montserrat", size=13, color="#F59E0B"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Poppins"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1E293B", title="Compound Score"),
            yaxis=dict(gridcolor="#1E293B", title="Count"),
            margin=dict(l=10, r=10, t=42, b=10),
            height=300
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    ic3, ic4 = st.columns(2)

    # ── Rating vs sentiment score scatter ──────────────────────────────────────
    with ic3:
        fig_scat = px.scatter(
            insights_df, x="rating", y="Score",
            color="Sentiment",
            color_discrete_map={"POSITIVE": "#10B981", "NEGATIVE": "#F43F5E", "NEUTRAL": "#F59E0B"},
            size="Confidence (%)",
            title="STAR RATING vs VADER SCORE",
            labels={"rating": "Star Rating", "Score": "VADER Score"}
        )
        fig_scat.update_layout(
            title_font=dict(family="Montserrat", size=13, color="#10B981"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Poppins"),
            legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=10, r=10, t=42, b=10),
            height=300
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Subjectivity vs confidence ─────────────────────────────────────────────
    with ic4:
        try:
            fig_sub = px.scatter(
                insights_df, x="Subjectivity (%)", y="Confidence (%)",
                color="Sentiment",
                color_discrete_map={"POSITIVE": "#10B981", "NEGATIVE": "#F43F5E", "NEUTRAL": "#F59E0B"},
                title="SUBJECTIVITY vs CONFIDENCE",
                trendline="ols"
            )
        except Exception:
            fig_sub = px.scatter(
                insights_df, x="Subjectivity (%)", y="Confidence (%)",
                color="Sentiment",
                color_discrete_map={"POSITIVE": "#10B981", "NEGATIVE": "#F43F5E", "NEUTRAL": "#F59E0B"},
                title="SUBJECTIVITY vs CONFIDENCE"
            )
        fig_sub.update_layout(
            title_font=dict(family="Montserrat", size=13, color="#8B5CF6"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Poppins"),
            legend=dict(bgcolor="rgba(0,0,0,0)", title_text=""),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=10, r=10, t=42, b=10),
            height=300
        )
        st.plotly_chart(fig_sub, use_container_width=True)

    # ── Top-words bar charts ───────────────────────────────────────────────────
    pos_texts = insights_df[insights_df["Sentiment"] == "POSITIVE"]["review_text"].tolist()
    neg_texts = insights_df[insights_df["Sentiment"] == "NEGATIVE"]["review_text"].tolist()
    wf_pos    = get_word_frequencies(pos_texts, 12)
    wf_neg    = get_word_frequencies(neg_texts, 12)

    wc1, wc2 = st.columns(2)
    with wc1:
        fig_wp = px.bar(
            wf_pos, x="Frequency", y="Word", orientation="h",
            color="Frequency",
            color_continuous_scale=["#064e3b", "#10B981", "#d1fae5"],
            title="TOP WORDS — POSITIVE REVIEWS"
        )
        fig_wp.update_layout(
            title_font=dict(family="Montserrat", size=13, color="#10B981"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Poppins"),
            coloraxis_showscale=False,
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=10, r=10, t=42, b=10),
            height=320
        )
        st.plotly_chart(fig_wp, use_container_width=True)

    with wc2:
        fig_wn = px.bar(
            wf_neg, x="Frequency", y="Word", orientation="h",
            color="Frequency",
            color_continuous_scale=["#4c0519", "#F43F5E", "#fecdd3"],
            title="TOP WORDS — NEGATIVE REVIEWS"
        )
        fig_wn.update_layout(
            title_font=dict(family="Montserrat", size=13, color="#F43F5E"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Poppins"),
            coloraxis_showscale=False,
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=10, r=10, t=42, b=10),
            height=320
        )
        st.plotly_chart(fig_wn, use_container_width=True)


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
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                      font-size:0.82rem;color:#CBD5E1;">
            <div>🧠 Real-time sentiment scoring</div>
            <div>📂 Batch CSV processing</div>
            <div>📊 Polarity score breakdown</div>
            <div>📈 Subjectivity measurement</div>
            <div>🔍 Keyword extraction</div>
            <div>📉 Category-level insights</div>
            <div>⬇️ Downloadable results CSV</div>
            <div>🎯 Positive / Negative / Neutral tiering</div>
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
            🐙 <a href="https://github.com/thesouravburman" target="_blank"
                  style="color:#94A3B8;text-decoration:underline;">github.com/thesouravburman</a><br>
            📧 <a href="mailto:thesouravburman@gmail.com"
                  style="color:#94A3B8;text-decoration:underline;">thesouravburman@gmail.com</a>
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

        tech_stack = ["Python 3.11", "Streamlit", "VADER", "TextBlob",
                      "Plotly", "NLTK", "Pandas", "NumPy"]
        tech_badges = "".join([
            f'<span style="background:rgba(99,102,241,0.15);border:1px solid '
            f'rgba(99,102,241,0.3);border-radius:6px;padding:3px 10px;'
            f'font-size:0.73rem;color:#A5B4FC;">{t}</span>'
            for t in tech_stack
        ])
        st.markdown(f"""
        <div class="glass-card" style="margin-top:0;">
          <div style="font-family:'Montserrat',sans-serif;font-size:0.68rem;
                      letter-spacing:0.2em;color:#6366F1;margin-bottom:12px;">⚡ TECH STACK</div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">{tech_badges}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="indigo-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:12px 0 4px;font-family:'Montserrat',sans-serif;
            font-size:0.68rem;letter-spacing:0.2em;color:#334155;">
  BUILT BY <span style="color:#6366F1;">SOURAV BURMAN</span> ·
  <span style="color:#10B981;">MARKET SENTIMENT ANALYSER</span> ·
  <span style="color:#F43F5E;">VADER · TEXTBLOB · STREAMLIT · PLOTLY</span>
</div>
""", unsafe_allow_html=True)
