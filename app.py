import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    clean_text,
    extract_skills,
    extract_text_from_pdf,
    extract_text_from_txt,
    jaccard_similarity,
    load_skills,
    skill_gap_analysis,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "resume_classifier.pkl"
SKILLS_PATH = BASE_DIR / "data" / "skills_list.txt"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
SAMPLE_JD_PATH = BASE_DIR / "data" / "sample_job_description.txt"

st.set_page_config(
    page_title="Resume Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        header[data-testid="stHeader"] { background: transparent !important; }
        [data-testid="collapsedControl"] { display: flex !important; }

        .stApp {
            background: #0d0d0d;
        }

        .block-container {
            padding: 1.5rem 2rem 3rem 2rem !important;
            max-width: 1100px;
        }

        section[data-testid="stSidebar"] > div {
            background: #111111;
            border-right: 1px solid #222;
            padding-top: 1.5rem !important;
        }

        /* ── Top bar ── */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0 1.2rem 0;
            border-bottom: 1px solid #222;
            margin-bottom: 1.6rem;
        }
        .topbar-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
        }
        .topbar-title span { color: #6366f1; }
        .topbar-sub {
            font-size: 0.82rem;
            color: #555;
        }

        /* ── Stat row ── */
        .stat-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        .stat-box {
            background: #161616;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }
        .stat-label {
            font-size: 0.72rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.4rem;
        }
        .stat-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: #fff;
            line-height: 1;
        }
        .stat-hint {
            font-size: 0.75rem;
            color: #444;
            margin-top: 0.35rem;
        }
        .stat-box.accent { border-color: #6366f1; }
        .stat-box.accent .stat-value { color: #818cf8; }

        /* ── Insight banner ── */
        .insight {
            background: #161616;
            border: 1px solid #2a2a2a;
            border-left: 3px solid #6366f1;
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            color: #aaa;
            font-size: 0.9rem;
            line-height: 1.65;
            margin-bottom: 1.5rem;
        }
        .insight strong { color: #e5e7eb; }

        /* ── Section header ── */
        .sec-head {
            font-size: 0.72rem;
            font-weight: 600;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 1.6rem 0 0.75rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .sec-head::after {
            content: '';
            flex: 1;
            height: 1px;
            background: #222;
        }

        /* ── Pills ── */
        .pill {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 6px;
            margin: 0.2rem 0.22rem 0.2rem 0;
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
        }
        .pill-default { background: #1a1a2e; border: 1px solid #2d2d5e; color: #a5b4fc; }
        .pill-green   { background: #0d2318; border: 1px solid #1a4731; color: #6ee7b7; }
        .pill-red     { background: #2a0d0d; border: 1px solid #4a1a1a; color: #fca5a5; }
        .pill-slate   { background: #1a1a1a; border: 1px solid #2a2a2a; color: #888; }

        /* ── Two-col result grid ── */
        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .result-card {
            background: #161616;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 1rem 1.1rem;
        }
        .result-card-title {
            font-size: 0.72rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.6rem;
        }

        /* ── Sidebar inputs ── */
        [data-testid="stFileUploader"] {
            background: #161616 !important;
            border: 1px solid #222 !important;
            border-radius: 10px !important;
            padding: 0.5rem !important;
        }
        [data-testid="stTextArea"] textarea {
            background: #161616 !important;
            color: #ccc !important;
            border: 1px solid #222 !important;
            border-radius: 10px !important;
            font-size: 0.85rem !important;
        }
        .stTextArea label, .stFileUploader label { color: #888 !important; font-size: 0.82rem !important; }

        /* ── Misc ── */
        .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #6366f1; }
        .muted { color: #555; font-size: 0.85rem; }
        .tag {
            display: inline-block;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            color: #666;
            font-size: 0.72rem;
            padding: 0.18rem 0.5rem;
            border-radius: 4px;
            margin-right: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def get_skills():
    return load_skills(str(SKILLS_PATH))


@st.cache_data
def get_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_sample_jd():
    return SAMPLE_JD_PATH.read_text(encoding="utf-8") if SAMPLE_JD_PATH.exists() else ""


def file_to_text(file):
    suffix = Path(file.name).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file)
    if suffix == ".txt":
        return extract_text_from_txt(file)
    return ""


def pills(items, tone="default"):
    if not items:
        return '<span class="muted">—</span>'
    cls = {"green": "pill-green", "red": "pill-red", "slate": "pill-slate"}.get(tone, "pill-default")
    return "".join(f'<span class="pill {cls}">{s}</span>' for s in items)


def get_top_predictions(text, top_n=5):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
        df = pd.DataFrame({"Role": classes, "Probability": probs}).sort_values("Probability", ascending=False).head(top_n)
        df["Confidence %"] = (df["Probability"] * 100).round(2)
        return df[["Role", "Confidence %"]]
    return pd.DataFrame()


inject_css()
model = load_model()
skills_list = get_skills()
metrics = get_metrics()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="topbar-title">⚡ Resume<span>Intel</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="muted" style="margin-bottom:1.2rem">NLP · ML · Skill Analysis</div>', unsafe_allow_html=True)

    st.markdown("**Upload Resume**")
    uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
    st.caption("Use text-based PDFs for best results.")

    st.markdown("---")
    use_sample = st.toggle("Load sample job description", value=False)
    default_jd = load_sample_jd() if use_sample else ""
    st.markdown("**Job Description**")
    job_description = st.text_area(
        "Paste JD",
        value=default_jd,
        height=220,
        label_visibility="collapsed",
        placeholder="Paste a job description to enable matching and gap analysis…",
    )

    st.markdown("---")
    accuracy = metrics.get("accuracy", 0)
    classes = metrics.get("report", {}).keys()
    clean_classes = [c for c in classes if c not in {"accuracy", "macro avg", "weighted avg"}]
    st.markdown(f'<div class="stat-label">Model accuracy</div><div class="mono">{accuracy:.2%}</div>', unsafe_allow_html=True)
    st.caption(f"{len(clean_classes)} supported roles")

# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="topbar-title">Resume <span>Intelligence</span> Dashboard</div>
            <div class="topbar-sub">Classification · Skill Extraction · JD Matching · Gap Analysis</div>
        </div>
        <div>
            <span class="tag">scikit-learn</span>
            <span class="tag">TF-IDF</span>
            <span class="tag">Streamlit</span>
            <span class="tag">Azure</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.markdown(
        """
        <div style="text-align:center; padding: 5rem 2rem; color: #333;">
            <div style="font-size:3rem; margin-bottom:1rem;">⚡</div>
            <div style="font-size:1.1rem; color:#555; font-weight:600;">Upload a resume to begin</div>
            <div style="font-size:0.85rem; color:#333; margin-top:0.5rem;">
                Prediction · Confidence · Skills · Match Score · Gap Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

resume_text = file_to_text(uploaded_file)
resume_clean = clean_text(resume_text)

if not resume_clean.strip():
    st.error("Could not extract readable text. Try a text-based PDF or TXT file.")
    st.stop()

predicted_role = model.predict([resume_clean])[0]
top_predictions = get_top_predictions(resume_clean)
confidence = top_predictions.iloc[0]["Confidence %"] if not top_predictions.empty else None
confidence_display = f"{confidence:.1f}%" if confidence else "N/A"

resume_skills = extract_skills(resume_clean, skills_list)
jd_clean = clean_text(job_description) if job_description.strip() else ""
jd_skills = extract_skills(jd_clean, skills_list) if jd_clean else []
match_score = jaccard_similarity(resume_skills, jd_skills) if jd_skills else 0.0
gap = (
    skill_gap_analysis(resume_skills, jd_skills)
    if jd_skills
    else {"matched": [], "missing": [], "extra": sorted(resume_skills)}
)

matched_count = len(gap["matched"])
missing_count = len(gap["missing"])

# ── Stat row ──
fit = "Strong" if match_score >= 0.65 else ("Moderate" if match_score >= 0.35 else "Limited")
st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-box accent">
            <div class="stat-label">Predicted Role</div>
            <div class="stat-value" style="font-size:1.1rem">{predicted_role}</div>
            <div class="stat-hint">Top classification</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Confidence</div>
            <div class="stat-value">{confidence_display}</div>
            <div class="stat-hint">Highest class probability</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Resume Skills</div>
            <div class="stat-value">{len(resume_skills)}</div>
            <div class="stat-hint">Extracted from resume</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">JD Match</div>
            <div class="stat-value">{match_score:.0%}</div>
            <div class="stat-hint">{fit} fit · {matched_count} matched · {missing_count} missing</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Insight ──
fit_color = {"Strong": "#6ee7b7", "Moderate": "#fcd34d", "Limited": "#fca5a5"}[fit]
st.markdown(
    f"""
    <div class="insight">
        Classified as <strong>{predicted_role}</strong> with <strong>{confidence_display}</strong> confidence.
        Shows a <strong style="color:{fit_color}">{fit.lower()} fit</strong> against the provided job description —
        <strong>{matched_count}</strong> skills matched, <strong>{missing_count}</strong> missing.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Role predictions + Match analytics ──
st.markdown('<div class="sec-head">Prediction Analytics</div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2, gap="medium")

with col_a:
    st.caption("Top predicted roles")
    if not top_predictions.empty:
        st.bar_chart(top_predictions.set_index("Role")["Confidence %"], height=200)
    else:
        st.write("Probability output unavailable.")

with col_b:
    st.caption("Skill match breakdown")
    if jd_skills:
        breakdown = pd.DataFrame(
            {"Category": ["Matched", "Missing", "Extra"], "Count": [matched_count, missing_count, len(gap["extra"])]}
        )
        st.bar_chart(breakdown.set_index("Category"), height=200)
    else:
        st.info("Paste a job description to see match analytics.")

# ── Skills ──
st.markdown('<div class="sec-head">Skills Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="result-grid">
        <div class="result-card">
            <div class="result-card-title">Resume Skills ({len(resume_skills)})</div>
            {pills(resume_skills)}
        </div>
        <div class="result-card">
            <div class="result-card-title">Matched Skills ({matched_count})</div>
            {pills(gap["matched"], "green")}
        </div>
        <div class="result-card">
            <div class="result-card-title">Missing Skills ({missing_count})</div>
            {pills(gap["missing"], "red")}
        </div>
        <div class="result-card">
            <div class="result-card-title">Extra Resume Skills ({len(gap["extra"])})</div>
            {pills(gap["extra"], "slate")}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Recruiter take ──
st.markdown('<div class="sec-head">Recruiter Interpretation</div>', unsafe_allow_html=True)
rec_col1, rec_col2 = st.columns(2, gap="medium")

with rec_col1:
    if match_score >= 0.65:
        st.success(f"**Strong alignment** — {predicted_role} profile matches the target role well.")
    elif match_score >= 0.35:
        st.warning(f"**Moderate alignment** — Some skill gaps exist for the target role.")
    else:
        st.error(f"**Limited alignment** — Significant skill gaps relative to the job description.")

with rec_col2:
    if missing_count > 0:
        st.info(f"**Recommended:** Focus on acquiring — {', '.join(gap['missing'][:6])}{'…' if missing_count > 6 else ''}")
    else:
        st.success("**No skill gaps detected.** Strengthen achievement-based bullet points next.")

# ── Resume + JD preview ──
st.markdown('<div class="sec-head">Document Preview</div>', unsafe_allow_html=True)
prev_col1, prev_col2 = st.columns(2, gap="medium")

with prev_col1:
    st.caption("Resume text")
    st.text_area("resume", resume_text[:6000], height=260, label_visibility="collapsed")

with prev_col2:
    st.caption("Job description")
    if job_description.strip():
        st.text_area("jd", job_description[:4000], height=260, label_visibility="collapsed")
    else:
        st.markdown('<div class="muted" style="padding-top:1rem">No job description provided.</div>', unsafe_allow_html=True)

# ── Model details ──
with st.expander("Model & Pipeline Details"):
    st.code("TF-IDF Vectorizer (1-2 grams, English stop words) → Logistic Regression", language="text")
    if metrics:
        st.json({"accuracy": metrics.get("accuracy"), "supported_roles": clean_classes})
    else:
        st.info("No metrics file found.")

st.markdown('<div class="muted" style="margin-top:2rem;text-align:center">Built with Streamlit · scikit-learn · pypdf · Deployed on Azure</div>', unsafe_allow_html=True)
