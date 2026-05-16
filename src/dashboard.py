"""
PCOS Digital Twin — Dashboard
==============================
Run with:  streamlit run src/dashboard.py

Setup:
    1. Create a .env file in the project root with:
           GROQ_API_KEY=gsk_...
    2. pip install streamlit pdfplumber groq joblib matplotlib numpy pandas python-dotenv
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
from explainability import render_explainability_tab

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Load .env (optional) ───────────────────────────────────────────────────────
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SRC, ".."))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass
ENV_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Paths ──────────────────────────────────────────────────────────────────────
STATE_MODEL_PATH = os.path.join(_ROOT, "models", "state_estimator.pkl")
RISK_MODEL_PATH  = os.path.join(_ROOT, "models", "risk_predictor.pkl")
SCALER_PATH      = os.path.join(_ROOT, "models", "scaler.pkl")
DATA_PATH        = os.path.join(_ROOT, "data", "processed", "pcos_processed.csv")

# ── Colour tokens (single source of truth) ─────────────────────────────────────
C = {
    "bg":        "#f4f7fb",
    "surface":   "#ffffff",
    "border":    "#d8e4ef",
    "border2":   "#b8cfe0",
    "teal":      "#0b6b54",
    "teal_mid":  "#0f8068",
    "teal_lite": "#e0f4ee",
    "teal_xlt":  "#f0faf6",
    "text":      "#1a3040",
    "muted":     "#4f6a7a",
    "hint":      "#7a9aaa",
    "red":       "#c0392b",
    "orange":    "#d35400",
    "green":     "#1e8449",
    "grid":      "#dde8f0",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, .stApp {{
  background: {C['bg']} !important;
  font-family: 'DM Sans', sans-serif !important;
  color: {C['text']} !important;
}}

/* ── Header bar ── */
.stApp > header {{
  background: {C['surface']} !important;
  border-bottom: 1px solid {C['border']} !important;
  box-shadow: 0 1px 4px rgba(11,107,84,0.06) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: {C['surface']} !important;
  border-right: 1px solid {C['border']} !important;
}}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

/* ── Main container ── */
.main .block-container {{
  background: {C['bg']} !important;
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1200px !important;
}}

/* ── Typography ── */
h1 {{ font-size: 1.75rem !important; font-weight: 600 !important;
      color: {C['teal']} !important; letter-spacing: -0.3px !important; }}
h2 {{ font-size: 1.25rem !important; font-weight: 600 !important;
      color: {C['teal']} !important; }}
h3 {{ font-size: 1rem !important; font-weight: 600 !important;
      color: {C['teal']} !important; }}
p, li {{ color: {C['text']} !important; line-height: 1.6 !important; }}
.stCaption {{ color: {C['muted']} !important; font-size: 0.82rem !important; }}

/* ── Section header ── */
.sec-hdr {{
  font-size: 0.82rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: {C['teal']};
  border-bottom: 1.5px solid {C['teal_lite']};
  padding-bottom: 6px;
  margin: 24px 0 14px;
}}

/* ── Step card ── */
.step-card {{
  background: {C['surface']};
  border: 1px solid {C['border']};
  border-radius: 12px;
  padding: 20px 24px 18px;
  margin-bottom: 14px;
  box-shadow: 0 1px 4px rgba(11,107,84,0.05);
  transition: box-shadow 0.2s;
}}
.step-card:hover {{ box-shadow: 0 2px 10px rgba(11,107,84,0.09); }}

/* ── Step number badge ── */
.step-num {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: {C['teal']};
  color: #fff;
  border-radius: 50%;
  width: 24px; height: 24px;
  font-size: 0.78rem; font-weight: 600;
  margin-right: 10px;
  vertical-align: middle;
}}
.step-title {{
  font-size: 0.97rem;
  font-weight: 600;
  color: {C['teal']};
  vertical-align: middle;
}}
.step-sub {{
  font-size: 0.8rem;
  color: {C['muted']};
  margin: 4px 0 0 34px;
}}

/* ── Upload box labels ── */
.up-label {{
  font-size: 0.82rem;
  font-weight: 600;
  color: {C['teal']};
  margin-bottom: 3px;
  display: block;
}}
.up-sub {{
  font-size: 0.76rem;
  color: {C['muted']};
  margin-bottom: 6px;
  display: block;
}}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {{
  background: {C['teal_xlt']} !important;
  border: 1.5px dashed #5dcaa5 !important;
  border-radius: 10px !important;
  padding: 4px !important;
}}
div[data-testid="stFileUploader"] * {{ color: {C['teal']} !important; }}
div[data-testid="stFileUploader"] section {{ padding: 8px 12px !important; }}

/* ── Risk card ── */
.risk-card {{
  background: {C['surface']};
  border: 1px solid {C['border']};
  border-top: 3px solid var(--rc, {C['teal']});
  border-radius: 10px;
  padding: 16px 12px 14px;
  text-align: center;
  margin-bottom: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  transition: transform 0.15s, box-shadow 0.15s;
}}
.risk-card:hover {{ transform: translateY(-2px);
                   box-shadow: 0 4px 14px rgba(0,0,0,0.08); }}
.risk-lbl  {{ font-size: 0.72rem; font-weight: 500; text-transform: uppercase;
              letter-spacing: 0.6px; color: {C['muted']}; margin-bottom: 6px; }}
.risk-val  {{ font-size: 2rem; font-weight: 600; line-height: 1; }}
.risk-tier {{ font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.8px; margin-top: 4px; }}
.bar-bg  {{ background: {C['grid']}; border-radius: 4px; height: 5px; margin-top: 10px; }}
.bar-fg  {{ border-radius: 4px; height: 5px; transition: width 0.8s ease; }}

/* ── Info / notice box ── */
.notice {{
  background: {C['teal_lite']};
  border-left: 3px solid {C['teal']};
  border-radius: 0 8px 8px 0;
  padding: 10px 14px;
  font-size: 0.83rem;
  color: {C['teal']};
  margin: 8px 0;
  line-height: 1.55;
}}
.notice strong {{ color: {C['teal']}; }}
.notice.warn {{
  background: #fef9ec;
  border-left-color: #d4a017;
  color: #7a5c00;
}}

/* ── Buttons ── */
.stButton > button {{
  background: {C['teal']} !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  padding: 10px 0 !important;
  letter-spacing: 0.2px !important;
  transition: background 0.15s, transform 0.1s !important;
}}
.stButton > button:hover {{
  background: {C['teal_mid']} !important;
  transform: translateY(-1px) !important;
}}
.stButton > button:active {{
  background: #085041 !important;
  transform: none !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
  background: {C['surface']} !important;
  border-bottom: 1.5px solid {C['border']} !important;
  border-radius: 0 !important;
  gap: 2px !important;
  padding: 0 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.85rem !important;
  color: {C['muted']} !important;
  font-weight: 500 !important;
  border-radius: 6px 6px 0 0 !important;
  padding: 9px 18px !important;
  transition: color 0.15s, background 0.15s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: {C['teal']} !important;
  background: {C['teal_xlt']} !important;
}}
.stTabs [aria-selected="true"] {{
  color: {C['teal']} !important;
  font-weight: 600 !important;
  background: {C['teal_lite']} !important;
  border-bottom: 2px solid {C['teal']} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
  border: 1px solid {C['border']} !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}}

/* ── Inputs ── */
.stTextInput > div > input,
.stSelectbox > div > div {{
  background: {C['surface']} !important;
  border: 1px solid {C['border2']} !important;
  border-radius: 8px !important;
  color: {C['text']} !important;
  font-family: 'DM Sans', sans-serif !important;
}}
.stTextInput > div > input:focus {{
  border-color: {C['teal']} !important;
  box-shadow: 0 0 0 2px rgba(11,107,84,0.12) !important;
}}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] > div > div {{
  background: {C['teal']} !important;
}}
.stSlider [data-testid="stThumbValue"] {{
  color: {C['teal']} !important;
  font-weight: 600 !important;
}}

/* ── Alerts ── */
.stSuccess {{
  background: {C['teal_lite']} !important;
  color: {C['teal']} !important;
  border-color: {C['teal']} !important;
  border-radius: 8px !important;
}}
.stError  {{ border-radius: 8px !important; }}
.stInfo   {{
  background: {C['teal_lite']} !important;
  color: {C['teal']} !important;
  border-radius: 8px !important;
}}
.stWarning {{ border-radius: 8px !important; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {C['teal']} !important; }}

/* ── Multiselect tags ── */
.stMultiSelect span[data-baseweb="tag"] {{
  background: {C['teal_lite']} !important;
  color: {C['teal']} !important;
  border: 1px solid #9fe1cb !important;
  border-radius: 6px !important;
}}

/* ── Expander ── */
details {{
  border: 1px solid {C['border']} !important;
  border-radius: 10px !important;
  background: {C['surface']} !important;
  overflow: hidden !important;
}}
details summary {{
  color: {C['teal']} !important;
  font-weight: 500 !important;
  padding: 10px 14px !important;
  font-size: 0.9rem !important;
}}

/* ── Divider ── */
hr {{ border-color: {C['border']} !important; margin: 20px 0 !important; }}

/* ── API key hint ── */
.api-hint {{
  font-size: 0.78rem;
  color: {C['muted']};
  margin-top: 6px;
}}
.api-hint a {{ color: {C['teal']}; text-decoration: none; }}
.api-hint a:hover {{ text-decoration: underline; }}

/* ── Metric ── */
[data-testid="metric-container"] {{
  background: {C['surface']} !important;
  border: 1px solid {C['border']} !important;
  border-radius: 10px !important;
  padding: 14px 16px !important;
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    missing = [p for p in [STATE_MODEL_PATH, RISK_MODEL_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, f"Missing model files: {missing}"
    try:
        state_model = joblib.load(STATE_MODEL_PATH)
        risk_models = joblib.load(RISK_MODEL_PATH)
        scaler      = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        return state_model, risk_models, scaler, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def load_feature_cols():
    if not os.path.exists(DATA_PATH):
        return []
    df = pd.read_csv(DATA_PATH)
    return (df.drop(columns=["PCOS (Y/N)"], errors="ignore")
              .select_dtypes(include=[np.number]).columns.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(uploaded_file) -> str:
    try:
        import pdfplumber
    except ImportError:
        st.error("Run: pip install pdfplumber")
        st.stop()
    parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                parts.append(f"--- Page {i+1} ---\n{t}")
    return "\n\n".join(parts)


EXTRACTION_PROMPT = """You are a medical data extraction assistant for a PCOS Digital Twin system.
Extract values from the lab report text below. Return ONLY a valid JSON object with these exact keys.
Use null for any value not found. No markdown, no explanation — raw JSON only.

{
  "Age (yrs)": null, "Weight (Kg)": null, "Height(Cm) ": null, "BMI": null,
  "Cycle(R/I)": null, "Cycle length(days)": null, "AMH(ng/mL)": null,
  "RBS(mg/dl)": null, "BP _Systolic (mmHg)": null, "BP _Diastolic (mmHg)": null,
  "Follicle No. (L)": null, "Follicle No. (R)": null,
  "Avg. F size (L) (mm)": null, "Avg. F size (R) (mm)": null,
  "Endometrium (mm)": null, "Waist(inch)": null, "Hip(inch)": null,
  "TSH (mIU/L)": null, "LH(mIU/mL)": null, "FSH(mIU/mL)": null,
  "PRG(ng/mL)": null, "PRL(ng/mL)": null, "Vit D3 (ng/mL)": null,
  "Hb(g/dl)": null, "Reg.Exercise(Y/N)": null, "Fast food (Y/N)": null
}

Rules: Cycle(R/I)=1 for Regular/2 for Irregular. Reg.Exercise & Fast food: 1=Yes/0=No.
All values numeric or null.

Report text:
"""


def extract_with_groq(text: str, api_key: str) -> dict:
    from groq import Groq
    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": EXTRACTION_PROMPT + text}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ══════════════════════════════════════════════════════════════════════════════
# RISK SCORING & SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

def scale_patient(patient, scaler, feature_cols):
    if scaler is None:
        return patient
    df = pd.DataFrame([patient]).reindex(columns=feature_cols, fill_value=0).fillna(0)
    return dict(zip(feature_cols, scaler.transform(df)[0]))

def get_risk_scores(risk_models, patient, feature_cols, scaler=None):
    p = scale_patient(patient, scaler, feature_cols) if scaler else patient
    scores = {}
    for name, info in risk_models.items():
        df_in = pd.DataFrame([p]).reindex(columns=feature_cols, fill_value=0).fillna(0)
        scores[name] = round(info["model"].predict_proba(df_in)[0][1] * 100, 1)
    return scores

SCENARIOS = {
    "Weight loss (5 kg)": {"BMI": -2.0, "Weight (Kg)": -5.0, "RBS(mg/dl)": -10.0},
    "Regular exercise":   {"Reg.Exercise(Y/N)": 1, "BP _Systolic (mmHg)": -8.0,
                           "RBS(mg/dl)": -15.0, "Cycle(R/I)": 1},
    "Dietary changes":    {"Fast food (Y/N)": 0, "RBS(mg/dl)": -12.0,
                           "BMI": -1.0, "Weight (Kg)": -2.0},
    "Metformin":          {"RBS(mg/dl)": -20.0, "Cycle(R/I)": 1,
                           "Cycle length(days)": 30, "BMI": -1.5},
    "Combined lifestyle": {"BMI": -3.5, "Weight (Kg)": -8.0, "RBS(mg/dl)": -25.0,
                           "BP _Systolic (mmHg)": -10.0, "Cycle(R/I)": 1,
                           "Fast food (Y/N)": 0, "Reg.Exercise(Y/N)": 1,
                           "Symptom_burden": -2.0},
}
FLOORS = {"BMI": 18.5, "Weight (Kg)": 45, "RBS(mg/dl)": 70,
          "BP _Systolic (mmHg)": 90, "Symptom_burden": 0}

def apply_scenario(patient, sc):
    p = deepcopy(patient)
    for field, delta in SCENARIOS[sc].items():
        if isinstance(delta, float) and delta < 0:
            p[field] = max(FLOORS.get(field, -9999), p.get(field, 0) + delta)
        else:
            p[field] = (delta if not isinstance(delta, float)
                        else max(FLOORS.get(field, -9999), p.get(field, 0) + delta))
    b = p.get("BMI", 25)
    p["BMI_category"] = 1 if b < 25 else (2 if b < 30 else 3)
    return p

def estimate_trajectory(state_model, patient, feature_cols, scaler=None, months=12):
    scores, bmi0, rbs0 = [], patient.get("BMI", 25), patient.get("RBS(mg/dl)", 100)
    for m in range(months + 1):
        sp = deepcopy(patient)
        sp["BMI"]        = max(18.5, bmi0 - 0.15 * m)
        sp["RBS(mg/dl)"] = max(70,   rbs0 - 1.2  * m)
        b = sp["BMI"]
        sp["BMI_category"] = 1 if b < 25 else (2 if b < 30 else 3)
        if scaler:
            sp = scale_patient(sp, scaler, feature_cols)
        df_in = pd.DataFrame([sp]).reindex(columns=feature_cols, fill_value=0).fillna(0)
        try:
            score = (float(state_model.predict_proba(df_in)[0][0]) * 100
                     if hasattr(state_model, "predict_proba")
                     else float(state_model.predict(df_in)[0]))
        except Exception:
            score = max(0, 100 - (bmi0 - 18.5) * 3 - 0.15 * m * 3)
        scores.append(score)
    return list(range(months + 1)), scores


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def risk_color(s):
    return C["red"] if s >= 70 else (C["orange"] if s >= 40 else C["green"])

def risk_label(s):
    return "HIGH" if s >= 70 else ("MODERATE" if s >= 40 else "LOW")

def fmt_risk(k):
    return k.replace("_risk","").replace("_"," ").title()

def sec(title):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)

def notice(html, warn=False):
    cls = "notice warn" if warn else "notice"
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)

FIELDS = {
    "Age (yrs)":            ("Age (years)",           15,  55,  28,   1),
    "Weight (Kg)":          ("Weight (kg)",           40, 120,  68,   1),
    "Height(Cm) ":          ("Height (cm)",          140, 185, 162,   1),
    "AMH(ng/mL)":           ("AMH (ng/mL)",          0.1,20.0, 4.5, 0.1),
    "RBS(mg/dl)":           ("Blood Sugar (mg/dl)",   60, 250, 105,   1),
    "BP _Systolic (mmHg)":  ("BP Systolic (mmHg)",    80, 180, 122,   1),
    "BP _Diastolic (mmHg)": ("BP Diastolic (mmHg)",   50, 120,  80,   1),
    "Follicle No. (L)":     ("Follicles — Left",        0,  40,  12,   1),
    "Follicle No. (R)":     ("Follicles — Right",       0,  40,  13,   1),
    "Avg. F size (L) (mm)": ("Follicle Size L (mm)",    4,  35,  15,   1),
    "Avg. F size (R) (mm)": ("Follicle Size R (mm)",    4,  35,  15,   1),
    "Endometrium (mm)":     ("Endometrium (mm)",         3,  20,   8,   1),
    "LH(mIU/mL)":           ("LH (mIU/mL)",           0.1,40.0, 8.0, 0.1),
    "FSH(mIU/mL)":          ("FSH (mIU/mL)",          0.1,25.0, 6.0, 0.1),
    "TSH (mIU/L)":          ("TSH (mIU/L)",           0.1,10.0, 2.5, 0.1),
    "PRG(ng/mL)":           ("Progesterone (ng/mL)",  0.1,30.0, 1.0, 0.1),
    "PRL(ng/mL)":           ("Prolactin (ng/mL)",     1.0,50.0,15.0, 0.5),
    "Hb(g/dl)":             ("Haemoglobin (g/dl)",    6.0,18.0,12.0, 0.1),
    "Vit D3 (ng/mL)":       ("Vitamin D3 (ng/mL)",     5, 100,  25,   1),
    "Waist(inch)":          ("Waist (inches)",         20,  60,  32,   1),
    "Hip(inch)":            ("Hip (inches)",           25,  65,  38,   1),
    "Cycle length(days)":   ("Cycle length (days)",    21,  90,  35,   1),
}


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS  (light theme)
# ══════════════════════════════════════════════════════════════════════════════

CHART_STYLE = dict(
    fig_bg   = C["surface"],
    ax_bg    = C["bg"],
    text     = C["text"],
    muted    = C["muted"],
    grid     = C["grid"],
    spine    = C["border"],
)

def _style_ax(ax, fig):
    fig.patch.set_facecolor(CHART_STYLE["fig_bg"])
    ax.set_facecolor(CHART_STYLE["ax_bg"])
    ax.tick_params(colors=CHART_STYLE["muted"], labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(CHART_STYLE["spine"])
        sp.set_linewidth(0.6)
    ax.grid(axis="y", color=CHART_STYLE["grid"], linewidth=0.6, alpha=0.8)

def make_risk_chart(all_results):
    risks     = list(list(all_results.values())[0].keys())
    scenarios = list(all_results.keys())
    x         = np.arange(len(risks))
    width     = 0.72 / len(scenarios)
    palette   = [C["teal"], "#2980b9", "#27ae60", "#d35400", "#8e44ad", C["red"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    _style_ax(ax, fig)

    for i, (sc, scores) in enumerate(all_results.items()):
        vals   = [scores[r] for r in risks]
        offset = (i - (len(scenarios)-1)/2) * width
        bars   = ax.bar(x + offset, vals, width, label=sc,
                        color=palette[i % len(palette)],
                        alpha=0.85, edgecolor=C["surface"], linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 7:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.8, f"{v:.0f}",
                        ha="center", va="bottom",
                        fontsize=7, color=CHART_STYLE["text"], alpha=0.75)

    ax.axhline(70, color=C["red"],    linestyle="--", lw=1, alpha=0.5, label="High risk (70%)")
    ax.axhline(40, color=C["orange"], linestyle="--", lw=1, alpha=0.4, label="Moderate (40%)")
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_risk(r) for r in risks], rotation=16,
                       ha="right", fontsize=9, color=CHART_STYLE["muted"])
    ax.set_ylabel("Risk probability (%)", color=CHART_STYLE["muted"], fontsize=9)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, facecolor=C["surface"], edgecolor=C["border"],
              labelcolor=CHART_STYLE["text"], loc="upper right", ncol=2,
              framealpha=1)
    fig.tight_layout(pad=1.5)
    return fig

def make_trajectory_chart(months_list, scores):
    fig, ax = plt.subplots(figsize=(10, 4))
    _style_ax(ax, fig)

    ax.fill_between(months_list, scores, alpha=0.10, color=C["teal"])
    ax.plot(months_list, scores, color=C["teal"], lw=2.5,
            marker="o", markersize=5, markerfacecolor=C["surface"],
            markeredgecolor=C["teal"], markeredgewidth=1.5)

    ax.axhspan(70, 100, alpha=0.04, color=C["green"])
    ax.axhspan(40,  70, alpha=0.04, color=C["orange"])
    ax.axhspan(0,   40, alpha=0.04, color=C["red"])

    n     = len(months_list) - 1
    delta = scores[-1] - scores[0]
    sign  = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"

    ax.annotate(f"M0: {scores[0]:.0f}%",
                xy=(0, scores[0]), xytext=(0.6, scores[0] + 6),
                color=CHART_STYLE["text"], fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=C["muted"], lw=0.8))
    ax.annotate(f"M{n}: {scores[-1]:.0f}% ({sign})",
                xy=(n, scores[-1]), xytext=(n - 3, scores[-1] + 6),
                color=C["teal"], fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["teal"], lw=0.8))

    step = max(1, n // 4)
    ax.set_xticks(range(0, n + 1, step))
    ax.set_xticklabels([f"M{m}" for m in range(0, n + 1, step)],
                       color=CHART_STYLE["muted"], fontsize=9)
    ax.set_xlim(-0.3, n + 2)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Month", color=CHART_STYLE["muted"], fontsize=9)
    ax.set_ylabel("Health score (%)", color=CHART_STYLE["muted"], fontsize=9)
    ax.grid(color=C["grid"], linewidth=0.6, alpha=0.8)
    fig.tight_layout(pad=1.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CONFIRMATION FORM
# ══════════════════════════════════════════════════════════════════════════════

def show_confirmation_form(extracted: dict) -> dict:
    sec("Review & confirm extracted values")
    notice("🟢 = found in your report &nbsp;&nbsp;|&nbsp;&nbsp; "
           "🟠 = not found — default shown, adjust if known.")

    confirmed = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        raw = extracted.get("Cycle(R/I)")
        tag = "🟢" if raw is not None else "🟠"
        default = "Irregular" if raw == 2 else ("Regular" if raw == 1 else "Irregular")
        sel = st.selectbox(f"{tag} Menstrual cycle", ["Irregular","Regular"],
                           index=0 if default == "Irregular" else 1)
        confirmed["Cycle(R/I)"] = 2 if sel == "Irregular" else 1
    with c2:
        raw = extracted.get("Reg.Exercise(Y/N)")
        tag = "🟢" if raw is not None else "🟠"
        sel = st.selectbox(f"{tag} Regular exercise", ["No","Yes"],
                           index=1 if raw == 1 else 0)
        confirmed["Reg.Exercise(Y/N)"] = 1 if sel == "Yes" else 0
    with c3:
        raw = extracted.get("Fast food (Y/N)")
        tag = "🟢" if raw is not None else "🟠"
        sel = st.selectbox(f"{tag} Fast food intake", ["Yes","No"],
                           index=0 if raw == 1 else 1)
        confirmed["Fast food (Y/N)"] = 1 if sel == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    keys = list(FIELDS.keys())
    for i in range(0, len(keys), 3):
        cols = st.columns(3)
        for col, key in zip(cols, keys[i:i+3]):
            label, mn, mx, default, step = FIELDS[key]
            ext_val = extracted.get(key)
            found   = ext_val is not None
            tag     = "🟢" if found else "🟠"
            val     = max(float(mn), min(float(mx),
                          float(ext_val) if found else float(default)))
            with col:
                if isinstance(step, float):
                    confirmed[key] = st.slider(f"{tag} {label}",
                                               float(mn), float(mx), val, step)
                else:
                    confirmed[key] = st.slider(f"{tag} {label}",
                                               int(mn), int(mx), int(val), int(step))

    # Derived fields
    h = confirmed.get("Height(Cm) ", 162)
    w = confirmed.get("Weight (Kg)", 68)
    confirmed["BMI"]            = round(w / (h/100)**2, 1) if h > 0 else float(extracted.get("BMI") or 25)
    waist = confirmed.get("Waist(inch)", 32)
    hip   = confirmed.get("Hip(inch)",   38)
    confirmed["Waist:Hip Ratio"] = round(waist/hip, 3) if hip else 0
    lh    = confirmed.get("LH(mIU/mL)",  8)
    fsh   = confirmed.get("FSH(mIU/mL)", 6)
    confirmed["LH/FSH ratio"]   = round(lh/fsh, 3) if fsh else 0
    b = confirmed.get("BMI", 25)
    confirmed["BMI_category"]   = 1 if b < 25 else (2 if b < 30 else 3)

    all_keys = list(FIELDS.keys()) + ["Cycle(R/I)", "Reg.Exercise(Y/N)", "Fast food (Y/N)"]
    found_n  = sum(1 for k in all_keys if extracted.get(k) is not None)
    total_n  = len(all_keys)
    notice(
        f"📄 <strong>{found_n}/{total_n}</strong> fields extracted "
        f"({found_n/total_n*100:.0f}% coverage) &nbsp;|&nbsp; "
        f"BMI = <strong>{confirmed['BMI']:.1f}</strong> &nbsp;|&nbsp; "
        f"W:H = <strong>{confirmed['Waist:Hip Ratio']:.3f}</strong> &nbsp;|&nbsp; "
        f"LH/FSH = <strong>{confirmed['LH/FSH ratio']:.2f}</strong>"
    )
    return confirmed


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABS
# ══════════════════════════════════════════════════════════════════════════════

def show_results(state_model, risk_models, feature_cols, scaler, patient):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊  Risk Dashboard",
        "🔮  What-If Simulator",
        "📈  Health Trajectory",
        "💊  Recommendations",
        "🧠  Why This Score?",
    ])
    baseline = get_risk_scores(risk_models, patient, feature_cols, scaler)

    # ── Tab 1: Risk Dashboard ─────────────────────────────────────────────────
    with tab1:
        sec("Baseline risk scores")
        cols = st.columns(len(baseline))
        for col, (rk, score) in zip(cols, baseline.items()):
            color = risk_color(score)
            with col:
                st.markdown(f"""
                <div class="risk-card" style="--rc:{color}">
                  <div class="risk-lbl">{fmt_risk(rk)}</div>
                  <div class="risk-val" style="color:{color}">{score:.1f}%</div>
                  <div class="risk-tier" style="color:{color}">{risk_label(score)}</div>
                  <div class="bar-bg">
                    <div class="bar-fg" style="width:{int(score)}%;background:{color}"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        avg = np.mean(list(baseline.values()))
        sev_icon  = "🔴" if avg >= 70 else ("🟠" if avg >= 40 else "🟢")
        sev_label = "High concern" if avg >= 70 else ("Moderate concern" if avg >= 40 else "Low concern")
        notice(f"Overall average risk: <strong>{avg:.1f}%</strong> — "
               f"{sev_icon} {sev_label}")

        sec("Patient summary")
        bmi_v = patient.get("BMI", 0)
        bmi_cat = 'Normal' if bmi_v < 25 else ('Overweight' if bmi_v < 30 else 'Obese')
        summ = {
            "BMI":           f"{bmi_v:.1f}  ({bmi_cat})",
            "Age":           f"{patient.get('Age (yrs)',0):.0f} yrs",
            "Weight":        f"{patient.get('Weight (Kg)',0):.0f} kg",
            "Cycle":         "Irregular" if patient.get("Cycle(R/I)") == 2 else "Regular",
            "AMH":           f"{patient.get('AMH(ng/mL)',0):.2f} ng/mL",
            "Blood Sugar":   f"{patient.get('RBS(mg/dl)',0):.0f} mg/dl",
            "BP":            f"{patient.get('BP _Systolic (mmHg)',0):.0f} / "
                             f"{patient.get('BP _Diastolic (mmHg)',0):.0f} mmHg",
            "LH / FSH":      f"{patient.get('LH/FSH ratio',0):.2f}",
            "Follicles L/R": f"{patient.get('Follicle No. (L)',0):.0f} / "
                             f"{patient.get('Follicle No. (R)',0):.0f}",
            "Haemoglobin":   f"{patient.get('Hb(g/dl)',0):.1f} g/dl",
            "Vitamin D3":    f"{patient.get('Vit D3 (ng/mL)',0):.1f} ng/mL",
            "Waist : Hip":   f"{patient.get('Waist:Hip Ratio',0):.3f}",
            "Exercise":      "Yes" if patient.get("Reg.Exercise(Y/N)") else "No",
            "Fast food":     "Yes" if patient.get("Fast food (Y/N)") else "No",
        }
        st.dataframe(pd.DataFrame(summ.items(), columns=["Parameter", "Value"]),
                     use_container_width=True, hide_index=True)

    # ── Tab 2: What-If Simulator ──────────────────────────────────────────────
    with tab2:
        sec("Select intervention scenarios")
        selected = st.multiselect("Scenarios:", list(SCENARIOS.keys()),
                                  default=list(SCENARIOS.keys()))
        if not selected:
            st.info("Select at least one scenario to simulate.")
        else:
            all_res = {"Baseline": baseline}
            for sc in selected:
                all_res[sc] = get_risk_scores(risk_models,
                                              apply_scenario(patient, sc),
                                              feature_cols, scaler)
            st.pyplot(make_risk_chart(all_res), use_container_width=True)

            sec("Score changes vs baseline")
            risks = list(baseline.keys())
            rows  = [{"Scenario": sc,
                      **{fmt_risk(r): f"{all_res[sc][r]-baseline[r]:+.1f}%" for r in risks},
                      "Avg Δ": f"{np.mean([all_res[sc][r]-baseline[r] for r in risks]):+.1f}%"}
                     for sc in selected]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            best = min(selected, key=lambda s: np.mean(list(all_res[s].values())))
            ba   = np.mean(list(baseline.values()))
            bst  = np.mean(list(all_res[best].values()))
            notice(f"Best scenario: <strong>{best}</strong> — "
                   f"average risk {ba:.1f}% → <strong>{bst:.1f}%</strong> "
                   f"(↓ {ba-bst:.1f} percentage points)")

    # ── Tab 3: Health Trajectory ──────────────────────────────────────────────
    with tab3:
        sec("12-month health state trajectory")
        st.caption("Projection assumes gradual BMI (−0.15/month) and "
                   "blood sugar (−1.2 mg/dl/month) improvement.")
        horizon = st.slider("Projection horizon (months)", 3, 24, 12)
        months_list, traj = estimate_trajectory(state_model, patient,
                                                feature_cols, scaler, horizon)
        st.pyplot(make_trajectory_chart(months_list, traj), use_container_width=True)

        sec("Score at key checkpoints")
        step_  = max(1, horizon // 4)
        df_t   = pd.DataFrame(
            [(m, round(traj[m], 1),
              "🟢 Good" if traj[m] >= 70
              else ("🟠 Moderate" if traj[m] >= 40 else "🔴 High risk"))
             for m in range(0, horizon + 1, step_)],
            columns=["Month", "Health Score (%)", "Status"])
        st.dataframe(df_t, use_container_width=True, hide_index=True)

    # ── Tab 4: Recommendations ────────────────────────────────────────────────
    with tab4:
        sec("Personalised recommendations")
        sc_res = {sc: get_risk_scores(risk_models, apply_scenario(patient, sc),
                                      feature_cols, scaler)
                  for sc in SCENARIOS}
        for rk, base_score in baseline.items():
            label_ = fmt_risk(rk)
            color  = risk_color(base_score)
            best   = min(sc_res, key=lambda s: sc_res[s][rk])
            b_sc   = sc_res[best][rk]
            red    = base_score - b_sc
            with st.expander(f"{label_} Risk — {base_score:.1f}%  ({risk_label(base_score)})",
                             expanded=(base_score >= 60)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div style="text-align:center;padding:14px 8px">
                      <div style="font-size:2.4rem;font-weight:600;
                                  color:{color};line-height:1">{base_score:.1f}%</div>
                      <div style="font-size:0.7rem;font-weight:600;
                                  text-transform:uppercase;letter-spacing:0.8px;
                                  color:{color};margin-top:4px">{risk_label(base_score)}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if red > 0.5:
                        st.success(f"Best: **{best}** → {b_sc:.1f}%  (↓ {red:.1f} pp)")
                    else:
                        st.success("Already well managed.")
                    for sc, res in sorted(sc_res.items(), key=lambda kv: kv[1][rk]):
                        d   = base_score - res[rk]
                        clr = C["green"] if d > 0 else (C["red"] if d < 0 else C["muted"])
                        tag = f"↓{d:.1f}%" if d > 0 else (f"↑{abs(d):.1f}%" if d < 0 else "—")
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:5px 0;border-bottom:1px solid {C["border"]};'
                            f'font-size:0.85rem;">'
                            f'<span style="color:{C["text"]}">{sc}</span>'
                            f'<span style="color:{clr};font-weight:600">'
                            f'{res[rk]:.1f}%&nbsp;&nbsp;{tag}</span></div>',
                            unsafe_allow_html=True)

        notice("""<strong>General PCOS guidelines</strong><br><br>
          • 5–7% body weight reduction can restore menstrual regularity.<br>
          • 150 min/week moderate aerobic exercise (WHO recommendation).<br>
          • Metformin most effective when combined with dietary change.<br>
          • Low-GI diet reduces insulin resistance and androgen levels.<br>
          • Reassess clinically every 3–6 months.<br><br>
          <em style="font-size:0.8rem;opacity:0.8">Model-based estimates —
          not a medical diagnosis. Always consult a clinician.</em>""")

    # ── Tab 5: Explainability ─────────────────────────────────────────────────
    with tab5:
        render_explainability_tab(risk_models, patient, feature_cols, scaler)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Page header ───────────────────────────────────────────────────────────
    col_h, col_badge = st.columns([4, 1])
    with col_h:
        st.title("🩺 PCOS Digital Twin")
        st.caption("Upload clinical lab reports · AI extracts patient data · "
                   "Risk analysis · What-if simulation · Health trajectory")
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:right;padding-top:6px">
          <span style="background:{C['teal_lite']};color:{C['teal']};
                       border:1px solid #9fe1cb;border-radius:20px;
                       padding:4px 12px;font-size:0.75rem;font-weight:600">
            NIT Srinagar
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Load models ───────────────────────────────────────────────────────────
    state_model, risk_models, scaler, err = load_models()
    feature_cols = load_feature_cols()
    if err:
        st.error(f"⚠️ Models not found: {err}")
        st.info("Run: data_pipeline.py → state_estimator.py → risk_predictor.py")
        st.stop()

    # ── Step 1: API Key ───────────────────────────────────────────────────────
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown(
        '<span class="step-num">1</span>'
        '<span class="step-title">Groq API Key</span>',
        unsafe_allow_html=True)

    if ENV_API_KEY:
        # Key found in .env — show a masked confirmation, no input needed
        st.markdown(
            f'<div class="step-sub">✅ API key loaded from <code>.env</code> file — '
            f'<code>gsk_...{ENV_API_KEY[-6:]}</code></div>',
            unsafe_allow_html=True)
        api_key = ENV_API_KEY
        # Allow override in expander
        with st.expander("Use a different key"):
            override = st.text_input("Override API key", type="password",
                                     placeholder="gsk_...",
                                     label_visibility="collapsed")
            if override.strip():
                api_key = override.strip()
    else:
        st.markdown(
            '<div class="step-sub">Enter your free Groq key below</div>',
            unsafe_allow_html=True)
        api_key = st.text_input("Groq API key", type="password",
                                placeholder="gsk_...",
                                label_visibility="collapsed")
        st.markdown(
            '<div class="api-hint">Free key at '
            '<a href="https://console.groq.com" target="_blank">console.groq.com</a> '
            '— or add <code>GROQ_API_KEY=gsk_...</code> to a <code>.env</code> '
            'file in the project root to skip this step permanently.</div>',
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if not api_key:
        st.info("👆 Enter your Groq API key to continue.")
        st.stop()

    # ── Step 2: Upload three reports ──────────────────────────────────────────
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown(
        '<span class="step-num">2</span>'
        '<span class="step-title">Upload Lab Reports</span>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Upload all three for best extraction accuracy. '
        'Digital (typed) PDFs only — not scanned images.</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<span class="up-label">🩸 Blood Test Report</span>', unsafe_allow_html=True)
        st.markdown('<span class="up-sub">CBC · blood sugar · lipid panel · vitals</span>',
                    unsafe_allow_html=True)
        blood_file = st.file_uploader("blood", type=["pdf"], key="blood",
                                      label_visibility="collapsed")
        if blood_file:
            st.success(f"✅  {blood_file.name}")

    with col2:
        st.markdown('<span class="up-label">🧪 Hormone Panel Report</span>', unsafe_allow_html=True)
        st.markdown('<span class="up-sub">LH · FSH · AMH · TSH · prolactin · Vit D3</span>',
                    unsafe_allow_html=True)
        hormone_file = st.file_uploader("hormone", type=["pdf"], key="hormone",
                                        label_visibility="collapsed")
        if hormone_file:
            st.success(f"✅  {hormone_file.name}")

    with col3:
        st.markdown('<span class="up-label">🔬 Ultrasound Report</span>', unsafe_allow_html=True)
        st.markdown('<span class="up-sub">Follicle count · sizes · endometrium</span>',
                    unsafe_allow_html=True)
        ultrasound_file = st.file_uploader("ultrasound", type=["pdf"], key="ultrasound",
                                           label_visibility="collapsed")
        if ultrasound_file:
            st.success(f"✅  {ultrasound_file.name}")

    st.markdown('</div>', unsafe_allow_html=True)

    uploaded = {
        "Blood Test":    blood_file,
        "Hormone Panel": hormone_file,
        "Ultrasound":    ultrasound_file,
    }
    if not any(v is not None for v in uploaded.values()):
        st.info("👆 Upload at least one PDF report to continue.")
        st.stop()

    # ── Step 3: Extract ───────────────────────────────────────────────────────
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown(
        '<span class="step-num">3</span>'
        '<span class="step-title">Extract Data from Reports</span>',
        unsafe_allow_html=True)
    ready = [name for name, f in uploaded.items() if f is not None]
    st.markdown(
        f'<div class="step-sub">Ready: {", ".join(ready)}</div>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    extract_btn = st.button("🔍  Extract values using AI",
                            use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if extract_btn:
        combined = ""
        with st.spinner("Reading PDF reports..."):
            for report_name, f in uploaded.items():
                if f is not None:
                    text = extract_pdf_text(f)
                    if text.strip():
                        combined += f"\n\n=== {report_name}: {f.name} ===\n\n{text}"

        if not combined.strip():
            st.error("No text found. Ensure PDFs are digital (not scanned images).")
            st.stop()

        with st.spinner("AI is reading your reports..."):
            try:
                extracted = extract_with_groq(combined, api_key)
                st.session_state["extracted"] = extracted
                found = sum(v is not None for v in extracted.values())
                st.success(
                    f"✅  Done — **{found}/{len(extracted)}** fields extracted "
                    f"from {len(ready)} report(s)."
                )
            except json.JSONDecodeError:
                st.error("Unexpected response format. Check your API key and try again.")
                st.stop()
            except Exception as e:
                st.error(f"Extraction error: {e}")
                st.stop()

    if "extracted" not in st.session_state:
        st.stop()

    # ── Step 4: Confirm values ────────────────────────────────────────────────
    st.markdown("---")
    confirmed = show_confirmation_form(st.session_state["extracted"])

    # ── Step 5: Run Analysis ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown(
        '<span class="step-num">4</span>'
        '<span class="step-title">Run Risk Analysis</span>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="step-sub">Confirm the values above, then run the full analysis.</div>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  Run Analysis on confirmed values", use_container_width=True):
        st.session_state["patient"] = confirmed
    st.markdown('</div>', unsafe_allow_html=True)

    if "patient" not in st.session_state:
        st.info("👆 Click **Run Analysis** after confirming the values above.")
        st.stop()

    st.markdown("---")
    show_results(state_model, risk_models, feature_cols, scaler,
                 st.session_state["patient"])

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;padding:8px 0;font-size:0.78rem;color:{C['hint']}">
      PCOS Digital Twin &nbsp;·&nbsp; NIT Srinagar &nbsp;·&nbsp;
      Dept. of Computer Science &amp; Engineering
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()