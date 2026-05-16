"""
PCOS Digital Twin — PDF Upload Dashboard
src/dashboard.py

Flow:
  Step 1 → Enter Anthropic API key
  Step 2 → Upload PDF lab reports
  Step 3 → Extract values via Claude API
  Step 4 → Review / edit extracted values
  Step 5 → Run Analysis → 4 result tabs
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import streamlit as st
import joblib
import pdfplumber
import anthropic

warnings.filterwarnings("ignore")

# ── Paths (relative to src/) ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
DATA_DIR   = os.path.join(BASE_DIR, "..", "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Digital Twin",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── global ── */
  [data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
  [data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
  h1,h2,h3,h4                        { color: #e6edf3 !important; }

  /* ── metric cards ── */
  .metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; margin-bottom: 10px;
  }
  .metric-card .label { font-size:12px; color:#8b949e; margin-bottom:4px; }
  .metric-card .value { font-size:22px; font-weight:700; color:#58a6ff; }

  /* ── status badges ── */
  .badge-found    { background:#1a3a2a; color:#3fb950; border:1px solid #238636;
                    border-radius:4px; padding:2px 8px; font-size:11px; }
  .badge-missing  { background:#3a2a1a; color:#d29922; border:1px solid #9e6a03;
                    border-radius:4px; padding:2px 8px; font-size:11px; }

  /* ── step header ── */
  .step-header {
    background: linear-gradient(135deg,#1f2937,#111827);
    border-left: 4px solid #58a6ff; border-radius:8px;
    padding:12px 16px; margin-bottom:16px;
    font-size:15px; font-weight:600; color:#e6edf3;
  }

  /* ── risk bar ── */
  .risk-bar-container { background:#21262d; border-radius:6px; height:12px; overflow:hidden; }
  .risk-bar-fill      { height:12px; border-radius:6px; transition:width .4s ease; }

  /* ── stButton override ── */
  div.stButton > button {
    background: #238636; color:#fff; border:none; border-radius:6px;
    padding:8px 20px; font-weight:600; cursor:pointer;
  }
  div.stButton > button:hover { background:#2ea043; }

  /* ── tabs ── */
  [data-baseweb="tab-list"] { gap:4px; }
  [data-baseweb="tab"]      { background:#161b22 !important; border-radius:6px 6px 0 0;
                               color:#8b949e !important; border:1px solid #30363d !important; }
  [aria-selected="true"]    { background:#1f6feb !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helper: load models + feature columns
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    rp = joblib.load(os.path.join(MODEL_DIR, "risk_predictor.pkl"))
    se = joblib.load(os.path.join(MODEL_DIR, "state_estimator.pkl"))
    return rp, se

@st.cache_data
def load_feature_columns():
    df = pd.read_csv(os.path.join(DATA_DIR, "pcos_processed.csv"))
    drop = ["PCOS (Y/N)"]
    num_cols = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore") \
                 .select_dtypes(include=[np.number]).columns.tolist()
    return num_cols

# ── Column name aliases: dashboard label → possible CSV column names ──────────
FIELD_ALIASES: dict[str, list[str]] = {
    "Age (yrs)":               ["Age (yrs)", "Age"],
    "Weight (Kg)":             ["Weight (Kg)", "Weight"],
    "Height(Cm)":              ["Height(Cm)", "Height"],
    "BMI":                     ["BMI"],
    "Blood Group":             ["Blood Group"],
    "Pulse rate(bpm)":         ["Pulse rate(bpm)", "Pulse rate"],
    "RR (breaths/min)":        ["RR (breaths/min)", "RR"],
    "Hb(g/dl)":                ["Hb(g/dl)", "Hb"],
    "Cycle(R/I)":              ["Cycle(R/I)", "Cycle"],
    "Cycle length(days)":      ["Cycle length(days)"],
    "Marraige Status (Yrs)":   ["Marraige Status (Yrs)", "Marriage Status"],
    "Pregnant(Y/N)":           ["Pregnant(Y/N)", "Pregnant"],
    "No. of aborptions":       ["No. of aborptions", "No. of abortions"],
    "  I   beta-HCG(mIU/mL)": ["  I   beta-HCG(mIU/mL)", "beta-HCG I"],
    "II    beta-HCG(mIU/mL)":  ["II    beta-HCG(mIU/mL)", "beta-HCG II"],
    "FSH(mIU/mL)":             ["FSH(mIU/mL)", "FSH"],
    "LH(mIU/mL)":              ["LH(mIU/mL)", "LH"],
    "FSH/LH":                  ["FSH/LH"],
    "Hip(inch)":               ["Hip(inch)", "Hip"],
    "Waist(inch)":             ["Waist(inch)", "Waist"],
    "Waist:Hip Ratio":         ["Waist:Hip Ratio", "WHR"],
    "TSH (mIU/L)":             ["TSH (mIU/L)", "TSH"],
    "AMH(ng/mL)":              ["AMH(ng/mL)", "AMH"],
    "PRL(ng/mL)":              ["PRL(ng/mL)", "Prolactin", "PRL"],
    "Vit D3 (ng/mL)":          ["Vit D3 (ng/mL)", "Vit D3"],
    "PRG(ng/mL)":              ["PRG(ng/mL)", "Progesterone", "PRG"],
    "RBS(mg/dl)":              ["RBS(mg/dl)", "RBS"],
    "Weight gain(Y/N)":        ["Weight gain(Y/N)"],
    "hair growth(Y/N)":        ["hair growth(Y/N)"],
    "Skin darkening (Y/N)":    ["Skin darkening (Y/N)"],
    "Hair loss(Y/N)":          ["Hair loss(Y/N)"],
    "Pimples(Y/N)":            ["Pimples(Y/N)"],
    "Fast food (Y/N)":         ["Fast food (Y/N)"],
    "Reg.Exercise(Y/N)":       ["Reg.Exercise(Y/N)"],
    "BP _Systolic (mmHg)":     ["BP _Systolic (mmHg)", "BP Systolic"],
    "BP _Diastolic (mmHg)":    ["BP _Diastolic (mmHg)", "BP Diastolic"],
    "Follicle No. (L)":        ["Follicle No. (L)", "Follicles L"],
    "Follicle No. (R)":        ["Follicle No. (R)", "Follicles R"],
    "Avg. F size (L) (mm)":    ["Avg. F size (L) (mm)", "Avg F size L"],
    "Avg. F size (R) (mm)":    ["Avg. F size (R) (mm)", "Avg F size R"],
    "Endometrium (mm)":        ["Endometrium (mm)", "Endometrium"],
}

# Default / typical values used as fallback
DEFAULTS: dict[str, float] = {
    "Age (yrs)": 27, "Weight (Kg)": 65, "Height(Cm)": 160, "BMI": 25.0,
    "Blood Group": 2, "Pulse rate(bpm)": 76, "RR (breaths/min)": 16,
    "Hb(g/dl)": 12.5, "Cycle(R/I)": 2, "Cycle length(days)": 28,
    "Marraige Status (Yrs)": 0, "Pregnant(Y/N)": 0,
    "No. of aborptions": 0,
    "  I   beta-HCG(mIU/mL)": 0.5, "II    beta-HCG(mIU/mL)": 0.5,
    "FSH(mIU/mL)": 5.0, "LH(mIU/mL)": 5.0, "FSH/LH": 1.0,
    "Hip(inch)": 38, "Waist(inch)": 30, "Waist:Hip Ratio": 0.79,
    "TSH (mIU/L)": 2.5, "AMH(ng/mL)": 3.0, "PRL(ng/mL)": 15.0,
    "Vit D3 (ng/mL)": 30.0, "PRG(ng/mL)": 1.0, "RBS(mg/dl)": 90.0,
    "Weight gain(Y/N)": 0, "hair growth(Y/N)": 0,
    "Skin darkening (Y/N)": 0, "Hair loss(Y/N)": 0, "Pimples(Y/N)": 0,
    "Fast food (Y/N)": 0, "Reg.Exercise(Y/N)": 0,
    "BP _Systolic (mmHg)": 120, "BP _Diastolic (mmHg)": 80,
    "Follicle No. (L)": 6, "Follicle No. (R)": 6,
    "Avg. F size (L) (mm)": 6.0, "Avg. F size (R) (mm)": 6.0,
    "Endometrium (mm)": 7.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# PDF text extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_pdf_text(uploaded_file) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)

# ══════════════════════════════════════════════════════════════════════════════
# Claude extraction
# ══════════════════════════════════════════════════════════════════════════════
EXTRACTION_SYSTEM = """You are a medical data extraction assistant specialising in PCOS lab reports.
Extract ONLY the values explicitly present in the report text.
Return a single JSON object with exactly these keys (use null for missing values):

age, weight_kg, height_cm, bmi, pulse_rate, rr, hb, cycle, cycle_length,
marriage_status_yrs, pregnant, abortions, beta_hcg_i, beta_hcg_ii,
fsh, lh, fsh_lh_ratio, hip_inch, waist_inch, whr, tsh, amh, prolactin,
vit_d3, progesterone, rbs, weight_gain, hair_growth, skin_darkening,
hair_loss, pimples, fast_food, exercise,
bp_systolic, bp_diastolic,
follicles_l, follicles_r, avg_f_size_l, avg_f_size_r, endometrium,
blood_group

Rules:
- cycle: 2 = irregular, 4 = regular (integer)
- pregnant, weight_gain, hair_growth, skin_darkening, hair_loss,
  pimples, fast_food, exercise: 0 or 1 (integer)
- blood_group: encode as integer (O+=1,A+=2,B+=3,AB+=4,O-=5,A-=6,B-=7,AB-=8)
- All numeric values as floats/ints; null if not found.
- Return ONLY the JSON object — no markdown, no explanation."""

def claude_extract(api_key: str, combined_text: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=EXTRACTION_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Extract PCOS values from these lab reports:\n\n{combined_text}"}],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Map Claude keys → dashboard field labels ──────────────────────────────────
CLAUDE_TO_FIELD: dict[str, str] = {
    "age":             "Age (yrs)",
    "weight_kg":       "Weight (Kg)",
    "height_cm":       "Height(Cm)",
    "bmi":             "BMI",
    "blood_group":     "Blood Group",
    "pulse_rate":      "Pulse rate(bpm)",
    "rr":              "RR (breaths/min)",
    "hb":              "Hb(g/dl)",
    "cycle":           "Cycle(R/I)",
    "cycle_length":    "Cycle length(days)",
    "marriage_status_yrs": "Marraige Status (Yrs)",
    "pregnant":        "Pregnant(Y/N)",
    "abortions":       "No. of aborptions",
    "beta_hcg_i":      "  I   beta-HCG(mIU/mL)",
    "beta_hcg_ii":     "II    beta-HCG(mIU/mL)",
    "fsh":             "FSH(mIU/mL)",
    "lh":              "LH(mIU/mL)",
    "fsh_lh_ratio":    "FSH/LH",
    "hip_inch":        "Hip(inch)",
    "waist_inch":      "Waist(inch)",
    "whr":             "Waist:Hip Ratio",
    "tsh":             "TSH (mIU/L)",
    "amh":             "AMH(ng/mL)",
    "prolactin":       "PRL(ng/mL)",
    "vit_d3":          "Vit D3 (ng/mL)",
    "progesterone":    "PRG(ng/mL)",
    "rbs":             "RBS(mg/dl)",
    "weight_gain":     "Weight gain(Y/N)",
    "hair_growth":     "hair growth(Y/N)",
    "skin_darkening":  "Skin darkening (Y/N)",
    "hair_loss":       "Hair loss(Y/N)",
    "pimples":         "Pimples(Y/N)",
    "fast_food":       "Fast food (Y/N)",
    "exercise":        "Reg.Exercise(Y/N)",
    "bp_systolic":     "BP _Systolic (mmHg)",
    "bp_diastolic":    "BP _Diastolic (mmHg)",
    "follicles_l":     "Follicle No. (L)",
    "follicles_r":     "Follicle No. (R)",
    "avg_f_size_l":    "Avg. F size (L) (mm)",
    "avg_f_size_r":    "Avg. F size (R) (mm)",
    "endometrium":     "Endometrium (mm)",
}

def claude_dict_to_fields(raw: dict) -> dict[str, tuple[float, bool]]:
    """Returns {field_label: (value, found_in_pdf)}"""
    result: dict[str, tuple[float, bool]] = {}
    for ckey, flabel in CLAUDE_TO_FIELD.items():
        val = raw.get(ckey)
        if val is not None:
            result[flabel] = (float(val), True)
        else:
            result[flabel] = (DEFAULTS.get(flabel, 0.0), False)
    # Derived: FSH/LH if both present
    if "FSH(mIU/mL)" in result and "LH(mIU/mL)" in result:
        fsh, fsh_found = result["FSH(mIU/mL)"]
        lh, lh_found  = result["LH(mIU/mL)"]
        if lh != 0:
            result["FSH/LH"] = (round(fsh / lh, 3), fsh_found and lh_found)
    # Derived: WHR
    if "Waist(inch)" in result and "Hip(inch)" in result:
        w, wf = result["Waist(inch)"]
        h, hf = result["Hip(inch)"]
        if h != 0:
            result["Waist:Hip Ratio"] = (round(w / h, 3), wf and hf)
    # Derived: BMI from weight + height if not found
    bmi_val, bmi_found = result.get("BMI", (0.0, False))
    if not bmi_found:
        wkg, wf = result.get("Weight (Kg)", (0.0, False))
        hcm, hf = result.get("Height(Cm)", (0.0, False))
        if wkg > 0 and hcm > 0:
            result["BMI"] = (round(wkg / ((hcm / 100) ** 2), 1), wf and hf)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# Build input row for models
# ══════════════════════════════════════════════════════════════════════════════
def build_input_row(fields: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    row = {}
    for col in feature_columns:
        # Try exact match first
        if col in fields:
            row[col] = fields[col]
            continue
        # Try aliases
        found = False
        for flabel, aliases in FIELD_ALIASES.items():
            if col in aliases and flabel in fields:
                row[col] = fields[flabel]
                found = True
                break
        if not found:
            row[col] = DEFAULTS.get(col, 0.0)
    return pd.DataFrame([row])[feature_columns]

# ══════════════════════════════════════════════════════════════════════════════
# Risk scoring
# ══════════════════════════════════════════════════════════════════════════════
def compute_risks(risk_predictor, input_df: pd.DataFrame) -> dict[str, float]:
    risks = {}
    for risk_name, info in risk_predictor.items():
        model = info["model"]
        try:
            score = model.predict_proba(input_df)[0][1] * 100
        except Exception:
            score = 50.0
        risks[risk_name] = round(score, 1)
    return risks

RISK_COLORS = {
    "low":    ("#238636", "#3fb950"),
    "medium": ("#9e6a03", "#d29922"),
    "high":   ("#b91c1c", "#ef4444"),
}

def risk_level(score: float) -> str:
    if score < 35:  return "low"
    if score < 65:  return "medium"
    return "high"

# ══════════════════════════════════════════════════════════════════════════════
# What-If Scenarios
# ══════════════════════════════════════════════════════════════════════════════
SCENARIOS = {
    "Weight Loss (-5 kg)": {
        "Weight (Kg)": -5, "BMI": -1.8, "Waist(inch)": -1.5,
        "Waist:Hip Ratio": -0.02,
    },
    "Regular Exercise": {
        "Reg.Exercise(Y/N)": 1, "RBS(mg/dl)": -8, "BP _Systolic (mmHg)": -3,
    },
    "Dietary Changes": {
        "Fast food (Y/N)": 0, "RBS(mg/dl)": -10, "Weight (Kg)": -2,
    },
    "Metformin": {
        "RBS(mg/dl)": -15, "LH(mIU/mL)": -2, "FSH/LH": 0.3,
    },
    "Combined Lifestyle": {
        "Weight (Kg)": -7, "BMI": -2.5, "Reg.Exercise(Y/N)": 1,
        "Fast food (Y/N)": 0, "RBS(mg/dl)": -18, "Waist(inch)": -2,
        "Waist:Hip Ratio": -0.03,
    },
}

def apply_scenario(base_fields: dict, delta: dict) -> dict:
    new = base_fields.copy()
    for k, d in delta.items():
        if k in new:
            new[k] = max(0.0, new[k] + d)
    # Recompute WHR
    if "Waist(inch)" in new and "Hip(inch)" in new and new["Hip(inch)"] > 0:
        new["Waist:Hip Ratio"] = round(new["Waist(inch)"] / new["Hip(inch)"], 3)
    return new

# ══════════════════════════════════════════════════════════════════════════════
# Matplotlib figures
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_CLR = "#21262d"
TXT_CLR  = "#e6edf3"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
AMBER    = "#d29922"
RED      = "#ef4444"

def _style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TXT_CLR, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.yaxis.label.set_color(TXT_CLR)
    ax.xaxis.label.set_color(TXT_CLR)
    if title:
        ax.set_title(title, color=TXT_CLR, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.6, alpha=0.7)

# ── Risk Dashboard ────────────────────────────────────────────────────────────
def plot_risk_dashboard(risks: dict[str, float]) -> plt.Figure:
    names = list(risks.keys())
    scores = list(risks.values())
    colors = [RED if s >= 65 else AMBER if s >= 35 else GREEN for s in scores]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    facecolor=DARK_BG,
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Horizontal bar chart
    bars = ax1.barh(names, scores, color=colors, height=0.55, zorder=3)
    ax1.set_xlim(0, 100)
    ax1.axvline(35, color=AMBER, lw=1.2, ls="--", alpha=0.6)
    ax1.axvline(65, color=RED,   lw=1.2, ls="--", alpha=0.6)
    for bar, sc in zip(bars, scores):
        ax1.text(sc + 1, bar.get_y() + bar.get_height() / 2,
                 f"{sc:.1f}%", va="center", color=TXT_CLR, fontsize=9)
    _style_ax(ax1, "Risk Profile")
    ax1.set_xlabel("Risk Score (%)", color=TXT_CLR)

    # Pie: risk distribution
    low  = sum(1 for s in scores if s < 35)
    med  = sum(1 for s in scores if 35 <= s < 65)
    high = sum(1 for s in scores if s >= 65)
    vals  = [x for x in [low, med, high] if x > 0]
    lbls  = [l for l, x in zip(["Low", "Medium", "High"], [low, med, high]) if x > 0]
    clrs  = [c for c, x in zip([GREEN, AMBER, RED], [low, med, high]) if x > 0]
    ax2.pie(vals, labels=lbls, colors=clrs, autopct="%1.0f%%",
            textprops={"color": TXT_CLR, "fontsize": 9},
            wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5})
    ax2.set_facecolor(DARK_BG)
    ax2.set_title("Risk Distribution", color=TXT_CLR, fontsize=11, fontweight="bold")

    fig.tight_layout(pad=2)
    return fig

# ── What-If ───────────────────────────────────────────────────────────────────
def plot_whatif(
    base_risks: dict[str, float],
    scenario_risks: dict[str, dict[str, float]],
) -> plt.Figure:
    risk_names = list(base_risks.keys())
    sc_names   = list(scenario_risks.keys())

    fig, axes = plt.subplots(1, len(risk_names),
                             figsize=(max(14, 3 * len(risk_names)), 5),
                             facecolor=DARK_BG)
    if len(risk_names) == 1:
        axes = [axes]

    for ax, rname in zip(axes, risk_names):
        baseline = base_risks[rname]
        sc_vals  = [scenario_risks[sc][rname] for sc in sc_names]
        all_vals = [baseline] + sc_vals
        bar_colors = [RED if v >= 65 else AMBER if v >= 35 else GREEN for v in all_vals]
        labels = ["Baseline"] + sc_names
        bars = ax.bar(range(len(all_vals)), all_vals, color=bar_colors,
                      width=0.6, zorder=3)
        for bar, v in zip(bars, all_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.0f}%", ha="center", va="bottom",
                    color=TXT_CLR, fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 110)
        _style_ax(ax, rname[:20])
        ax.axhline(35, color=AMBER, lw=0.8, ls="--", alpha=0.5)
        ax.axhline(65, color=RED,   lw=0.8, ls="--", alpha=0.5)

    fig.suptitle("What-If Scenario Analysis", color=TXT_CLR,
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(pad=2)
    return fig

# ── Health Trajectory ─────────────────────────────────────────────────────────
def plot_trajectory(fields: dict[str, float]) -> plt.Figure:
    months = np.arange(0, 13)
    bmi_0  = fields.get("BMI", 25.0)
    rbs_0  = fields.get("RBS(mg/dl)", 90.0)

    bmi_traj = [max(18.5, bmi_0 - 0.15 * m) for m in months]
    rbs_traj = [max(70.0, rbs_0 - 1.2  * m) for m in months]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)

    ax1.plot(months, bmi_traj, color=BLUE, lw=2.5, marker="o", ms=5)
    ax1.axhline(25,   color=GREEN, lw=1.2, ls="--", alpha=0.7, label="Normal BMI (<25)")
    ax1.axhline(18.5, color=AMBER, lw=1.2, ls="--", alpha=0.5, label="Min safe (18.5)")
    ax1.fill_between(months, bmi_traj, 18.5, alpha=0.08, color=BLUE)
    _style_ax(ax1, "BMI Trajectory (12 Months)")
    ax1.set_xlabel("Month", color=TXT_CLR)
    ax1.set_ylabel("BMI", color=TXT_CLR)
    ax1.legend(fontsize=8, labelcolor=TXT_CLR, facecolor=PANEL_BG, edgecolor=GRID_CLR)

    ax2.plot(months, rbs_traj, color="#f97316", lw=2.5, marker="s", ms=5)
    ax2.axhline(100, color=GREEN, lw=1.2, ls="--", alpha=0.7, label="Normal RBS (<100)")
    ax2.axhline(70,  color=AMBER, lw=1.2, ls="--", alpha=0.5, label="Lower bound (70)")
    ax2.fill_between(months, rbs_traj, 70, alpha=0.08, color="#f97316")
    _style_ax(ax2, "Blood Sugar Trajectory (12 Months)")
    ax2.set_xlabel("Month", color=TXT_CLR)
    ax2.set_ylabel("RBS (mg/dl)", color=TXT_CLR)
    ax2.legend(fontsize=8, labelcolor=TXT_CLR, facecolor=PANEL_BG, edgecolor=GRID_CLR)

    fig.tight_layout(pad=2.5)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# Recommendations engine
# ══════════════════════════════════════════════════════════════════════════════
def generate_recommendations(risks: dict[str, float], fields: dict[str, float]) -> list[dict]:
    recs = []

    bmi = fields.get("BMI", 0)
    if bmi > 25:
        recs.append({
            "category": "🏃 Weight Management",
            "priority": "High" if bmi > 30 else "Medium",
            "action": f"BMI is {bmi:.1f}. Target 0.5–1 kg/week loss through calorie deficit.",
            "detail": "Aim for BMI < 25. Even 5–10% weight reduction improves PCOS symptoms.",
        })

    rbs = fields.get("RBS(mg/dl)", 0)
    if rbs > 100:
        recs.append({
            "category": "🩸 Blood Sugar",
            "priority": "High" if rbs > 126 else "Medium",
            "action": f"RBS {rbs:.0f} mg/dl — reduce refined carbs, add fibre.",
            "detail": "Consider HbA1c test. Low-GI diet and walking 30 min/day helps.",
        })

    vit_d = fields.get("Vit D3 (ng/mL)", 0)
    if vit_d < 20:
        recs.append({
            "category": "💊 Vitamin D",
            "priority": "Medium",
            "action": f"Vit D3 is {vit_d:.1f} ng/mL (deficient). Supplement 2000–4000 IU/day.",
            "detail": "Vit D deficiency worsens insulin resistance. Re-test in 3 months.",
        })

    amh = fields.get("AMH(ng/mL)", 0)
    if amh > 5:
        recs.append({
            "category": "🔬 Hormonal",
            "priority": "High",
            "action": f"Elevated AMH ({amh:.2f} ng/mL) — consistent with PCOS. OB-GYN consult advised.",
            "detail": "High AMH indicates polycystic ovarian reserve. Monitor annually.",
        })

    lh = fields.get("LH(mIU/mL)", 0)
    fsh = fields.get("FSH(mIU/mL)", 1)
    if lh > 0 and fsh > 0 and (lh / fsh) > 2:
        recs.append({
            "category": "⚗️ LH/FSH Ratio",
            "priority": "Medium",
            "action": f"LH/FSH ratio is {lh/fsh:.2f} (elevated). Hormonal panel recheck in 3 months.",
            "detail": "LH:FSH > 2:1 is a common PCOS marker. Lifestyle changes can normalise it.",
        })

    follicles = max(
        fields.get("Follicle No. (L)", 0),
        fields.get("Follicle No. (R)", 0),
    )
    if follicles >= 12:
        recs.append({
            "category": "🔍 Ultrasound",
            "priority": "High",
            "action": f"Polycystic morphology detected ({follicles:.0f} follicles). Annual ultrasound follow-up.",
            "detail": "≥12 follicles/ovary is a key PCOS criterion. Correlate with symptoms.",
        })

    if not recs:
        recs.append({
            "category": "✅ Overall",
            "priority": "Low",
            "action": "Values are within acceptable ranges. Maintain healthy lifestyle.",
            "detail": "Annual screening recommended. Keep up regular exercise and balanced diet.",
        })

    return recs

# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
for key, default in [
    ("step", 1),
    ("api_key", ""),
    ("extracted_fields", {}),
    ("confirmed_fields", {}),
    ("analysis_done", False),
    ("risks", {}),
    ("scenario_risks", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧬 PCOS Digital Twin")
    st.markdown("---")

    step_labels = {
        1: "🔑 API Key",
        2: "📄 Upload PDFs",
        3: "⚡ Extract",
        4: "✏️ Review",
        5: "📊 Analysis",
    }
    for s, lbl in step_labels.items():
        active = st.session_state.step == s
        done   = st.session_state.step > s
        icon   = "✅" if done else ("▶" if active else "○")
        color  = "#58a6ff" if active else ("#3fb950" if done else "#8b949e")
        st.markdown(
            f'<div style="color:{color};padding:4px 0;font-size:13px;">'
            f'{icon} Step {s}: {lbl}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#8b949e;">Built with Streamlit + Claude API</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧬 PCOS Digital Twin")
st.markdown("*AI-powered PCOS risk analysis from lab reports*")
st.markdown("---")

# ── STEP 1: API Key ───────────────────────────────────────────────────────────
if st.session_state.step == 1:
    st.markdown('<div class="step-header">Step 1 — Enter your Anthropic API Key</div>',
                unsafe_allow_html=True)
    st.info("Your API key is used only in this session to call Claude for PDF extraction. It is never stored.")

    key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        value=st.session_state.api_key,
    )

    if st.button("✅ Confirm API Key"):
        if key_input.startswith("sk-ant-") or key_input.startswith("sk-"):
            st.session_state.api_key = key_input
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("Invalid key format. Anthropic keys start with `sk-ant-`.")

# ── STEP 2: Upload PDFs ───────────────────────────────────────────────────────
elif st.session_state.step == 2:
    st.markdown('<div class="step-header">Step 2 — Upload Lab Report PDFs</div>',
                unsafe_allow_html=True)
    st.markdown("Upload one or more PDFs (blood test, hormone panel, ultrasound). All will be combined for extraction.")

    uploaded = st.file_uploader(
        "Upload PDF lab reports",
        type=["pdf"],
        accept_multiple_files=True,
        help="Blood test report, Hormone panel, Ultrasound report",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        back = st.button("← Back")
    with col2:
        proceed = st.button("➡️ Proceed to Extract")

    if back:
        st.session_state.step = 1
        st.rerun()

    if proceed:
        if not uploaded:
            st.error("Please upload at least one PDF.")
        else:
            st.session_state.uploaded_pdfs = uploaded
            st.session_state.step = 3
            st.rerun()

# ── STEP 3: Extract ───────────────────────────────────────────────────────────
elif st.session_state.step == 3:
    st.markdown('<div class="step-header">Step 3 — Extract Values via Claude AI</div>',
                unsafe_allow_html=True)

    pdfs = st.session_state.get("uploaded_pdfs", [])
    st.markdown(f"**{len(pdfs)} PDF(s) ready for extraction.**")
    for p in pdfs:
        st.markdown(f"• 📄 `{p.name}`")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        extract_btn = st.button("⚡ Extract with Claude")

    if extract_btn:
        with st.spinner("Extracting text from PDFs…"):
            combined = ""
            for p in pdfs:
                p.seek(0)
                combined += f"\n\n--- {p.name} ---\n"
                combined += extract_pdf_text(p)

        with st.spinner("Calling Claude API to parse PCOS fields…"):
            try:
                raw = claude_extract(st.session_state.api_key, combined)
                fields_with_flags = claude_dict_to_fields(raw)
                st.session_state.extracted_fields = fields_with_flags
                st.session_state.step = 4
                st.success("Extraction complete! Review the values below.")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error from Claude: {e}. Try again.")
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Go back to Step 1 and re-enter.")
            except Exception as e:
                st.error(f"Extraction error: {e}")

# ── STEP 4: Review / Edit ─────────────────────────────────────────────────────
elif st.session_state.step == 4:
    st.markdown('<div class="step-header">Step 4 — Review & Edit Extracted Values</div>',
                unsafe_allow_html=True)
    st.markdown("🟢 = Found in PDF &nbsp;&nbsp;&nbsp; 🟠 = Not found (default used) — edit any value before running analysis.")

    extracted: dict[str, tuple[float, bool]] = st.session_state.extracted_fields
    confirmed: dict[str, float] = {}

    # Group fields for display
    groups = {
        "🩺 Vitals & Anthropometrics": [
            "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI",
            "Pulse rate(bpm)", "RR (breaths/min)", "Hb(g/dl)",
            "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
            "Waist(inch)", "Hip(inch)", "Waist:Hip Ratio",
        ],
        "🔬 Hormones": [
            "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
            "AMH(ng/mL)", "PRL(ng/mL)", "PRG(ng/mL)",
            "TSH (mIU/L)", "Vit D3 (ng/mL)",
            "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
        ],
        "🩸 Metabolic": [
            "RBS(mg/dl)", "Blood Group",
        ],
        "🔍 Ultrasound": [
            "Follicle No. (L)", "Follicle No. (R)",
            "Avg. F size (L) (mm)", "Avg. F size (R) (mm)",
            "Endometrium (mm)", "Cycle(R/I)", "Cycle length(days)",
        ],
        "📝 Symptoms & Lifestyle": [
            "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
            "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)",
            "Reg.Exercise(Y/N)", "Pregnant(Y/N)", "No. of aborptions",
            "Marraige Status (Yrs)",
        ],
    }

    for group_name, field_list in groups.items():
        with st.expander(group_name, expanded=(group_name == "🩺 Vitals & Anthropometrics")):
            cols_per_row = 3
            field_items = [(f, extracted.get(f, (DEFAULTS.get(f, 0.0), False)))
                           for f in field_list]
            for i in range(0, len(field_items), cols_per_row):
                row_fields = field_items[i:i + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, (fname, (val, found)) in zip(cols, row_fields):
                    badge = "🟢" if found else "🟠"
                    new_val = col.number_input(
                        f"{badge} {fname}",
                        value=float(val),
                        step=0.01,
                        key=f"field_{fname}",
                        format="%.3f",
                    )
                    confirmed[fname] = new_val

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        run_btn = st.button("🚀 Run Analysis")

    if run_btn:
        st.session_state.confirmed_fields = confirmed
        # ── run models ──
        try:
            risk_predictor, state_estimator = load_models()
            feature_columns = load_feature_columns()
            input_df = build_input_row(confirmed, feature_columns)

            risks = compute_risks(risk_predictor, input_df)
            st.session_state.risks = risks

            # Scenario risks
            sc_risks = {}
            for sc_name, delta in SCENARIOS.items():
                sc_fields = apply_scenario(confirmed, delta)
                sc_df = build_input_row(sc_fields, feature_columns)
                sc_risks[sc_name] = compute_risks(risk_predictor, sc_df)
            st.session_state.scenario_risks = sc_risks
            st.session_state.analysis_done = True
            st.session_state.step = 5
            st.rerun()
        except Exception as e:
            st.error(f"Model error: {e}. Check that model files exist in ../models/")

# ── STEP 5: Results ───────────────────────────────────────────────────────────
elif st.session_state.step == 5 and st.session_state.analysis_done:
    st.markdown('<div class="step-header">Step 5 — Analysis Results</div>',
                unsafe_allow_html=True)

    confirmed = st.session_state.confirmed_fields
    risks     = st.session_state.risks
    sc_risks  = st.session_state.scenario_risks

    # ── KPI strip ──────────────────────────────────────────────────────────
    kpi_fields = [
        ("BMI",              confirmed.get("BMI", 0),             ""),
        ("RBS (mg/dl)",      confirmed.get("RBS(mg/dl)", 0),      "mg/dl"),
        ("AMH (ng/mL)",      confirmed.get("AMH(ng/mL)", 0),      "ng/mL"),
        ("Follicles (L/R)",
         f'{confirmed.get("Follicle No. (L)", 0):.0f} / {confirmed.get("Follicle No. (R)", 0):.0f}',
         ""),
    ]
    kpi_cols = st.columns(len(kpi_fields))
    for col, (label, val, unit) in zip(kpi_cols, kpi_fields):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val} <span style="font-size:13px;color:#8b949e;">{unit}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Risk Dashboard",
        "🔄 What-If Simulator",
        "📈 Health Trajectory",
        "💡 Recommendations",
    ])

    # Tab 1 — Risk Dashboard
    with tab1:
        st.subheader("Risk Profile")

        # Mini risk badges
        risk_cols = st.columns(len(risks))
        for col, (rname, score) in zip(risk_cols, risks.items()):
            level = risk_level(score)
            bg, fg = RISK_COLORS[level]
            col.markdown(
                f'<div style="background:{bg};border:1px solid {fg};border-radius:8px;'
                f'padding:10px;text-align:center;margin-bottom:8px;">'
                f'<div style="font-size:11px;color:{fg};">{rname}</div>'
                f'<div style="font-size:24px;font-weight:700;color:{fg};">{score:.0f}%</div>'
                f'<div style="font-size:10px;color:{fg};">{level.upper()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        fig = plot_risk_dashboard(risks)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Save
        fig2 = plot_risk_dashboard(risks)
        fig2.savefig(os.path.join(OUTPUT_DIR, "risk_dashboard.png"),
                     dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig2)
        st.success("💾 Saved → outputs/risk_dashboard.png")

    # Tab 2 — What-If
    with tab2:
        st.subheader("What-If Scenario Comparison")
        st.markdown("Each bar shows projected risk if you adopt a given intervention.")

        fig = plot_whatif(risks, sc_risks)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Table summary
        sc_table_data = []
        for sc_name, sc_r in sc_risks.items():
            row = {"Scenario": sc_name}
            for rname, score in sc_r.items():
                delta = score - risks[rname]
                row[rname] = f"{score:.1f}% ({delta:+.1f})"
            sc_table_data.append(row)
        df_sc = pd.DataFrame(sc_table_data).set_index("Scenario")
        st.dataframe(df_sc, use_container_width=True)

        # Save
        fig3 = plot_whatif(risks, sc_risks)
        fig3.savefig(os.path.join(OUTPUT_DIR, "simulation_results.png"),
                     dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig3)

    # Tab 3 — Trajectory
    with tab3:
        st.subheader("12-Month Health Trajectory")
        st.markdown("Projected improvement assuming consistent lifestyle changes (BMI −0.15/month, RBS −1.2 mg/dl/month).")

        fig = plot_trajectory(confirmed)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Trajectory data table
        months = list(range(13))
        bmi_0  = confirmed.get("BMI", 25.0)
        rbs_0  = confirmed.get("RBS(mg/dl)", 90.0)
        traj_df = pd.DataFrame({
            "Month": months,
            "BMI":   [round(max(18.5, bmi_0 - 0.15 * m), 2) for m in months],
            "RBS":   [round(max(70.0, rbs_0 - 1.2  * m), 1) for m in months],
        })
        st.dataframe(traj_df, use_container_width=True, hide_index=True)

    # Tab 4 — Recommendations
    with tab4:
        st.subheader("Personalised Recommendations")
        recs = generate_recommendations(risks, confirmed)

        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 3))

        for rec in recs:
            pri = rec["priority"]
            border = RED if pri == "High" else AMBER if pri == "Medium" else GREEN
            st.markdown(
                f'<div style="border-left:4px solid {border};background:#161b22;'
                f'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:12px;">'
                f'<div style="font-size:14px;font-weight:700;color:#e6edf3;">'
                f'{rec["category"]} '
                f'<span style="font-size:11px;background:{border};color:#fff;'
                f'border-radius:4px;padding:2px 8px;margin-left:8px;">{pri}</span>'
                f'</div>'
                f'<div style="font-size:13px;color:#c9d1d9;margin-top:6px;">{rec["action"]}</div>'
                f'<div style="font-size:12px;color:#8b949e;margin-top:4px;">{rec["detail"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Reset button
    st.markdown("---")
    if st.button("🔄 Start New Analysis"):
        for key in ["step", "api_key", "extracted_fields", "confirmed_fields",
                    "analysis_done", "risks", "scenario_risks", "uploaded_pdfs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

else:
    # Fallback — should not normally reach here
    st.session_state.step = 1
    st.rerun()