"""
PCOS Digital Twin — Streamlit Dashboard
========================================
Run with:  streamlit run src/dashboard.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="PCOS Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
# Always resolve relative to THIS file's location — works regardless of
# where streamlit is launched from.
_SRC      = os.path.dirname(os.path.abspath(__file__))   # .../pcos_digital_twin/src
_ROOT     = os.path.normpath(os.path.join(_SRC, ".."))   # .../pcos_digital_twin

STATE_MODEL_PATH = os.path.join(_ROOT, "models", "state_estimator.pkl")
RISK_MODEL_PATH  = os.path.join(_ROOT, "models", "risk_predictor.pkl")
SCALER_PATH      = os.path.join(_ROOT, "models", "scaler.pkl")
DATA_PATH        = os.path.join(_ROOT, "data", "processed", "pcos_processed.csv")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }
    [data-testid="stSidebar"] * { color: #c8cdd8 !important; }
    [data-testid="metric-container"] {
        background: #1a1d27; border: 1px solid #2d3142;
        border-radius: 10px; padding: 16px;
    }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #a78bfa;
        border-bottom: 1px solid #2d3142; padding-bottom: 6px; margin: 18px 0 12px;
    }
    .risk-card {
        background: #1a1d27; border: 1px solid #2d3142;
        border-radius: 10px; padding: 16px 12px;
        text-align: center; margin-bottom: 8px;
    }
    .risk-label  { font-size: 0.78rem; color: #888; margin-bottom: 4px; }
    .risk-value  { font-size: 1.8rem; font-weight: 700; }
    .risk-bar-bg { background: #2d3142; border-radius: 4px; height: 6px; margin-top: 8px; }
    .risk-bar-fg { border-radius: 4px; height: 6px; }
    .scenario-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 0; border-bottom: 1px solid #2d3142; font-size: 0.88rem;
    }
    .delta-good { color: #2ecc71; font-weight: 600; }
    .delta-bad  { color: #e74c3c; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1d27; border-radius: 8px; }
    .stTabs [data-baseweb="tab"]      { color: #888; }
    .stTabs [aria-selected="true"]    { color: #a78bfa !important; }
    .stSlider > div > div > div { background: #a78bfa; }
    .stButton > button {
        background: #7c3aed; color: white; border: none;
        border-radius: 8px; font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background: #6d28d9; }
    .info-box {
        background: #1e2235; border-left: 3px solid #a78bfa;
        border-radius: 0 8px 8px 0; padding: 10px 14px;
        font-size: 0.85rem; color: #c8cdd8; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    missing = [p for p in [STATE_MODEL_PATH, RISK_MODEL_PATH, SCALER_PATH]
               if not os.path.exists(p)]
    if missing:
        return None, None, None, f"Model files not found: {missing}"
    try:
        state_model = joblib.load(STATE_MODEL_PATH)
        risk_models = joblib.load(RISK_MODEL_PATH)
        scaler      = joblib.load(SCALER_PATH)
        return state_model, risk_models, scaler, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data
def load_feature_cols():
    if not os.path.exists(DATA_PATH):
        return []
    df = pd.read_csv(DATA_PATH)
    return (df.drop(columns=["PCOS (Y/N)"], errors="ignore")
              .select_dtypes(include=[np.number])
              .columns.tolist())


# ══════════════════════════════════════════════════════════════════════════════
# SCALING HELPER
# ══════════════════════════════════════════════════════════════════════════════

def scale_patient(patient: dict, scaler, feature_cols: list) -> dict:
    """Scale raw patient values using the fitted StandardScaler."""
    df = pd.DataFrame([patient])
    df = df.reindex(columns=feature_cols, fill_value=0).fillna(0)
    scaled = scaler.transform(df)
    return dict(zip(feature_cols, scaled[0]))


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def get_risk_scores(risk_models, patient: dict, feature_cols: list, scaler=None) -> dict:
    """Return risk probability (%) for each risk category."""
    if scaler is not None:
        patient = scale_patient(patient, scaler, feature_cols)
    scores = {}
    for name, info in risk_models.items():
        df_in = pd.DataFrame([patient]).reindex(columns=feature_cols, fill_value=0).fillna(0)
        proba = info["model"].predict_proba(df_in)[0][1]
        scores[name] = round(proba * 100, 1)
    return scores


SCENARIOS = {
    "Weight loss (5 kg)":  {"BMI": -2.0, "Weight (Kg)": -5.0, "RBS(mg/dl)": -10.0},
    "Regular exercise":    {"Reg.Exercise(Y/N)": 1, "BP _Systolic (mmHg)": -8.0,
                            "RBS(mg/dl)": -15.0, "Cycle(R/I)": 1},
    "Dietary changes":     {"Fast food (Y/N)": 0, "RBS(mg/dl)": -12.0,
                            "BMI": -1.0, "Weight (Kg)": -2.0},
    "Metformin":           {"RBS(mg/dl)": -20.0, "Cycle(R/I)": 1,
                            "Cycle length(days)": 30, "BMI": -1.5},
    "Combined lifestyle":  {"BMI": -3.5, "Weight (Kg)": -8.0, "RBS(mg/dl)": -25.0,
                            "BP _Systolic (mmHg)": -10.0, "Cycle(R/I)": 1,
                            "Fast food (Y/N)": 0, "Reg.Exercise(Y/N)": 1,
                            "Symptom_burden": -2.0},
}

FLOORS = {"BMI": 18.5, "Weight (Kg)": 45, "RBS(mg/dl)": 70,
          "BP _Systolic (mmHg)": 90, "Symptom_burden": 0}


def apply_scenario(patient: dict, scenario_name: str) -> dict:
    p = deepcopy(patient)
    for field, delta in SCENARIOS[scenario_name].items():
        if isinstance(delta, float) and delta < 0:
            floor = FLOORS.get(field, -9999)
            p[field] = max(floor, p.get(field, 0) + delta)
        else:
            p[field] = (delta if not isinstance(delta, float)
                        else max(FLOORS.get(field, -9999), p.get(field, 0) + delta))
    b = p.get("BMI", 25)
    p["BMI_category"] = 1 if b < 25 else (2 if b < 30 else 3)
    return p


def estimate_trajectory(state_model, patient: dict, feature_cols: list,
                         scaler=None, months: int = 12):
    scores = []
    bmi0 = patient.get("BMI", 25)
    rbs0 = patient.get("RBS(mg/dl)", 100)
    for m in range(months + 1):
        sp = deepcopy(patient)
        sp["BMI"]        = max(18.5, bmi0 - 0.15 * m)
        sp["RBS(mg/dl)"] = max(70,   rbs0 - 1.2  * m)
        b = sp["BMI"]
        sp["BMI_category"] = 1 if b < 25 else (2 if b < 30 else 3)
        if scaler is not None:
            sp = scale_patient(sp, scaler, feature_cols)
        df_in = pd.DataFrame([sp]).reindex(columns=feature_cols, fill_value=0).fillna(0)
        try:
            if hasattr(state_model, "predict_proba"):
                score = float(state_model.predict_proba(df_in)[0][0]) * 100
            else:
                score = float(state_model.predict(df_in)[0])
        except Exception:
            score = max(0, 100 - (bmi0 - 18.5) * 3 - 0.15 * m * 3)
        scores.append(score)
    return list(range(months + 1)), scores


def risk_color(score):
    if score >= 70: return "#e74c3c"
    if score >= 40: return "#e67e22"
    return "#2ecc71"

def risk_label(score):
    if score >= 70: return "HIGH"
    if score >= 40: return "MODERATE"
    return "LOW"

def fmt_risk(key):
    return key.replace("_risk", "").replace("_", " ").title()


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def make_risk_chart(all_results):
    risks     = list(list(all_results.values())[0].keys())
    scenarios = list(all_results.keys())
    x         = np.arange(len(risks))
    width     = 0.75 / len(scenarios)
    palette   = ["#7c3aed", "#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for i, (sc, scores) in enumerate(all_results.items()):
        vals   = [scores[r] for r in risks]
        offset = (i - (len(scenarios) - 1) / 2) * width
        bars   = ax.bar(x + offset, vals, width, label=sc,
                        color=palette[i % len(palette)], alpha=0.88,
                        edgecolor="#0f1117", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v > 6:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color="white", alpha=0.75)

    ax.axhline(70, color="#e74c3c", linestyle="--", lw=0.8, alpha=0.6, label="High (70%)")
    ax.axhline(40, color="#e67e22", linestyle="--", lw=0.8, alpha=0.5, label="Moderate (40%)")
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_risk(r) for r in risks], rotation=18, ha="right",
                       fontsize=9, color="#c8cdd8")
    ax.set_ylabel("Risk probability (%)", color="#c8cdd8", fontsize=10)
    ax.set_ylim(0, 115)
    ax.tick_params(colors="#c8cdd8")
    for sp in ax.spines.values(): sp.set_color("#2d3142")
    ax.grid(axis="y", color="#2d3142", linewidth=0.5)
    ax.legend(fontsize=8, facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#c8cdd8", loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def make_trajectory_chart(months, scores):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    ax.fill_between(months, scores, alpha=0.12, color="#2ecc71")
    ax.plot(months, scores, color="#2ecc71", lw=2, marker="o", markersize=4)

    ax.axhspan(70, 100, alpha=0.06, color="#2ecc71")
    ax.axhspan(40, 70,  alpha=0.06, color="#e67e22")
    ax.axhspan(0,  40,  alpha=0.06, color="#e74c3c")
    ax.text(12.2, 85, "Good",      color="#2ecc71", fontsize=8, va="center")
    ax.text(12.2, 55, "Moderate",  color="#e67e22", fontsize=8, va="center")
    ax.text(12.2, 20, "High risk", color="#e74c3c", fontsize=8, va="center")

    delta = scores[-1] - scores[0]
    sign  = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
    ax.annotate(f"M0: {scores[0]:.0f}%",
                xy=(0, scores[0]), xytext=(0.5, scores[0] + 6),
                color="white", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax.annotate(f"M12: {scores[-1]:.0f}% ({sign})",
                xy=(12, scores[-1]), xytext=(9, scores[-1] + 6),
                color="#2ecc71", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=0.8))

    ax.set_xlim(-0.3, 13.8)
    ax.set_ylim(0, 108)
    ax.set_xticks(range(0, 13, 3))
    ax.set_xticklabels([f"M{m}" for m in range(0, 13, 3)], color="#c8cdd8", fontsize=9)
    ax.set_xlabel("Month", color="#c8cdd8", fontsize=10)
    ax.set_ylabel("Health score (%)", color="#c8cdd8", fontsize=10)
    ax.tick_params(colors="#c8cdd8")
    for sp in ax.spines.values(): sp.set_color("#2d3142")
    ax.grid(color="#2d3142", linewidth=0.5)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Patient Input Form
# ══════════════════════════════════════════════════════════════════════════════

def sidebar_inputs():
    st.sidebar.image("https://img.icons8.com/color/96/stethoscope.png", width=48)
    st.sidebar.title("🩺 Patient Profile")
    st.sidebar.markdown("---")

    with st.sidebar.expander("📋 Demographics", expanded=True):
        age    = st.slider("Age (years)", 15, 55, 28)
        weight = st.slider("Weight (kg)", 40, 120, 68)
        height = st.slider("Height (cm)", 140, 185, 162)
        bmi    = round(weight / (height / 100) ** 2, 1)
        st.info(f"BMI: **{bmi}** ({'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'})")

    with st.sidebar.expander("🔬 Clinical Markers", expanded=True):
        cycle     = st.selectbox("Menstrual cycle", ["Irregular", "Regular"])
        cycle_len = st.slider("Cycle length (days)", 21, 60, 35)
        amh       = st.slider("AMH (ng/mL)", 0.1, 15.0, 4.5, step=0.1)
        rbs       = st.slider("Random Blood Sugar (mg/dl)", 60, 200, 105)
        bp_sys    = st.slider("BP Systolic (mmHg)", 80, 160, 122)
        bp_dia    = st.slider("BP Diastolic (mmHg)", 50, 110, 80)

    with st.sidebar.expander("🫁 Follicle & Hormones", expanded=False):
        foll_l    = st.slider("Follicles — left ovary", 0, 30, 13)
        foll_r    = st.slider("Follicles — right ovary", 0, 30, 14)
        foll_sz_l = st.slider("Avg follicle size L (mm)", 5, 30, 15)
        foll_sz_r = st.slider("Avg follicle size R (mm)", 5, 30, 15)

    with st.sidebar.expander("🥗 Lifestyle", expanded=False):
        exercise  = st.selectbox("Regular exercise", ["No", "Yes"])
        fast_food = st.selectbox("Fast food intake", ["Yes", "No"])
        symptom   = st.slider("Symptom burden (0–5)", 0, 5, 3)
        waist     = st.slider("Waist (inches)", 22, 55, 32)
        hip       = st.slider("Hip (inches)", 28, 60, 38)

    patient = {
        "Age (yrs)":            age,
        "Weight (Kg)":          weight,
        "Height(Cm) ":          height,          # trailing space matches dataset column
        "BMI":                  bmi,
        "Cycle(R/I)":           2 if cycle == "Irregular" else 1,
        "Cycle length(days)":   cycle_len,
        "AMH(ng/mL)":           amh,
        "RBS(mg/dl)":           rbs,
        "BP _Systolic (mmHg)":  bp_sys,
        "BP _Diastolic (mmHg)": bp_dia,
        "Follicle No. (L)":     foll_l,
        "Follicle No. (R)":     foll_r,
        "Avg. F size (L) (mm)": foll_sz_l,
        "Avg. F size (R) (mm)": foll_sz_r,
        "Reg.Exercise(Y/N)":    1 if exercise == "Yes" else 0,
        "Fast food (Y/N)":      1 if fast_food == "Yes" else 0,
        "Symptom_burden":       symptom,
        "Waist(inch)":          waist,
        "Hip(inch)":            hip,
        "Waist:Hip Ratio":      round(waist / hip, 3),
        "BMI_category":         1 if bmi < 25 else (2 if bmi < 30 else 3),
    }
    return patient


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("🩺 PCOS Digital Twin")
        st.caption("AI-powered risk prediction · what-if simulation · health trajectory")
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Analysis", use_container_width=True)

    st.markdown("---")

    # ── Load models ──────────────────────────────────────────────────────────
    state_model, risk_models, scaler, err = load_models()
    feature_cols = load_feature_cols()

    if err:
        st.error(f"⚠️ Could not load models: {err}")
        st.info("Make sure you've run data_pipeline.py, state_estimator.py, and risk_predictor.py first.")
        st.stop()

    # ── Patient inputs ───────────────────────────────────────────────────────
    patient = sidebar_inputs()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Risk Dashboard",
        "🔮  What-If Simulator",
        "📈  Health Trajectory",
        "💊  Recommendations",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Risk Dashboard
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Baseline risk scores</div>', unsafe_allow_html=True)

        if run_btn or True:   # auto-compute on load
            with st.spinner("Computing risk scores..."):
                try:
                    scores = get_risk_scores(risk_models, patient, feature_cols, scaler=scaler)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            cols = st.columns(len(scores))
            for col, (risk_key, score) in zip(cols, scores.items()):
                color = risk_color(score)
                label = fmt_risk(risk_key)
                lvl   = risk_label(score)
                with col:
                    pct = int(score)
                    st.markdown(f"""
                    <div class="risk-card">
                        <div class="risk-label">{label}</div>
                        <div class="risk-value" style="color:{color}">{score:.1f}%</div>
                        <div style="font-size:0.7rem;color:{color};margin-top:2px">{lvl}</div>
                        <div class="risk-bar-bg">
                            <div class="risk-bar-fg" style="width:{pct}%;background:{color}"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            avg      = np.mean(list(scores.values()))
            severity = ("🔴 High concern" if avg >= 70
                        else "🟠 Moderate concern" if avg >= 40 else "🟢 Low concern")
            st.markdown(f"""
            <div class="info-box">
                <strong>Overall average risk: {avg:.1f}%</strong> — {severity}<br>
                Based on {len(scores)} risk categories assessed by the trained model.
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Patient summary</div>', unsafe_allow_html=True)
            summary = {
                "Age":          f"{patient['Age (yrs)']:.0f} yrs",
                "BMI":          f"{patient['BMI']:.1f}",
                "Weight":       f"{patient['Weight (Kg)']:.0f} kg",
                "Cycle":        "Irregular" if patient["Cycle(R/I)"] == 2 else "Regular",
                "AMH":          f"{patient['AMH(ng/mL)']:.1f} ng/mL",
                "RBS":          f"{patient['RBS(mg/dl)']:.0f} mg/dl",
                "BP":           f"{patient['BP _Systolic (mmHg)']:.0f}/{patient['BP _Diastolic (mmHg)']:.0f}",
                "Follicles L/R":f"{patient['Follicle No. (L)']:.0f}/{patient['Follicle No. (R)']:.0f}",
                "Exercise":     "Yes" if patient["Reg.Exercise(Y/N)"] else "No",
                "Fast food":    "Yes" if patient["Fast food (Y/N)"] else "No",
            }
            df_summary = pd.DataFrame(summary.items(), columns=["Parameter", "Value"])
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — What-If Simulator
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Select intervention scenarios</div>',
                    unsafe_allow_html=True)

        selected = st.multiselect(
            "Choose one or more scenarios to simulate:",
            list(SCENARIOS.keys()),
            default=list(SCENARIOS.keys()),
        )

        if not selected:
            st.info("Select at least one scenario above.")
        else:
            with st.spinner("Running simulations..."):
                baseline    = get_risk_scores(risk_models, patient, feature_cols, scaler=scaler)
                all_results = {"Baseline": baseline}
                for sc in selected:
                    mod = apply_scenario(patient, sc)
                    all_results[sc] = get_risk_scores(risk_models, mod, feature_cols, scaler=scaler)

            st.markdown('<div class="section-header">Risk comparison chart</div>',
                        unsafe_allow_html=True)
            st.pyplot(make_risk_chart(all_results), use_container_width=True)

            st.markdown('<div class="section-header">Score changes vs baseline</div>',
                        unsafe_allow_html=True)
            risks = list(baseline.keys())
            rows  = []
            for sc in selected:
                row = {"Scenario": sc}
                for r in risks:
                    delta = all_results[sc][r] - baseline[r]
                    row[fmt_risk(r)] = f"{delta:+.1f}%"
                row["Avg Δ"] = f"{np.mean([all_results[sc][r] - baseline[r] for r in risks]):+.1f}%"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            best_sc  = min(selected, key=lambda s: np.mean(list(all_results[s].values())))
            best_avg = np.mean(list(all_results[best_sc].values()))
            base_avg = np.mean(list(baseline.values()))
            st.markdown(f"""
            <div class="info-box">
                Best scenario: <strong>{best_sc}</strong><br>
                Average risk drops from <strong>{base_avg:.1f}%</strong>
                → <strong>{best_avg:.1f}%</strong>
                (↓ {base_avg - best_avg:.1f} percentage points)
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Health Trajectory
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">12-month health state trajectory</div>',
                    unsafe_allow_html=True)
        st.caption("Projection assumes gradual BMI (−0.15/month) and RBS (−1.2 mg/dl/month) improvement.")

        horizon = st.slider("Projection horizon (months)", 3, 24, 12)

        with st.spinner("Estimating trajectory..."):
            months_list, traj_scores = estimate_trajectory(
                state_model, patient, feature_cols, scaler=scaler, months=horizon
            )

        st.pyplot(make_trajectory_chart(months_list, traj_scores), use_container_width=True)

        st.markdown('<div class="section-header">Score at key months</div>',
                    unsafe_allow_html=True)
        checkpoints = [(m, traj_scores[m]) for m in range(0, horizon + 1, 3)]
        df_traj = pd.DataFrame(checkpoints, columns=["Month", "Health Score (%)"])
        df_traj["Status"] = df_traj["Health Score (%)"].apply(
            lambda s: "🟢 Good" if s >= 70 else ("🟠 Moderate" if s >= 40 else "🔴 High risk"))
        df_traj["Health Score (%)"] = df_traj["Health Score (%)"].round(1)
        st.dataframe(df_traj, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — Recommendations
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">Personalised recommendations</div>',
                    unsafe_allow_html=True)

        baseline = get_risk_scores(risk_models, patient, feature_cols, scaler=scaler)

        all_sc_results = {}
        for sc in SCENARIOS:
            mod = apply_scenario(patient, sc)
            all_sc_results[sc] = get_risk_scores(risk_models, mod, feature_cols, scaler=scaler)

        for risk_key, base_score in baseline.items():
            label      = fmt_risk(risk_key)
            color      = risk_color(base_score)
            lvl        = risk_label(base_score)
            best       = min(all_sc_results, key=lambda s: all_sc_results[s][risk_key])
            best_score = all_sc_results[best][risk_key]
            reduction  = base_score - best_score

            with st.expander(f"{label} Risk — {base_score:.1f}% ({lvl})",
                             expanded=(base_score >= 60)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div style="text-align:center;padding:12px">
                        <div style="font-size:2.2rem;font-weight:700;color:{color}">{base_score:.1f}%</div>
                        <div style="font-size:0.8rem;color:{color}">{lvl}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    if reduction > 0.5:
                        st.success(f"Best intervention: **{best}**  →  {best_score:.1f}% (↓ {reduction:.1f} pp)")
                    else:
                        st.success("This risk factor is already well managed.")
                    st.markdown("**All scenarios ranked:**")
                    for sc, res in sorted(all_sc_results.items(),
                                          key=lambda kv: kv[1][risk_key]):
                        d     = base_score - res[risk_key]
                        tag   = f"↓{d:.1f}%" if d > 0 else (f"↑{abs(d):.1f}%" if d < 0 else "—")
                        color2 = "#2ecc71" if d > 0 else ("#e74c3c" if d < 0 else "#888")
                        st.markdown(
                            f'<div class="scenario-row"><span>{sc}</span>'
                            f'<span style="color:{color2};font-weight:600">'
                            f'{res[risk_key]:.1f}%  {tag}</span></div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <strong>General PCOS Management Guidelines</strong><br><br>
            • Even 5–7% body weight reduction can restore menstrual regularity in PCOS.<br>
            • Target 150 min/week of moderate-intensity aerobic activity (WHO recommendation).<br>
            • Metformin is most effective when combined with dietary intervention.<br>
            • Low glycaemic index (GI) diet helps reduce insulin resistance and androgen levels.<br>
            • Reassess clinically every 3–6 months.<br><br>
            <em>These predictions are model-based estimates, not medical diagnoses.
            Always consult a qualified healthcare professional.</em>
        </div>""", unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("PCOS Digital Twin  ·  NIT Srinagar  ·  Dept. of Computer Science & Engineering")


if __name__ == "__main__":
    main()