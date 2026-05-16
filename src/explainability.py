"""
PCOS Digital Twin — SHAP Explainability Module
================================================
Drop this file into src/ and add one import + one tab call in dashboard.py.

Install dependency:
    pip install shap

How to integrate into dashboard.py:
    1. from explainability import render_explainability_tab
    2. Add a 5th tab in show_results():
       tab5 = st.tabs([..., "🧠  Why This Score?"])
       with tab5:
           render_explainability_tab(risk_models, patient, feature_cols, scaler)
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ══════════════════════════════════════════════════════════════════════════════
# SHAP COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_shap_values(_model, patient_df, feature_cols):
    """
    Compute SHAP values for a single patient.
    Tries TreeExplainer first (fast, exact for tree models),
    falls back to KernelExplainer (model-agnostic, slower).
    Returns: (shap_values_array, base_value, explainer_type_str)
    """
    try:
        import shap
    except ImportError:
        return None, None, "not_installed"

    try:
        # TreeExplainer — works for RandomForest, XGBoost, LightGBM, etc.
        explainer   = shap.TreeExplainer(_model)
        shap_vals   = explainer.shap_values(patient_df)
        base_val    = explainer.expected_value

        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
            base_val  = base_val[1] if hasattr(base_val, "__len__") else base_val

        return shap_vals[0], float(base_val), "tree"

    except Exception:
        pass

    try:
        # KernelExplainer — model-agnostic fallback
        bg = patient_df.values

        explainer = shap.KernelExplainer(
            lambda x: _model.predict_proba(
                pd.DataFrame(x, columns=patient_df.columns)
            )[:, 1],
            bg,
            link="identity"
        )

        shap_vals = explainer.shap_values(patient_df.values, nsamples=100)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        return (
            shap_vals[0] if shap_vals.ndim > 1 else shap_vals,
            float(explainer.expected_value),
            "kernel"
        )

    except Exception as e:
        return None, None, f"error:{e}"


def get_shap_for_patient(risk_models, patient, feature_cols, scaler=None):
    """
    Run SHAP for every risk category.
    Returns dict: risk_name -> {shap_vals, base_val, feature_names, feature_values}
    """
    from copy import deepcopy

    results = {}
    p = deepcopy(patient)

    if scaler is not None:
        df_raw = (
            pd.DataFrame([p])
            .reindex(columns=feature_cols, fill_value=0)
            .fillna(0)
        )

        scaled = scaler.transform(df_raw)
        p_input = dict(zip(feature_cols, scaled[0]))

    else:
        p_input = p

    df_in = (
        pd.DataFrame([p_input])
        .reindex(columns=feature_cols, fill_value=0)
        .fillna(0)
    )

    for risk_name, info in risk_models.items():
        model = info["model"]

        shap_vals, base_val, etype = compute_shap_values(
            model,
            df_in,
            feature_cols
        )

        results[risk_name] = {
            "shap_vals":     shap_vals,
            "base_val":      base_val,
            "explainer":     etype,
            "feature_names": feature_cols,
            "feature_vals":  df_in.iloc[0].to_dict(),
            "raw_vals":      patient,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# HUMAN-READABLE FEATURE LABELS
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "AMH(ng/mL)":              "AMH level",
    "LH(mIU/mL)":              "LH hormone",
    "FSH(mIU/mL)":             "FSH hormone",
    "LH/FSH ratio":            "LH/FSH ratio",
    "Follicle No. (L)":        "Follicles (left ovary)",
    "Follicle No. (R)":        "Follicles (right ovary)",
    "Avg. F size (L) (mm)":    "Avg follicle size (L)",
    "Avg. F size (R) (mm)":    "Avg follicle size (R)",
    "BMI":                     "BMI",
    "Weight (Kg)":             "Body weight",
    "RBS(mg/dl)":              "Blood sugar (RBS)",
    "BP _Systolic (mmHg)":     "Systolic BP",
    "BP _Diastolic (mmHg)":    "Diastolic BP",
    "Cycle(R/I)":              "Menstrual cycle",
    "Cycle length(days)":      "Cycle length",
    "Hb(g/dl)":                "Haemoglobin",
    "TSH (mIU/L)":             "TSH (thyroid)",
    "PRL(ng/mL)":              "Prolactin",
    "PRG(ng/mL)":              "Progesterone",
    "Vit D3 (ng/mL)":          "Vitamin D3",
    "Waist(inch)":             "Waist circumference",
    "Hip(inch)":               "Hip circumference",
    "Waist:Hip Ratio":         "Waist-hip ratio",
    "Reg.Exercise(Y/N)":       "Regular exercise",
    "Fast food (Y/N)":         "Fast food intake",
    "Age (yrs)":               "Age",
    "BMI_category":            "BMI category",
    "Endometrium (mm)":        "Endometrial thickness",
    "Symptom_burden":          "Symptom burden",
}


def label(f):
    return FEATURE_LABELS.get(f, f.replace("_", " ").title())


# ══════════════════════════════════════════════════════════════════════════════
# CHART — SHAP WATERFALL FOR ONE RISK CATEGORY
# ══════════════════════════════════════════════════════════════════════════════

def make_shap_chart(
    shap_vals,
    base_val,
    feature_names,
    raw_patient,
    risk_name,
    final_score,
    top_n=10
):

    shap_pct = np.array(shap_vals) * 100

    pairs = [
        (f, v)
        for f, v in zip(feature_names, shap_pct)
        if abs(v) >= 0.05
    ]

    if len(pairs) < 3:
        pairs = list(zip(feature_names, shap_pct))

    pairs = sorted(
        pairs,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    pairs = sorted(pairs, key=lambda x: x[1])

    if not pairs:
        return None

    names  = [label(p[0]) for p in pairs]
    values = [p[1] for p in pairs]
    raws   = [raw_patient.get(p[0], "") for p in pairs]

    colors = [
        "#2ecc71" if v < 0 else "#e74c3c"
        for v in values
    ]

    fig, ax = plt.subplots(
        figsize=(10, max(5, len(pairs) * 0.6 + 2))
    )

    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    y_pos = np.arange(len(pairs))

    ax.barh(
        y_pos,
        values,
        color=colors,
        alpha=0.88,
        edgecolor="#0f1117",
        linewidth=0.4,
        height=0.6
    )

    for i, (v, raw) in enumerate(zip(values, raws)):

        raw_str = (
            f"  (val: {raw:.1f})"
            if isinstance(raw, (int, float))
            else ""
        )

        x_pos = (
            v + (max(abs(v) for v in values) * 0.03)
            if v >= 0
            else v - (max(abs(v) for v in values) * 0.03)
        )

        ha = "left" if v >= 0 else "right"

        ax.text(
            x_pos,
            i,
            f"{v:+.2f} pp{raw_str}",
            ha=ha,
            va="center",
            fontsize=8.5,
            color="white",
            alpha=0.9
        )

    ax.axvline(0, color="#4a4a6a", linewidth=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        names,
        fontsize=9.5,
        color="#c8cdd8"
    )

    ax.set_xlabel(
        "Contribution to risk score (percentage points)",
        color="#c8cdd8",
        fontsize=9
    )

    risk_label_str = (
        risk_name
        .replace("_risk", "")
        .replace("_", " ")
        .title()
    )

    ax.set_title(
        f"{risk_label_str} Risk — factor contributions\n"
        f"Base rate: {base_val*100:.1f}%  →  Patient score: {final_score:.1f}%",
        color="white",
        fontsize=11,
        pad=10
    )

    ax.tick_params(colors="#c8cdd8")

    for sp in ax.spines.values():
        sp.set_color("#2d3142")

    ax.grid(
        axis="x",
        color="#2d3142",
        linewidth=0.5,
        alpha=0.6
    )

    inc = mpatches.Patch(
        color="#e74c3c",
        alpha=0.88,
        label="Increases risk ↑"
    )

    dec = mpatches.Patch(
        color="#2ecc71",
        alpha=0.88,
        label="Decreases risk ↓"
    )

    ax.legend(
        handles=[inc, dec],
        fontsize=8,
        facecolor="#1a1d27",
        edgecolor="#2d3142",
        labelcolor="#c8cdd8"
    )

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PLAIN-ENGLISH SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plain_english_summary(
    shap_vals,
    feature_names,
    raw_patient,
    risk_name,
    score
):
    """
    Generate a 3-bullet doctor-readable explanation of the top drivers.
    """

    # Convert SHAP values to percentage points
    shap_pct = [
        (f, v * 100)
        for f, v in zip(feature_names, shap_vals)
    ]

    # Ignore tiny contributions
    top_up = sorted(
        [(f, v) for f, v in shap_pct if v > 0.01],
        key=lambda x: -x[1]
    )[:3]

    top_dn = sorted(
        [(f, v) for f, v in shap_pct if v < -0.01],
        key=lambda x: x[1]
    )[:2]

    risk_str = (
        risk_name
        .replace("_risk", "")
        .replace("_", " ")
        .title()
    )

    lines = [
        f"**{risk_str} risk is {score:.0f}%** — driven primarily by:"
    ]

    for f, v in top_up:

        raw = raw_patient.get(f, None)

        raw_str = (
            f" (value: {raw:.1f})"
            if isinstance(raw, (int, float))
            else ""
        )

        lines.append(
            f"- 🔴 **{label(f)}**{raw_str} is raising risk "
            f"(contribution: +{v:.2f} pp)"
        )

    if top_dn:

        lines.append("Partially offset by:")

        for f, v in top_dn:

            raw = raw_patient.get(f, None)

            raw_str = (
                f" (value: {raw:.1f})"
                if isinstance(raw, (int, float))
                else ""
            )

            lines.append(
                f"- 🟢 **{label(f)}**{raw_str} is lowering risk "
                f"(contribution: {v:.2f} pp)"
            )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MINI SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════

def render_top3_cards(shap_vals, feature_names, raw_patient):
    """
    Show top 3 risk drivers as compact coloured cards.
    """

    pairs = list(zip(feature_names, shap_vals))

    top3 = sorted(
        pairs,
        key=lambda x: -abs(x[1])
    )[:3]

    cols = st.columns(3)

    for col, (feat, val) in zip(cols, top3):

        raw = raw_patient.get(feat, "")

        raw_s = (
            f"{raw:.1f}"
            if isinstance(raw, (int, float))
            else str(raw)
        )

        direction = (
            "↑ raises risk"
            if val > 0
            else "↓ lowers risk"
        )

        color = (
            "#e74c3c"
            if val > 0
            else "#2ecc71"
        )

        factor_num = top3.index((feat, val)) + 1
        impact_val = f"{val*100:+.2f}"

        with col:
            html_card = f"""
<div style="background:#1a1d27; border:1px solid #2d3142; border-left:4px solid {color}; border-radius:8px; padding:14px 16px;">
  <div style="font-size:0.75rem; color:#999; margin-bottom:6px; font-weight:500;">Top factor #{factor_num}</div>
  <div style="font-size:1rem; font-weight:700; color:#fff; margin-bottom:4px;">{label(feat)}</div>
  <div style="font-size:0.85rem; color:{color}; font-weight:600; margin-bottom:8px;">{direction}</div>
  <div style="border-top:1px solid #2d3142; padding-top:8px; font-size:0.85rem; color:#aaa;">
    <div style="margin-bottom:4px;">Patient value: <span style="color:#fff; font-weight:600;">{raw_s}</span></div>
    <div>Impact: <span style="color:{color}; font-weight:600;">{impact_val} pp</span></div>
  </div>
</div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

def render_explainability_tab(risk_models, patient, feature_cols, scaler=None):
    """
    Full explainability tab. Call inside a `with tab:` block in dashboard.py.
    """

    st.markdown("""
    <div style="background:#1e2235;border-left:3px solid #a78bfa;
                border-radius:0 8px 8px 0;padding:10px 14px;
                font-size:0.85rem;color:#c8cdd8;margin-bottom:16px">
      <strong>Why did the model give this score?</strong><br>
      Each bar shows how much a specific clinical factor pushed the risk score
      up (🔴 red) or down (🟢 green) for <em>this patient</em>.
      This is not a generic feature ranking — it is specific to the values
      extracted from this patient's lab reports.
    </div>""", unsafe_allow_html=True)

    # Check if SHAP is available
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False

    if not shap_available:
        render_fallback_explainability(
            risk_models,
            patient,
            feature_cols,
            scaler
        )
        return

    # Risk selector
    risk_names = list(risk_models.keys())

    selected = st.selectbox(
        "Select risk category to explain:",
        risk_names,
        format_func=lambda k: (
            k.replace("_risk", "")
             .replace("_", " ")
             .title()
        )
    )

    model = risk_models[selected]["model"]

    from copy import deepcopy
    p = deepcopy(patient)

    if scaler:
        df_raw = (
            pd.DataFrame([p])
            .reindex(columns=feature_cols, fill_value=0)
            .fillna(0)
        )

        scaled = scaler.transform(df_raw)
        p_in = dict(zip(feature_cols, scaled[0]))

    else:
        p_in = p

    df_in = (
        pd.DataFrame([p_in])
        .reindex(columns=feature_cols, fill_value=0)
        .fillna(0)
    )

    with st.spinner("Computing SHAP values..."):

        shap_vals, base_val, etype = compute_shap_values(
            model,
            df_in,
            feature_cols
        )

    if shap_vals is None:

        if "not_installed" in str(etype):
            render_fallback_explainability(
                risk_models,
                patient,
                feature_cols,
                scaler
            )

        else:
            st.error(f"Could not compute explanations: {etype}")

        return

    # Final probability
    try:
        final_prob = model.predict_proba(df_in)[0][1] * 100

    except Exception:
        final_prob = (
            base_val + float(np.sum(shap_vals))
        ) * 100

    # ── Top cards ─────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:1rem;font-weight:600;color:#a78bfa;'
        'border-bottom:1px solid #2d3142;padding-bottom:6px;'
        'margin:0 0 12px">'
        'Top 3 factors for this patient</div>',
        unsafe_allow_html=True
    )

    render_top3_cards(
        shap_vals,
        feature_cols,
        patient
    )

    # ── Plain English ────────────────────────────────────────
    st.markdown("---")

    summary = plain_english_summary(
        shap_vals,
        feature_cols,
        patient,
        selected,
        final_prob
    )

    st.markdown(summary)

    # ── Waterfall chart ──────────────────────────────────────
    st.markdown("---")

    st.markdown(
        '<div style="font-size:1rem;font-weight:600;color:#a78bfa;'
        'border-bottom:1px solid #2d3142;padding-bottom:6px;'
        'margin:0 0 12px">'
        'Full factor breakdown</div>',
        unsafe_allow_html=True
    )

    top_n = st.slider(
        "Number of factors to show",
        5,
        min(20, len(feature_cols)),
        10
    )

    fig = make_shap_chart(
        shap_vals,
        base_val,
        feature_cols,
        patient,
        selected,
        final_prob,
        top_n=top_n
    )

    st.pyplot(fig, use_container_width=True)

    st.caption(
        f"Explainer: {etype}  |  "
        f"Base rate: {base_val*100:.1f}%  |  "
        f"Sum of contributions: {np.sum(shap_vals)*100:+.1f}pp  |  "
        f"Final score: {final_prob:.1f}%"
    )

    # ── Full table ───────────────────────────────────────────
    with st.expander("📋 Full factor table (all features)"):

        pairs = sorted(
            zip(feature_cols, shap_vals),
            key=lambda x: -abs(x[1])
        )

        rows = []

        for feat, val in pairs:

            raw = patient.get(feat, "N/A")

            rows.append({
                "Clinical factor": label(feat),

                "Patient value":
                    f"{raw:.1f}"
                    if isinstance(raw, (int, float))
                    else str(raw),

                "Impact on risk":
                    f"{val*100:+.2f} pp",

                "Direction":
                    "↑ Raises risk"
                    if val > 0
                    else (
                        "↓ Lowers risk"
                        if val < 0
                        else "—"
                    ),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    # ── Clinical note ────────────────────────────────────────
    st.markdown("""
    <div style="background:#1e2235;border-left:3px solid #2d3142;
                border-radius:0 8px 8px 0;padding:10px 14px;
                font-size:0.8rem;color:#888;margin-top:16px">

      <strong>Clinical note:</strong>

      SHAP (SHapley Additive exPlanations) values are computed
      per patient using the trained Random Forest model.

      Each value represents the marginal contribution of that
      feature to the final risk score.

      Values are additive:
      base rate + sum of all SHAP values = final predicted score.

    </div>
    """, unsafe_allow_html=True)