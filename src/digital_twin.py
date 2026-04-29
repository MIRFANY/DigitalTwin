import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════
# PCOS DIGITAL TWIN — COMPLETE INTEGRATED SYSTEM
# ═══════════════════════════════════════════════════


# ─────────────────────────────────────────
# 1. LOAD EVERYTHING
# ─────────────────────────────────────────
def load_all(patient_index=2):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load models
    state_model = joblib.load(os.path.normpath(
        os.path.join(script_dir, '../models/state_estimator.pkl')))
    risk_models = joblib.load(os.path.normpath(
        os.path.join(script_dir, '../models/risk_predictor.pkl')))

    # Load processed data
    path = os.path.normpath(
        os.path.join(script_dir, '../data/processed/pcos_processed.csv'))
    df = pd.read_csv(path)

    target = 'PCOS (Y/N)'
    X = df.drop(columns=[target])
    y = df[target]
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled = X_scaled.fillna(0)

    feature_names    = X.columns.tolist()
    patient_original = X.iloc[patient_index].to_dict()
    patient_scaled   = X_scaled.iloc[patient_index].to_dict()
    actual_label     = y.iloc[patient_index]

    return (state_model, risk_models, scaler,
            feature_names, patient_original,
            patient_scaled, actual_label, patient_index)


# ─────────────────────────────────────────
# 2. STATE ESTIMATION
# ─────────────────────────────────────────
def run_state_estimator(state_model, patient_scaled, feature_names):
    model    = state_model['model']
    df_input = pd.DataFrame([patient_scaled])
    df_input = df_input.reindex(columns=feature_names, fill_value=0).fillna(0)

    proba      = model.predict_proba(df_input)[0][1]
    prediction = 1 if proba >= 0.5 else 0

    if proba >= 0.7:   severity = "Severe"
    elif proba >= 0.5: severity = "Moderate"
    elif proba >= 0.3: severity = "Mild Risk"
    else:              severity = "Low Risk"

    return {
        'prediction': prediction,
        'probability': proba,
        'severity': severity,
        'label': "PCOS Positive ⚠️" if prediction == 1 else "PCOS Negative ✅"
    }


# ─────────────────────────────────────────
# 3. RISK PREDICTION
# ─────────────────────────────────────────
def run_risk_predictor(risk_models, patient_scaled, feature_names):
    scores = {}
    for risk_name, model_info in risk_models.items():
        model    = model_info['model']
        df_input = pd.DataFrame([patient_scaled])
        df_input = df_input.reindex(
            columns=feature_names, fill_value=0).fillna(0)
        proba = model.predict_proba(df_input)[0][1]
        scores[risk_name] = proba
    return scores


# ─────────────────────────────────────────
# 4. WHAT-IF SIMULATION
# ─────────────────────────────────────────
def apply_scenario(patient_original, scaler, feature_names, scenario):
    modified = patient_original.copy()

    if scenario == 'weight_loss':
        w = max(modified.get('Weight (Kg)', 70) - 5, 45)
        h = modified.get('Height(Cm)', 160) / 100
        modified['Weight (Kg)'] = w
        modified['BMI']         = w / (h ** 2)

    elif scenario == 'exercise':
        modified['Reg.Exercise(Y/N)']    = 1
        modified['BP _Systolic (mmHg)']  = max(modified.get('BP _Systolic (mmHg)', 120)  - 8,  90)
        modified['BP _Diastolic (mmHg)'] = max(modified.get('BP _Diastolic (mmHg)', 80)  - 5,  60)
        modified['RBS(mg/dl)']           = max(modified.get('RBS(mg/dl)', 100) - 15, 70)
        modified['Cycle(R/I)']           = 1

    elif scenario == 'diet':
        modified['Fast food (Y/N)'] = 0
        modified['RBS(mg/dl)']      = max(modified.get('RBS(mg/dl)', 100) - 10, 70)
        w = max(modified.get('Weight (Kg)', 70) - 2, 45)
        h = modified.get('Height(Cm)', 160) / 100
        modified['Weight (Kg)'] = w
        modified['BMI']         = w / (h ** 2)

    elif scenario == 'medication':
        modified['RBS(mg/dl)']  = max(modified.get('RBS(mg/dl)', 100)  - 20, 70)
        modified['Cycle(R/I)']  = 1
        modified['AMH(ng/mL)']  = max(modified.get('AMH(ng/mL)', 3)   - 1,  0.5)
        modified['LH(mIU/mL)']  = max(modified.get('LH(mIU/mL)', 10)  - 3,  1)

    elif scenario == 'combined':
        w = max(modified.get('Weight (Kg)', 70) - 7, 45)
        h = modified.get('Height(Cm)', 160) / 100
        modified['Weight (Kg)']          = w
        modified['BMI']                  = w / (h ** 2)
        modified['Reg.Exercise(Y/N)']    = 1
        modified['Fast food (Y/N)']      = 0
        modified['RBS(mg/dl)']           = max(modified.get('RBS(mg/dl)', 100) - 25, 70)
        modified['BP _Systolic (mmHg)']  = max(modified.get('BP _Systolic (mmHg)', 120) - 10, 90)
        modified['BP _Diastolic (mmHg)'] = max(modified.get('BP _Diastolic (mmHg)', 80)  - 7,  60)
        modified['Cycle(R/I)']           = 1
        modified['AMH(ng/mL)']           = max(modified.get('AMH(ng/mL)', 3)   - 1.5, 0.5)
        modified['LH(mIU/mL)']           = max(modified.get('LH(mIU/mL)', 10)  - 4,   1)

    df_mod     = pd.DataFrame([modified])
    df_mod     = df_mod.reindex(columns=feature_names, fill_value=0).fillna(0)
    arr_scaled = scaler.transform(df_mod)
    return dict(zip(feature_names, arr_scaled[0]))


def run_simulator(patient_original, patient_scaled,
                  risk_models, scaler, feature_names):
    scenarios = {
        'Current':   patient_scaled,
        'Weight\nLoss':  apply_scenario(patient_original, scaler, feature_names, 'weight_loss'),
        'Exercise':  apply_scenario(patient_original, scaler, feature_names, 'exercise'),
        'Diet':      apply_scenario(patient_original, scaler, feature_names, 'diet'),
        'Medication':apply_scenario(patient_original, scaler, feature_names, 'medication'),
        'Combined':  apply_scenario(patient_original, scaler, feature_names, 'combined'),
    }
    results = {}
    for name, data in scenarios.items():
        results[name] = run_risk_predictor(risk_models, data, feature_names)
    return results


# ─────────────────────────────────────────
# 5. GENERATE FULL REPORT (VISUAL)
# ─────────────────────────────────────────
def generate_report(patient_index, state_result,
                    risk_scores, sim_results, actual_label):

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.5, wspace=0.4)

    BLUE   = '#3498db'
    GREEN  = '#2ecc71'
    RED    = '#e74c3c'
    ORANGE = '#f39c12'
    WHITE  = '#ffffff'
    GRAY   = '#888888'
    BG     = '#1a1d27'

    # ── Panel 1: Diagnosis Card ──────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(BG)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
    ax0.axis('off')

    color = RED if state_result['prediction'] == 1 else GREEN
    ax0.text(0.5, 0.92, f"Patient #{patient_index}",
             ha='center', va='top', color=GRAY,
             fontsize=11, transform=ax0.transAxes)
    ax0.text(0.5, 0.75, state_result['label'],
             ha='center', va='top', color=color,
             fontsize=14, fontweight='bold',
             transform=ax0.transAxes)
    ax0.text(0.5, 0.54,
             f"Confidence: {state_result['probability']*100:.1f}%",
             ha='center', va='top', color=WHITE,
             fontsize=12, transform=ax0.transAxes)
    ax0.text(0.5, 0.36,
             f"Severity: {state_result['severity']}",
             ha='center', va='top', color=ORANGE,
             fontsize=12, fontweight='bold',
             transform=ax0.transAxes)
    actual_txt = "Actual: Positive ⚠️" if actual_label == 1 else "Actual: Negative ✅"
    ax0.text(0.5, 0.15, actual_txt,
             ha='center', va='top', color=GRAY,
             fontsize=10, transform=ax0.transAxes)
    ax0.set_title('🩺 DIAGNOSIS', color=WHITE,
                  fontsize=12, fontweight='bold', pad=8)
    for spine in ax0.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)

    # ── Panel 2: Risk Gauge ──────────────────
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.set_facecolor(BG)
    ax1.axis('off')
    ax1.set_title('⚠️ COMPLICATION RISK PROFILE',
                  color=WHITE, fontsize=12,
                  fontweight='bold', pad=8)

    risk_labels = [r.replace('_risk','').replace('_',' ').title()
                   for r in risk_scores]
    risk_vals   = [v * 100 for v in risk_scores.values()]
    bar_colors  = [RED if v >= 70 else ORANGE
                   if v >= 40 else GREEN for v in risk_vals]

    y_pos = np.arange(len(risk_labels))
    bars  = ax1.barh(y_pos, risk_vals, color=bar_colors,
                     height=0.5, left=0)
    ax1.set_xlim(0, 115)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(risk_labels, color=WHITE, fontsize=11)
    ax1.axvline(70, color=RED,    linestyle='--', alpha=0.5)
    ax1.axvline(40, color=ORANGE, linestyle='--', alpha=0.5)
    ax1.tick_params(colors=GRAY)
    for bar, val in zip(bars, risk_vals):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}%', va='center',
                 color=WHITE, fontsize=11, fontweight='bold')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')

    # ── Panels 3-7: Scenario charts ──────────
    risk_keys = list(risk_scores.keys())
    scen_names = list(sim_results.keys())
    panel_colors = [BLUE, GREEN, ORANGE, '#9b59b6', RED]

    for idx, risk in enumerate(risk_keys):
        row = 1 + idx // 3
        col = idx % 3
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(BG)

        vals  = [sim_results[s][risk] * 100 for s in scen_names]
        bars2 = ax.bar(range(len(scen_names)), vals,
                       color=panel_colors[idx],
                       alpha=0.85, edgecolor='white',
                       linewidth=0.5)
        ax.set_title(risk.replace('_risk','').replace('_',' ').title(),
                     color=WHITE, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(scen_names)))
        ax.set_xticklabels(scen_names, fontsize=7,
                           color=GRAY, rotation=15)
        ax.set_ylim(0, 115)
        ax.set_ylabel('%', color=GRAY, fontsize=9)
        ax.axhline(70, color=RED,    linestyle='--',
                   alpha=0.4, linewidth=1)
        ax.axhline(40, color=ORANGE, linestyle='--',
                   alpha=0.4, linewidth=1)
        ax.tick_params(colors=GRAY)
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        for bar, val in zip(bars2, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{val:.0f}', ha='center',
                    color=WHITE, fontsize=7)

    # ── Main Title ───────────────────────────
    fig.suptitle('PCOS Digital Twin — Complete Patient Report',
                 fontsize=16, fontweight='bold',
                 color=WHITE, y=0.98)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.normpath(os.path.join(
        script_dir, f'../outputs/dt_report_patient{patient_index}.png'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Full report saved → outputs/dt_report_patient{patient_index}.png")
    return out_path


# ─────────────────────────────────────────
# 6. PRINT TEXT REPORT
# ─────────────────────────────────────────
def print_report(patient_index, state_result,
                 risk_scores, sim_results):
    print("\n" + "╔" + "═"*53 + "╗")
    print(f"║{'PCOS DIGITAL TWIN — PATIENT REPORT':^53}║")
    print(f"║{'Patient #' + str(patient_index):^53}║")
    print("╠" + "═"*53 + "╣")

    # Diagnosis
    print(f"║  🩺 DIAGNOSIS                                       ║")
    print(f"║     Result:     {state_result['label']:<36}║")
    print(f"║     Confidence: {state_result['probability']*100:.1f}%{'':<36}║")
    print(f"║     Severity:   {state_result['severity']:<36}║")
    print("╠" + "═"*53 + "╣")

    # Risks
    print(f"║  ⚠️  COMPLICATION RISKS                             ║")
    for risk, score in risk_scores.items():
        label = risk.replace('_risk','').replace('_',' ').title()
        pct   = score * 100
        level = "🔴" if pct >= 70 else "🟡" if pct >= 40 else "🟢"
        print(f"║     {level} {label:<28} {pct:5.1f}%      ║")
    print("╠" + "═"*53 + "╣")

    # Best intervention
    print(f"║  💊 INTERVENTION IMPACT (Combined vs Current)      ║")
    current  = sim_results['Current']
    combined = sim_results['Combined']
    for risk in current:
        label    = risk.replace('_risk','').replace('_',' ').title()
        before   = current[risk]  * 100
        after    = combined[risk] * 100
        change   = before - after
        arrow    = "↓" if change > 0 else "→"
        print(f"║     {label:<22} {before:.0f}% {arrow} {after:.0f}% "
              f"({change:+.0f}%){'':>3}║")
    print("╚" + "═"*53 + "╝")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--patient', type=int, default=2,
                        help='Patient index to analyze')
    args = parser.parse_args()

    print("═" * 55)
    print("   PCOS DIGITAL TWIN — INTEGRATED SYSTEM")
    print("═" * 55)

    # Load everything
    (state_model, risk_models, scaler,
     feature_names, patient_original,
     patient_scaled, actual_label,
     patient_index) = load_all(args.patient)

    print(f"\n✅ Patient #{patient_index} loaded")
    print(f"   Actual PCOS Status: "
          f"{'Positive ⚠️' if actual_label == 1 else 'Negative ✅'}")

    # Run all modules
    print("\n🔄 Running Digital Twin analysis...")

    state_result = run_state_estimator(
        state_model, patient_scaled, feature_names)

    risk_scores  = run_risk_predictor(
        risk_models, patient_scaled, feature_names)

    sim_results  = run_simulator(
        patient_original, patient_scaled,
        risk_models, scaler, feature_names)

    # Print text report
    print_report(patient_index, state_result,
                 risk_scores, sim_results)

    # Generate visual report
    generate_report(patient_index, state_result,
                    risk_scores, sim_results, actual_label)

    print("\n" + "═" * 55)
    print("✅ DIGITAL TWIN ANALYSIS COMPLETE")
    print("═" * 55)