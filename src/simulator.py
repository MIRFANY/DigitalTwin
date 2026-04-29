import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# STEP 1: LOAD SAVED MODELS
# ─────────────────────────────────────────
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    state_path = os.path.normpath(os.path.join(script_dir, '../models/state_estimator.pkl'))
    risk_path  = os.path.normpath(os.path.join(script_dir, '../models/risk_predictor.pkl'))

    state_model = joblib.load(state_path)
    risk_models = joblib.load(risk_path)

    print("✅ Models loaded successfully")
    return state_model, risk_models


# ─────────────────────────────────────────
# STEP 2: LOAD A SAMPLE PATIENT
# ─────────────────────────────────────────
def load_sample_patient(index=0):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(script_dir, '../data/processed/pcos_processed.csv'))
    df   = pd.read_csv(path)

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    target = 'PCOS (Y/N)'
    X = df.drop(columns=[target])
    y = df[target]
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled = X_scaled.fillna(0)

    # Store original (unscaled) values for modification
    patient_original = X.iloc[index].to_dict()
    patient_scaled   = X_scaled.iloc[index].to_dict()
    feature_names    = X.columns.tolist()

    # Store scaler params for re-scaling modified values
    scaler_mean = scaler.mean_
    scaler_std  = scaler.scale_

    print(f"✅ Patient {index} loaded")
    print(f"   PCOS Status: {'Positive ⚠️' if y.iloc[index] == 1 else 'Negative ✅'}")

    return patient_original, patient_scaled, feature_names, scaler, y.iloc[index]


# ─────────────────────────────────────────
# STEP 3: GET CURRENT RISK SCORES
# ─────────────────────────────────────────
def get_risk_scores(risk_models, patient_scaled, feature_names):
    scores = {}
    for risk_name, model_info in risk_models.items():
        model    = model_info['model']
        df_input = pd.DataFrame([patient_scaled])
        df_input = df_input.reindex(columns=feature_names, fill_value=0).fillna(0)
        proba    = model.predict_proba(df_input)[0][1]
        scores[risk_name] = proba
    return scores


# ─────────────────────────────────────────
# STEP 4: DEFINE SCENARIOS
# Each scenario modifies specific features
# to simulate lifestyle/treatment changes
# ─────────────────────────────────────────
def apply_scenario(patient_original, scaler, feature_names, scenario_name):
    modified = patient_original.copy()

    if scenario_name == "weight_loss_5kg":
        # Simulate losing 5kg
        current_weight = modified.get('Weight (Kg)', 70)
        current_height = modified.get('Height(Cm)', 160) / 100
        new_weight     = max(current_weight - 5, 45)
        modified['Weight (Kg)'] = new_weight
        modified['BMI']         = new_weight / (current_height ** 2)

    elif scenario_name == "regular_exercise":
        # Exercise improves cycle regularity,
        # reduces BP and RBS
        modified['Reg.Exercise(Y/N)']  = 1
        modified['BP _Systolic (mmHg)'] = max(
            modified.get('BP _Systolic (mmHg)', 120) - 8, 90)
        modified['BP _Diastolic (mmHg)'] = max(
            modified.get('BP _Diastolic (mmHg)', 80) - 5, 60)
        modified['RBS(mg/dl)'] = max(
            modified.get('RBS(mg/dl)', 100) - 15, 70)
        modified['Cycle(R/I)'] = 1   # cycles become regular

    elif scenario_name == "healthy_diet":
        # Diet reduces RBS, BMI, symptoms
        modified['Fast food (Y/N)'] = 0
        modified['RBS(mg/dl)']      = max(
            modified.get('RBS(mg/dl)', 100) - 10, 70)
        current_weight = modified.get('Weight (Kg)', 70)
        current_height = modified.get('Height(Cm)', 160) / 100
        new_weight     = max(current_weight - 2, 45)
        modified['Weight (Kg)'] = new_weight
        modified['BMI']         = new_weight / (current_height ** 2)

    elif scenario_name == "medication":
        # Medication (e.g. Metformin) improves
        # insulin resistance and cycle regularity
        modified['RBS(mg/dl)']    = max(
            modified.get('RBS(mg/dl)', 100) - 20, 70)
        modified['Cycle(R/I)']    = 1
        modified['AMH(ng/mL)']    = max(
            modified.get('AMH(ng/mL)', 3) - 1, 0.5)
        modified['LH(mIU/mL)']   = max(
            modified.get('LH(mIU/mL)', 10) - 3, 1)

    elif scenario_name == "combined":
        # All interventions combined
        current_weight = modified.get('Weight (Kg)', 70)
        current_height = modified.get('Height(Cm)', 160) / 100
        new_weight     = max(current_weight - 7, 45)
        modified['Weight (Kg)']          = new_weight
        modified['BMI']                  = new_weight / (current_height ** 2)
        modified['Reg.Exercise(Y/N)']    = 1
        modified['Fast food (Y/N)']      = 0
        modified['RBS(mg/dl)']           = max(
            modified.get('RBS(mg/dl)', 100) - 25, 70)
        modified['BP _Systolic (mmHg)']  = max(
            modified.get('BP _Systolic (mmHg)', 120) - 10, 90)
        modified['BP _Diastolic (mmHg)'] = max(
            modified.get('BP _Diastolic (mmHg)', 80) - 7, 60)
        modified['Cycle(R/I)']           = 1
        modified['AMH(ng/mL)']           = max(
            modified.get('AMH(ng/mL)', 3) - 1.5, 0.5)
        modified['LH(mIU/mL)']          = max(
            modified.get('LH(mIU/mL)', 10) - 4, 1)

    # Re-scale the modified patient
    df_mod    = pd.DataFrame([modified])
    df_mod    = df_mod.reindex(columns=feature_names, fill_value=0).fillna(0)
    arr_scaled = scaler.transform(df_mod)
    scaled_dict = dict(zip(feature_names,
                           arr_scaled[0]))

    return scaled_dict


# ─────────────────────────────────────────
# STEP 5: RUN ALL SCENARIOS
# ─────────────────────────────────────────
def run_all_scenarios(patient_original, patient_scaled,
                      risk_models, scaler, feature_names):

    scenarios = {
        'Current State':    patient_scaled,
        'Weight Loss 5kg':  apply_scenario(patient_original, scaler,
                                           feature_names, 'weight_loss_5kg'),
        'Regular Exercise': apply_scenario(patient_original, scaler,
                                           feature_names, 'regular_exercise'),
        'Healthy Diet':     apply_scenario(patient_original, scaler,
                                           feature_names, 'healthy_diet'),
        'Medication':       apply_scenario(patient_original, scaler,
                                           feature_names, 'medication'),
        'All Combined':     apply_scenario(patient_original, scaler,
                                           feature_names, 'combined'),
    }

    results = {}
    print("\n🔄 Running simulations...")
    print("=" * 55)

    for scenario_name, patient_data in scenarios.items():
        scores = get_risk_scores(risk_models, patient_data, feature_names)
        results[scenario_name] = scores

        print(f"\n📍 {scenario_name}")
        for risk, score in scores.items():
            label = risk.replace('_risk', '').replace('_', ' ').title()
            bar   = '█' * int(score * 20)
            print(f"   {label:<25} {score*100:5.1f}%  {bar}")

    return results


# ─────────────────────────────────────────
# STEP 6: PLOT COMPARISON CHART
# ─────────────────────────────────────────
def plot_scenario_comparison(results):
    scenarios  = list(results.keys())
    risk_names = list(list(results.values())[0].keys())

    fig, axes = plt.subplots(1, len(risk_names),
                             figsize=(18, 6), sharey=False)

    colors = ['#3498db', '#2ecc71', '#e67e22',
              '#9b59b6', '#e74c3c', '#1abc9c']

    for i, risk in enumerate(risk_names):
        ax     = axes[i]
        values = [results[s][risk] * 100 for s in scenarios]
        bars   = ax.bar(range(len(scenarios)), values,
                        color=colors, edgecolor='white', width=0.6)

        ax.set_title(risk.replace('_risk', '').replace('_', '\n').title(),
                     fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.replace(' ', '\n')
                            for s in scenarios], fontsize=7)
        ax.set_ylim(0, 110)
        ax.set_ylabel('Risk %' if i == 0 else '')
        ax.axhline(y=70, color='red',    linestyle='--',
                   alpha=0.4, linewidth=1)
        ax.axhline(y=40, color='orange', linestyle='--',
                   alpha=0.4, linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{val:.0f}%', ha='center',
                    fontsize=7, fontweight='bold')

    plt.suptitle('PCOS Digital Twin — What-If Scenario Simulation\n'
                 'How Different Interventions Affect Complication Risks',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.normpath(os.path.join(
        script_dir, '../outputs/scenario_simulation.png'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Simulation chart saved to outputs/scenario_simulation.png")


# ─────────────────────────────────────────
# STEP 7: PRINT RECOMMENDATION
# ─────────────────────────────────────────
def print_recommendation(results):
    print("\n" + "=" * 55)
    print("💊 PERSONALIZED RECOMMENDATIONS")
    print("=" * 55)

    current   = results['Current State']
    combined  = results['All Combined']

    for risk in current:
        label    = risk.replace('_risk', '').replace('_', ' ').title()
        before   = current[risk]  * 100
        after    = combined[risk] * 100
        change   = before - after

        if before >= 70:
            urgency = "🔴 URGENT"
        elif before >= 40:
            urgency = "🟡 MONITOR"
        else:
            urgency = "🟢 HEALTHY"

        print(f"\n{urgency} — {label}")
        print(f"   Current risk:      {before:.1f}%")
        print(f"   After intervention:{after:.1f}%")
        print(f"   Potential reduction:{change:.1f}%")

    print("\n📋 Best Single Intervention:")
    scenario_totals = {}
    for scenario, scores in results.items():
        if scenario == 'Current State':
            continue
        total = sum(scores.values())
        scenario_totals[scenario] = total

    best = min(scenario_totals, key=scenario_totals.get)
    print(f"   → {best} reduces overall risk the most")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   PCOS DIGITAL TWIN — WHAT-IF SIMULATOR")
    print("=" * 55)

    # Load models
    state_model, risk_models = load_models()

    # Load patient
    patient_original, patient_scaled, feature_names, scaler, pcos_status = \
        load_sample_patient(index=2)

    # Run all scenarios
    results = run_all_scenarios(
        patient_original, patient_scaled,
        risk_models, scaler, feature_names)

    # Plot comparison
    plot_scenario_comparison(results)

    # Print recommendations
    print_recommendation(results)

    print("\n" + "=" * 55)
    print("✅ SIMULATOR COMPLETE")
    print("=" * 55)