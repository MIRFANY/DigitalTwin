import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(script_dir, '../data/processed/pcos_processed.csv'))
    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} patients, {df.shape[1]} features")
    return df


# ─────────────────────────────────────────
# STEP 2: CREATE RISK LABELS
# These are derived from existing clinical
# features — since our dataset doesn't have
# explicit future outcome columns, we use
# clinically validated thresholds
# ─────────────────────────────────────────
def create_risk_labels(df):
    risks = pd.DataFrame()

    # Risk 1: Insulin Resistance
    # Relaxed: High RBS OR high BMI OR PCOS positive
    risks['insulin_resistance_risk'] = (
        (df['RBS(mg/dl)'] > 100) |
        (df['BMI'] > 25)
    ).astype(int)

    # Risk 2: Menstrual Irregularity
    # Relaxed: irregular cycle OR high follicle count
    risks['menstrual_risk'] = (
        (df['Cycle(R/I)'] == 2) |
        (df['Follicle No. (L)'] + df['Follicle No. (R)'] > 10)
    ).astype(int)

    # Risk 3: Infertility Risk
    # Relaxed: high AMH OR irregular cycle OR PCOS positive
    risks['infertility_risk'] = (
        (df['AMH(ng/mL)'] > 2.5) |
        (df['Cycle(R/I)'] == 2) |
        (df['PCOS (Y/N)'] == 1)
    ).astype(int)

    # Risk 4: Metabolic Risk
    # Relaxed: any BP elevation OR overweight OR high RBS
    risks['metabolic_risk'] = (
        (df['BP _Systolic (mmHg)'] > 120) |
        (df['BP _Diastolic (mmHg)'] > 80) |
        (df['BMI'] > 25)
    ).astype(int)

    # Risk 5: Hyperandrogenism
    # Relaxed: any androgen symptom
    risks['hyperandrogenism_risk'] = (
        (df['hair growth(Y/N)'] == 1) |
        (df['Skin darkening (Y/N)'] == 1) |
        (df['Hair loss(Y/N)'] == 1)
    ).astype(int)

    print(f"\n📊 Risk Label Distribution:")
    for col in risks.columns:
        pct = risks[col].mean() * 100
        print(f"   {col}: {risks[col].sum()} patients at risk ({pct:.1f}%)")

    return risks
# ─────────────────────────────────────────
# STEP 3: PREPARE FEATURES
# ─────────────────────────────────────────
def prepare_features(df):
    drop_cols = ['PCOS (Y/N)']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled = X_scaled.fillna(0)

    return X_scaled, X.columns.tolist()


# ─────────────────────────────────────────
# STEP 4: TRAIN ONE MODEL PER RISK
# ─────────────────────────────────────────
def train_risk_models(X, risk_labels):
    models = {}
    print(f"\n🔄 Training risk prediction models...")
    print("=" * 55)

    for risk_name in risk_labels.columns:
        y = risk_labels[risk_name]

        # Skip if too few positive cases
        if y.sum() < 10:
            print(f"   ⚠️  {risk_name}: too few cases, skipping")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        cv  = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

        models[risk_name] = {
            'model': model,
            'auc':   auc,
            'cv_auc': cv.mean()
        }

        print(f"\n🔹 {risk_name.replace('_', ' ').title()}")
        print(f"   AUC-ROC:    {auc:.3f}")
        print(f"   CV AUC:     {cv.mean():.3f} ± {cv.std():.3f}")

    return models


# ─────────────────────────────────────────
# STEP 5: PREDICT RISK FOR ONE PATIENT
# ─────────────────────────────────────────
def predict_patient_risks(models, patient_data, feature_names):
    print(f"\n👤 Patient Risk Profile:")
    print("=" * 55)

    risk_scores = {}
    for risk_name, model_info in models.items():
        model = model_info['model']

        df_input = pd.DataFrame([patient_data])
        df_input = df_input.reindex(columns=feature_names, fill_value=0)
        df_input = df_input.fillna(0)

        proba = model.predict_proba(df_input)[0][1]
        risk_scores[risk_name] = proba

        if proba >= 0.7:
            level = "🔴 HIGH"
        elif proba >= 0.4:
            level = "🟡 MODERATE"
        else:
            level = "🟢 LOW"

        label = risk_name.replace('_', ' ').title()
        print(f"   {level}  {label}: {proba*100:.1f}%")

    return risk_scores

# ─────────────────────────────────────────
# STEP 6: PLOT RISK DASHBOARD
# ─────────────────────────────────────────
def plot_risk_dashboard(risk_scores):
    labels = [k.replace('_risk', '').replace('_', '\n').title()
              for k in risk_scores.keys()]
    values = [v * 100 for v in risk_scores.values()]
    colors = ['#e74c3c' if v >= 70 else '#f39c12'
              if v >= 40 else '#2ecc71' for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    # Risk zones
    ax.axvline(x=40, color='#f39c12', linestyle='--', alpha=0.5, label='Moderate threshold')
    ax.axvline(x=70, color='#e74c3c', linestyle='--', alpha=0.5, label='High threshold')

    ax.set_xlabel('Risk Probability (%)', fontsize=12)
    ax.set_title('PCOS Patient — Complication Risk Dashboard',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.normpath(os.path.join(script_dir, '../outputs/risk_dashboard.png'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Risk dashboard saved to outputs/risk_dashboard.png")


# ─────────────────────────────────────────
# STEP 7: SAVE ALL RISK MODELS
# ─────────────────────────────────────────
def save_models(models):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(script_dir, '../models/risk_predictor.pkl'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(models, path)
    print(f"✅ Risk models saved to models/risk_predictor.pkl")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   PCOS DIGITAL TWIN — RISK PREDICTOR")
    print("=" * 55)

    # Load
    df = load_data()

    # Create risk labels
    risk_labels = create_risk_labels(df)

    # Prepare features
    X, feature_names = prepare_features(df)

    # Train one model per risk
    models = train_risk_models(X, risk_labels)

    # Test on a sample patient (first patient in dataset)
    sample = {col: float(X.iloc[0][col]) for col in feature_names}
    risk_scores = predict_patient_risks(models, sample, feature_names)

    # Plot dashboard
    plot_risk_dashboard(risk_scores)

    # Save
    save_models(models)

    print("\n" + "=" * 55)
    print("✅ RISK PREDICTOR COMPLETE")
    print("=" * 55)