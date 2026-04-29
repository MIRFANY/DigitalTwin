import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────
# STEP 1: LOAD PROCESSED DATA
# ─────────────────────────────────────────
def load_processed_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(script_dir, '../data/processed/pcos_processed.csv'))
    df = pd.read_csv(path)

    target = 'PCOS (Y/N)'
    X = df.drop(columns=[target])
    y = df[target]

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled = X_scaled.fillna(X_scaled.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print(f"✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────
# STEP 2: TRAIN ALL 3 MODELS
# ─────────────────────────────────────────
def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=2, eval_metric='logloss', verbosity=0),
        'Logistic Regression': LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=1000)
    }

    trained = {}
    print("\n🔄 Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        trained[name] = model
        print(f"   {name}: CV AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return trained


# ─────────────────────────────────────────
# STEP 3: EVALUATE ALL MODELS
# ─────────────────────────────────────────
def evaluate_models(trained_models, X_test, y_test):
    print("\n📊 Model Evaluation on Test Set:")
    print("=" * 55)

    results = {}
    for name, model in trained_models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        report  = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'model':       model,
            'auc':         auc,
            'sensitivity': report['1']['recall'],
            'specificity': report['0']['recall'],
            'accuracy':    report['accuracy'],
            'y_proba':     y_proba
        }

        print(f"\n🔹 {name}")
        print(f"   AUC-ROC:     {auc:.3f}")
        print(f"   Accuracy:    {report['accuracy']:.3f}")
        print(f"   Sensitivity: {report['1']['recall']:.3f}  (catching PCOS cases)")
        print(f"   Specificity: {report['0']['recall']:.3f}  (avoiding false alarms)")

    return results


# ─────────────────────────────────────────
# STEP 4: PICK BEST MODEL
# ─────────────────────────────────────────
def pick_best_model(results):
    # Best = highest AUC (best overall discrimination)
    best_name = max(results, key=lambda x: results[x]['auc'])
    best      = results[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   AUC-ROC:     {best['auc']:.3f}")
    print(f"   Sensitivity: {best['sensitivity']:.3f}")
    print(f"   Specificity: {best['specificity']:.3f}")

    return best_name, best['model']


# ─────────────────────────────────────────
# STEP 5: SHAP EXPLAINABILITY
# ─────────────────────────────────────────
def explain_model(model, X_test, feature_names, model_name):
    print(f"\n🔍 Generating SHAP explanations for {model_name}...")

    # Use TreeExplainer for tree models, LinearExplainer for others
    if 'Forest' in model_name or 'XGB' in model_name or 'Boost' in model_name:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values
    else:
        # LinearExplainer for Logistic Regression
        explainer   = shap.LinearExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)
        sv = shap_values

    # Plot top 15 most important features
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_test, feature_names=feature_names,
                      max_display=15, show=False, plot_type='bar')
    plt.title(f'Top 15 Features Driving PCOS Prediction\n({model_name})',
              fontsize=13, fontweight='bold')
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.normpath(os.path.join(script_dir, '../outputs/shap_importance.png'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP plot saved to outputs/shap_importance.png")
# ─────────────────────────────────────────
# STEP 6: SAVE BEST MODEL
# ─────────────────────────────────────────
def save_model(model, model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(script_dir, '../models/state_estimator.pkl'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'name': model_name}, path)
    print(f"✅ Best model saved to models/state_estimator.pkl")


# ─────────────────────────────────────────
# STEP 7: PREDICT NEW PATIENT
# ─────────────────────────────────────────
def predict_patient(model, scaler_cols, sample_input):
    """
    Give the model a single patient's data and get PCOS prediction.
    sample_input: dict of feature values
    """
    df_input = pd.DataFrame([sample_input])
    df_input = df_input.reindex(columns=scaler_cols, fill_value=0)

    proba      = model.predict_proba(df_input)[0][1]
    prediction = "PCOS Positive ⚠️" if proba >= 0.5 else "PCOS Negative ✅"

    print(f"\n👤 Patient Prediction:")
    print(f"   Result:      {prediction}")
    print(f"   Confidence:  {proba*100:.1f}%")

    if proba >= 0.7:
        severity = "Severe"
    elif proba >= 0.5:
        severity = "Moderate"
    elif proba >= 0.3:
        severity = "Mild Risk"
    else:
        severity = "Low Risk"

    print(f"   Severity:    {severity}")
    return prediction, proba


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   PCOS DIGITAL TWIN — STATE ESTIMATOR")
    print("=" * 55)

    # Load data
    X_train, X_test, y_train, y_test, features = load_processed_data()

    # Train
    trained_models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate_models(trained_models, X_test, y_test)

    # Pick best
    best_name, best_model = pick_best_model(results)

    # Explain
    explain_model(best_model, X_test, features, best_name)

    # Save
    save_model(best_model, best_name)

    # Test on a sample patient
    sample_patient = {col: float(X_test.iloc[0][col]) for col in features}
    predict_patient(best_model, features, sample_patient)

    print("\n" + "=" * 55)
    print("✅ STATE ESTIMATOR COMPLETE")
    print("=" * 55)