import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
def load_data(path):
    xl = pd.ExcelFile(path)
    print(f"📋 Sheets found: {xl.sheet_names}")
    
    # 'Full_new' is the actual data sheet
    df = pd.read_excel(path, sheet_name='Full_new')
    
    df.columns = df.columns.str.strip()
    
    print(f"✅ Data loaded: {df.shape[0]} patients, {df.shape[1]} features")
    return df
# ─────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────
def clean_data(df):
    
    # 2a. Drop irrelevant columns
    drop_cols = ['Sl. No', 'Patient File No.', 'Unnamed: 44']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 2b. Fix data types
    # Some numeric columns may be read as strings
    df['AMH(ng/mL)']      = pd.to_numeric(df['AMH(ng/mL)'],      errors='coerce')
    df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(
        df['II    beta-HCG(mIU/mL)'], errors='coerce')

    # 2c. Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"   ↳ Filled {missing} missing values in '{col}' with median")

    # 2d. Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after  = len(df)
    if before != after:
        print(f"   ↳ Removed {before - after} duplicate rows")

    print(f"✅ Data cleaned: {df.shape[0]} patients, {df.shape[1]} features")
    return df


# ─────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):

    # 3a. LH:FSH ratio — classic PCOS indicator (>2 strongly suggests PCOS)
    df['LH_FSH_ratio'] = df['LH(mIU/mL)'] / (df['FSH(mIU/mL)'] + 1e-5)

    # 3b. Waist:Hip ratio already exists, but let's flag high risk
    # >0.85 indicates central obesity — PCOS risk factor
    df['High_WaistHip'] = (df['Waist:Hip Ratio'] > 0.85).astype(int)

    # 3c. Beta-HCG ratio
    df['betaHCG_I_II_ratio'] = df['I   beta-HCG(mIU/mL)'] / (df['II    beta-HCG(mIU/mL)'] + 1e-5)

    # 3d. BMI category
    def bmi_category(bmi):
        if bmi < 18.5: return 0   # Underweight
        elif bmi < 25: return 1   # Normal
        elif bmi < 30: return 2   # Overweight
        else:          return 3   # Obese
    df['BMI_category'] = df['BMI'].apply(bmi_category)

    # 3e. Symptom burden score (count of symptoms)
    symptom_cols = [
        'Weight gain(Y/N)', 'hair growth(Y/N)', 'Skin darkening (Y/N)',
        'Hair loss(Y/N)', 'Pimples(Y/N)', 'Fast food (Y/N)'
    ]
    existing = [c for c in symptom_cols if c in df.columns]
    df['Symptom_burden'] = df[existing].sum(axis=1)

    # 3f. AMH flag (>3.5 strongly associated with PCOS)
    df['High_AMH'] = (df['AMH(ng/mL)'] > 3.5).astype(int)

    # 3g. Log AMH (fixes skewed distribution)
    df['log_AMH'] = np.log1p(df['AMH(ng/mL)'])

    # 3h. Follicle count total (left + right ovary)
    df['Total_follicles'] = df['Follicle No. (L)'] + df['Follicle No. (R)']

    # 3i. High follicle flag (>12 total is a PCOS diagnostic criterion)
    df['High_follicle'] = (df['Total_follicles'] > 12).astype(int)

    # 3j. RBS flag (>140 mg/dl suggests impaired glucose — insulin resistance)
    df['High_RBS'] = (df['RBS(mg/dl)'] > 140).astype(int)

    print(f"✅ Features engineered: {df.shape[1]} total features now")
    return df
# ─────────────────────────────────────────
# STEP 4: PREPARE FOR MODELING
# ─────────────────────────────────────────
def prepare_for_modeling(df):

    # 4a. Separate target variable
    target = 'PCOS (Y/N)'
    X = df.drop(columns=[target])
    y = df[target]

    print(f"\n📊 Class Distribution:")
    print(f"   PCOS Positive: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   PCOS Negative: {(y==0).sum()} ({(1-y.mean())*100:.1f}%)")

    # 4b. Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # 4c. Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )


    import joblib
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.normpath(os.path.join(script_dir, '../models/scaler.pkl'))
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")



    # 4d. Train/Test Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y        # maintains class balance in both splits
    )

    print(f"\n✅ Data split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()


# ─────────────────────────────────────────
# STEP 5: SAVE PROCESSED DATA
# ─────────────────────────────────────────
def save_processed(df):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(script_dir, '../data/processed/pcos_processed.csv'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Processed data saved to {path}")


# ─────────────────────────────────────────
# MAIN — RUN THE FULL PIPELINE
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   PCOS DIGITAL TWIN — DATA PIPELINE")
    print("=" * 50)

    # Run pipeline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../data/raw/PCOS_data_without_infertility.xlsx')
    



    df = load_data(data_path)
    df = clean_data(df)



    df = engineer_features(df)
    save_processed(df)

    X_train, X_test, y_train, y_test, scaler, features = prepare_for_modeling(df)

    print("\n" + "=" * 50)
    print("✅ PIPELINE COMPLETE — Ready for modeling")
    print("=" * 50)