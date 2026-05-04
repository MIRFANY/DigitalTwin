# PCOS Digital Twin: AI-Powered Health State Estimation & Risk Prediction

A machine learning-based digital twin system for Polycystic Ovary Syndrome (PCOS) that estimates patient health states and predicts clinical risks using advanced ML models and interactive visualization.

## 🎯 Project Overview

This project implements a **Digital Twin** framework for PCOS, enabling:

- **State Estimation**: Real-time assessment of patient health conditions based on clinical features
- **Risk Prediction**: ML-based prediction of PCOS-related health risks (Diabetes, CVD, Infertility)
- **Interactive Dashboard**: Streamlit-based UI for visualization and what-if scenario analysis
- **Simulation**: Temporal dynamics modeling of PCOS progression

The system uses cleaned clinical data and trained machine learning models to provide actionable insights for healthcare professionals.

## ✨ Key Features

- 🏥 **Clinical Data Processing**: Automated data pipeline for preprocessing PCOS patient records
- 🤖 **ML-Based State Estimator**: Random Forest model for patient health state classification
- 📊 **Multi-Risk Predictor**: Separate models for Diabetes, Cardiovascular Disease, and Infertility risks
- 📈 **Interactive Dashboard**: Real-time visualization with scenario simulation capabilities
- 🔍 **What-If Analysis**: Modify patient parameters to see potential health outcomes
- 📉 **Risk Visualization**: Color-coded risk assessment and clinical recommendations
- 🎛️ **Parameter Adjustment**: User-friendly sliders for exploring different health scenarios

## 📋 Requirements

- **Python**: 3.8+
- **Key Dependencies**:
  - pandas >= 2.1.0
  - numpy >= 1.26.0
  - scikit-learn >= 1.3.0
  - streamlit >= 1.28.0
  - matplotlib >= 3.7.0
  - joblib >= 1.3.0
  - shap >= 0.42.0

See `requirements.txt` for complete dependency list.

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
cd pcos_digital_twin
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**On Windows:**

```powershell
venv\Scripts\Activate.ps1
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
pcos_digital_twin/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── raw/                          # Original PCOS dataset
│   └── processed/
│       └── pcos_processed.csv         # Cleaned, preprocessed patient records
│
├── models/                           # Trained ML models
│   ├── state_estimator.pkl           # Health state classification model
│   ├── risk_predictor.pkl            # Risk prediction models
│   └── scaler.pkl                    # Feature standardization scaler
│
├── src/
│   ├── dashboard.py                  # Main Streamlit interactive application
│   ├── digital_twin.py               # Core digital twin logic & integration
│   ├── data_pipeline.py              # Data loading & preprocessing
│   ├── risk_predictor.py             # Risk prediction model training & evaluation
│   ├── state_estimator.py            # Health state model training & evaluation
│   └── simulator.py                  # Temporal dynamics simulation
│
├── notebooks/                        # Jupyter notebooks for analysis & experiments
└── outputs/                          # Generated visualizations & results
```

## 🔧 How to Use

### Launch the Interactive Dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501` and provides:

- **Patient Selection**: Choose a patient from the dataset for analysis
- **Current State Assessment**: View current health state classification
- **Risk Dashboard**: See predicted risks for multiple conditions (Diabetes, CVD, Infertility)
- **What-If Scenarios**: Adjust patient parameters (BMI, hormones, etc.) using sliders
- **Visualizations**: Real-time charts showing health metrics and risk evolution
- **Clinical Recommendations**: Suggestions based on current risk levels

### Run Analysis Scripts

**Train/Evaluate State Estimator:**

```bash
python src/state_estimator.py
```

**Train/Evaluate Risk Predictor:**

```bash
python src/risk_predictor.py
```

**Simulate Patient Progression:**

```bash
python src/simulator.py
```

**Process Raw Data:**

```bash
python src/data_pipeline.py
```

## 📊 Data Format

The processed dataset (`data/processed/pcos_processed.csv`) contains:

- **Rows**: Individual PCOS patients
- **Columns**: Clinical features (hormones, metabolic markers, demographics)
- **Target**: PCOS diagnosis (Y/N)

All features are standardized during model training.

## 🎓 Model Architecture

### State Estimator

- **Algorithm**: Random Forest Classifier
- **Purpose**: Classify current patient health state
- **Input**: 30+ clinical features (standardized)
- **Output**: Health state category (Class 0-3)

### Risk Predictor

- **Algorithm**: Ensemble of Random Forest models
- **Purpose**: Predict probability of clinical complications
- **Risks Predicted**:
  - Type 2 Diabetes
  - Cardiovascular Disease (CVD)
  - Infertility
- **Input**: Clinical features
- **Output**: Risk probability [0, 1] for each condition

## 📈 How It Works

1. **Data Processing**: Raw PCOS patient data is cleaned, normalized, and standardized
2. **Model Loading**: Pre-trained ML models are loaded from the models directory
3. **State Estimation**: Patient clinical features are input to the state estimator
4. **Risk Calculation**: Individual risk models predict probabilities for each complication
5. **Visualization**: Results are displayed interactively with confidence indicators
6. **Scenario Analysis**: Users can modify parameters to simulate different health outcomes

## 🔍 Dashboard Features

### Main Sections

1. **Sidebar Controls**
   - Patient selection (dropdown)
   - Parameter adjustment sliders
   - Scenario comparison

2. **Metrics Display**
   - Current health state
   - Risk scores (Diabetes, CVD, Infertility)
   - Clinical recommendations

3. **Visualizations**
   - Risk bar charts
   - Feature importance plots
   - Scenario comparison graphs
   - Historical trend lines

## 🛠️ Development & Extension

To extend the project:

1. **Add New Models**: Train additional risk predictors and add to `models/`
2. **Enhance Dashboard**: Modify `src/dashboard.py` with new visualizations
3. **Improve Data Pipeline**: Update `src/data_pipeline.py` for additional preprocessing
4. **Add New Features**: Implement new analysis in new Python modules in `src/`

## ⚠️ Disclaimer

This system is designed for **educational and research purposes**. It should not be used as a standalone diagnostic tool without validation by healthcare professionals. Always consult qualified medical practitioners for clinical decision-making.

## 📚 References

- PCOS epidemiology and clinical management literature
- Machine Learning for Healthcare best practices
- Scikit-learn documentation
- Streamlit framework documentation

## 👨‍💼 Author

Academic Project - PCOS Digital Twin Implementation

## 📝 Notes for Professor

- All source code is modular and well-documented
- Models are pre-trained and ready to use
- Dashboard provides interactive demonstration of the system
- Data pipeline can be extended with additional datasets
- Fully reproducible: run `streamlit run src/dashboard.py` to see live demo

---

**Last Updated**: May 2026
