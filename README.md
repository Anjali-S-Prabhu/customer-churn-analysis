# Customer Churn Analysis & Prediction

An end-to-end Machine Learning classification system designed to analyze customer behavior, identify key factors associated with customer attrition, and predict customer churn risk using scikit-learn pipelines.

---

## 1. Project Title
**CUSTOMER CHURN ANALYSIS & PREDICTION SYSTEM**  
*Academic & Engineering Portfolio Project (BE / B.Tech / AIML)*

---

## 2. Project Overview
Customer attrition (or churn) represents the loss of subscribers or customers who discontinue paying for a company's services. In subscription-based businesses such as telecommunications, SaaS, and utilities, retaining an existing customer is estimated to be 5 to 7 times less expensive than acquiring a new one.

This project delivers a complete, production-ready machine learning solution:
- Comprehensive exploratory data analysis (EDA) across behavioral, demographic, and billing dimensions.
- Robust data cleaning and leakage-free preprocessing with scikit-learn `ColumnTransformer`.
- Comparative training of multiple classification models: **Logistic Regression**, **Decision Tree**, and **Random Forest**.
- In-depth model evaluation emphasizing business-critical metrics (**Recall**, **Precision**, **F1-Score**, **ROC-AUC**) over raw accuracy.
- Visualizations including Confusion Matrix heatmaps, ROC curves, and Feature Importance / Coefficient plots.
- Model persistence and a standalone, reusable inference component for scoring new customer profiles.

---

## 3. Problem Statement
Telecommunications service providers face high annual turnover as customers switch to competitors offering promotional rates or perceived superior value. Business managers often lack early-warning indicators to identify which subscribers are at imminent risk of leaving. Without predictive insights, retention campaigns are distributed inefficiently or triggered after the customer has already decided to defect. 

The core challenge is to construct a predictive classification model that:
1. Accurately segments high-risk vs. low-risk customers before churn occurs.
2. Identifies the primary observable factors associated with churn.
3. Provides calibrated churn probabilities to inform personalized retention strategies.

---

## 4. Objectives
- **Data Ingestion & Verification:** Automatically acquire, validate, and inspect real-world customer churn data.
- **Data Cleaning & Quality Assurance:** Resolve incorrect data types (e.g., whitespace strings in financial columns), impute missing values rationally, and remove duplicate profiles.
- **Exploratory Data Analysis (EDA):** Quantify churn rates across contract types, tenure stages, monthly pricing tiers, payment methods, and service subscriptions.
- **Data Pipeline Engineering:** Build a reproducible preprocessing pipeline using `StandardScaler` for numerical attributes and `OneHotEncoder` for categorical attributes.
- **Model Training & Benchmarking:** Train Logistic Regression, Decision Tree, and Random Forest classifiers with stratified cross-validation.
- **Multi-Metric Evaluation:** Evaluate models using Confusion Matrices, Classification Reports, and ROC-AUC curves, explaining metric significance in business context.
- **Model Persistence & Inference:** Serialize the best pipeline using `joblib` and build an easy-to-use inference script for individual or batch predictions.

---

## 5. Dataset Description
The project utilizes the industry-standard **IBM Telco Customer Churn Dataset** (7,043 rows, 21 columns).

### Dataset Schema:
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `customerID` | Categorical / String | Unique customer identifier (dropped during preprocessing) |
| `gender` | Categorical | Customer gender (`Male`, `Female`) |
| `SeniorCitizen` | Numeric / Binary | Senior citizen indicator (`0`: No, `1`: Yes) |
| `Partner` | Categorical | Whether customer has a partner (`Yes`, `No`) |
| `Dependents` | Categorical | Whether customer has dependents (`Yes`, `No`) |
| `tenure` | Numeric (Integer) | Months the customer has stayed with the company |
| `PhoneService` | Categorical | Whether customer has phone service (`Yes`, `No`) |
| `MultipleLines` | Categorical | Whether customer has multiple lines (`Yes`, `No`, `No phone service`) |
| `InternetService` | Categorical | Customer's internet service provider (`DSL`, `Fiber optic`, `No`) |
| `OnlineSecurity` | Categorical | Whether customer has online security add-on (`Yes`, `No`, `No internet service`) |
| `OnlineBackup` | Categorical | Whether customer has online backup add-on (`Yes`, `No`, `No internet service`) |
| `DeviceProtection`| Categorical | Whether customer has tech device protection (`Yes`, `No`, `No internet service`) |
| `TechSupport` | Categorical | Whether customer has tech support add-on (`Yes`, `No`, `No internet service`) |
| `StreamingTV` | Categorical | Whether customer has streaming TV (`Yes`, `No`, `No internet service`) |
| `StreamingMovies` | Categorical | Whether customer has streaming movies (`Yes`, `No`, `No internet service`) |
| `Contract` | Categorical | Contract term (`Month-to-month`, `One year`, `Two year`) |
| `PaperlessBilling`| Categorical | Whether customer has paperless billing (`Yes`, `No`) |
| `PaymentMethod` | Categorical | Payment method (`Electronic check`, `Mailed check`, `Bank transfer (auto)`, `Credit card (auto)`) |
| `MonthlyCharges` | Numeric (Float) | Amount charged to customer monthly (USD) |
| `TotalCharges` | Numeric (Float) | Total amount charged to customer over entire tenure (USD) |
| `Churn` | Categorical / Target | Whether customer defected (`Yes`: 1, `No`: 0) |

---

## 6. Technologies Used
- **Language:** Python 3.10+
- **Platform:** Windows / Linux / macOS
- **Environment:** Jupyter Notebook / Interactive Python Shell / Command Line

---

## 7. Libraries Used
- **`pandas`:** Tabular data structures, cleaning, grouping, and aggregations.
- **`numpy`:** Vectorized computations, array transformations, and numeric handling.
- **`matplotlib`:** Low-level charting, plot customization, and figure styling.
- **`seaborn`:** Statistical graphics, distributions (KDE), countplots, and heatmaps.
- **`scikit-learn`:** Data splitting, scaling, encoding, machine learning estimators, and evaluation metrics.
- **`joblib`:** Model pipeline serialization and deserialization.

---

## 8. Project Structure
```
Customer_Churn_Analysis/
│
├── data/
│   └── customer_churn.csv                     # Ingested customer churn dataset (7,043 rows)
│
├── notebooks/
│   └── customer_churn_analysis.ipynb          # End-to-end, fully executed Jupyter Notebook (20 sections)
│
├── src/
│   ├── download_data.py                       # Dataset verification and automated downloader
│   ├── data_preprocessing.py                  # Cleaning, type conversion, and ColumnTransformer pipeline
│   ├── eda.py                                 # Statistical visualization generation module
│   ├── train_model.py                         # Model training, tuning, and pipeline serialization
│   ├── evaluate_model.py                      # Metrics, confusion matrices, ROC curves, feature importance
│   └── predict.py                             # Reusable inference script for new customer profiles
│
├── models/
│   ├── churn_model.pkl                        # Serialized best performing scikit-learn pipeline
│   └── model_metadata.pkl                     # Pipeline metadata, column indices, and settings
│
├── visualizations/                            # High-resolution saved charts & plots
│   ├── 01_churn_distribution.png
│   ├── 02_churn_by_gender.png
│   ├── 03_churn_by_contract.png
│   ├── 04_churn_by_tenure.png
│   ├── 05_churn_by_monthly_charges.png
│   ├── 06_churn_by_payment_method.png
│   ├── 07_churn_by_internet_service.png
│   ├── 08_churn_by_senior_citizen.png
│   ├── 09_numerical_correlation_heatmap.png
│   ├── 10_confusion_matrices.png
│   ├── 11_roc_curves.png
│   ├── 12_feature_importance_rf.png
│   └── 13_logistic_regression_coefficients.png
│
├── requirements.txt                           # Project dependencies
├── README.md                                  # Complete academic & architectural documentation
└── main.py                                    # Master script executing the end-to-end pipeline
```

---

## 9. Data Preprocessing
To adhere to professional machine learning standards and prevent data leakage:
1. **Identifier Elimination:** The non-predictive `customerID` attribute was dropped.
2. **Datatype Correction & Missing Values:** `TotalCharges` initially arrived as an `object` type because customers with `tenure = 0` possessed whitespace strings (`" "`). We coerced these values to `NaN` and imputed them with `0.0` (as 0 tenure implies 0 accumulated charges).
3. **Duplicate Detection:** Identified and removed duplicate profiles to ensure data hygiene.
4. **Target Encoding:** Mapped `Churn` from categorical string (`Yes` / `No`) to binary integers (`1` / `0`).
5. **Stratified Train-Test Split:** Partitioned data into 80% training (`5,616` records) and 20% testing (`1,405` records) using `stratify=y` to preserve the ~26.5% positive churn ratio across splits.
6. **Scikit-Learn ColumnTransformer:**
   - Numerical Pipeline: Scaled `tenure`, `MonthlyCharges`, and `TotalCharges` using `StandardScaler`.
   - Categorical Pipeline: Encoded all nominal features using `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.

---

## 10. Exploratory Data Analysis (EDA)
The analysis generated 9 comprehensive visualization charts in `visualizations/`:
1. **Overall Churn Distribution:** Approximately **26.5%** of customers churned while **73.5%** remained, indicating moderate class imbalance.
2. **Churn by Contract Type:** Customers on Month-to-Month contracts showed a churn rate of **~42.7%**, whereas customers on One-Year and Two-Year contracts exhibited churn rates of **~11.3%** and **~2.8%** respectively.
3. **Churn by Tenure:** Churn is heavily concentrated in the early lifecycle (tenure < 12 months). Customers surviving beyond 24 months demonstrate significantly higher loyalty.
4. **Churn by Monthly Charges:** Churners display a higher median monthly charge (~$79.65) compared to retained customers (~$64.40).
5. **Churn by Internet Service:** Fiber optic subscribers exhibited an elevated churn rate of **~41.9%**, compared to **~19.0%** for DSL subscribers.
6. **Churn by Payment Method:** Customers paying via Electronic Check churn at **~45.3%**, whereas automated bank transfer and credit card users churn at **~15.2% - 16.7%**.
7. **Churn by Senior Citizen Status:** Senior citizens churn at **~41.7%**, compared to non-seniors at **~23.6%**.
8. **Numerical Correlations:** `tenure` and `TotalCharges` show a strong positive correlation (+0.826), while `tenure` is negatively correlated with `Churn` (-0.352).

---

## 11. Machine Learning Models
Three distinct classification paradigms were implemented using scikit-learn:

1. **Logistic Regression:**
   - Linear probabilistic classifier with L2 regularization (`C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`).
   - Serves as an interpretable baseline modeling log-odds of customer defection.
2. **Decision Tree Classifier:**
   - Non-linear tree-based classifier with regularization (`max_depth=5`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight='balanced'`).
   - Provides clear decision rules and split points.
3. **Random Forest Classifier:**
   - Ensemble bagging model with 100 decision trees (`n_estimators=100`, `max_depth=8`, `min_samples_split=15`, `class_weight='balanced'`).
   - Reduces variance and captures intricate feature interactions.

---

## 12. Evaluation Metrics

Because churn datasets contain class imbalance, models were evaluated across multiple metrics:

| Metric | Business Meaning in Customer Churn Context |
| :--- | :--- |
| **Accuracy** | Fraction of all predictions that were correct. Misleading when evaluated in isolation because predicting "No Churn" universally yields ~73.5% accuracy. |
| **Precision** | Percentage of customers flagged as "At Risk" who actually defected. High precision minimizes false alarms, preventing unnecessary retention discounts. |
| **Recall (Sensitivity)** | Percentage of actual churners correctly caught by the model. **Highest business priority** because missing an at-risk customer results in lost recurring revenue. |
| **F1-Score** | Harmonic mean of Precision and Recall. Balances false positives and false negatives under class imbalance. |
| **ROC-AUC** | Area under the True Positive Rate vs False Positive Rate curve. Measures the model's ranking capability across all decision thresholds. |

---

## 13. Results & Model Comparison

Evaluated on the held-out test dataset (**1,405 customers**):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **0.7580** | **0.5288** | **0.7903** | **0.6336** | **0.8413** |
| **Logistic Regression** | 0.7395 | 0.5052 | 0.7769 | 0.6123 | 0.8398 |
| **Decision Tree Classifier** | 0.7210 | 0.4834 | 0.7823 | 0.5975 | 0.8211 |

### Model Selection:
**Random Forest Classifier** was selected as the final production model:
- **Top ROC-AUC (0.8413):** Superior ability to rank churn risk accurately.
- **Top Recall (79.03%):** Successfully captures ~4 out of every 5 churning customers.
- **Highest F1-Score (0.6336):** Optimal balance between catching churners and limiting false alarms.

---

## 14. How to Run the Project

### Prerequisites:
- Python 3.10 or higher installed.

### Step 1: Clone or Navigate to Project Directory
```bash
cd c:\Customer_Churn_Analysis
```

### Step 2: Install Required Libraries
```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the Complete Master Pipeline
```bash
python main.py
```
This single command runs data validation, cleaning, EDA figure generation, model training, evaluation, comparison, serialization, and sample predictions.

### Step 4: Run Individual Modules (Optional)
```bash
# Data Acquisition
python src/download_data.py

# Data Preprocessing
python src/data_preprocessing.py

# Exploratory Data Analysis & Plots
python src/eda.py

# Model Training & Pipeline Serialization
python src/train_model.py

# Run Predictions on Sample Customer Profiles
python src/predict.py

# Run Interactive Real-Time Prediction Prompt
python src/predict.py --interactive
```

### Step 5: Run the Jupyter Notebook
```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```
*(Or open the `.ipynb` file in VS Code / Antigravity IDE).*

---

## 15. Example Prediction
The reusable inference component (`src/predict.py`) accepts raw customer records and yields predictions, calibrated churn probabilities, and risk levels:

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()

# Example new customer record
new_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.50,
    'TotalCharges': 191.00
}

result = predictor.predict(new_customer)[0]
print(result)
# Output:
# {
#   'prediction': 'Churn',
#   'prediction_code': 1,
#   'churn_probability': 0.871,
#   'churn_probability_pct': '87.1%',
#   'risk_level': 'High Risk'
# }
```

---

## 16. Key Findings & Business Insights
*Note: Findings represent statistical associations identified by the analysis rather than direct causal claims.*

1. **Contract Term Association:**
   - Month-to-month subscribers exhibit the highest churn tendency (~42.7%). Long-term contracts (One-year ~11.3%, Two-year ~2.8%) act as a strong retention buffer.
   - *Recommendation:* Introduce loyalty incentives and discounted annual contracts to encourage migration away from month-to-month commitments.

2. **Critical Initial Onboarding Window:**
   - Tenure demonstrates a pronounced negative association with churn. Attrition is heavily concentrated in the first 0–12 months.
   - *Recommendation:* Implement an intensive 90-day customer onboarding journey, with proactive customer support check-ins.

3. **Fiber Optic & Pricing Disconnect:**
   - Fiber optic users churn at ~41.9% despite paying premium rates. This often indicates service stability friction or dissatisfaction with support.
   - *Recommendation:* Bundle tech support and cyber-security protection into standard fiber optic tiers.

4. **Payment Method Retention Gap:**
   - Customers using Electronic Check exhibit high defection rates (~45.3%) compared to automated payment options (~15%–17%).
   - *Recommendation:* Provide a one-time bill credit for customers enrolling in automated credit card or bank transfer billing.

---

## 17. Limitations
- **Cross-Sectional Dataset:** The data provides a historical snapshot rather than a time-series log of usage trends (e.g., changes in call drops, bandwidth usage drops).
- **Synthetic/Benchmark Source:** While based on the IBM Telco dataset, real-world telecom data includes unstructured customer support tickets, network telemetry, and NPS survey feedback.
- **Unexplored Non-Linear Algorithms:** Advanced gradient boosting frameworks (XGBoost, LightGBM, CatBoost) were not included in order to strictly respect core library constraints.

---

## 18. Future Improvements
- **Threshold Optimization:** Tune classification probability thresholds based on actual dollar costs of false positives (unnecessary discounts) versus false negatives (lost customer lifetime value).
- **Customer Lifetime Value (CLV) Weighting:** Incorporate customer CLV into the objective function so retention agents prioritize higher-value accounts.
- **Explainability with SHAP:** Implement SHAP (SHapley Additive exPlanations) values for individualized, local feature explanations in client dashboards.
- **Interactive Web Application:** Deploy a lightweight web user interface (FastAPI / Streamlit / Flask) for real-time customer churn scoring by customer service agents.

---

## 19. Conclusion
This project successfully designed, implemented, and validated an end-to-end Machine Learning Customer Churn Analysis system. By integrating data engineering, statistical EDA, scikit-learn `ColumnTransformer` pipelines, and balanced multi-metric evaluation, the system achieved a **0.8413 ROC-AUC** and a **79.03% Recall** on unseen customer records. The project is fully modular, reproducible, and ready for deployment in academic presentations and technical portfolios.
