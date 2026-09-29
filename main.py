"""
Master Execution Script: main.py
Project: CUSTOMER CHURN ANALYSIS
Author: BE / AIML Academic Project
Description: Runs the entire machine learning pipeline end-to-end:
             1. Data loading & validation
             2. Data cleaning & missing value resolution
             3. Exploratory Data Analysis (EDA) & figure generation
             4. Feature engineering & ColumnTransformer preprocessing
             5. Stratified train/test splitting
             6. Model training (Logistic Regression, Decision Tree, Random Forest)
             7. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
             8. Feature importance & coefficient analysis
             9. Model persistence (.pkl)
             10. Sample customer churn prediction inference
"""

import sys
import os

# Add src to path to enable modular imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from download_data import ensure_dataset
from data_preprocessing import load_dataset, inspect_dataset, clean_dataset, prepare_data
from eda import run_full_eda
from train_model import train_and_evaluate_models, save_pipeline
from predict import ChurnPredictor


def main():
    print("\n" + "#"*70)
    print("#" + " "*21 + "CUSTOMER CHURN ANALYSIS" + " "*22 + "#")
    print("#" + " "*17 + "End-to-End Machine Learning System" + " "*17 + "#")
    print("#"*70)
    
    # 1. Dataset Verification & Acquisition
    print("\n>>> STEP 1: Dataset Verification & Acquisition")
    data_path = ensure_dataset()
    
    # 2. Inspect Dataset
    print("\n>>> STEP 2: Dataset Overview & Inspection")
    raw_df = load_dataset(data_path)
    summary = inspect_dataset(raw_df)
    print(f"Total Rows:       {summary['num_rows']:,}")
    print(f"Total Columns:    {summary['num_cols']}")
    print(f"Duplicate Rows:   {summary['duplicate_rows']}")
    print("Columns Overview:")
    for col, dtype in summary['dtypes'].items():
        missing = summary['missing_values'][col]
        print(f"  - {col:<18}: {str(dtype):<10} | Missing: {missing}")
        
    # 3. Clean Dataset
    print("\n>>> STEP 3: Data Cleaning & Preprocessing Preparation")
    clean_df = clean_dataset(raw_df, target_col='Churn')
    print(f"Cleaned Data Dimensions: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
    
    # 4. Exploratory Data Analysis (EDA)
    print("\n>>> STEP 4: Exploratory Data Analysis (EDA)")
    run_full_eda(clean_df, target_col='Churn')
    
    # 5. Data Preparation & Stratified Train/Test Split
    print("\n>>> STEP 5: Data Preparation & Stratified Split")
    data_artifacts = prepare_data(data_path, target_col='Churn', test_size=0.2, random_state=42)
    
    # 6 & 7. Model Training, Evaluation, and Comparison
    print("\n>>> STEP 6 & 7: Model Training, Evaluation & Comparison")
    results, best_model_name, best_pipeline, comp_df = train_and_evaluate_models(data_artifacts)
    
    # 8. Model Persistence
    print("\n>>> STEP 8: Model Serialization")
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    save_pipeline(
        best_pipeline,
        best_model_name,
        data_artifacts['numerical_cols'],
        data_artifacts['categorical_cols'],
        save_dir=models_dir
    )
    
    # 9. Inference on Sample Customer Profiles
    print("\n>>> STEP 9: Inference & Customer Scoring Demonstration")
    predictor = ChurnPredictor(model_path=os.path.join(models_dir, 'churn_model.pkl'))
    
    sample_customers = [
        {
            'name': 'High-Risk Customer (Alex)',
            'data': {
                'gender': 'Male',
                'SeniorCitizen': 0,
                'Partner': 'No',
                'Dependents': 'No',
                'tenure': 3,
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
                'TotalCharges': 286.50
            }
        },
        {
            'name': 'Low-Risk Customer (Samantha)',
            'data': {
                'gender': 'Female',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'Yes',
                'tenure': 60,
                'PhoneService': 'Yes',
                'MultipleLines': 'Yes',
                'InternetService': 'DSL',
                'OnlineSecurity': 'Yes',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'Yes',
                'TechSupport': 'Yes',
                'StreamingTV': 'No',
                'StreamingMovies': 'Yes',
                'Contract': 'Two year',
                'PaperlessBilling': 'No',
                'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 64.00,
                'TotalCharges': 3840.00
            }
        }
    ]
    
    for sample in sample_customers:
        pred_res = predictor.predict(sample['data'])[0]
        print(f"\nProfile: {sample['name']}")
        print(f"  Predicted Outcome:   {pred_res['prediction']} (Code: {pred_res['prediction_code']})")
        print(f"  Churn Probability:   {pred_res['churn_probability_pct']}")
        print(f"  Risk Category:       {pred_res['risk_level']}")
        
    print("\n" + "="*70)
    print(" [COMPLETE] End-to-end Customer Churn Analysis executed successfully!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
