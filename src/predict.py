"""
Module: predict.py
Description: Reusable customer churn inference module.
             Accepts raw customer attributes and returns churn predictions,
             churn probabilities, and risk classifications using the saved pipeline.
"""

import os
import joblib
import pandas as pd
import numpy as np


# Pre-defined mapping for case-insensitive categorical normalization
VALID_CATEGORIES = {
    'gender': {'female': 'Female', 'male': 'Male'},
    'Partner': {'yes': 'Yes', 'no': 'No'},
    'Dependents': {'yes': 'Yes', 'no': 'No'},
    'PhoneService': {'yes': 'Yes', 'no': 'No'},
    'MultipleLines': {'no': 'No', 'yes': 'Yes', 'no phone service': 'No phone service'},
    'InternetService': {'dsl': 'DSL', 'fiber optic': 'Fiber optic', 'fiber': 'Fiber optic', 'no': 'No'},
    'OnlineSecurity': {'no': 'No', 'yes': 'Yes', 'no internet service': 'No internet service'},
    'OnlineBackup': {'yes': 'Yes', 'no': 'No', 'no internet service': 'No internet service'},
    'DeviceProtection': {'no': 'No', 'yes': 'Yes', 'no internet service': 'No internet service'},
    'TechSupport': {'no': 'No', 'yes': 'Yes', 'no internet service': 'No internet service'},
    'StreamingTV': {'no': 'No', 'yes': 'Yes', 'no internet service': 'No internet service'},
    'StreamingMovies': {'no': 'No', 'yes': 'Yes', 'no internet service': 'No internet service'},
    'Contract': {'month-to-month': 'Month-to-month', 'one year': 'One year', 'two year': 'Two year', '1 year': 'One year', '2 year': 'Two year'},
    'PaperlessBilling': {'yes': 'Yes', 'no': 'No'},
    'PaymentMethod': {
        'electronic check': 'Electronic check',
        'mailed check': 'Mailed check',
        'bank transfer (automatic)': 'Bank transfer (automatic)',
        'bank transfer': 'Bank transfer (automatic)',
        'credit card (automatic)': 'Credit card (automatic)',
        'credit card': 'Credit card (automatic)'
    }
}


class ChurnPredictor:
    """
    Inference class to load saved model pipeline and score customer churn risk.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(base_dir, 'models', 'churn_model.pkl')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model pipeline not found at: {model_path}. "
                "Please run train_model.py first to generate the saved pipeline."
            )
            
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        print(f"[INFERENCE] Loaded model pipeline from: {model_path}")

    def predict(self, customer_input):
        """
        Predict churn and probability for one or multiple customers.
        
        Args:
            customer_input: dict, list of dicts, or pd.DataFrame of customer records.
            
        Returns:
            list of dicts containing prediction results.
        """
        # Convert single dict or list of dicts to DataFrame
        if isinstance(customer_input, dict):
            df = pd.DataFrame([customer_input])
        elif isinstance(customer_input, list):
            df = pd.DataFrame(customer_input)
        elif isinstance(customer_input, pd.DataFrame):
            df = customer_input.copy()
        else:
            raise TypeError("customer_input must be a dict, list of dicts, or pandas DataFrame.")
            
        # Ensure 'customerID' is dropped if supplied by caller
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        # Case-insensitive normalization for categorical features
        for col, mapping in VALID_CATEGORIES.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower().map(mapping).fillna(df[col])
                
        # Type casting and imputation for numerical features
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
        for num_col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if num_col in df.columns:
                df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0.0)
            
        # Generate predictions and probabilities
        preds = self.pipeline.predict(df)
        has_proba = hasattr(self.pipeline, "predict_proba")
        probs = self.pipeline.predict_proba(df)[:, 1] if has_proba else [None] * len(df)
        
        results = []
        for i in range(len(df)):
            pred_val = int(preds[i])
            pred_label = "Churn" if pred_val == 1 else "No Churn"
            prob_val = float(probs[i]) if probs[i] is not None else None
            
            # Risk segmentation
            if prob_val is not None:
                if prob_val >= 0.65:
                    risk = "High Risk"
                elif prob_val >= 0.35:
                    risk = "Moderate Risk"
                else:
                    risk = "Low Risk"
            else:
                risk = "Unknown"
                
            res = {
                'prediction': pred_label,
                'prediction_code': pred_val,
                'churn_probability': prob_val,
                'churn_probability_pct': f"{prob_val * 100:.1f}%" if prob_val is not None else "N/A",
                'risk_level': risk
            }
            results.append(res)
            
        return results


def run_sample_predictions():
    """
    Run sample predictions on contrasting customer profiles to demonstrate inference.
    """
    predictor = ChurnPredictor()
    
    # Profile 1: High-Risk Profile (New customer, month-to-month, fiber optic, electronic check)
    high_risk_customer = {
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
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.50,
        'TotalCharges': 179.00
    }
    
    # Profile 2: Low-Risk Profile (Long-term customer, two-year contract, bundled security, credit card)
    low_risk_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'Yes',
        'tenure': 68,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Credit card (automatic)',
        'MonthlyCharges': 55.20,
        'TotalCharges': 3753.60
    }
    
    print("\n" + "="*65)
    print("        TESTING CUSTOMER CHURN PREDICTION INFERENCE")
    print("="*65)
    
    # Evaluate Profile 1
    res1 = predictor.predict(high_risk_customer)[0]
    print("\n[CUSTOMER PROFILE 1: High-Risk Scenario]")
    print("Attributes: 2 months tenure, Month-to-month contract, Fiber optic, $89.50/mo")
    print(f"-> Prediction:         {res1['prediction']}")
    print(f"-> Churn Probability:  {res1['churn_probability_pct']}")
    print(f"-> Risk Assessment:    {res1['risk_level']}")
    
    # Evaluate Profile 2
    res2 = predictor.predict(low_risk_customer)[0]
    print("\n[CUSTOMER PROFILE 2: Low-Risk Scenario]")
    print("Attributes: 68 months tenure, Two-year contract, DSL + Security, $55.20/mo")
    print(f"-> Prediction:         {res2['prediction']}")
    print(f"-> Churn Probability:  {res2['churn_probability_pct']}")
    print(f"-> Risk Assessment:    {res2['risk_level']}")
    print("="*65 + "\n")


def interactive_cli():
    """
    Interactive prompt allowing the user to score any custom customer profile.
    """
    print("\n" + "="*65)
    print("     INTERACTIVE CUSTOMER CHURN RISK SCORER")
    print("="*65)
    print("Please enter customer details (press Enter to accept default):")
    
    predictor = ChurnPredictor()
    
    def prompt(text, default):
        val = input(f"  {text} [{default}]: ").strip()
        return val if val else default
        
    cust = {
        'gender': prompt("Gender (Male/Female)", "Female"),
        'SeniorCitizen': int(prompt("Senior Citizen (0/1)", "0")),
        'Partner': prompt("Has Partner (Yes/No)", "No"),
        'Dependents': prompt("Has Dependents (Yes/No)", "No"),
        'tenure': int(prompt("Tenure in Months (0-72)", "6")),
        'PhoneService': prompt("Phone Service (Yes/No)", "Yes"),
        'MultipleLines': prompt("Multiple Lines (Yes/No/No phone service)", "No"),
        'InternetService': prompt("Internet Service (DSL/Fiber optic/No)", "Fiber optic"),
        'OnlineSecurity': prompt("Online Security (Yes/No/No internet service)", "No"),
        'OnlineBackup': prompt("Online Backup (Yes/No/No internet service)", "No"),
        'DeviceProtection': prompt("Device Protection (Yes/No/No internet service)", "No"),
        'TechSupport': prompt("Tech Support (Yes/No/No internet service)", "No"),
        'StreamingTV': prompt("Streaming TV (Yes/No/No internet service)", "Yes"),
        'StreamingMovies': prompt("Streaming Movies (Yes/No/No internet service)", "No"),
        'Contract': prompt("Contract (Month-to-month/One year/Two year)", "Month-to-month"),
        'PaperlessBilling': prompt("Paperless Billing (Yes/No)", "Yes"),
        'PaymentMethod': prompt("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))", "Electronic check"),
        'MonthlyCharges': float(prompt("Monthly Charges in $", "85.0")),
        'TotalCharges': float(prompt("Total Charges in $", "510.0"))
    }
    
    res = predictor.predict(cust)[0]
    print("\n" + "-"*65)
    print("PREDICTION RESULT:")
    print(f"  -> Predicted Outcome:  {res['prediction']}")
    print(f"  -> Churn Probability:  {res['churn_probability_pct']}")
    print(f"  -> Risk Category:      {res['risk_level']}")
    print("-"*65 + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Component")
    parser.add_argument('--interactive', action='store_true', help="Run interactive customer scoring prompt")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_cli()
    else:
        run_sample_predictions()
