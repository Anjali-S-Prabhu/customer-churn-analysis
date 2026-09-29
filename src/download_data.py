"""
Data acquisition script for Telco Customer Churn Analysis.
Downloads the standard IBM Telco Customer Churn dataset if not already present.
"""
import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
DATA_PATH = os.path.join(DATA_DIR, 'customer_churn.csv')
TELCO_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'

def ensure_dataset():
    """Ensure that the customer churn dataset is available locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 10000:
        print(f"Dataset already exists at {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset successfully with shape: {df.shape}")
        return DATA_PATH

    print(f"Downloading dataset from {TELCO_URL}...")
    try:
        urllib.request.urlretrieve(TELCO_URL, DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        print(f"Successfully downloaded dataset to {DATA_PATH} (Shape: {df.shape})")
        return DATA_PATH
    except Exception as e:
        print(f"Network download failed: {e}. Generating fallback sample dataset...")
        # Create standard schema fallback dataset if offline
        create_sample_dataset(DATA_PATH)
        return DATA_PATH

def create_sample_dataset(dest_path):
    """Fallback generator for standard customer churn dataset if offline."""
    import numpy as np
    np.random.seed(42)
    n_samples = 1000
    
    genders = np.random.choice(['Male', 'Female'], size=n_samples)
    senior = np.random.choice([0, 1], p=[0.84, 0.16], size=n_samples)
    partner = np.random.choice(['Yes', 'No'], size=n_samples)
    dependents = np.random.choice(['Yes', 'No'], p=[0.3, 0.7], size=n_samples)
    tenure = np.random.randint(1, 73, size=n_samples)
    phone = np.random.choice(['Yes', 'No'], p=[0.9, 0.1], size=n_samples)
    multiple = np.random.choice(['Yes', 'No', 'No phone service'], size=n_samples)
    internet = np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.4, 0.45, 0.15], size=n_samples)
    security = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    backup = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    device_prot = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    tv = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    movies = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.55, 0.25, 0.20], size=n_samples)
    paperless = np.random.choice(['Yes', 'No'], p=[0.6, 0.4], size=n_samples)
    payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], size=n_samples)
    
    monthly = np.where(internet == 'Fiber optic', np.random.uniform(70, 110, size=n_samples),
              np.where(internet == 'DSL', np.random.uniform(40, 75, size=n_samples),
                       np.random.uniform(18, 25, size=n_samples)))
    monthly = np.round(monthly, 2)
    total = np.round(monthly * tenure + np.random.normal(0, 10, size=n_samples), 2)
    total = np.maximum(total, monthly)
    
    # Churn probability based on realistic risk factors (month-to-month, high charges, low tenure, fiber optic, electronic check)
    churn_score = (
        (contract == 'Month-to-month') * 1.5 +
        (tenure < 12) * 1.2 +
        (internet == 'Fiber optic') * 0.8 +
        (payment == 'Electronic check') * 0.7 +
        (tech_support == 'No') * 0.5 +
        (monthly > 70) * 0.6 -
        (contract == 'Two year') * 2.0 -
        (tenure > 48) * 1.5 - 1.2
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = np.where(np.random.rand(n_samples) < churn_prob, 'Yes', 'No')
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n_samples)]
    
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': device_prot,
        'TechSupport': tech_support,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'Churn': churn
    })
    df.to_csv(dest_path, index=False)
    print(f"Generated sample dataset at {dest_path} with {n_samples} records.")

if __name__ == '__main__':
    ensure_dataset()
