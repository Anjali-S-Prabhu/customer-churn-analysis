"""
Module: data_preprocessing.py
Description: Data loading, cleaning, validation, feature separation,
             and preprocessing pipeline creation for Customer Churn Analysis.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load customer dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset from '{filepath}' with shape: {df.shape}")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Inspect dataset properties: dimensions, datatypes, missing values, duplicates.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        dict: Inspection summary.
    """
    summary = {
        'num_rows': df.shape[0],
        'num_cols': df.shape[1],
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
    }
    return summary


def clean_dataset(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Clean the customer dataset:
    - Handle whitespace/missing values in TotalCharges
    - Drop duplicate rows if present
    - Standardize binary target variable to integer (1 for Churn, 0 for No Churn)
    - Remove non-predictive identifiers like 'customerID'
    
    Args:
        df: Input raw DataFrame.
        target_col: Name of the target churn column.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # 1. Drop customerID column if present
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        print("[INFO] Dropped 'customerID' column (non-predictive identifier).")
        
    # 2. Fix TotalCharges data type
    # In standard Telco dataset, TotalCharges contains whitespace strings " " for tenure=0 customers
    if 'TotalCharges' in df_clean.columns:
        # Convert to numeric, turning spaces or unparseable values to NaN
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        nan_count = df_clean['TotalCharges'].isnull().sum()
        if nan_count > 0:
            # Impute NaN with 0.0 because these records have tenure = 0 (new customers)
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
            print(f"[INFO] Converted 'TotalCharges' to numeric; imputed {nan_count} missing values with 0.0 (tenure=0).")
            
    # 3. Handle duplicate records
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"[INFO] Removed {dup_count} duplicate rows.")
    else:
        print("[INFO] No duplicate rows detected.")
        
    # 4. Standardize Target Variable
    if target_col in df_clean.columns:
        target_map = {
            'Yes': 1, 'No': 0,
            'yes': 1, 'no': 0,
            'True': 1, 'False': 0,
            True: 1, False: 0,
            1: 1, 0: 0
        }
        # Check current unique values
        unique_vals = df_clean[target_col].unique()
        print(f"[INFO] Raw target values in '{target_col}': {unique_vals}")
        df_clean[target_col] = df_clean[target_col].map(target_map)
        
        # Check for unmapped values
        if df_clean[target_col].isnull().any():
            unmapped = df[df_clean[target_col].isnull()][target_col].unique()
            raise ValueError(f"Target column contains unmapped values: {unmapped}")
        print(f"[INFO] Standardized target column '{target_col}' to binary (0 = No Churn, 1 = Churn).")
        
    return df_clean


def get_feature_lists(df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Identify numerical and categorical feature column names.
    
    Args:
        df: Input DataFrame.
        target_col: Target column name.
        
    Returns:
        tuple: (numerical_cols, categorical_cols)
    """
    feature_df = df.drop(columns=[target_col]) if target_col in df.columns else df
    
    # Define explicit known numerical columns for customer churn
    known_numerical = ['tenure', 'MonthlyCharges', 'TotalCharges']
    numerical_cols = [col for col in known_numerical if col in feature_df.columns]
    
    # Any other column is treated as categorical
    categorical_cols = [col for col in feature_df.columns if col not in numerical_cols]
    
    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer for preprocessing:
    - Numerical: StandardScaler
    - Categorical: OneHotEncoder(drop='first', handle_unknown='ignore')
    
    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        
    Returns:
        ColumnTransformer: Preprocessor object.
    """
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def prepare_data(filepath: str, target_col: str = 'Churn', test_size: float = 0.2, random_state: int = 42):
    """
    End-to-end data preparation function:
    Loads, cleans, splits into stratified train/test sets, and creates preprocessor.
    
    Args:
        filepath: Path to dataset.
        target_col: Target column name.
        test_size: Proportion of test data.
        random_state: Seed for reproducibility.
        
    Returns:
        dict: Prepared data artifacts and metadata.
    """
    raw_df = load_dataset(filepath)
    clean_df = clean_dataset(raw_df, target_col=target_col)
    
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]
    
    numerical_cols, categorical_cols = get_feature_lists(clean_df, target_col=target_col)
    
    # Stratified split to preserve churn ratio in train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    print(f"[INFO] Train split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples")
    print(f"[INFO] Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"[INFO] Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    return {
        'raw_df': raw_df,
        'clean_df': clean_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'preprocessor': preprocessor
    }


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'customer_churn.csv')
    artifacts = prepare_data(data_path)
    print("[SUCCESS] Data preprocessing completed successfully.")
