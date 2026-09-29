"""
Module: train_model.py
Description: Model training, hyperparameter configuration, pipeline construction,
             comparative evaluation, and model serialization for Customer Churn Analysis.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import prepare_data
from evaluate_model import (
    calculate_metrics,
    display_classification_report,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_tree_feature_importance,
    plot_logistic_coefficients,
    generate_comparison_table,
    explain_metrics,
    generate_business_insights
)


def get_models():
    """
    Define classification models with reproducible random states and balanced parameters.
    
    Returns:
        dict: Mapping of model names to scikit-learn estimator instances.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def train_and_evaluate_models(data_artifacts: dict):
    """
    Build scikit-learn Pipelines, train all candidate models, and evaluate on test set.
    
    Args:
        data_artifacts: Output dictionary from prepare_data().
        
    Returns:
        tuple: (results_dict, best_model_name, best_pipeline)
    """
    X_train = data_artifacts['X_train']
    y_train = data_artifacts['y_train']
    X_test = data_artifacts['X_test']
    y_test = data_artifacts['y_test']
    preprocessor = data_artifacts['preprocessor']
    
    candidate_models = get_models()
    results = {}
    
    print("\n" + "="*60)
    print("           TRAINING CLASSIFICATION MODELS")
    print("="*60)
    
    for name, estimator in candidate_models.items():
        print(f"\n[TRAINING] {name}...")
        
        # Build complete pipeline (Preprocessing + Classifier)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', estimator)
        ])
        
        # Train on training partition
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        results[name] = {
            'pipeline': pipeline,
            'estimator': estimator,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics
        }
        
        display_classification_report(name, y_test, y_pred)
        
    # Generate Comparison Table
    comp_df = generate_comparison_table(results)
    print("\n" + "="*60)
    print("              MODEL PERFORMANCE COMPARISON")
    print("="*60)
    print(comp_df.to_string(index=False))
    print("="*60)
    
    # Visualizations
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    
    # Feature Importances
    plot_tree_feature_importance(
        results['Random Forest']['pipeline'],
        data_artifacts['numerical_cols'],
        data_artifacts['categorical_cols'],
        model_name='Random Forest'
    )
    
    # Logistic Regression Coefficients
    plot_logistic_coefficients(
        results['Logistic Regression']['pipeline'],
        data_artifacts['numerical_cols'],
        data_artifacts['categorical_cols']
    )
    
    # Print metrics explanation and business insights
    print(explain_metrics())
    print(generate_business_insights())
    
    # Select best model based on ROC-AUC and F1-Score
    best_model_name = comp_df.iloc[0]['Model']
    best_pipeline = results[best_model_name]['pipeline']
    print(f"\n[SELECTION] Best Performing Model: '{best_model_name}' (ROC-AUC: {results[best_model_name]['metrics']['ROC-AUC']:.4f}, F1: {results[best_model_name]['metrics']['F1 Score']:.4f})")
    
    return results, best_model_name, best_pipeline, comp_df


def save_pipeline(pipeline, model_name: str, numerical_cols: list, categorical_cols: list, save_dir: str = None):
    """
    Save the trained scikit-learn Pipeline and metadata for reuse.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'churn_model.pkl')
    metadata = {
        'model_name': model_name,
        'pipeline': pipeline,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }
    
    # Save the pipeline itself
    joblib.dump(pipeline, model_path)
    
    # Also save metadata dictionary
    meta_path = os.path.join(save_dir, 'model_metadata.pkl')
    joblib.dump(metadata, meta_path)
    
    print(f"[SAVE] Saved final model pipeline to: '{model_path}'")
    print(f"[SAVE] Saved model metadata to: '{meta_path}'")
    return model_path


def run_pipeline():
    """
    Full training run script.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'customer_churn.csv')
    
    data_artifacts = prepare_data(data_path)
    results, best_model_name, best_pipeline, comp_df = train_and_evaluate_models(data_artifacts)
    
    save_pipeline(
        best_pipeline,
        best_model_name,
        data_artifacts['numerical_cols'],
        data_artifacts['categorical_cols']
    )
    
    return results, best_pipeline, comp_df


if __name__ == '__main__':
    run_pipeline()
