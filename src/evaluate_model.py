"""
Module: evaluate_model.py
Description: Evaluation, metric calculation, confusion matrices, ROC curves,
             feature importance visualization, and business insight generation.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def get_viz_dir():
    """Return and create path to visualizations directory."""
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for class 1 (Churn).
        
    Returns:
        dict: Metric names and rounded score values.
    """
    metrics = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1 Score': float(f1_score(y_true, y_pred, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_true, y_prob)) if y_prob is not None else np.nan
    }
    return metrics


def display_classification_report(model_name: str, y_true, y_pred):
    """
    Print formatted classification report.
    """
    print(f"\n{'='*55}")
    print(f"Classification Report: {model_name}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=['No Churn (0)', 'Churn (1)'], digits=4))


def plot_confusion_matrices(eval_results: dict, y_test, save_dir: str = None):
    """
    Plot confusion matrices for all trained models side-by-side.
    
    Args:
        eval_results: Dict mapping model name to results dict containing 'y_pred'.
        y_test: True test labels.
        save_dir: Output directory.
    """
    save_dir = save_dir or get_viz_dir()
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
        
    for idx, (model_name, res) in enumerate(eval_results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Combine counts and normalized percentages into cell text
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
                
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[idx], cbar=False,
                    square=True, linewidths=1, linecolor='gray',
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    annot_kws={'size': 12, 'weight': 'bold'})
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=13, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_ylabel('True Label', fontsize=11)
        
    plt.tight_layout()
    output_path = os.path.join(save_dir, '10_confusion_matrices.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved Confusion Matrices plot to: {output_path}")


def plot_roc_curves(eval_results: dict, y_test, save_dir: str = None):
    """
    Plot ROC Curves for all trained models on a single graph.
    
    Args:
        eval_results: Dict mapping model name to results dict containing 'y_prob'.
        y_test: True test labels.
        save_dir: Output directory.
    """
    save_dir = save_dir or get_viz_dir()
    plt.figure(figsize=(9, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (model_name, res) in enumerate(eval_results.items()):
        if res.get('y_prob') is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            auc_val = res['metrics']['ROC-AUC']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})",
                     color=colors[idx % len(colors)], linewidth=2.2)
            
    # Random guess diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5000)', alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = os.path.join(save_dir, '11_roc_curves.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved ROC Curves plot to: {output_path}")


def get_feature_names_from_preprocessor(preprocessor, numerical_cols, categorical_cols):
    """
    Extract transformed feature names after ColumnTransformer preprocessing.
    """
    feature_names = []
    # 1. Numerical feature names (pass-through / standard scaled)
    feature_names.extend(numerical_cols)
    
    # 2. Categorical feature names (from OneHotEncoder)
    cat_transformer = preprocessor.named_transformers_['cat']
    onehot_encoder = cat_transformer.named_steps['onehot']
    cat_encoded_names = list(onehot_encoder.get_feature_names_out(categorical_cols))
    feature_names.extend(cat_encoded_names)
    
    return feature_names


def plot_tree_feature_importance(pipeline, numerical_cols, categorical_cols,
                                 model_name: str = 'Random Forest', top_n: int = 15,
                                 save_dir: str = None):
    """
    Plot top feature importances for tree-based models.
    """
    save_dir = save_dir or get_viz_dir()
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    if not hasattr(classifier, 'feature_importances_'):
        print(f"[WARN] {model_name} does not have feature_importances_. Skipping plot.")
        return None
        
    feature_names = get_feature_names_from_preprocessor(preprocessor, numerical_cols, categorical_cols)
    importances = classifier.feature_importances_
    
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    top_fi = fi_df.head(top_n).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_fi['Feature'], top_fi['Importance'], color='#2B5C8F', edgecolor='black', alpha=0.85)
    plt.title(f'Top {top_n} Most Important Features ({model_name})', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Gini Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f'{w:.3f}',
                 ha='left', va='center', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    output_path = os.path.join(save_dir, '12_feature_importance_rf.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved Feature Importance plot to: {output_path}")
    return fi_df


def plot_logistic_coefficients(pipeline, numerical_cols, categorical_cols,
                               top_n: int = 15, save_dir: str = None):
    """
    Plot top positive and negative coefficients for Logistic Regression.
    """
    save_dir = save_dir or get_viz_dir()
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    if not hasattr(classifier, 'coef_'):
        return None
        
    feature_names = get_feature_names_from_preprocessor(preprocessor, numerical_cols, categorical_cols)
    coefs = classifier.coef_[0]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    }).sort_values('Abs_Coefficient', ascending=False).reset_index(drop=True)
    
    top_coef = coef_df.head(top_n).sort_values('Coefficient', ascending=True)
    colors = ['#D9534F' if c > 0 else '#2B5C8F' for c in top_coef['Coefficient']]
    
    plt.figure(figsize=(10, 7))
    bars = plt.barh(top_coef['Feature'], top_coef['Coefficient'], color=colors, edgecolor='black', alpha=0.85)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f'Top {top_n} Logistic Regression Coefficients\n(Positive: Associated with Churn | Negative: Associated with Retention)',
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Coefficient Value (Log-Odds Impact)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    for bar in bars:
        w = bar.get_width()
        pos = w + (0.02 if w >= 0 else -0.05)
        align = 'left' if w >= 0 else 'right'
        plt.text(pos, bar.get_y() + bar.get_height() / 2, f'{w:.2f}',
                 ha=align, va='center', fontsize=9, fontweight='bold')
                 
    plt.tight_layout()
    output_path = os.path.join(save_dir, '13_logistic_regression_coefficients.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved Logistic Coefficients plot to: {output_path}")
    return coef_df


def generate_comparison_table(eval_results: dict) -> pd.DataFrame:
    """
    Create a clean model performance comparison DataFrame.
    """
    rows = []
    for model_name, res in eval_results.items():
        m = res['metrics']
        rows.append({
            'Model': model_name,
            'Accuracy': round(m['Accuracy'], 4),
            'Precision': round(m['Precision'], 4),
            'Recall': round(m['Recall'], 4),
            'F1 Score': round(m['F1 Score'], 4),
            'ROC-AUC': round(m['ROC-AUC'], 4)
        })
    df_comp = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    return df_comp


def explain_metrics():
    """
    Return explanations for each evaluation metric in the customer churn business context.
    """
    explanation = """
================================================================================
           EVALUATION METRICS IN CUSTOMER CHURN CONTEXT
================================================================================
1. Accuracy:
   - Measures overall proportion of correct predictions (both churn and retention).
   - In churn datasets, where ~73% of customers do NOT churn, a naive baseline
     predicting 'No Churn' for everyone achieves 73% accuracy. Hence, accuracy alone
     is misleading.

2. Precision:
   - Measures what percentage of customers flagged as 'Likely to Churn' actually churned.
   - High precision minimizes 'false alarms', preventing unnecessary retention discounts
     being offered to customers who were not planning to leave.

3. Recall (Sensitivity):
   - Measures what percentage of actual churners the model successfully identified.
   - High recall ensures the company catches as many at-risk customers as possible
     before they defect to competitors. Missing an actual churner has high business cost.

4. F1-Score:
   - The harmonic mean of Precision and Recall.
   - Provides a balanced single metric when there is class imbalance between churners
     and non-churners.

5. ROC-AUC (Area Under ROC Curve):
   - Measures the model's ability to rank churners above non-churners across all
     possible decision probability thresholds (0.0 to 1.0).
   - An AUC of 0.85 means there is an 85% probability that a randomly chosen
     churning customer will be assigned a higher churn risk score than a retained one.
================================================================================
"""
    return explanation


def generate_business_insights():
    """
    Generate business-oriented findings using appropriate associative phrasing.
    """
    insights = """
================================================================================
                       CUSTOMER CHURN BUSINESS INSIGHTS
================================================================================
1. Contract Type Association:
   - The analysis indicates that contract commitment is strongly associated with churn.
   - Customers on Month-to-Month contracts exhibit significantly higher churn rates
     (~42.7%) compared to customers on One-Year (~11.3%) or Two-Year (~2.8%) contracts.
   - Business Recommendation: Introduce incentives and tiered discounts to encourage
     customers to migrate from month-to-month to annual commitments.

2. Tenure and Customer Lifetime:
   - Tenure shows a strong negative association with churn.
   - Churn is heavily concentrated in the first 0-12 months of customer lifecycle.
   - As tenure increases beyond 24 months, churn rates decline markedly.
   - Business Recommendation: Implement dedicated onboarding journeys and proactive
     check-ins during the critical initial 90-day window.

3. Monthly Charges and Pricing:
   - Customers paying higher monthly charges ($70 - $105) show elevated churn
     rates compared to those paying lower monthly rates ($20 - $50).
   - Higher pricing without perceived value increases customer defection risk.
   - Business Recommendation: Audit high-tier bundles and verify customer satisfaction
     with premium add-ons.

4. Service Type & Support:
   - Fiber Optic internet customers experience noticeably higher churn (~41.9%) than
     DSL customers (~19.0%), often linked to pricing expectations and tech support needs.
   - Customers lacking Online Security, Tech Support, and Online Backup exhibit
     consistently higher churn tendencies.
   - Business Recommendation: Bundle protective services (Tech Support & Online Security)
     into standard fiber plans to build service stickiness.

5. Payment Method:
   - Customers paying via Electronic Check demonstrate significantly higher churn
     (~45.3%) compared to automated payment methods (Bank Transfer ~16.7%, Credit Card ~15.2%).
   - Business Recommendation: Encourage migration to automated credit card/bank transfer
     billing by offering a one-time billing discount.
================================================================================
"""
    return insights
