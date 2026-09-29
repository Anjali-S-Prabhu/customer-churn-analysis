"""
Module: eda.py
Description: Exploratory Data Analysis (EDA) module for Customer Churn Analysis.
             Generates publication-quality visualizations and statistical summaries.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.autolayout'] = True

# Standard color palette for churn
# 0 (No Churn / Retained) -> Deep Blue / Teal (#2B5C8F)
# 1 (Churned) -> Warm Coral / Red (#D9534F)
CHURN_PALETTE = {0: '#2B5C8F', 1: '#D9534F', 'No': '#2B5C8F', 'Yes': '#D9534F'}


def get_viz_dir():
    """Return and create path to visualizations directory."""
    viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


def plot_churn_distribution(df: pd.DataFrame, target_col: str = 'Churn', save_dir: str = None):
    """
    Plot overall customer churn distribution (Bar chart & Pie chart side by side).
    """
    save_dir = save_dir or get_viz_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Counts and percentages
    churn_counts = df[target_col].value_counts()
    churn_labels = ['No Churn (0)', 'Churn (1)'] if set(df[target_col].unique()) == {0, 1} else ['No Churn', 'Churn']
    
    # 1. Bar Chart
    bars = axes[0].bar(churn_labels, churn_counts.values, color=['#2B5C8F', '#D9534F'], edgecolor='black', alpha=0.85)
    axes[0].set_title('Customer Churn Count Distribution', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Customer Status', fontsize=12)
    axes[0].set_ylabel('Number of Customers', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{height:,}\n({height/len(df)*100:.1f}%)',
                         xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                         xytext=(0, 0), textcoords="offset points",
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    # 2. Pie Chart
    axes[1].pie(churn_counts.values, labels=churn_labels, autopct='%1.1f%%', startangle=90,
                colors=['#2B5C8F', '#D9534F'], explode=(0, 0.08),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1.2},
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Customer Churn Proportion', fontsize=14, fontweight='bold', pad=12)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, '01_churn_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Churn Distribution plot to: {output_path}")


def plot_churn_by_categorical(df: pd.DataFrame, column: str, title: str, filename: str,
                              target_col: str = 'Churn', save_dir: str = None):
    """
    Plot churn breakdown for any categorical feature:
    Countplot + Percentage Churn Rate by category.
    """
    if column not in df.columns:
        print(f"[EDA] Column '{column}' not found. Skipping plot.")
        return
        
    save_dir = save_dir or get_viz_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Stacked/Grouped Count
    order = sorted(df[column].dropna().unique())
    sns.countplot(data=df, x=column, hue=target_col, order=order,
                  palette=CHURN_PALETTE, ax=axes[0], edgecolor='black')
    axes[0].set_title(f'Customer Counts by {title}', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel(title, fontsize=11)
    axes[0].set_ylabel('Customer Count', fontsize=11)
    axes[0].tick_params(axis='x', rotation=20 if len(order) > 3 else 0)
    axes[0].legend(title='Churn', labels=['No Churn (0)', 'Churn (1)'])
    
    # Right: Churn Rate per category
    churn_rates = df.groupby(column)[target_col].mean().loc[order] * 100
    bars = axes[1].bar(churn_rates.index.astype(str), churn_rates.values,
                       color='#E67E22', edgecolor='black', alpha=0.85)
    axes[1].set_title(f'Churn Rate (%) by {title}', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel(title, fontsize=11)
    axes[1].set_ylabel('Churn Rate (%)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=20 if len(order) > 3 else 0)
    axes[1].set_ylim(0, max(churn_rates.values) * 1.25 if max(churn_rates.values) > 0 else 100)
    
    # Annotate percentage on bars
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Baseline churn reference line
    baseline_churn = df[target_col].mean() * 100
    axes[1].axhline(baseline_churn, color='red', linestyle='--', linewidth=1.5,
                    label=f'Overall Avg ({baseline_churn:.1f}%)')
    axes[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved '{title}' analysis plot to: {output_path}")


def plot_churn_by_tenure(df: pd.DataFrame, target_col: str = 'Churn', save_dir: str = None):
    """
    Plot churn distribution across customer tenure (months).
    """
    if 'tenure' not in df.columns:
        return
    save_dir = save_dir or get_viz_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KDE distribution
    sns.kdeplot(data=df[df[target_col] == 0]['tenure'], ax=axes[0], color='#2B5C8F',
                fill=True, label='No Churn (Retained)', alpha=0.4, linewidth=2)
    sns.kdeplot(data=df[df[target_col] == 1]['tenure'], ax=axes[0], color='#D9534F',
                fill=True, label='Churned', alpha=0.4, linewidth=2)
    axes[0].set_title('Tenure Distribution by Churn Status', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Tenure (Months)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].legend()
    
    # Boxplot
    sns.boxplot(data=df, x=target_col, y='tenure', ax=axes[1],
                palette=CHURN_PALETTE, hue=target_col, legend=False)
    axes[1].set_title('Tenure Spread by Churn Status', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['No Churn (0)', 'Churn (1)'])
    axes[1].set_xlabel('Customer Status', fontsize=11)
    axes[1].set_ylabel('Tenure (Months)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, '04_churn_by_tenure.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Tenure vs Churn plot to: {output_path}")


def plot_churn_by_monthly_charges(df: pd.DataFrame, target_col: str = 'Churn', save_dir: str = None):
    """
    Plot churn distribution across Monthly Charges.
    """
    if 'MonthlyCharges' not in df.columns:
        return
    save_dir = save_dir or get_viz_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # KDE
    sns.kdeplot(data=df[df[target_col] == 0]['MonthlyCharges'], ax=axes[0], color='#2B5C8F',
                fill=True, label='No Churn (Retained)', alpha=0.4, linewidth=2)
    sns.kdeplot(data=df[df[target_col] == 1]['MonthlyCharges'], ax=axes[0], color='#D9534F',
                fill=True, label='Churned', alpha=0.4, linewidth=2)
    axes[0].set_title('Monthly Charges Distribution by Churn Status', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Monthly Charges ($)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].legend()
    
    # Boxplot
    sns.boxplot(data=df, x=target_col, y='MonthlyCharges', ax=axes[1],
                palette=CHURN_PALETTE, hue=target_col, legend=False)
    axes[1].set_title('Monthly Charges Spread by Churn Status', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['No Churn (0)', 'Churn (1)'])
    axes[1].set_xlabel('Customer Status', fontsize=11)
    axes[1].set_ylabel('Monthly Charges ($)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, '05_churn_by_monthly_charges.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Monthly Charges vs Churn plot to: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, target_col: str = 'Churn', save_dir: str = None):
    """
    Plot Pearson correlation matrix for numerical features and the target variable.
    """
    save_dir = save_dir or get_viz_dir()
    num_cols = [c for c in ['tenure', 'MonthlyCharges', 'TotalCharges', target_col] if c in df.columns]
    
    if len(num_cols) < 2:
        return
        
    corr = df[num_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='Blues', cbar=True,
                square=True, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 11, 'weight': 'bold'})
    plt.title('Numerical Features Correlation Matrix', fontsize=14, fontweight='bold', pad=14)
    plt.tight_layout()
    output_path = os.path.join(save_dir, '09_numerical_correlation_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Correlation Heatmap to: {output_path}")


def run_full_eda(clean_df: pd.DataFrame, target_col: str = 'Churn'):
    """
    Run complete exploratory data analysis and generate all required plots.
    """
    save_dir = get_viz_dir()
    print("\n" + "="*60)
    print("        RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)
    
    # 1. Overall Churn Distribution
    plot_churn_distribution(clean_df, target_col=target_col, save_dir=save_dir)
    
    # 2. Churn by Gender (if exists)
    if 'gender' in clean_df.columns:
        plot_churn_by_categorical(clean_df, 'gender', 'Gender', '02_churn_by_gender.png',
                                  target_col=target_col, save_dir=save_dir)
                                  
    # 3. Churn by Contract Type (if exists)
    if 'Contract' in clean_df.columns:
        plot_churn_by_categorical(clean_df, 'Contract', 'Contract Type', '03_churn_by_contract.png',
                                  target_col=target_col, save_dir=save_dir)
                                  
    # 4. Churn by Tenure
    plot_churn_by_tenure(clean_df, target_col=target_col, save_dir=save_dir)
    
    # 5. Churn by Monthly Charges
    plot_churn_by_monthly_charges(clean_df, target_col=target_col, save_dir=save_dir)
    
    # 6. Churn by Payment Method (if exists)
    if 'PaymentMethod' in clean_df.columns:
        plot_churn_by_categorical(clean_df, 'PaymentMethod', 'Payment Method', '06_churn_by_payment_method.png',
                                  target_col=target_col, save_dir=save_dir)
                                  
    # 7. Churn by Internet Service (if exists)
    if 'InternetService' in clean_df.columns:
        plot_churn_by_categorical(clean_df, 'InternetService', 'Internet Service Type', '07_churn_by_internet_service.png',
                                  target_col=target_col, save_dir=save_dir)
                                  
    # 8. Churn by Senior Citizen Status (if exists)
    if 'SeniorCitizen' in clean_df.columns:
        plot_churn_by_categorical(clean_df, 'SeniorCitizen', 'Senior Citizen Status', '08_churn_by_senior_citizen.png',
                                  target_col=target_col, save_dir=save_dir)
                                  
    # 9. Correlation Analysis
    plot_correlation_heatmap(clean_df, target_col=target_col, save_dir=save_dir)
    
    print("="*60)
    print(f"[SUCCESS] All EDA visualizations generated in '{save_dir}'")
    print("="*60 + "\n")


if __name__ == '__main__':
    from data_preprocessing import prepare_data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'customer_churn.csv')
    artifacts = prepare_data(data_path)
    run_full_eda(artifacts['clean_df'])
