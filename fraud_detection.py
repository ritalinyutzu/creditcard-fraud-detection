"""
Credit Card Fraud Detection - Simplified Implementation (Without Deep Learning)
Dataset: Kaggle Credit Card Fraud Detection
Methods: Tomek Links, SMOTE, Undersampling, Advanced Ensemble Models
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                           precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Imbalanced data handling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.cluster import KMeans

# Advanced models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("Credit Card Fraud Detection - Advanced Implementation")
print("="*80)

# ============================================================================
# PART 1: Data Loading and Initial Exploration
# ============================================================================

def download_data():
    """Download dataset from Kaggle"""
    print("\n[1] Downloading dataset from Kaggle...")
    os.system('kaggle datasets download -d mlg-ulb/creditcardfraud')
    os.system('unzip -o creditcardfraud.zip')
    print("✓ Dataset downloaded successfully!")

def load_and_explore_data():
    """Load and perform initial data exploration"""
    print("\n[2] Loading and exploring data...")
    
    # Load data
    df = pd.read_csv('creditcard.csv')
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Total Transactions: {len(df):,}")
    print(f"   Features: {df.shape[1]}")
    
    # Class distribution
    fraud_count = df['Class'].value_counts()
    print(f"\n📈 Class Distribution:")
    print(f"   Normal (0): {fraud_count[0]:,} ({fraud_count[0]/len(df)*100:.2f}%)")
    print(f"   Fraud (1):  {fraud_count[1]:,} ({fraud_count[1]/len(df)*100:.2f}%)")
    print(f"   Imbalance Ratio: 1:{fraud_count[0]/fraud_count[1]:.1f}")
    
    # Missing values
    print(f"\n🔍 Missing Values: {df.isnull().sum().sum()}")
    
    # Basic statistics
    print("\n📉 Amount Statistics:")
    print(df['Amount'].describe())
    
    print("\n⏰ Time Statistics:")
    print(df['Time'].describe())
    
    return df

# ============================================================================
# PART 2: Exploratory Data Analysis (EDA)
# ============================================================================

def perform_eda(df):
    """Comprehensive EDA with visualizations"""
    print("\n[3] Performing Exploratory Data Analysis...")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # 1. Class distribution pie chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Class distribution
    class_counts = df['Class'].value_counts()
    axes[0, 0].pie(class_counts, labels=['Normal', 'Fraud'], 
                   autopct='%1.2f%%', startangle=90,
                   colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
    
    # Amount distribution by class
    axes[0, 1].hist([df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']], 
                    bins=50, label=['Normal', 'Fraud'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0, 1].set_xlabel('Amount')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Amount Distribution by Class', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Time distribution
    axes[1, 0].hist([df[df['Class']==0]['Time'], df[df['Class']==1]['Time']], 
                    bins=50, label=['Normal', 'Fraud'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Time Distribution by Class', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    
    # Correlation heatmap for V features
    fraud_corr = df[df['Class']==1][['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']].corr()
    sns.heatmap(fraud_corr, ax=axes[1, 1], cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    axes[1, 1].set_title('Correlation Matrix (V1-V10) - Fraud Only', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/eda_overview.png', dpi=300, bbox_inches='tight')
    print("✓ EDA plots saved to 'plots/eda_overview.png'")
    plt.close()
    
    # 2. Feature importance via variance
    features = [col for col in df.columns if col.startswith('V')]
    variances = df[features].var().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(variances)), variances.values, color='steelblue')
    plt.xlabel('Feature Index (Sorted by Variance)')
    plt.ylabel('Variance')
    plt.title('Feature Variance Analysis (V1-V28)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/feature_variance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature variance plot saved")
    plt.close()

# ============================================================================
# PART 3: Data Preprocessing
# ============================================================================

def preprocess_data(df):
    """Preprocess data with scaling"""
    print("\n[4] Preprocessing data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Time and Amount (V features are already scaled from PCA)
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    print("✓ Data preprocessed successfully!")
    return X, y

# ============================================================================
# PART 4: Imbalanced Data Handling
# ============================================================================

def apply_kmeans_undersampling(X, y, n_clusters=7):
    """Apply KMeans undersampling (from your PDF)"""
    print(f"\n[5.1] Applying KMeans Undersampling (K={n_clusters})...")
    
    # Get majority and minority indices
    majority_idx = y[y==0].index
    minority_idx = y[y==1].index
    
    X_majority = X.loc[majority_idx]
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    kmeans.fit(X_majority)
    
    # Select samples closest to centroids
    cluster_labels = kmeans.labels_
    selected_indices = []
    
    for i in range(n_clusters):
        cluster_indices = majority_idx[cluster_labels == i]
        cluster_center = kmeans.cluster_centers_[i]
        
        # Calculate distances to center
        distances = np.linalg.norm(X_majority.loc[cluster_indices] - cluster_center, axis=1)
        
        # Select samples proportional to cluster size
        n_samples = len(minority_idx) // n_clusters
        closest_idx = cluster_indices[np.argsort(distances)[:n_samples]]
        selected_indices.extend(closest_idx)
    
    # Combine with minority class
    final_indices = list(selected_indices) + list(minority_idx)
    X_resampled = X.loc[final_indices]
    y_resampled = y.loc[final_indices]
    
    print(f"✓ KMeans Undersampling completed!")
    print(f"   Original: {len(y)} | Resampled: {len(y_resampled)}")
    print(f"   Class 0: {sum(y_resampled==0)} | Class 1: {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled

def apply_tomek_links(X, y):
    """Apply Tomek Links to remove boundary samples"""
    print("\n[5.2] Applying Tomek Links...")
    
    tomek = TomekLinks(sampling_strategy='majority')
    X_resampled, y_resampled = tomek.fit_resample(X, y)
    
    print(f"✓ Tomek Links completed!")
    print(f"   Samples removed: {len(y) - len(y_resampled)}")
    print(f"   Class 0: {sum(y_resampled==0)} | Class 1: {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled

def apply_smote_tomek(X, y):
    """Apply SMOTE + Tomek Links combination"""
    print("\n[5.3] Applying SMOTE + Tomek Links...")
    
    smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    
    print(f"✓ SMOTE + Tomek completed!")
    print(f"   Original: {len(y)} | Resampled: {len(y_resampled)}")
    print(f"   Class 0: {sum(y_resampled==0)} | Class 1: {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled

def apply_adasyn(X, y):
    """Apply ADASYN (Adaptive Synthetic Sampling)"""
    print("\n[5.4] Applying ADASYN...")
    
    adasyn = ADASYN(random_state=RANDOM_STATE)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
    print(f"✓ ADASYN completed!")
    print(f"   Original: {len(y)} | Resampled: {len(y_resampled)}")
    print(f"   Class 0: {sum(y_resampled==0)} | Class 1: {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled

# ============================================================================
# PART 5: Model Training and Evaluation
# ============================================================================

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Comprehensive model evaluation"""
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, y_prob)
    
    results = {
        'Model': model_name,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'AUC-ROC': auc
    }
    
    return results

def train_traditional_models(X_train, X_test, y_train, y_test, sampling_method):
    """Train traditional ML models"""
    print(f"\n[6] Training Traditional Models ({sampling_method})...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        result = evaluate_model(y_test, y_pred, y_prob, f"{name} ({sampling_method})")
        results.append(result)
    
    return results

def train_boosting_models(X_train, X_test, y_train, y_test, sampling_method):
    """Train advanced boosting models"""
    print(f"\n[7] Training Boosting Models ({sampling_method})...")
    
    results = []
    
    # XGBoost
    print("   Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate_model(y_test, y_pred, y_prob, f"XGBoost ({sampling_method})"))
    
    # LightGBM
    print("   Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    y_prob = lgb_model.predict_proba(X_test)[:, 1]
    results.append(evaluate_model(y_test, y_pred, y_prob, f"LightGBM ({sampling_method})"))
    
    # CatBoost
    print("   Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=0
    )
    cat_model.fit(X_train, y_train)
    y_pred = cat_model.predict(X_test)
    y_prob = cat_model.predict_proba(X_test)[:, 1]
    results.append(evaluate_model(y_test, y_pred, y_prob, f"CatBoost ({sampling_method})"))
    
    return results

# ============================================================================
# PART 6: Main Pipeline
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Download data (comment out if already downloaded)
    # download_data()
    
    # Load and explore
    df = load_and_explore_data()
    
    # EDA
    perform_eda(df)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Train-Test Split (20-80 as in PDF)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"   Train: {len(X_train_orig):,} samples")
    print(f"   Test:  {len(X_test):,} samples")
    
    # Store all results
    all_results = []
    
    # Strategy 1: KMeans Undersampling
    X_train_kmeans, y_train_kmeans = apply_kmeans_undersampling(X_train_orig, y_train_orig, n_clusters=7)
    all_results.extend(train_traditional_models(X_train_kmeans, X_test, y_train_kmeans, y_test, "KMeans"))
    all_results.extend(train_boosting_models(X_train_kmeans, X_test, y_train_kmeans, y_test, "KMeans"))
    
    # Strategy 2: Tomek Links
    X_train_tomek, y_train_tomek = apply_tomek_links(X_train_orig, y_train_orig)
    all_results.extend(train_traditional_models(X_train_tomek, X_test, y_train_tomek, y_test, "Tomek"))
    all_results.extend(train_boosting_models(X_train_tomek, X_test, y_train_tomek, y_test, "Tomek"))
    
    # Strategy 3: SMOTE + Tomek
    X_train_smote, y_train_smote = apply_smote_tomek(X_train_orig, y_train_orig)
    all_results.extend(train_traditional_models(X_train_smote, X_test, y_train_smote, y_test, "SMOTE+Tomek"))
    all_results.extend(train_boosting_models(X_train_smote, X_test, y_train_smote, y_test, "SMOTE+Tomek"))
    
    # Strategy 4: ADASYN
    X_train_adasyn, y_train_adasyn = apply_adasyn(X_train_orig, y_train_orig)
    all_results.extend(train_boosting_models(X_train_adasyn, X_test, y_train_adasyn, y_test, "ADASYN"))
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL MODELS")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    # Display results
    print("\n📊 Top 10 Models by F1-Score:")
    print(results_df[['Model', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC-ROC']].head(10).to_string(index=False))
    
    # Save complete results
    results_df.to_csv('model_results_complete.csv', index=False)
    print(f"\n✓ Complete results saved to 'model_results_complete.csv'")
    
    # Comparison with original PDF results
    print("\n" + "="*80)
    print("COMPARISON WITH ORIGINAL RESULTS (from PDF)")
    print("="*80)
    print("\nOriginal Best Results:")
    print("   SVM Model: F1=75.61%, Precision=93.94%, Recall=63.27%")
    print("   Xgboost:   F1=70.18%, Precision=82.19%, Recall=61.22%")
    print("   Ensemble:  F1=73.66%, Precision=64.8%,  Recall=85.66%")
    
    best_model = results_df.iloc[0]
    print(f"\nNew Best Result:")
    print(f"   {best_model['Model']}")
    print(f"   F1={best_model['F1-Score']:.2%}, Precision={best_model['Precision']:.2%}, Recall={best_model['Recall']:.2%}")
    
    improvement = ((best_model['F1-Score'] - 0.7561) / 0.7561) * 100
    print(f"\n🎯 F1-Score Improvement: {improvement:+.2f}%")
    
    # Visualize top models
    plt.figure(figsize=(14, 8))
    top_10 = results_df.head(10)
    
    x = range(len(top_10))
    width = 0.25
    
    plt.bar([i - width for i in x], top_10['Precision'], width, label='Precision', alpha=0.8)
    plt.bar(x, top_10['Recall'], width, label='Recall', alpha=0.8)
    plt.bar([i + width for i in x], top_10['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.title('Top 10 Models Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, top_10['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/top_models_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to 'plots/top_models_comparison.png'")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("   📁 plots/eda_overview.png - Exploratory data analysis")
    print("   📁 plots/feature_variance.png - Feature importance")
    print("   📁 plots/top_models_comparison.png - Model comparison")
    print("   📄 model_results_complete.csv - Complete results table")
    

if __name__ == "__main__":
    main()