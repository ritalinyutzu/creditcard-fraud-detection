"""
Credit Card Fraud Detection - Fast Implementation
優化版本：移除慢速模型，專注於高效能模型
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                           f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Imbalanced data handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Advanced models (快速的boosting模型)
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("Credit Card Fraud Detection - Fast Implementation")
print("="*80)

def load_and_explore_data():
    """Load and explore data"""
    print("\n[1] Loading data...")
    df = pd.read_csv('creditcard.csv')
    
    print(f"\n📊 Dataset: {df.shape[0]:,} transactions, {df.shape[1]} features")
    fraud_count = df['Class'].value_counts()
    print(f"📈 Normal: {fraud_count[0]:,} ({fraud_count[0]/len(df)*100:.2f}%)")
    print(f"📈 Fraud:  {fraud_count[1]:,} ({fraud_count[1]/len(df)*100:.2f}%)")
    
    return df

def preprocess_data(df):
    """Preprocess with scaling"""
    print("\n[2] Preprocessing...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    print("✓ Preprocessing done!")
    return X, y

def apply_sampling(X, y, method='smote'):
    """Apply sampling method"""
    print(f"\n[3] Applying {method.upper()}...")
    
    if method == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(random_state=RANDOM_STATE)
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    print(f"✓ Original: {len(y):,} → Resampled: {len(y_resampled):,}")
    print(f"  Class 0: {sum(y_resampled==0):,} | Class 1: {sum(y_resampled==1):,}")
    
    return X_resampled, y_resampled

def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Evaluate model performance"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(y_true, y_prob)
    
    return {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy,
        'AUC-ROC': auc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }

def train_fast_models(X_train, X_test, y_train, y_test, sampling_method):
    """Train only fast models"""
    print(f"\n[4] Training Models ({sampling_method})...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, 
                                               n_jobs=-1, random_state=RANDOM_STATE),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                                    n_jobs=-1, random_state=RANDOM_STATE, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                                      n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"   Training {name}...", end=' ')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        result = evaluate_model(y_test, y_pred, y_prob, f"{name} ({sampling_method})")
        results.append(result)
        print(f"✓ F1={result['F1-Score']:.4f}")
    
    return results

def visualize_results(results_df):
    """Create comparison plots"""
    print("\n[5] Creating visualizations...")
    os.makedirs('plots', exist_ok=True)
    
    # Top 10 models
    top_10 = results_df.head(10)
    
    plt.figure(figsize=(14, 8))
    x = range(len(top_10))
    width = 0.25
    
    plt.bar([i - width for i in x], top_10['Precision'], width, 
            label='Precision', alpha=0.8, color='#3498db')
    plt.bar(x, top_10['Recall'], width, 
            label='Recall', alpha=0.8, color='#2ecc71')
    plt.bar([i + width for i in x], top_10['F1-Score'], width, 
            label='F1-Score', alpha=0.8, color='#e74c3c')
    
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.title('Top 10 Models Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(x, top_10['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Plot saved to plots/model_comparison.png")
    plt.close()

def main():
    """Main pipeline"""
    
    # Load data
    df = load_and_explore_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Train-Test Split
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📊 Train: {len(X_train_orig):,} | Test: {len(X_test):,}")
    
    # Store all results
    all_results = []
    
    # Strategy 1: Random Undersampling (fastest)
    X_train_under, y_train_under = apply_sampling(X_train_orig, y_train_orig, 'undersample')
    all_results.extend(train_fast_models(X_train_under, X_test, y_train_under, y_test, "Undersample"))
    
    # Strategy 2: SMOTE
    X_train_smote, y_train_smote = apply_sampling(X_train_orig, y_train_orig, 'smote')
    all_results.extend(train_fast_models(X_train_smote, X_test, y_train_smote, y_test, "SMOTE"))
    
    # Strategy 3: SMOTE + Tomek
    X_train_st, y_train_st = apply_sampling(X_train_orig, y_train_orig, 'smote_tomek')
    all_results.extend(train_fast_models(X_train_st, X_test, y_train_st, y_test, "SMOTE+Tomek"))
    
    # Results summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    print("\n📊 Top 10 Models by F1-Score:")
    print(results_df[['Model', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].head(10).to_string(index=False))
    
    # Save results
    results_df.to_csv('model_results_fast.csv', index=False)
    print(f"\n✓ Results saved to model_results_fast.csv")
    
    # Visualize
    visualize_results(results_df)
    
    # Best model
    best = results_df.iloc[0]
    print(f"\n🏆 Best Model: {best['Model']}")
    print(f"   F1-Score:  {best['F1-Score']:.4f}")
    print(f"   Precision: {best['Precision']:.4f}")
    print(f"   Recall:    {best['Recall']:.4f}")
    print(f"   AUC-ROC:   {best['AUC-ROC']:.4f}")
    
    print("\n" + "="*80)
    print("✅ COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("   📊 plots/model_comparison.png")
    print("   📄 model_results_fast.csv")

if __name__ == "__main__":
    main()
