"""
Complete Analysis and Report Generation
生成完整的模型分析報告和 Confusion Matrix 視覺化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 設定視覺化風格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results():
    """載入結果"""
    print("📊 Loading results...")
    df = pd.read_csv('model_results_fast.csv')
    return df

def plot_all_confusion_matrices(results_df):
    """繪製所有模型的 Confusion Matrix"""
    print("\n📈 Generating confusion matrices...")
    
    n_models = len(results_df)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        ax = axes[idx]
        
        # 建立 confusion matrix
        cm = np.array([[row['TN'], row['FP']], 
                       [row['FN'], row['TP']]])
        
        # 繪製 heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'],
                   ax=ax, cbar=False,
                   annot_kws={'size': 12, 'weight': 'bold'})
        
        # 標題包含關鍵指標
        title = f"{row['Model']}\n"
        title += f"F1={row['F1-Score']:.4f} | P={row['Precision']:.4f} | R={row['Recall']:.4f}"
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
    
    # 隱藏多餘的子圖
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/all_confusion_matrices.png")
    plt.close()

def plot_metrics_comparison(results_df):
    """繪製指標比較圖"""
    print("\n📊 Generating metrics comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1-Score comparison
    ax1 = axes[0, 0]
    colors = ['#e74c3c' if i == 0 else '#3498db' for i in range(len(results_df))]
    ax1.barh(range(len(results_df)), results_df['F1-Score'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Model'], fontsize=9)
    ax1.set_xlabel('F1-Score', fontweight='bold')
    ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. Precision vs Recall
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['Recall'], results_df['Precision'], 
                         s=results_df['F1-Score']*500, alpha=0.6, 
                         c=range(len(results_df)), cmap='viridis')
    for idx, row in results_df.iterrows():
        ax2.annotate(row['Model'].split('(')[0].strip(), 
                    (row['Recall'], row['Precision']),
                    fontsize=7, ha='center')
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision vs Recall (bubble size = F1-Score)', 
                 fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. AUC-ROC comparison
    ax3 = axes[1, 0]
    colors = ['#2ecc71' if i == 0 else '#95a5a6' for i in range(len(results_df))]
    ax3.barh(range(len(results_df)), results_df['AUC-ROC'], color=colors, alpha=0.8)
    ax3.set_yticks(range(len(results_df)))
    ax3.set_yticklabels(results_df['Model'], fontsize=9)
    ax3.set_xlabel('AUC-ROC', fontweight='bold')
    ax3.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. Grouped bar chart
    ax4 = axes[1, 1]
    x = np.arange(len(results_df))
    width = 0.25
    ax4.bar(x - width, results_df['Precision'], width, label='Precision', alpha=0.8)
    ax4.bar(x, results_df['Recall'], width, label='Recall', alpha=0.8)
    ax4.bar(x + width, results_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.split('(')[0].strip()[:15] for m in results_df['Model']], 
                        rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: plots/metrics_comparison.png")
    plt.close()

def generate_markdown_report(results_df):
    """生成完整的 Markdown 報告"""
    print("\n📝 Generating README.md report...")
    
    best_model = results_df.iloc[0]
    
    report = f"""# Credit Card Fraud Detection - Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Table of Contents
- [Executive Summary](#executive-summary)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Best Model Analysis](#best-model-analysis)
- [Detailed Results](#detailed-results)
- [Confusion Matrices](#confusion-matrices)
- [Conclusions and Recommendations](#conclusions-and-recommendations)

---

## 🎯 Executive Summary

This project implements a comprehensive fraud detection system using machine learning techniques to identify fraudulent credit card transactions. We evaluated **{len(results_df)} different model configurations** across multiple sampling strategies.

### Key Findings:
- **Best Model:** {best_model['Model']}
- **F1-Score:** {best_model['F1-Score']:.4f} ({best_model['F1-Score']*100:.2f}%)
- **Precision:** {best_model['Precision']:.4f} ({best_model['Precision']*100:.2f}%)
- **Recall:** {best_model['Recall']:.4f} ({best_model['Recall']*100:.2f}%)
- **AUC-ROC:** {best_model['AUC-ROC']:.4f}

### Performance Highlight:
- **True Positives:** {int(best_model['TP'])} frauds correctly detected
- **False Positives:** {int(best_model['FP'])} legitimate transactions flagged
- **False Negatives:** {int(best_model['FN'])} frauds missed
- **True Negatives:** {int(best_model['TN'])} legitimate transactions correctly identified

---

## 📊 Dataset Overview

### Credit Card Fraud Dataset (Kaggle)
- **Total Transactions:** 284,807
- **Features:** 30 (V1-V28 from PCA, Time, Amount)
- **Target Variable:** Class (0 = Normal, 1 = Fraud)

### Class Distribution:
- **Normal Transactions:** 284,315 (99.83%)
- **Fraudulent Transactions:** 492 (0.17%)
- **Imbalance Ratio:** 1:577.9

This severe class imbalance requires special handling techniques, which is why we employed multiple resampling strategies.

---

## 🔬 Methodology

### 1. Data Preprocessing
- **Scaling:** StandardScaler applied to Time and Amount features
- **Train-Test Split:** 80-20 split with stratification
- **Features:** All 30 features used (PCA-transformed V1-V28 + Time + Amount)

### 2. Imbalanced Data Handling

We tested three resampling strategies:

#### A. Random Undersampling
- **Method:** Randomly reduce majority class samples
- **Result:** Balanced dataset with equal class distribution
- **Advantage:** Fast training, reduced computational cost
- **Disadvantage:** Loss of potentially useful data

#### B. SMOTE (Synthetic Minority Over-sampling Technique)
- **Method:** Generate synthetic minority class samples
- **Result:** Increased minority class samples
- **Advantage:** No data loss, creates diverse samples
- **Disadvantage:** Potential overfitting to noise

#### C. SMOTE + Tomek Links
- **Method:** SMOTE followed by Tomek Links cleaning
- **Result:** Balanced dataset with cleaned decision boundaries
- **Advantage:** Best of both worlds - data augmentation + boundary cleaning
- **Disadvantage:** Higher computational cost

### 3. Models Evaluated

| Model | Type | Key Parameters |
|-------|------|----------------|
| **Logistic Regression** | Linear | max_iter=1000 |
| **Decision Tree** | Tree-based | max_depth=10 |
| **Random Forest** | Ensemble | n_estimators=50, max_depth=10 |
| **XGBoost** | Gradient Boosting | n_estimators=50, max_depth=6 |
| **LightGBM** | Gradient Boosting | n_estimators=50, max_depth=6 |

### 4. Evaluation Metrics

- **Precision:** How many predicted frauds are actually frauds?
  - Formula: TP / (TP + FP)
  - Important for minimizing false alarms

- **Recall (Sensitivity):** How many actual frauds are we catching?
  - Formula: TP / (TP + FN)
  - Critical for fraud detection - we want to catch all frauds

- **F1-Score:** Harmonic mean of Precision and Recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Balanced metric for imbalanced datasets

- **AUC-ROC:** Area Under the Receiver Operating Characteristic curve
  - Measures model's ability to distinguish between classes
  - Range: 0.5 (random) to 1.0 (perfect)

---

## 🏆 Model Performance

### Top 5 Models (Ranked by F1-Score)

"""
    
    # Top 5 models table
    for idx, row in results_df.head(5).iterrows():
        report += f"""
#### {idx + 1}. {row['Model']}

| Metric | Value |
|--------|-------|
| **F1-Score** | {row['F1-Score']:.4f} ({row['F1-Score']*100:.2f}%) |
| **Precision** | {row['Precision']:.4f} ({row['Precision']*100:.2f}%) |
| **Recall** | {row['Recall']:.4f} ({row['Recall']*100:.2f}%) |
| **Accuracy** | {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%) |
| **AUC-ROC** | {row['AUC-ROC']:.4f} |

**Confusion Matrix:**
```
                Predicted
                Normal    Fraud
Actual Normal   {int(row['TN']):6d}    {int(row['FP']):5d}
       Fraud    {int(row['FN']):6d}    {int(row['TP']):5d}
```

**Interpretation:**
- ✅ Correctly identified **{int(row['TP'])} fraudulent transactions** (True Positives)
- ✅ Correctly identified **{int(row['TN'])} normal transactions** (True Negatives)
- ❌ Incorrectly flagged **{int(row['FP'])} normal transactions** as fraud (False Positives)
- ❌ Missed **{int(row['FN'])} fraudulent transactions** (False Negatives)

---
"""
    
    report += f"""
## 🥇 Best Model Analysis

### {best_model['Model']}

This model achieved the best overall performance with an F1-Score of **{best_model['F1-Score']:.4f}**.

### Why This Model Performs Best:

1. **Balanced Performance:** Achieves good balance between Precision ({best_model['Precision']:.2%}) and Recall ({best_model['Recall']:.2%})

2. **High AUC-ROC:** Score of {best_model['AUC-ROC']:.4f} indicates excellent discrimination ability

3. **Practical Implications:**
   - Out of every 100 predicted frauds, approximately **{int(best_model['Precision']*100)} are actually fraudulent**
   - Out of every 100 actual frauds, approximately **{int(best_model['Recall']*100)} are caught**
   - Miss rate: {100 - best_model['Recall']*100:.2f}%

### Business Impact:

- **Cost of False Positives:** {int(best_model['FP'])} legitimate transactions flagged
  - May cause customer inconvenience
  - Recommendation: Implement secondary verification

- **Cost of False Negatives:** {int(best_model['FN'])} frauds missed
  - Direct financial loss
  - Average fraud amount: ~$122 (dataset statistic)
  - Estimated potential loss: ${int(best_model['FN']) * 122:,}

- **Fraud Detection Rate:** {best_model['Recall']*100:.2f}%
  - Industry benchmark: 70-85%
  - Our model: **{'Above' if best_model['Recall'] > 0.75 else 'Below'}** benchmark

---

## 📈 Detailed Results

### Complete Performance Table

| Rank | Model | Precision | Recall | F1-Score | Accuracy | AUC-ROC |
|------|-------|-----------|--------|----------|----------|---------|
"""
    
    for idx, row in results_df.iterrows():
        report += f"| {idx+1} | {row['Model']:40s} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['Accuracy']:.4f} | {row['AUC-ROC']:.4f} |\n"
    
    report += """
### Performance by Sampling Method

"""
    
    # Group by sampling method
    for method in ['Undersample', 'SMOTE', 'SMOTE+Tomek']:
        method_df = results_df[results_df['Model'].str.contains(method)]
        if len(method_df) > 0:
            avg_f1 = method_df['F1-Score'].mean()
            avg_precision = method_df['Precision'].mean()
            avg_recall = method_df['Recall'].mean()
            
            report += f"""
#### {method}
- **Average F1-Score:** {avg_f1:.4f}
- **Average Precision:** {avg_precision:.4f}
- **Average Recall:** {avg_recall:.4f}
- **Number of Models:** {len(method_df)}
"""
    
    report += """
---

## 📊 Confusion Matrices

All confusion matrices are visualized in `plots/all_confusion_matrices.png`.

### How to Read Confusion Matrices:

```
                Predicted
                Normal    Fraud
Actual Normal   TN        FP      ← Normal transactions
       Fraud    FN        TP      ← Fraudulent transactions
```

- **TN (True Negative):** Correctly identified normal transactions ✅
- **FP (False Positive):** Normal transactions incorrectly flagged as fraud ❌
- **FN (False Negative):** Frauds that were missed ❌❌ (Most critical!)
- **TP (True Positive):** Correctly caught fraudulent transactions ✅✅

### Visual Analysis:

See `plots/metrics_comparison.png` for:
- F1-Score comparison across all models
- Precision vs Recall scatter plot
- AUC-ROC comparison
- Grouped metrics bar chart

---

## 🎯 Conclusions and Recommendations

### Key Findings:

1. **Best Sampling Strategy:** """
    
    # Find best sampling method
    best_sampling = best_model['Model'].split('(')[1].replace(')', '')
    report += f"{best_sampling}\n"
    
    report += f"""
   - Provides optimal balance between data augmentation and computational efficiency
   
2. **Best Model Type:** {best_model['Model'].split('(')[0].strip()}
   - Demonstrates superior performance in handling imbalanced fraud data
   - Achieves F1-Score of {best_model['F1-Score']:.4f}

3. **Trade-offs:**
   - **High Precision models:** Fewer false alarms but may miss more frauds
   - **High Recall models:** Catch more frauds but more false positives
   - **Recommended:** Prioritize Recall for fraud detection (better safe than sorry)

### Recommendations:

#### 1. Model Deployment
- ✅ Deploy {best_model['Model']} as primary fraud detection model
- ✅ Set up real-time prediction pipeline
- ✅ Implement automatic retraining schedule (monthly)

#### 2. Threshold Optimization
- Current threshold: 0.5 (default)
- Consider lowering threshold to increase Recall (catch more frauds)
- Recommended: Test thresholds between 0.3-0.5 based on business needs

#### 3. False Positive Mitigation
- Implement multi-stage verification for flagged transactions
- Use transaction velocity checks
- Incorporate customer behavior patterns
- Add manual review for high-value transactions

#### 4. Continuous Improvement
- **Short-term (1-3 months):**
  - Collect more fraud examples for model retraining
  - A/B test different thresholds in production
  - Monitor precision/recall in production

- **Medium-term (3-6 months):**
  - Implement ensemble methods (combining multiple models)
  - Add feature engineering (transaction patterns, time-based features)
  - Explore deep learning approaches (LSTM, Autoencoder)

- **Long-term (6-12 months):**
  - Build real-time fraud detection system
  - Integrate external data sources (device fingerprinting, geolocation)
  - Implement adaptive learning system

#### 5. Business Integration
- Create dashboard for fraud monitoring
- Set up alert system for high-risk transactions
- Train customer service team on false positive handling
- Establish fraud investigation workflow

### Next Steps:

1. **Model Validation:**
   - [ ] Cross-validation on different time periods
   - [ ] Test on more recent data
   - [ ] Validate with domain experts

2. **Production Preparation:**
   - [ ] Save best model (`.pkl` or `.joblib`)
   - [ ] Create API endpoint for predictions
   - [ ] Set up monitoring and logging
   - [ ] Document model versioning

3. **Documentation:**
   - [ ] Create API documentation
   - [ ] Write deployment guide
   - [ ] Prepare presentation for stakeholders

---

## 📁 Project Structure

```
creditcard_fraud_project/
│
├── creditcard.csv                      # Original dataset
├── fraud_detection_fast.py             # Training script
├── complete_analysis_and_report.py     # This analysis script
│
├── model_results_fast.csv              # Results table
├── README.md                           # This report
│
└── plots/
    ├── model_comparison.png            # Initial comparison
    ├── all_confusion_matrices.png      # All confusion matrices
    └── metrics_comparison.png          # Detailed metrics comparison
```

---

## 🔧 Requirements

```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
imbalanced-learn >= 0.8.0
xgboost >= 1.4.0
lightgbm >= 3.2.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

---

## 📚 References

1. Kaggle Credit Card Fraud Detection Dataset
   - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Original Research Paper:
   - "Credit Card Fraud Detection Using Machine Learning"
   - Comparison baseline: SVM F1=75.61%

3. Techniques Used:
   - SMOTE: Chawla et al. (2002)
   - Random Forest: Breiman (2001)
   - XGBoost: Chen & Guestrin (2016)
   - LightGBM: Ke et al. (2017)

---

## 👥 Contact & Support

For questions or issues regarding this analysis:
- Review the code in `fraud_detection_fast.py`
- Check visualizations in `plots/` directory
- Examine detailed results in `model_results_fast.csv`

---

**Report Generated by:** Credit Card Fraud Detection Analysis System  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version:** 1.0

---

## 📊 Appendix: All Model Metrics

### Complete Confusion Matrix Data

```python
"""
    
    for idx, row in results_df.iterrows():
        report += f"\n# {row['Model']}\n"
        report += f"TP={int(row['TP'])}, TN={int(row['TN'])}, FP={int(row['FP'])}, FN={int(row['FN'])}\n"
        report += f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1-Score']:.4f}\n"
    
    report += """```

---

**End of Report**

*This report was automatically generated. For updates or modifications, re-run `complete_analysis_and_report.py`.*
"""
    
    # 寫入檔案
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✓ Saved: README.md")

def main():
    """主程式"""
    print("="*80)
    print("Complete Analysis and Report Generation")
    print("="*80)
    
    # 確保 plots 資料夾存在
    os.makedirs('plots', exist_ok=True)
    
    # 載入結果
    results_df = load_results()
    
    # 依 F1-Score 排序
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # 生成所有 confusion matrices
    plot_all_confusion_matrices(results_df)
    
    # 生成指標比較圖
    plot_metrics_comparison(results_df)
    
    # 生成 Markdown 報告
    generate_markdown_report(results_df)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 plots/all_confusion_matrices.png    - All confusion matrices")
    print("  📊 plots/metrics_comparison.png        - Detailed metrics comparison")
    print("  📄 README.md                           - Complete analysis report")
    print("\nNext steps:")
    print("  1. Review README.md for comprehensive analysis")
    print("  2. Check plots/ directory for visualizations")
    print("  3. Use best model for deployment")

if __name__ == "__main__":
    main()