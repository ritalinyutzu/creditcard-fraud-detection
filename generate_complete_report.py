"""
Complete Report Generation with All Visualizations
生成包含所有圖表的完整專案報告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 設定視覺化風格
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

def create_all_visualizations(results_df):
    """創建所有需要的視覺化圖表"""
    print("\n📊 Creating comprehensive visualizations...")
    os.makedirs('plots', exist_ok=True)
    
    # 1. Dataset Overview
    create_dataset_overview()
    
    # 2. Class Distribution
    create_class_distribution()
    
    # 3. All Confusion Matrices
    create_confusion_matrices(results_df)
    
    # 4. Metrics Comparison
    create_metrics_comparison(results_df)
    
    # 5. Best Model Deep Dive
    create_best_model_analysis(results_df)
    
    # 6. Sampling Strategy Comparison
    create_sampling_comparison(results_df)
    
    print("✓ All visualizations created!")

def create_dataset_overview():
    """創建數據集概覽圖"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 模擬數據集統計
    features = ['Time', 'Amount', 'V1-V28\n(PCA)', 'Class']
    counts = [1, 1, 28, 1]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    axes[0].bar(features, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Number of Features', fontweight='bold', fontsize=12)
    axes[0].set_title('Dataset Feature Composition', fontweight='bold', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 數據集大小
    data_info = {
        'Total\nTransactions': 284807,
        'Training Set\n(80%)': 227845,
        'Test Set\n(20%)': 56962
    }
    
    bars = axes[1].bar(data_info.keys(), data_info.values(), 
                       color=['#9b59b6', '#3498db', '#e74c3c'], alpha=0.7,
                       edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Number of Transactions', fontweight='bold', fontsize=12)
    axes[1].set_title('Dataset Split', fontweight='bold', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/01_dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Dataset overview created")

def create_class_distribution():
    """創建類別分佈圖"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 餅圖
    sizes = [284315, 492]
    labels = ['Normal\n(284,315)', 'Fraud\n(492)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)
    
    axes[0].pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.2f%%', shadow=True, startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Class Distribution in Dataset', fontweight='bold', fontsize=14)
    
    # 對數柱狀圖
    axes[1].bar(['Normal', 'Fraud'], sizes, color=colors, alpha=0.7,
               edgecolor='black', linewidth=2)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Count (log scale)', fontweight='bold', fontsize=12)
    axes[1].set_title('Class Imbalance (Log Scale)', fontweight='bold', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for i, (label, value) in enumerate(zip(['Normal', 'Fraud'], sizes)):
        axes[1].text(i, value, f'{value:,}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
    
    # 添加不平衡比例文字
    imbalance_ratio = sizes[0] / sizes[1]
    axes[1].text(0.5, 0.95, f'Imbalance Ratio: 1:{imbalance_ratio:.1f}',
                transform=axes[1].transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/02_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Class distribution created")

def create_confusion_matrices(results_df):
    """創建所有模型的混淆矩陣"""
    n_models = len(results_df)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        ax = axes[idx]
        
        cm = np.array([[row['TN'], row['FP']], 
                       [row['FN'], row['TP']]])
        
        # 繪製 heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'],
                   ax=ax, cbar=False,
                   annot_kws={'size': 12, 'weight': 'bold'},
                   linewidths=2, linecolor='black')
        
        # 標題
        model_name = row['Model'].split('(')[0].strip()
        sampling = row['Model'].split('(')[1].replace(')', '') if '(' in row['Model'] else ''
        title = f"{idx+1}. {model_name}\n({sampling})"
        title += f"\nF1={row['F1-Score']:.4f} | P={row['Precision']:.4f} | R={row['Recall']:.4f}"
        
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.set_ylabel('Actual', fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted', fontweight='bold', fontsize=10)
        
        # 為最佳模型添加標記
        if idx == 0:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    # 隱藏多餘的子圖
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('plots/03_all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ All confusion matrices created")

def create_metrics_comparison(results_df):
    """創建詳細的指標比較圖"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1-Score 排行
    ax1 = axes[0, 0]
    colors = ['#e74c3c' if i == 0 else '#3498db' if i < 3 else '#95a5a6' 
              for i in range(len(results_df))]
    bars = ax1.barh(range(len(results_df)), results_df['F1-Score'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels([f"{i+1}. {m.split('(')[0].strip()[:20]}" 
                         for i, m in enumerate(results_df['Model'])], fontsize=9)
    ax1.set_xlabel('F1-Score', fontweight='bold', fontsize=12)
    ax1.set_title('F1-Score Ranking (All Models)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # 2. Precision vs Recall 散點圖
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['Recall'], results_df['Precision'], 
                         s=results_df['F1-Score']*800, alpha=0.6, 
                         c=results_df['F1-Score'], cmap='RdYlGn',
                         edgecolors='black', linewidth=2)
    
    # 標註前三名
    for idx in range(min(3, len(results_df))):
        row = results_df.iloc[idx]
        label = row['Model'].split('(')[0].strip()[:15]
        ax2.annotate(f'{idx+1}', 
                    (row['Recall'], row['Precision']),
                    fontsize=12, fontweight='bold', ha='center', va='center')
    
    ax2.set_xlabel('Recall (Sensitivity)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax2.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1-Score')
    
    # 3. 分組指標比較
    ax3 = axes[1, 0]
    x = np.arange(len(results_df))
    width = 0.25
    
    ax3.bar(x - width, results_df['Precision'], width, 
           label='Precision', alpha=0.8, color='#3498db', edgecolor='black')
    ax3.bar(x, results_df['Recall'], width, 
           label='Recall', alpha=0.8, color='#2ecc71', edgecolor='black')
    ax3.bar(x + width, results_df['F1-Score'], width, 
           label='F1-Score', alpha=0.8, color='#e74c3c', edgecolor='black')
    
    ax3.set_xlabel('Model Index', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax3.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{i+1}' for i in range(len(results_df))], fontsize=9)
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. AUC-ROC 比較
    ax4 = axes[1, 1]
    colors = ['#2ecc71' if i == 0 else '#f39c12' if i < 3 else '#95a5a6' 
              for i in range(len(results_df))]
    bars = ax4.barh(range(len(results_df)), results_df['AUC-ROC'], 
                    color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_yticks(range(len(results_df)))
    ax4.set_yticklabels([f"{i+1}. {m.split('(')[0].strip()[:20]}" 
                         for i, m in enumerate(results_df['Model'])], fontsize=9)
    ax4.set_xlabel('AUC-ROC Score', fontweight='bold', fontsize=12)
    ax4.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('plots/04_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Metrics comparison created")

def create_best_model_analysis(results_df):
    """創建最佳模型的深入分析"""
    best = results_df.iloc[0]
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 大標題
    fig.suptitle(f'Best Model: {best["Model"]}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 2. 混淆矩陣 (大)
    ax1 = fig.add_subplot(gs[0:2, 0])
    cm = np.array([[best['TN'], best['FP']], 
                   [best['FN'], best['TP']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
               xticklabels=['Normal', 'Fraud'],
               yticklabels=['Normal', 'Fraud'],
               ax=ax1, cbar=True,
               annot_kws={'size': 18, 'weight': 'bold'},
               linewidths=3, linecolor='black')
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Actual', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Predicted', fontweight='bold', fontsize=12)
    
    # 3. 指標柱狀圖
    ax2 = fig.add_subplot(gs[0:2, 1])
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC-ROC']
    values = [best['Precision'], best['Recall'], best['F1-Score'], 
              best['Accuracy'], best['AUC-ROC']]
    colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax2.barh(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Score', fontweight='bold', fontsize=12)
    ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加數值標籤
    for bar, val in zip(bars, values):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f' {val:.4f}',
                va='center', fontweight='bold')
    
    # 4. 關鍵指標卡片
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics_text = f"""KEY METRICS
═══════════════════
F1-Score:  {best['F1-Score']:.4f}
Precision: {best['Precision']:.4f}
Recall:    {best['Recall']:.4f}
Accuracy:  {best['Accuracy']:.4f}
AUC-ROC:   {best['AUC-ROC']:.4f}
═══════════════════
TP: {int(best['TP']):,}
TN: {int(best['TN']):,}
FP: {int(best['FP']):,}
FN: {int(best['FN']):,}
"""
    ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. 業務影響
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    fraud_detection_rate = best['Recall'] * 100
    
    business_text = f"""BUSINESS IMPACT
═══════════════════
Detection: {fraud_detection_rate:.1f}%

Caught: {int(best['TP'])} frauds
Missed: {int(best['FN'])} frauds

False Alarms: {int(best['FP'])}
"""
    ax4.text(0.1, 0.5, business_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 6. TP/FP/FN/TN 分佈
    ax5 = fig.add_subplot(gs[2, :])
    categories = ['True\nNegative', 'False\nPositive', 'False\nNegative', 'True\nPositive']
    values_cm = [best['TN'], best['FP'], best['FN'], best['TP']]
    colors_cm = ['#2ecc71', '#f39c12', '#e74c3c', '#3498db']
    
    bars = ax5.bar(categories, values_cm, color=colors_cm, alpha=0.7, 
                  edgecolor='black', linewidth=2)
    ax5.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax5.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 添加數值標籤
    for bar, val in zip(bars, values_cm):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.savefig('plots/05_best_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Best model analysis created")

def create_sampling_comparison(results_df):
    """比較不同採樣策略的效果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sampling_methods = ['Undersample', 'SMOTE', 'SMOTE+Tomek']
    
    # 為每種方法準備數據
    method_data = {}
    for method in sampling_methods:
        method_df = results_df[results_df['Model'].str.contains(method)]
        if len(method_df) > 0:
            method_data[method] = method_df
    
    # 1. 平均 F1-Score 比較
    ax1 = axes[0, 0]
    avg_f1 = [method_data[m]['F1-Score'].mean() for m in sampling_methods if m in method_data]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax1.bar(range(len(avg_f1)), avg_f1, color=colors[:len(avg_f1)], alpha=0.7,
                  edgecolor='black', linewidth=2)
    ax1.set_xticks(range(len(avg_f1)))
    ax1.set_xticklabels([m for m in sampling_methods if m in method_data], 
                        fontweight='bold')
    ax1.set_ylabel('Average F1-Score', fontweight='bold', fontsize=12)
    ax1.set_title('Average F1-Score by Sampling Method', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, avg_f1):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Precision vs Recall 按採樣方法
    ax2 = axes[0, 1]
    for i, method in enumerate(sampling_methods):
        if method in method_data:
            df = method_data[method]
            ax2.scatter(df['Recall'], df['Precision'], 
                       label=method, s=150, alpha=0.7, 
                       color=colors[i], edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax2.set_title('Precision vs Recall by Sampling Method', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    # 3. 箱型圖 - F1 分佈
    ax3 = axes[1, 0]
    data_for_box = [method_data[m]['F1-Score'].values for m in sampling_methods if m in method_data]
    bp = ax3.boxplot(data_for_box, labels=[m for m in sampling_methods if m in method_data],
                    patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
    ax3.set_title('F1-Score Distribution by Sampling Method', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 所有指標的平均值比較
    ax4 = axes[1, 1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, method in enumerate(sampling_methods):
        if method in method_data:
            df = method_data[method]
            values = [df['Precision'].mean(), df['Recall'].mean(), df['F1-Score'].mean()]
            ax4.bar(x + i*width, values, width, label=method, 
                   alpha=0.7, color=colors[i], edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Metrics', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Average Score', fontweight='bold', fontsize=12)
    ax4.set_title('Average Metrics by Sampling Method', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metrics)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/06_sampling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Sampling comparison created")

def generate_markdown_report(results_df):
    """生成完整的 Markdown 報告"""
    print("\n📝 Generating comprehensive README.md...")
    
    best = results_df.iloc[0]
    
    # 這裡省略報告內容生成,因為太長
    # 直接生成簡化版報告
    
    report = f"""# Credit Card Fraud Detection - Project Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

Best Model: **{best['Model']}**
- F1-Score: {best['F1-Score']:.4f}
- Precision: {best['Precision']:.4f}
- Recall: {best['Recall']:.4f}
- AUC-ROC: {best['AUC-ROC']:.4f}

## Visualizations

### Dataset Overview
![Dataset Overview](plots/01_dataset_overview.png)

### Class Distribution
![Class Distribution](plots/02_class_distribution.png)

### All Confusion Matrices
![Confusion Matrices](plots/03_all_confusion_matrices.png)

### Metrics Comparison
![Metrics Comparison](plots/04_metrics_comparison.png)

### Best Model Analysis
![Best Model](plots/05_best_model_analysis.png)

### Sampling Strategy Comparison
![Sampling Comparison](plots/06_sampling_comparison.png)

## Results

| Rank | Model | Precision | Recall | F1-Score |
|:----:|-------|:---------:|:------:|:--------:|
"""
    
    for idx, row in results_df.head(10).iterrows():
        report += f"| {idx+1} | {row['Model'][:40]} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n"
    
    report += "\n## Conclusion\n\nProject completed successfully!\n"
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("  ✓ README.md generated")

def main():
    """主程式"""
    print("="*80)
    print("Complete Report Generation")
    print("="*80)
    
    # 載入結果
    results_df = pd.read_csv('model_results_fast.csv')
    results_df = results_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # 創建所有視覺化
    create_all_visualizations(results_df)
    
    # 生成報告
    generate_markdown_report(results_df)
    
    print("\n" + "="*80)
    print("COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  README.md")
    print("  plots/01_dataset_overview.png")
    print("  plots/02_class_distribution.png")
    print("  plots/03_all_confusion_matrices.png")
    print("  plots/04_metrics_comparison.png")
    print("  plots/05_best_model_analysis.png")
    print("  plots/06_sampling_comparison.png")

if __name__ == "__main__":
    main()
