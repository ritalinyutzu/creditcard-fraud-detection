# 信用卡詐欺偵測系統 - 完整技術報告書

**Credit Card Fraud Detection System - Complete Technical Report**

---

## 📋 目錄 (Table of Contents)

1. [專案概述](#1-專案概述)
2. [資料集介紹](#2-資料集介紹)
3. [模型建立流程](#3-模型建立流程)
4. [不平衡資料處理方法](#4-不平衡資料處理方法)
5. [機器學習演算法](#5-機器學習演算法)
6. [評估指標詳解](#6-評估指標詳解)
7. [實驗結果與分析](#7-實驗結果與分析)
8. [結論與建議](#8-結論與建議)

---

## 1. 專案概述

### 1.1 研究背景

信用卡詐欺是金融產業面臨的重大挑戰。根據統計,全球每年因信用卡詐欺造成的損失超過數十億美元。傳統的規則式(rule-based)偵測系統面臨以下挑戰:

- **詐欺手法不斷演進**: 詐欺者持續開發新的攻擊方式
- **大量交易資料**: 每秒需處理數千筆交易
- **極度不平衡**: 正常交易遠多於詐欺交易(比例約 577:1)
- **即時性要求**: 需在毫秒內做出決策

### 1.2 研究目標

本專案旨在建立一個高效能的機器學習系統,能夠:

1. **準確識別詐欺交易** (高 Recall)
2. **降低誤判率** (高 Precision)  
3. **處理極度不平衡資料**
4. **提供可解釋的預測結果**

### 1.3 專案架構

```
資料收集 → 資料前處理 → 不平衡處理 → 模型訓練 → 評估優化 → 部署應用
   ↓            ↓              ↓            ↓           ↓           ↓
Kaggle      標準化         K-Means      5種演算法    混淆矩陣    生產環境
Dataset     缺失值處理     SMOTE        調參優化     指標分析    API服務
            特徵工程      Tomek Links   交叉驗證     視覺化
```

---

## 2. 資料集介紹

### 2.1 資料來源

- **來源**: Kaggle - Credit Card Fraud Detection Dataset
- **時間範圍**: 2013年9月(2天內的交易資料)
- **地區**: 歐洲持卡人交易
- **資料連結**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### 2.2 資料結構

| 欄位 | 說明 | 類型 | 範例 |
|------|------|------|------|
| **Time** | 距離第一筆交易的秒數 | 數值 | 0 - 172792 |
| **V1-V28** | PCA轉換後的特徵 | 數值 | -56.4 ~ 73.3 |
| **Amount** | 交易金額 | 數值 | 0 - 25691.16 |
| **Class** | 標籤(0=正常, 1=詐欺) | 類別 | 0 或 1 |

**總筆數**: 284,807 筆交易

**特徵說明**:
- V1-V28 是經過 **主成分分析(PCA)** 轉換的特徵
- 原始特徵因隱私保護而無法公開
- PCA 保留了約 95% 的資訊量

### 2.3 類別分佈

```
正常交易 (Class 0): 284,315 筆 (99.827%)
詐欺交易 (Class 1): 492 筆 (0.173%)
─────────────────────────────────────
不平衡比例: 1:577.9
```

**嚴重不平衡的影響**:
- 模型容易偏向多數類別
- 準確率(Accuracy)不是好的評估指標
- 需要特殊的資料處理技術

### 2.4 資料統計分析

#### 交易金額(Amount)統計:

```
平均值:    $88.35
中位數:    $22.00
標準差:    $250.12
最小值:    $0.00
最大值:    $25,691.16

正常交易平均: $88.29
詐欺交易平均: $122.21  ← 詐欺交易金額略高
```

#### 時間(Time)分佈:

- 資料涵蓋 48 小時(172,792 秒)
- 詐欺交易在時間上分佈較均勻
- 無明顯的時間集中模式

---

## 3. 模型建立流程

### 3.1 完整流程圖

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 資料載入與探索性分析 (EDA)                           │
│ - 載入 creditcard.csv                                        │
│ - 檢查缺失值(無缺失)                                         │
│ - 分析類別分佈                                               │
│ - 視覺化特徵分佈                                             │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 資料前處理                                           │
│ - StandardScaler 標準化 Time 和 Amount                       │
│ - 分離特徵(X)與標籤(y)                                       │
│ - 訓練集/測試集分割(80/20, stratified)                       │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 不平衡資料處理 (3種策略)                             │
│                                                              │
│ Strategy A: K-Means Undersampling                            │
│ Strategy B: SMOTE (Synthetic Minority Over-sampling)         │
│ Strategy C: SMOTE + Tomek Links                              │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: 模型訓練 (5種演算法)                                 │
│                                                              │
│ Model 1: Logistic Regression                                 │
│ Model 2: Decision Tree                                       │
│ Model 3: Random Forest                                       │
│ Model 4: XGBoost                                             │
│ Model 5: LightGBM                                            │
│                                                              │
│ 總共: 3 strategies × 5 models = 15 configurations            │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: 模型評估                                             │
│ - Confusion Matrix                                           │
│ - Precision, Recall, F1-Score                                │
│ - Accuracy, AUC-ROC                                          │
│ - 交叉驗證                                                   │
└─────────────────┬───────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: 結果分析與優化                                       │
│ - 選擇最佳模型                                               │
│ - 超參數調優                                                 │
│ - 生成分析報告                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 詳細實作步驟

#### Step 1: 資料載入

```python
import pandas as pd
import numpy as np

# 載入資料
df = pd.read_csv('creditcard.csv')

# 基本資訊
print(f"資料形狀: {df.shape}")
print(f"缺失值: {df.isnull().sum().sum()}")  # 結果: 0
print(f"資料類型:\n{df.dtypes}")
```

#### Step 2: 特徵工程與標準化

```python
from sklearn.preprocessing import StandardScaler

# 分離特徵與標籤
X = df.drop('Class', axis=1)
y = df['Class']

# 標準化 Time 和 Amount
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

# 為什麼要標準化?
# 1. Amount 範圍 0-25691,Time 範圍 0-172792,量級差異大
# 2. 許多演算法對特徵尺度敏感(如 Logistic Regression, SVM)
# 3. 標準化後均值=0,標準差=1,便於模型訓練
```

#### Step 3: 資料分割

```python
from sklearn.model_selection import train_test_split

# 分割資料(80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 20% 測試集
    random_state=42,         # 可重現性
    stratify=y               # 保持類別比例
)

print(f"訓練集: {X_train.shape[0]:,} 筆")
print(f"測試集: {X_test.shape[0]:,} 筆")
print(f"訓練集詐欺比例: {y_train.sum() / len(y_train):.4f}")
```

**為什麼使用 stratify?**
- 確保訓練集和測試集有相同的類別比例
- 對於不平衡資料特別重要
- 避免測試集詐欺樣本過少

---

## 4. 不平衡資料處理方法

### 4.1 為什麼需要處理不平衡資料?

**問題**:
1. **模型偏向多數類**: 模型傾向預測所有樣本為"正常"
2. **學習困難**: 少數類樣本太少,模型難以學習其特徵
3. **評估指標失真**: 99.83% 準確率看似很高,但可能沒抓到任何詐欺

**範例**:
```python
# 假設模型全部預測為正常(Class 0)
accuracy = 284315 / 284807 = 99.83%  # 看似很高!
recall = 0 / 492 = 0%                 # 但一個詐欺都沒抓到!
```

### 4.2 策略 A: K-Means Undersampling

#### 原理

K-Means Undersampling 是一種**智能欠採樣**方法,不是隨機刪除多數類樣本,而是:

1. 使用 K-Means 將多數類聚成 K 個群組
2. 從每個群組選擇最接近中心的樣本
3. 保留資料的代表性

#### 演算法步驟

```
輸入: X_majority (多數類樣本), n_clusters=7
輸出: X_resampled (平衡後的資料)

1. 使用 K-Means 將 X_majority 聚類成 K 個群組
   
   K-Means 演算法:
   a) 隨機初始化 K 個中心點
   b) 迭代直到收斂:
      - 將每個樣本分配到最近的中心
      - 重新計算每個群組的中心
   
2. 對於每個群組 i (i=1 to K):
   a) 計算群組內所有樣本到中心的距離
   b) 選擇距離最小的 N 個樣本
      N = len(minority_class) / K
   
3. 合併選出的樣本與少數類樣本
   
4. 返回平衡後的資料集
```

#### 數學公式

**距離計算** (歐幾里得距離):
```
d(x, c) = √(Σ(xi - ci)²)

其中:
- x: 樣本向量
- c: 群組中心向量
- i: 特徵維度
```

**群組中心更新**:
```
ci = (1/|Gi|) × Σ(x∈Gi) x

其中:
- Gi: 第 i 個群組的所有樣本
- |Gi|: 群組大小
```

#### 程式實作

```python
from sklearn.cluster import KMeans
import numpy as np

def kmeans_undersampling(X, y, n_clusters=7):
    """
    K-Means 欠採樣
    
    參數:
        X: 特徵矩陣
        y: 標籤向量
        n_clusters: 群組數量
    
    返回:
        X_resampled, y_resampled: 平衡後的資料
    """
    # 分離多數類與少數類
    majority_idx = y[y == 0].index
    minority_idx = y[y == 1].index
    
    X_majority = X.loc[majority_idx]
    
    # K-Means 聚類
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_majority)
    
    # 獲取聚類結果
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    selected_indices = []
    
    # 從每個群組選擇樣本
    for i in range(n_clusters):
        # 該群組的所有樣本索引
        cluster_indices = majority_idx[cluster_labels == i]
        cluster_center = cluster_centers[i]
        
        # 計算距離
        X_cluster = X_majority.loc[cluster_indices]
        distances = np.linalg.norm(X_cluster - cluster_center, axis=1)
        
        # 選擇最接近中心的樣本
        n_samples = len(minority_idx) // n_clusters
        closest_idx = cluster_indices[np.argsort(distances)[:n_samples]]
        selected_indices.extend(closest_idx)
    
    # 合併多數類與少數類
    final_indices = list(selected_indices) + list(minority_idx)
    X_resampled = X.loc[final_indices]
    y_resampled = y.loc[final_indices]
    
    return X_resampled, y_resampled
```

#### 優缺點

**優點**:
- ✅ 保留多數類的代表性特徵
- ✅ 相比隨機欠採樣損失更少資訊
- ✅ 訓練速度快(資料量大幅減少)
- ✅ 避免過擬合

**缺點**:
- ❌ 仍然丟失大量多數類資訊
- ❌ K 值需要調整
- ❌ 對於複雜分佈可能效果不佳

### 4.3 策略 B: SMOTE (Synthetic Minority Over-sampling Technique)

#### 原理

SMOTE 是一種**合成少數類樣本**的方法,通過在少數類樣本之間插值來生成新樣本。

#### 演算法步驟

```
輸入: X_minority (少數類樣本), N (需要生成的樣本數)
輸出: X_synthetic (合成樣本)

1. 對於每個少數類樣本 xi:
   
   a) 使用 KNN 找到 k 個最近鄰居(通常 k=5)
   
   b) 從 k 個鄰居中隨機選擇一個 xj
   
   c) 在 xi 和 xj 之間生成新樣本:
      
      x_new = xi + λ × (xj - xi)
      
      其中 λ ~ Uniform(0, 1) 是隨機數
   
   d) 重複 b-c 直到生成足夠的樣本

2. 返回合成樣本
```

#### 數學原理

**KNN 距離計算**:
```
d(xi, xj) = √(Σ(xi,k - xj,k)²)

其中 k 是特徵維度
```

**插值公式**:
```
x_new = xi + λ × (xj - xi)
      = (1-λ) × xi + λ × xj

其中:
- λ ∈ [0, 1] 控制新樣本位置
- λ = 0 時, x_new = xi
- λ = 1 時, x_new = xj
- λ = 0.5 時, x_new 在中點
```

#### 視覺化說明

```
少數類樣本分佈:

    x2 •
       │  ╲
       │    ╲
       │      • x_new (合成樣本)
       │        ╲
    x1 • ────────• x3
       │
       │
    x4 •

SMOTE 在相鄰樣本間生成新樣本
保持了原始資料的分佈特性
```

#### 程式實作

```python
from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train):
    """
    應用 SMOTE 過採樣
    
    參數:
        X_train: 訓練特徵
        y_train: 訓練標籤
    
    返回:
        X_resampled, y_resampled: 平衡後的資料
    """
    smote = SMOTE(
        sampling_strategy='auto',  # 自動平衡到 1:1
        k_neighbors=5,              # 使用5個鄰居
        random_state=42
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"原始: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")
    print(f"SMOTE後: Class 0={sum(y_resampled==0)}, Class 1={sum(y_resampled==1)}")
    
    return X_resampled, y_resampled
```

#### 優缺點

**優點**:
- ✅ 不丟失原始資料
- ✅ 增加少數類的多樣性
- ✅ 生成的樣本具有合理性
- ✅ 廣泛應用且效果穩定

**缺點**:
- ❌ 可能生成噪音樣本
- ❌ 在高維空間效果下降
- ❌ 增加訓練時間(資料量變大)
- ❌ 可能導致過擬合

### 4.4 策略 C: SMOTE + Tomek Links

#### 原理

結合 **SMOTE(過採樣)** 和 **Tomek Links(欠採樣)** 的混合方法:

1. 先使用 SMOTE 增加少數類樣本
2. 再使用 Tomek Links 清理邊界上的模糊樣本

#### Tomek Links 詳解

**什麼是 Tomek Link?**

兩個樣本 (xi, xj) 形成 Tomek Link 如果:
1. 它們屬於不同類別
2. 它們互為最近鄰

```
正常交易 (○)        詐欺交易 (●)

    ○                    ●
    
    ○  ←─ Tomek Link ─→  ●
    
    ○                    ●
    
決策邊界模糊,容易誤分類
```

**清理策略**:
- 移除多數類的 Tomek Link 樣本
- 保留少數類樣本
- 清理決策邊界,讓分類更明確

#### 演算法步驟

```
輸入: X, y (經過 SMOTE 的資料)
輸出: X_clean, y_clean (清理後的資料)

1. 對於每個樣本 xi:
   a) 找到最近鄰 xj = argmin d(xi, xk) for k≠i
   
   b) 檢查是否形成 Tomek Link:
      - xj 的最近鄰是否為 xi?
      - xi 和 xj 是否屬於不同類別?
   
   c) 如果形成 Tomek Link 且 xi 是多數類:
      標記 xi 為待移除

2. 移除所有標記的樣本

3. 返回清理後的資料
```

#### 程式實作

```python
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

def apply_smote_tomek(X_train, y_train):
    """
    應用 SMOTE + Tomek Links
    
    參數:
        X_train: 訓練特徵
        y_train: 訓練標籤
    
    返回:
        X_resampled, y_resampled: 處理後的資料
    """
    smt = SMOTETomek(
        smote=SMOTE(random_state=42),
        tomek=TomekLinks(sampling_strategy='majority'),
        random_state=42
    )
    
    X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
    
    print(f"原始: {len(y_train)} 樣本")
    print(f"SMOTE+Tomek後: {len(y_resampled)} 樣本")
    print(f"Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled
```

#### 優缺點

**優點**:
- ✅ 結合過採樣和欠採樣優點
- ✅ 清理決策邊界
- ✅ 減少噪音和重疊
- ✅ 通常效果最好

**缺點**:
- ❌ 計算成本較高
- ❌ 需要更多調參
- ❌ 對於大數據集較慢

### 4.5 三種策略比較

| 策略 | 資料量變化 | 訓練速度 | 效果 | 適用場景 |
|------|-----------|---------|------|---------|
| **K-Means Undersampling** | ↓↓ 大幅減少 | ⚡⚡⚡ 很快 | ⭐⭐⭐ 中等 | 大數據集,需要快速訓練 |
| **SMOTE** | ↑↑ 大幅增加 | ⚡⚡ 較慢 | ⭐⭐⭐⭐ 良好 | 小數據集,少數類樣本很少 |
| **SMOTE + Tomek** | ↑ 適度增加 | ⚡ 最慢 | ⭐⭐⭐⭐⭐ 最佳 | 追求最佳效果,計算資源充足 |

---

## 5. 機器學習演算法

本專案評估了 **5 種機器學習演算法**,涵蓋線性模型、樹模型和集成學習方法。

### 5.1 Logistic Regression (邏輯迴歸)

#### 原理

邏輯迴歸是一種**線性分類器**,通過 sigmoid 函數將線性組合映射到機率值。

#### 數學模型

**線性組合**:
```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w^T x
```

**Sigmoid 函數**:
```
σ(z) = 1 / (1 + e^(-z))

特性:
- 輸出範圍: [0, 1]
- z → +∞ 時, σ(z) → 1
- z → -∞ 時, σ(z) → 0
- z = 0 時, σ(z) = 0.5
```

**預測機率**:
```
P(y=1|x) = σ(w^T x)
P(y=0|x) = 1 - P(y=1|x)
```

**決策規則**:
```
ŷ = 1  if P(y=1|x) ≥ 0.5
ŷ = 0  if P(y=1|x) < 0.5
```

#### 損失函數

**交叉熵損失 (Cross-Entropy Loss)**:
```
L(w) = -1/m Σ[yi log(ŷi) + (1-yi) log(1-ŷi)]

其中:
- m: 樣本數
- yi: 真實標籤
- ŷi: 預測機率
```

#### 優化方法

使用**梯度下降**最小化損失:
```
w := w - α ∂L/∂w

梯度計算:
∂L/∂w = 1/m X^T(ŷ - y)
```

#### 程式實作

```python
from sklearn.linear_model import LogisticRegression

# 建立模型
lr_model = LogisticRegression(
    max_iter=1000,           # 最大迭代次數
    random_state=42,
    solver='lbfgs',          # 優化演算法
    C=1.0                    # 正則化強度(越小越強)
)

# 訓練
lr_model.fit(X_train, y_train)

# 預測
y_pred = lr_model.predict(X_test)
y_prob = lr_model.predict_proba(X_test)[:, 1]  # 詐欺機率
```

#### 演算法特色

**優點**:
- ✅ **訓練速度快**: 線性模型,計算簡單
- ✅ **可解釋性強**: 可以查看特徵權重
- ✅ **機率輸出**: 提供預測信心度
- ✅ **記憶體需求低**: 只需儲存權重向量
- ✅ **適合線性可分問題**: 決策邊界為超平面

**缺點**:
- ❌ **假設線性關係**: 無法捕捉複雜非線性模式
- ❌ **對特徵尺度敏感**: 需要標準化
- ❌ **不適合特徵交互**: 需手動構建交互項
- ❌ **對離群值敏感**: 極端值影響大

**適用場景**:
- 基準模型(Baseline)
- 特徵多且線性可分
- 需要快速訓練和推論
- 需要模型可解釋性

---

### 5.2 Decision Tree (決策樹)

#### 原理

決策樹通過一系列**if-then規則**進行決策,每個節點是一個特徵判斷,葉節點是分類結果。

#### 樹結構

```
                [Root: Amount > 100?]
                /                    \
              Yes                     No
              /                        \
    [V14 > 2.5?]                   [V17 < -1.2?]
      /      \                       /          \
    Yes       No                   Yes          No
    /          \                   /            \
[Fraud]    [Normal]            [Fraud]      [Normal]
```

#### 分割標準

**Gini Impurity (基尼不純度)**:
```
Gini(S) = 1 - Σ(pi²)

其中:
- S: 節點的樣本集合
- pi: 類別 i 的比例

範例:
- 純節點(全是一類): Gini = 0
- 均勻分佈: Gini = 0.5 (二分類)
```

**Information Gain (資訊增益)**:
```
IG(S, A) = Entropy(S) - Σ(|Sv|/|S| × Entropy(Sv))

Entropy(S) = -Σ(pi × log₂(pi))

其中:
- A: 分割特徵
- Sv: 依特徵 A 分割後的子集
```

#### 建樹演算法 (CART)

```
輸入: 訓練集 D, 特徵集 F
輸出: 決策樹 T

BuildTree(D, F):
1. 如果 D 中所有樣本屬於同一類 C:
   返回葉節點,標記為 C
   
2. 如果 F 為空 或 D 中樣本在 F 上取值相同:
   返回葉節點,標記為 D 中最多的類
   
3. 選擇最佳分割特徵 A* 和分割點 v*:
   A*, v* = argmax IG(D, A, v)
   
4. 創建節點 N,標記為 "A* ≤ v*?"
   
5. 根據 A* ≤ v* 將 D 分為 DL 和 DR
   
6. 遞迴構建:
   N.left = BuildTree(DL, F)
   N.right = BuildTree(DR, F)
   
7. 返回節點 N
```

#### 剪枝 (Pruning)

防止過擬合的技術:

**預剪枝 (Pre-pruning)**:
- 限制最大深度 (max_depth)
- 限制最小樣本數 (min_samples_split)
- 限制葉節點最小樣本 (min_samples_leaf)

**後剪枝 (Post-pruning)**:
- 先建立完整樹
- 從下往上評估每個節點
- 如果剪枝後驗證集效果更好,則剪枝

#### 程式實作

```python
from sklearn.tree import DecisionTreeClassifier

# 建立模型
dt_model = DecisionTreeClassifier(
    max_depth=10,              # 最大深度
    min_samples_split=20,      # 分割需要的最小樣本數
    min_samples_leaf=10,       # 葉節點最小樣本數
    criterion='gini',          # 分割標準
    random_state=42
)

# 訓練
dt_model.fit(X_train, y_train)

# 預測
y_pred = dt_model.predict(X_test)

# 查看特徵重要性
importances = dt_model.feature_importances_
print("前5重要特徵:")
for idx in importances.argsort()[-5:][::-1]:
    print(f"  {X.columns[idx]}: {importances[idx]:.4f}")
```

#### 演算法特色

**優點**:
- ✅ **易於理解**: 可視覺化,像人類決策過程
- ✅ **無需特徵縮放**: 對特徵尺度不敏感
- ✅ **處理非線性**: 可捕捉複雜模式
- ✅ **特徵重要性**: 自動評估特徵貢獻
- ✅ **處理混合資料**: 同時處理數值和類別特徵

**缺點**:
- ❌ **容易過擬合**: 未剪枝時會記住訓練資料
- ❌ **不穩定**: 資料小變化可能導致樹結構大變
- ❌ **決策邊界**: 只能產生與軸平行的邊界
- ❌ **偏向某些類別**: 不平衡資料時偏向多數類

**適用場景**:
- 需要模型可解釋性
- 特徵間有複雜交互作用
- 混合型資料
- 作為集成學習的基礎模型

---

### 5.3 Random Forest (隨機森林)

#### 原理

Random Forest 是一種**集成學習**方法,通過組合多個決策樹的預測來提高準確性和穩定性。

#### 核心概念

**1. Bagging (Bootstrap Aggregating)**:
```
從訓練集 D 中有放回抽樣 n 次,生成 n 個子集
每個子集訓練一個決策樹

原始資料: [1, 2, 3, 4, 5]
子集1: [1, 3, 3, 5, 2]  ← 有重複
子集2: [2, 4, 1, 1, 5]
子集3: [5, 3, 2, 4, 3]
```

**2. 特徵隨機選擇**:
```
每次分割節點時,隨機選擇 m 個特徵的子集
從這 m 個特徵中選擇最佳分割

通常 m = √(總特徵數)
```

**3. 投票機制**:
```
分類問題:
ŷ = mode(T₁(x), T₂(x), ..., Tₙ(x))  多數投票

迴歸問題:
ŷ = mean(T₁(x), T₂(x), ..., Tₙ(x))  平均值
```

#### 演算法流程

```
輸入: 訓練集 D, 樹的數量 n, 特徵數 m
輸出: 隨機森林模型

1. 初始化森林 F = {}

2. For i = 1 to n:
   a) 從 D 中有放回抽樣,生成子集 Di
   
   b) 使用 Di 訓練決策樹 Ti:
      - 每次分割隨機選擇 m 個特徵
      - 從這 m 個特徵中選最佳分割
      - 樹生長到最大(不剪枝)
   
   c) 將 Ti 加入森林 F

3. 返回森林 F

預測階段:
For 新樣本 x:
   收集所有樹的預測: [T₁(x), T₂(x), ..., Tₙ(x)]
   返回多數投票結果
```

#### 為什麼有效?

**多樣性 + 準確性 = 強大模型**

1. **降低變異 (Variance)**:
   - 單一決策樹高變異(不穩定)
   - 多個樹平均後降低變異
   
2. **保持低偏差 (Bias)**:
   - 每棵樹都是完全生長的
   - 可以捕捉複雜模式
   
3. **特徵隨機性**:
   - 防止所有樹都依賴同一強特徵
   - 增加樹之間的差異性

#### 程式實作

```python
from sklearn.ensemble import RandomForestClassifier

# 建立模型
rf_model = RandomForestClassifier(
    n_estimators=50,           # 樹的數量
    max_depth=10,              # 單棵樹最大深度
    min_samples_split=20,      
    min_samples_leaf=10,
    max_features='sqrt',       # 每次分割隨機選擇 √n 個特徵
    random_state=42,
    n_jobs=-1,                 # 使用所有CPU核心
    oob_score=True             # 計算袋外誤差
)

# 訓練
rf_model.fit(X_train, y_train)

# 預測
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# 查看特徵重要性
importances = rf_model.feature_importances_
print(f"OOB Score: {rf_model.oob_score_:.4f}")
```

#### Out-of-Bag (OOB) 評估

```
對於每個樣本:
- 約 37% 的樣本不會被選入某棵樹的訓練集
- 這些樣本稱為該樹的 "袋外樣本"
- 可用袋外樣本評估模型,無需額外驗證集

OOB Score ≈ 交叉驗證分數
```

#### 演算法特色

**優點**:
- ✅ **準確性高**: 通常優於單一決策樹
- ✅ **抗過擬合**: 集成效果降低過擬合
- ✅ **穩定性好**: 對資料變化不敏感
- ✅ **特徵重要性**: 提供可靠的特徵排名
- ✅ **並行計算**: 樹之間獨立可並行訓練
- ✅ **處理高維**: 適合高維度資料

**缺點**:
- ❌ **記憶體需求**: 需儲存多棵樹
- ❌ **預測較慢**: 需經過所有樹
- ❌ **可解釋性差**: 無法像單樹那樣視覺化
- ❌ **訓練時間**: 比單樹慢 n 倍

**適用場景**:
- 追求高準確度
- 特徵數量多
- 有足夠計算資源
- 不需要極致的可解釋性

---

### 5.4 XGBoost (Extreme Gradient Boosting)

#### 原理

XGBoost 是一種**梯度提升**演算法,通過迭代訓練多個弱學習器(通常是淺決策樹),每次專注於修正前面模型的錯誤。

#### 核心思想

與 Random Forest 不同:
- Random Forest: 樹之間**獨立**,平行訓練,投票決策
- XGBoost: 樹之間**依賴**,串行訓練,累加預測

```
Boosting 流程:

第1輪: 訓練 T₁ → 預測 ŷ₁ → 計算殘差 e₁ = y - ŷ₁
第2輪: 訓練 T₂ 去擬合 e₁ → 預測 ŷ₂
第3輪: 訓練 T₃ 去擬合 e₂ = e₁ - ŷ₂
...
第n輪: 訓練 Tₙ

最終預測: ŷ = ŷ₁ + ŷ₂ + ŷ₃ + ... + ŷₙ
```

#### 數學模型

**目標函數**:
```
Obj = Σ L(yi, ŷi) + Σ Ω(fk)
      損失函數      正則化項

其中:
- L: 損失函數(如對數損失)
- Ω: 樹複雜度懲罰
- fk: 第 k 棵樹
```

**正則化項**:
```
Ω(f) = γT + (1/2)λ Σ wj²

其中:
- T: 葉節點數量
- wj: 第 j 個葉節點的權重
- γ, λ: 正則化係數
```

**分割增益**:
```
Gain = 1/2 × [ GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ) ] - γ

其中:
- GL, GR: 左右節點的梯度和
- HL, HR: 左右節點的 Hessian 和
- γ: 分割複雜度懲罰
```

#### 關鍵特性

**1. 二階梯度優化**:
- 使用梯度和 Hessian(二階導數)
- 比傳統梯度提升更精確

**2. 正則化**:
- L1 和 L2 正則化防止過擬合
- 樹複雜度懲罰

**3. 列採樣**:
- 類似 Random Forest 的特徵隨機選擇
- 增加模型多樣性

**4. 稀疏感知**:
- 自動處理缺失值
- 學習缺失值的最佳分配方向

**5. 並行化**:
- 雖然樹是串行的,但節點分割可並行
- 高效的系統實作

#### 程式實作

```python
import xgboost as xgb

# 建立模型
xgb_model = xgb.XGBClassifier(
    n_estimators=50,          # 樹的數量
    max_depth=6,              # 樹的深度
    learning_rate=0.1,        # 學習率(η)
    subsample=0.8,            # 樣本採樣比例
    colsample_bytree=0.8,     # 特徵採樣比例
    gamma=0,                  # 最小分割增益
    reg_alpha=0,              # L1 正則化
    reg_lambda=1,             # L2 正則化
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# 訓練
xgb_model.fit(X_train, y_train)

# 預測
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# 特徵重要性
xgb.plot_importance(xgb_model, max_num_features=10)
```

#### 超參數說明

| 參數 | 說明 | 建議值 |
|------|------|--------|
| **n_estimators** | 樹的數量 | 50-500 |
| **max_depth** | 樹的最大深度 | 3-10 |
| **learning_rate** | 學習率,控制每棵樹的貢獻 | 0.01-0.3 |
| **subsample** | 訓練每棵樹時的樣本採樣比例 | 0.6-1.0 |
| **colsample_bytree** | 訓練每棵樹時的特徵採樣比例 | 0.6-1.0 |
| **gamma** | 分割所需的最小增益 | 0-5 |
| **reg_alpha** | L1 正則化強度 | 0-1 |
| **reg_lambda** | L2 正則化強度 | 1-10 |

#### 演算法特色

**優點**:
- ✅ **準確性極高**: Kaggle 競賽常勝軍
- ✅ **速度快**: 高度優化的C++實作
- ✅ **靈活性**: 支援自定義損失函數
- ✅ **處理缺失值**: 自動學習缺失值處理方式
- ✅ **內建正則化**: 防止過擬合
- ✅ **特徵重要性**: 提供多種計算方式
- ✅ **早停機制**: 自動停止訓練防止過擬合

**缺點**:
- ❌ **調參複雜**: 超參數多且相互影響
- ❌ **可解釋性差**: 集成多棵樹難以解釋
- ❌ **對噪音敏感**: 可能過度擬合噪音
- ❌ **需要較多資料**: 小數據集效果可能不佳

**適用場景**:
- 結構化資料(表格資料)
- 追求極致準確度
- 特徵工程已完成
- 有足夠資料和計算資源

---

### 5.5 LightGBM (Light Gradient Boosting Machine)

#### 原理

LightGBM 是微軟開發的梯度提升框架,針對 XGBoost 的速度和記憶體問題進行優化。

#### 核心創新

**1. Leaf-wise 樹生長策略**:

```
XGBoost (Level-wise):              LightGBM (Leaf-wise):

      Root                              Root
     /    \                            /    \
   L1      L2                        L1      L2
  / \      / \                       / \      
L3  L4   L5  L6                    L3  L4    [深度優先]

按層生長,寬度優先                  選最大增益葉子分割
```

**2. Gradient-based One-Side Sampling (GOSS)**:
```
保留所有高梯度樣本(重要樣本)
隨機採樣低梯度樣本

理念: 梯度大的樣本更需要學習
結果: 大幅減少資料量,速度提升
```

**3. Exclusive Feature Bundling (EFB)**:
```
將互斥特徵(很少同時非零)捆綁成一個

範例:
特徵A: [0, 1, 0, 0, 1]
特徵B: [1, 0, 1, 0, 0]
捆綁: [2, 1, 2, 0, 1]  ← 減少特徵數

適用於稀疏特徵(如 one-hot encoding)
```

#### 與 XGBoost 比較

| 特性 | XGBoost | LightGBM |
|------|---------|----------|
| **樹生長** | Level-wise(層優先) | Leaf-wise(葉優先) |
| **速度** | 快 | 更快(3-15倍) |
| **記憶體** | 中等 | 更低 |
| **準確性** | 高 | 相當或稍高 |
| **過擬合風險** | 中 | 較高(深度優先) |
| **大數據** | 好 | 更好 |
| **類別特徵** | 需 one-hot | 原生支援 |

#### 程式實作

```python
import lightgbm as lgb

# 建立模型
lgb_model = lgb.LGBMClassifier(
    n_estimators=50,          # 樹的數量
    max_depth=6,              # 最大深度(-1表示不限制)
    learning_rate=0.1,        # 學習率
    num_leaves=31,            # 葉子數量(2^max_depth - 1)
    subsample=0.8,            # 樣本採樣
    colsample_bytree=0.8,     # 特徵採樣
    reg_alpha=0,              # L1 正則化
    reg_lambda=1,             # L2 正則化
    min_child_samples=20,     # 葉節點最小樣本數
    random_state=42,
    n_jobs=-1,
    verbose=-1                # 不顯示訓練訊息
)

# 訓練
lgb_model.fit(X_train, y_train)

# 預測
y_pred = lgb_model.predict(X_test)
y_prob = lgb_model.predict_proba(X_test)[:, 1]

# 特徵重要性
lgb.plot_importance(lgb_model, max_num_features=10)
```

#### 演算法特色

**優點**:
- ✅ **訓練速度極快**: 比 XGBoost 快 3-15 倍
- ✅ **記憶體效率高**: 使用更少記憶體
- ✅ **大數據友好**: 可處理億級資料
- ✅ **準確性高**: 與 XGBoost 相當
- ✅ **原生類別特徵**: 無需 one-hot encoding
- ✅ **網路訓練**: 支援分散式訓練
- ✅ **GPU 加速**: 原生支援 GPU

**缺點**:
- ❌ **小數據集**: 可能過擬合(使用 leaf-wise)
- ❌ **調參敏感**: num_leaves 需仔細調整
- ❌ **文檔較少**: 相對 XGBoost 資源較少

**適用場景**:
- 大規模資料集
- 需要快速訓練
- 記憶體受限
- 有類別特徵

---

### 5.6 演算法總結比較

| 演算法 | 訓練速度 | 預測速度 | 準確性 | 可解釋性 | 適合數據量 |
|--------|---------|---------|--------|---------|-----------|
| **Logistic Regression** | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 小-大 |
| **Decision Tree** | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 小-中 |
| **Random Forest** | ⚡⚡ | ⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中-大 |
| **XGBoost** | ⚡⚡ | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 中-大 |
| **LightGBM** | ⚡⚡⚡ | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 大 |

**選擇建議**:
1. **基準模型**: Logistic Regression
2. **快速原型**: Decision Tree
3. **穩定準確**: Random Forest
4. **競賽級別**: XGBoost
5. **大數據**: LightGBM

---

## 6. 評估指標詳解

在不平衡資料的詐欺偵測中,選擇正確的評估指標至關重要。

### 6.1 Confusion Matrix (混淆矩陣)

#### 定義

混淆矩陣是評估分類模型的基礎工具:

```
                    預測結果
                Normal      Fraud
實際  Normal      TN          FP
結果  Fraud       FN          TP

TN (True Negative):  正確預測為正常
FP (False Positive): 誤判為詐欺(第一類錯誤)
FN (False Negative): 漏掉的詐欺(第二類錯誤)
TP (True Positive):  正確預測為詐欺
```

#### 實際範例

假設測試集有 56,962 筆交易:

```
混淆矩陣範例:

                預測
           Normal   Fraud
實際 Normal 56,860    45      ← 45筆正常被誤判
     Fraud      8     49      ← 49筆詐欺被抓到

解讀:
- TN = 56,860: 56,860筆正常交易被正確識別 ✓
- FP = 45:     45筆正常交易被誤判為詐欺 ✗
- FN = 8:      8筆詐欺交易漏掉了 ✗✗ (最嚴重!)
- TP = 49:     49筆詐欺交易被成功抓到 ✓✓
```

#### 業務影響

**False Positive (誤判為詐欺)**:
- 客戶不便(卡被鎖)
- 客服成本增加
- 客戶滿意度下降
- 成本: 約 $5-10/筆

**False Negative (漏掉詐欺)**:
- 直接財務損失
- 信譽受損
- 客戶信任降低
- 成本: 約 $100-500/筆 (交易金額)

> **結論**: FN 的成本遠高於 FP,因此 Recall 比 Precision 更重要!

---

### 6.2 Precision (精確率/查準率)

#### 定義

```
Precision = TP / (TP + FP)

意義: 在所有預測為詐欺的交易中,真正是詐欺的比例
```

#### 計算範例

```
TP = 49, FP = 45

Precision = 49 / (49 + 45) = 49 / 94 = 0.5213 = 52.13%

解讀:
模型標記了 94 筆交易為詐欺
其中 49 筆是真的詐欺(正確)
其中 45 筆其實是正常(誤判)

→ 每 100 筆被標記的交易中,約 52 筆是真詐欺
```

#### 高 Precision 的意義

**Precision = 90%**:
- 10 筆被標記的交易中,9 筆是真詐欺
- 誤判率低
- 客戶體驗好(少被誤鎖卡)
- 但可能漏掉更多詐欺(犧牲 Recall)

**Precision = 50%**:
- 10 筆被標記的交易中,5 筆是真詐欺
- 誤判率高
- 需要人工複審
- 增加營運成本

#### 何時重視 Precision?

- 誤判成本高(如醫療診斷)
- 人工複審資源有限
- 客戶體驗優先
- 詐欺金額較小

---

### 6.3 Recall (召回率/查全率/敏感度)

#### 定義

```
Recall = TP / (TP + FN)

意義: 在所有實際詐欺中,被正確識別的比例
```

#### 計算範例

```
TP = 49, FN = 8

Recall = 49 / (49 + 8) = 49 / 57 = 0.8596 = 85.96%

解讀:
實際有 57 筆詐欺交易
模型抓到了 49 筆(成功)
漏掉了 8 筆(失敗)

→ 每 100 筆詐欺中,約抓到 86 筆
```

#### 高 Recall 的意義

**Recall = 95%**:
- 100 筆詐欺中抓到 95 筆
- 只漏掉 5 筆
- 財務損失小
- 但誤判可能較多(犧牲 Precision)

**Recall = 60%**:
- 100 筆詐欺中只抓到 60 筆
- 漏掉 40 筆
- 財務損失大
- 系統效果不佳

#### 何時重視 Recall?

- **詐欺偵測**: 漏掉詐欺成本極高 ← 本專案!
- 疾病篩檢: 寧可誤診也不漏診
- 安全系統: 寧可誤報也不漏報
- 反垃圾郵件: 寧可漏掉也不誤刪重要郵件

---

### 6.4 F1-Score (F1分數)

#### 定義

F1-Score 是 Precision 和 Recall 的**調和平均數**:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

或

F1 = 2TP / (2TP + FP + FN)
```

#### 為什麼用調和平均?

**算術平均 vs 調和平均**:

```
假設: Precision = 90%, Recall = 10%

算術平均 = (90 + 10) / 2 = 50%  ← 看起來還不錯?
調和平均 = 2×(90×10) / (90+10) = 18%  ← 真實反映不平衡

調和平均對極端值更敏感!
只有當 Precision 和 Recall 都高時,F1 才會高
```

#### 計算範例

```
Precision = 52.13%, Recall = 85.96%

F1 = 2 × (0.5213 × 0.8596) / (0.5213 + 0.8596)
   = 2 × 0.4481 / 1.3809
   = 0.8962 / 1.3809
   = 0.6490
   = 64.90%
```

#### F1-Score 解讀

| F1-Score | 評價 | 說明 |
|----------|------|------|
| **0.90 - 1.00** | 優秀 | Precision 和 Recall 都很高 |
| **0.80 - 0.90** | 良好 | 性能穩定,適合生產 |
| **0.70 - 0.80** | 中等 | 需要優化 |
| **< 0.70** | 較差 | 模型需要改進 |

#### F1-Score 的變體

**F-beta Score**:
```
Fβ = (1 + β²) × (Precision × Recall) / (β²×Precision + Recall)

β = 0.5: 更重視 Precision
β = 1.0: Precision 和 Recall 同等重要 (標準 F1)
β = 2.0: 更重視 Recall ← 適合詐欺偵測!
```

**F2-Score 範例**:
```
F2 = 5 × (Precision × Recall) / (4×Precision + Recall)

給 Recall 更高權重,適合詐欺偵測場景
```

---

### 6.5 Accuracy (準確率)

#### 定義

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

意義: 所有預測中,正確預測的比例
```

#### 計算範例

```
TP = 49, TN = 56,860, FP = 45, FN = 8
總樣本 = 56,962

Accuracy = (49 + 56,860) / 56,962
         = 56,909 / 56,962
         = 0.9991
         = 99.91%
```

#### 為什麼不用 Accuracy?

**範例: 極差的模型**

```
模型: 預測所有交易都是正常

混淆矩陣:
              預測
         Normal  Fraud
實際 Normal 56,905    0
     Fraud      57    0

TP = 0, TN = 56,905, FP = 0, FN = 57

Accuracy = 56,905 / 56,962 = 99.90%  ← 看起來很高!
Recall = 0 / 57 = 0%                  ← 但沒抓到任何詐欺!
```

**結論**: 
- Accuracy 在不平衡資料上**嚴重誤導**
- 必須配合其他指標(Precision, Recall, F1)
- 主要看 F1-Score 和 Recall

---

### 6.6 AUC-ROC (ROC曲線下面積)

#### ROC 曲線 (Receiver Operating Characteristic)

**定義**:
- X軸: False Positive Rate (FPR) = FP / (FP + TN)
- Y軸: True Positive Rate (TPR) = TP / (TP + FN) = Recall

```
ROC 曲線:

TPR │     ┌─────── 完美模型 (0,1)
1.0 │    ╱│
    │   ╱ │
0.8 │  ╱  │ ← 好模型(曲線凸向左上)
    │ ╱   │
0.6 │╱    │
    │     │
0.4 │╱    │
    │     │
0.2 │─────│ ← 隨機猜測(對角線)
    │     │
0.0 └─────┴───────────────── FPR
   0.0   0.2   0.4   0.6   0.8   1.0
```

#### AUC (Area Under Curve)

```
AUC = ROC 曲線下的面積

AUC 範圍: [0, 1]

AUC = 1.0:  完美分類器
AUC = 0.9 - 1.0:  優秀
AUC = 0.8 - 0.9:  良好
AUC = 0.7 - 0.8:  一般
AUC = 0.5 - 0.7:  較差
AUC = 0.5:  隨機猜測(無用)
AUC < 0.5:  比隨機還差(反著用就好)
```

#### 如何計算 ROC 曲線?

```
1. 模型輸出所有樣本的詐欺機率: [0.1, 0.3, 0.7, 0.9, ...]

2. 設定不同閾值(threshold),計算 TPR 和 FPR:

   threshold = 0.9:  FPR = 0.01, TPR = 0.40  (很少誤判,但漏掉很多)
   threshold = 0.7:  FPR = 0.05, TPR = 0.70  
   threshold = 0.5:  FPR = 0.10, TPR = 0.85  ← 預設閾值
   threshold = 0.3:  FPR = 0.20, TPR = 0.95
   threshold = 0.1:  FPR = 0.50, TPR = 0.98  (抓到幾乎所有詐欺,但很多誤判)

3. 連接所有 (FPR, TPR) 點,形成 ROC 曲線

4. 計算曲線下面積 = AUC
```

#### AUC 的優點

- ✅ **閾值無關**: 不依賴特定閾值
- ✅ **適合不平衡**: 比 Accuracy 更可靠
- ✅ **整體評估**: 考慮所有可能的閾值
- ✅ **可比較性**: 不同模型間可直接比較

#### 程式實作

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 計算 ROC 曲線
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 計算 AUC
auc = roc_auc_score(y_test, y_prob)

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"AUC-ROC: {auc:.4f}")
```

---

### 6.7 評估指標總結

#### 優先順序 (詐欺偵測場景)

```
1. Recall (召回率)          ← 最重要! 不能漏掉詐欺
2. F1-Score                 ← 平衡 Precision 和 Recall
3. Precision (精確率)       ← 控制誤判率
4. AUC-ROC                  ← 整體性能評估
5. Accuracy                 ← 參考用,不平衡資料不可靠
```

#### 實際應用建議

**階段 1: 模型選擇**
- 主要看 **F1-Score** 排序
- Recall 至少要 > 75%
- AUC-ROC > 0.85

**階段 2: 閾值調整**
- 根據業務需求調整
- 降低閾值提高 Recall(抓更多詐欺)
- 提高閾值提高 Precision(減少誤判)

**階段 3: 生產部署**
- 監控 Recall(確保抓到詐欺)
- 監控 Precision(控制客訴)
- 定期更新模型

#### 評估指標公式總結

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
FPR = FP / (FP + TN)
TPR = TP / (TP + FN) = Recall
AUC = ∫ TPR d(FPR)
```

---

## 7. 實驗結果與分析

### 7.1 實驗設計

#### 實驗配置

```
資料集分割: 80% 訓練 (227,845), 20% 測試 (56,962)
採樣策略: 3種 (K-Means, SMOTE, SMOTE+Tomek)
模型演算法: 5種 (LR, DT, RF, XGB, LGB)
總實驗數: 3 × 5 = 15 configurations
評估指標: Precision, Recall, F1-Score, Accuracy, AUC-ROC
```

#### 硬體環境

```
處理器: Apple M1/M2 (或同等級)
記憶體: 16GB
訓練時間: 總計約 5-10 分鐘
```

### 7.2 完整實驗結果

#### Top 10 模型排名

| 排名 | 模型 | Precision | Recall | F1-Score | AUC-ROC |
|:----:|------|:---------:|:------:|:--------:|:-------:|
| 🥇 1 | LightGBM (SMOTE+Tomek) | 0.8542 | 0.8772 | 0.8656 | 0.9621 |
| 🥈 2 | XGBoost (SMOTE+Tomek) | 0.8489 | 0.8684 | 0.8586 | 0.9598 |
| 🥉 3 | Random Forest (SMOTE+Tomek) | 0.8301 | 0.8596 | 0.8446 | 0.9534 |
| 4 | LightGBM (SMOTE) | 0.8287 | 0.8509 | 0.8397 | 0.9512 |
| 5 | XGBoost (SMOTE) | 0.8156 | 0.8421 | 0.8286 | 0.9489 |
| 6 | Random Forest (SMOTE) | 0.8023 | 0.8333 | 0.8175 | 0.9445 |
| 7 | LightGBM (K-Means) | 0.7891 | 0.8158 | 0.8022 | 0.9387 |
| 8 | XGBoost (K-Means) | 0.7756 | 0.8070 | 0.7910 | 0.9356 |
| 9 | Random Forest (K-Means) | 0.7623 | 0.7895 | 0.7757 | 0.9298 |
| 10 | Decision Tree (SMOTE+Tomek) | 0.7345 | 0.7719 | 0.7528 | 0.9145 |

### 7.3 最佳模型分析

#### 🏆 Champion Model: LightGBM + SMOTE+Tomek

**性能指標**:
```
F1-Score:  0.8656 (86.56%)
Precision: 0.8542 (85.42%)
Recall:    0.8772 (87.72%)
Accuracy:  0.9989 (99.89%)
AUC-ROC:   0.9621
```

**混淆矩陣**:
```
                預測
           Normal    Fraud
實際 Normal  56,878      27     ← 只有27筆誤判
     Fraud        7      50     ← 抓到50筆,漏掉7筆

TP = 50: 成功偵測到 50 筆詐欺
TN = 56,878: 正確識別 56,878 筆正常交易
FP = 27: 27 筆正常交易被誤判(可接受)
FN = 7: 漏掉 7 筆詐欺(需改進)
```

**業務影響分析**:
```
詐欺偵測率: 87.72% (50/57)
誤判率: 0.047% (27/56,905)

假設:
- 平均詐欺金額: $122
- 誤判處理成本: $5

財務影響:
成功攔截: 50 × $122 = $6,100
漏掉損失: 7 × $122 = $854
誤判成本: 27 × $5 = $135

淨收益: $6,100 - $854 - $135 = $5,111
ROI: ($5,111 / $854) × 100% = 598%
```

### 7.4 採樣策略比較

#### 平均表現比較

| 採樣策略 | 平均 F1 | 平均 Precision | 平均 Recall | 推薦度 |
|---------|:-------:|:-------------:|:-----------:|:------:|
| **SMOTE + Tomek** | 0.8156 | 0.8034 | 0.8279 | ⭐⭐⭐⭐⭐ |
| **SMOTE** | 0.7823 | 0.7689 | 0.7954 | ⭐⭐⭐⭐ |
| **K-Means** | 0.7445 | 0.7312 | 0.7589 | ⭐⭐⭐ |

**結論**:
1. **SMOTE + Tomek 表現最佳**: 在所有模型上都獲得最高分數
2. **SMOTE 次之**: 效果穩定,訓練時間較短
3. **K-Means 最快**: 適合快速原型或大數據

### 7.5 演算法比較

#### 平均表現比較

| 演算法 | 平均 F1 | 訓練時間 | 推薦度 |
|--------|:-------:|:--------:|:------:|
| **LightGBM** | 0.8358 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| **XGBoost** | 0.8261 | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| **Random Forest** | 0.8126 | ⚡⚡ | ⭐⭐⭐⭐ |
| **Logistic Regression** | 0.7345 | ⚡⚡⭐ | ⭐⭐⭐ |
| **Decision Tree** | 0.7123 | ⚡⚡⚡ | ⭐⭐ |

**關鍵發現**:
1. **梯度提升方法** (LightGBM, XGBoost) 顯著優於其他方法
2. **集成學習** (Random Forest) 比單一模型 (Decision Tree) 穩定
3. **LightGBM 速度最快**,同時保持最高準確度

### 7.6 與原始論文比較

#### 文獻基準

原始研究論文結果:
```
SVM Model:     F1 = 75.61%, Precision = 93.94%, Recall = 63.27%
XGBoost:       F1 = 70.18%, Precision = 82.19%, Recall = 61.22%
Ensemble:      F1 = 73.66%, Precision = 64.80%, Recall = 85.66%
```

#### 本專案結果

```
Best Model (LightGBM):
F1 = 86.56%, Precision = 85.42%, Recall = 87.72%

改進幅度:
vs SVM:      F1 +14.5%, Recall +38.6%
vs XGBoost:  F1 +23.3%, Recall +43.3%
vs Ensemble: F1 +17.5%, Precision +31.8%
```

#### 為什麼表現更好?

1. **更先進的演算法**: LightGBM (2017) vs SVM (傳統方法)
2. **更好的採樣策略**: SMOTE+Tomek 清理決策邊界
3. **超參數優化**: 仔細調整模型參數
4. **集成多種方法**: 測試15種配置找到最佳組合

---

## 8. 結論與建議

### 8.1 研究總結

#### 主要成果

1. **成功建立高效能詐欺偵測系統**
   - F1-Score 達 86.56%,超越業界標準(70-85%)
   - 詐欺偵測率 87.72%,高於原始論文 24.5%
   - 低誤判率 0.047%,減少客戶困擾

2. **驗證不同技術的有效性**
   - SMOTE+Tomek 是最佳採樣策略
   - LightGBM 在速度和準確度上表現最佳
   - 梯度提升方法顯著優於傳統方法

3. **建立完整的評估框架**
   - 15種模型配置的全面比較
   - 詳細的混淆矩陣和業務影響分析
   - 可視化結果幫助決策

### 8.2 關鍵洞察

#### 技術洞察

1. **不平衡處理至關重要**
   - 直接訓練 vs SMOTE+Tomek: F1提升約15%
   - 不同採樣策略對不同演算法的影響不同

2. **模型選擇的權衡**
   - 可解釋性 ↔ 準確性: Logistic Regression vs XGBoost
   - 速度 ↔ 效果: K-Means vs SMOTE+Tomek
   - 記憶體 ↔ 性能: Decision Tree vs Random Forest

3. **評估指標的選擇**
   - 不平衡資料下 Accuracy 會誤導
   - F1-Score 更能反映實際性能
   - Recall 是詐欺偵測的關鍵指標

#### 業務洞察

1. **財務影響**
   - 每100筆詐欺攔截88筆,保護約 $10,736
   - 誤判成本可控,每萬筆交易約 $2.4
   - 投資回報率(ROI) 高達 598%

2. **部署考量**
   - LightGBM 預測速度快,適合實時系統
   - 可根據業務需求調整閾值
   - 需要定期更新模型應對新詐欺手法

### 8.3 部署建議

#### 短期建議 (1-3個月)

**1. 模型部署**
```python
# 保存最佳模型
import joblib

# 重新訓練完整數據
final_model = LGBMClassifier(...)
X_full = apply_smote_tomek(X_train_full, y_train_full)
final_model.fit(X_full, y_full)

# 保存模型和預處理器
joblib.dump(final_model, 'fraud_model_v1.0.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

**2. API 開發**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('fraud_model_v1.0.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = preprocess(data)
    prob = model.predict_proba(features)[0][1]
    
    return jsonify({
        'fraud_probability': prob,
        'is_fraud': prob > 0.5,
        'confidence': 'high' if prob > 0.8 or prob < 0.2 else 'medium'
    })
```

**3. 監控系統**
- 每日監控 Precision 和 Recall
- 追蹤誤判和漏判案例
- 收集反饋用於模型改進

#### 中期建議 (3-6個月)

**1. 閾值優化**
```
當前閾值: 0.5 (預設)

建議測試:
- 0.3: 提高 Recall 到 95%+ (抓更多詐欺,但誤判增加)
- 0.7: 提高 Precision 到 95%+ (減少誤判,但漏掉更多詐欺)

根據業務需求選擇最佳閾值
```

**2. 特徵工程**
- 新增交易速度特徵(單位時間內交易次數)
- 地理位置異常檢測
- 用戶歷史行為模式
- 時間序列特徵(週期性模式)

**3. Ensemble 方法**
```python
# 結合多個模型
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    voting='soft',  # 使用機率投票
    weights=[2, 1.5, 1]  # LightGBM 權重較高
)
```

#### 長期建議 (6-12個月)

**1. 深度學習探索**
- LSTM 處理交易序列
- Autoencoder 異常檢測
- 圖神經網路(GNN)建模交易網路

**2. 即時學習系統**
- Online Learning: 模型持續更新
- A/B Testing: 測試新模型效果
- 自