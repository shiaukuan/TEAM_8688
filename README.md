# TEAM_8688 - 玉山銀行警示帳戶預測競賽

## 專案概述

本專案為 **T-Brain 2025 玉山銀行警示帳戶預測競賽**，目標是透過交易行為模式預測金融帳戶是否為警示帳戶（詐騙相關）。

### 競賽資訊
- **任務類型**：二元分類（Binary Classification）
- **評估指標**：F1 Score
- **資料規模**：443 萬+ 筆交易記錄（703 MB）
- **預測目標**：4,781 個帳戶中預測 240 個警示帳戶

### 資料集說明

#### 1. 警示帳戶清單 (`data/acct_alert.csv`)
- **筆數**：1,004 筆
- **標籤**：全部為正樣本（Y=1）
- **欄位**：`acct`, `event_date`
- **關鍵特性**：警示帳戶在 `event_date` 後無交易記錄

#### 2. 待預測帳戶 (`data/acct_predict.csv`)
- **筆數**：4,781 筆
- **真實分布**：其中約 240 個為警示帳戶（未知）
- **欄位**：`acct`, `label`（暫時標記為 0）

#### 3. 交易記錄 (`data/acct_transaction.csv`)
- **筆數**：4,436,929 筆
- **欄位**：
  - `from_acct`: 轉出帳戶
  - `to_acct`: 轉入帳戶
  - `txn_date`: 相對日期（第一日 = 1）
  - `txn_time`: 交易時間
  - `txn_amt`: 交易金額
  - `info_asset_code`: 交易類型（1=本行, 2=他行）

---

## 建模流程

整個專案採用**六步驟管線流程**，實現從原始資料到最終預測的完整pipeline：

```
Step 0: 帳戶編號簡化
   ↓
Step 1: 交易資料處理與合併
   ↓
Step 2: 特徵工程（V8-V14 六版本迭代）
   ↓
Step 3: AutoGluon 模型訓練
   ↓
Step 4: 全帳戶預測
   ↓
Step 5: 產生最終提交檔案
```

---

## 各步驟詳細說明

### Step 0: 帳戶編號簡化 (`step0_short_acct.py`)

**目的**：簡化帳戶編號以提高處理效率

**處理方式**：
- 將所有帳戶編號截取前 11 個字元
- 處理三個檔案：
  - `acct_transaction.csv`: 截取 `from_acct` 和 `to_acct`
  - `acct_alert.csv`: 截取 `acct`
  - `acct_predict.csv`: 截取 `acct`

**輸出位置**：`datas/` 目錄

**關鍵程式碼**：
```python
trans['from_acct'] = trans['from_acct'].apply(lambda x: x[:11])
trans['to_acct'] = trans['to_acct'].apply(lambda x: x[:11])
```

---

### Step 1: 交易資料處理與合併 (`step1_process_acct_transactions.py`)

**目的**：為每個帳戶建立完整的交易歷史，標記方向（IN/OUT）

#### 處理邏輯

1. **合併所有帳戶**：
   - `acct_alert`: 1,004 個帳戶 → label = 1
   - `acct_predict`: 4,780 個帳戶 → label = 0
   - **總計**：5,784 個帳戶

2. **交易方向定義**：
   - **IN**：其他帳戶轉入該帳戶（`to_acct == 該帳戶`）
   - **OUT**：該帳戶轉出到其他帳戶（`from_acct == 該帳戶`）

3. **新增欄位**：
   - `hour`: 從 `txn_time` 提取小時數（0-23）
   - `days_to_end`: 倒數天數（從帳戶最後交易日往回算）
     ```python
     max_date = group['txn_date'].max()
     group['days_to_end'] = max_date - group['txn_date']
     ```

#### 輸出檔案
- **檔名**：`train/all_acct_transactions.csv`
- **內容**：所有 5,784 個帳戶的交易記錄，每筆交易包含 `acct`, `label`, `direction`, `days_to_end` 等欄位

---

### Step 2: 特徵工程 (`step2_feature_engineering.py`)

這是整個專案的**核心步驟**，採用 **V8-V14 六版本迭代**的特徵工程策略。

#### 2.1 Label 修正策略（半監督學習）

**關鍵步驟**：
1. 載入 `train/all_acct_transactions.csv`（原始 label）
2. 讀取 `train/new_label_1.csv`（包含 42 個帳戶）
3. **修正 Label**：將這 42 個帳戶從 label=0 修正為 label=1

**new_label_1.csv 的來源**：
- 這 42 個帳戶是透過**多次模型預測機率 ≥ 0.9** 篩選出來的高可信度警示帳戶
- 採用半監督學習策略，從待預測集中挖掘隱藏的正樣本
- 提升模型對「隱藏警示帳戶」的識別能力

**修正後的 Label 分布**：
- **Label = 1**：1,004（原始）+ 42（修正）= **1,046 個**
- **Label = 0**：4,738 個

#### 2.2 特徵工程架構

特徵工程分為 **6 個版本迭代**（V8 → V9 → V10 → V12 → V13 → V14），累積產生約 **140 個特徵**。

---

### 詳細特徵列表

#### **V8 特徵組：基礎統計特徵（8 個）**

| 特徵名稱 | 計算方式 | 業務意義 |
|---------|---------|---------|
| `in_out_ratio` | IN 交易數 / OUT 交易數 | 收支比例，詐騙帳戶可能收多支少 |
| `txn_concentration_7d` | 最後 7 天交易數 / 總交易數 | 最近是否異常活躍 |
| `in_concentration_7d` | 最後 7 天 IN 交易 / 最後 7 天總交易 | 近期收款集中度 |
| `last5_in_pct` | 最後 5 筆交易中 IN 的比例 | 最近期收款模式 |
| `last5_type2_pct` | 最後 5 筆 IN 中來自其他銀行的比例 | 跨行收款模式 |
| `last5_amt_ratio` | 最後 5 筆平均金額 / 早期平均金額 | 金額變化趨勢 |
| `max_amt_ratio_7d` | 最後 7 天最大金額 / 早期最大金額 | 是否出現異常大額 |
| *(同樣計算 10、20 筆版本)* | - | - |

---

#### **V9 特徵組：短時間聚集特徵（45 個）**

**核心概念**：詐騙帳戶常在短時間內大量收款/轉帳

**時間窗口**：60 分鐘、120 分鐘、240 分鐘

| 特徵類型 | 特徵名稱範例 | 計算方式 |
|---------|------------|---------|
| **交易爆發** | `max_in_burst_60m` | 60 分鐘內最多 IN 交易數 |
|  | `max_out_burst_60m` | 60 分鐘內最多 OUT 交易數 |
|  | `max_in_amt_60m` | 60 分鐘內最大 IN 總金額 |
| **來源集中度** | `max_same_source_ratio_60m` | 60 分鐘內同一來源的最大占比 |
|  | `max_same_source_count_60m` | 60 分鐘內來自同一帳戶的最多次數 |
| **目標集中度** | `max_same_target_ratio_60m` | 60 分鐘內轉給同一目標的最大占比 |
|  | `max_same_target_count_60m` | 60 分鐘內轉給同一帳戶的最多次數 |
| **小額模式** | `max_small_amt_count` | 單日最多小額交易數 |
|  | `max_small_amt_ratio` | 單日小額交易占比 |
|  | `max_repeated_amt_max_count` | 單日重複金額最多次數 |
|  | `max_repeated_amt_types` | 單日重複金額種類數 |
| **時間間隔** | `max_min_time_gap` | 單日最小交易間隔（分鐘）|
|  | `max_short_gap_count` | 單日短間隔交易數（<10分鐘）|
|  | `max_burst_session_count` | 單日爆發會話數（連續3筆間隔<30分鐘）|

---

#### **V10 特徵組：詐騙模式特徵（20 個）**

| 特徵類型 | 特徵名稱 | 計算方式 | 檢測模式 |
|---------|---------|---------|---------|
| **Type2 集中** | `max_same_type2_in_60m` | 60分鐘內來自同一其他銀行帳戶的IN數 | 人頭帳戶收款 |
|  | `max_same_type2_in_ratio_60m` | Type2 IN 占總 IN 的比例 | - |
|  | `max_same_type2_amt_60m` | Type2 同源最大金額 | - |
| **異常大額匯入** | `max_in_amt_vs_history` | 最大IN / 平均IN | 突然大額收款 |
|  | `top3_in_amt_ratio` | 前3大IN占總IN比例 | 收款集中度 |
|  | `sudden_large_in_7d` | 最後7天最大IN / 早期最大IN | 近期異常收款 |
|  | `large_in_count_ratio` | 大額IN占比（>3倍平均）| - |
|  | `max_in_amt_std_score` | 最大IN的Z-score | 統計異常值 |
|  | `recent_large_in_spike` | 最近5筆中大額IN數 | - |
|  | `in_amt_iqr_outlier` | IQR離群值數量 | - |
|  | `max_single_day_in_amt` | 單日最大IN / 日均IN | 單日爆量收款 |
| **跨天模式** | `consecutive_in_days` | 最長連續收款天數 | 持續收款模式 |
|  | `daily_in_burst_count` | 單日IN>10筆的天數 | 高頻收款天數 |
|  | `cross_day_same_source` | 多天出現的同一Type2來源數 | 固定來源收款 |
|  | `rapid_clear_pattern` | 當日先IN後OUT的天數 | 快速清空資金 |

---

#### **V12 特徵組：帳戶多樣性與活躍度（18 個）**

| 特徵組 | 特徵名稱 | 計算方式 | 業務意義 |
|-------|---------|---------|---------|
| **帳戶多樣性** | `unique_from_acct_ratio` | 唯一來源數 / 總IN數 | 來源分散度 |
|  | `in_diversity_score` | IN來源的Gini係數 | 收款集中度（0=平等，1=集中）|
|  | `unique_to_acct_ratio` | 唯一目標數 / 總OUT數 | 目標分散度 |
|  | `out_diversity_score` | OUT目標的Gini係數 | 轉出集中度 |
| **多帳號匯出** | `max_out_accounts_1day` | 單日轉出到最多帳戶數 | 分散資金模式 |
|  | `max_out_accounts_60m` | 60分鐘內轉出到最多帳戶數 | 快速分散 |
|  | `out_burst_diversity_60m` | 60分鐘內OUT多樣性 | - |
|  | `rapid_multi_out_pattern` | 10分鐘內轉給≥3帳戶的次數 | 極速分散 |
| **活躍天數** | `active_days_ratio` | 活躍天數 / 總天數跨度 | 活躍密集度 |
|  | `consecutive_days_ratio` | 最長連續天數 / 總活躍天數 | 持續性 |
|  | `high_freq_ratio_5` | 單日>5筆的天數占比 | 高頻天占比 |
|  | `consecutive_high_freq_ratio_5` | 連續高頻天數占比 | - |
|  | `daily_txn_variability` | 日交易數變異係數（std/mean）| 交易量波動 |

---

#### **V13 特徵組：詐騙行為模式（23 個）**

基於詐騙案例統計設計的專家特徵。

| 特徵組 | 特徵名稱 | 計算方式 | 檢測目標 | 統計依據 |
|-------|---------|---------|---------|---------|
| **單向交易對象** | `one_way_counterparty_ratio` | 單向交易對象 / 總對象 | 大多數對象只單向往來 | 93.4%案例 |
|  | `bidirectional_counterparty_ratio` | 雙向交易對象 / 總對象 | - | - |
|  | `in_counterparties_count` | IN 對象數量 | - | - |
|  | `out_counterparties_count` | OUT 對象數量 | - | - |
|  | `one_way_counterparties_count` | 單向對象數量 | - | - |
|  | `bidirectional_counterparties_count` | 雙向對象數量 | - | - |
| **收款後轉出時間** | `avg_days_in_to_out` | IN後第一筆OUT的平均天數 | 快速轉出資金 | 43.4%案例平均3天內 |
|  | `median_days_in_to_out` | 中位數天數 | - | - |
|  | `min_days_in_to_out` | 最小天數 | - | - |
|  | `quick_turnover_ratio` | 1天內轉出的占比 | - | - |
|  | `quick_turnover_3d_ratio` | 3天內轉出的占比 | - | - |
|  | `first_in_to_first_out_days` | 第一筆IN到第一筆OUT的天數 | - | - |
| **資金流向不對稱** | `flow_asymmetry_ratio` | OUT金額 / IN金額 | 嚴重不對稱 | 42%案例 |
|  | `net_flow_ratio` | \|IN-OUT\| / (IN+OUT) | 淨流量占比 | - |
|  | `flow_count_asymmetry` | OUT筆數 / IN筆數 | - | - |
|  | `avg_amt_in_out_ratio` | 平均OUT金額 / 平均IN金額 | - | - |
| **分散度** | `in_dispersion_score` | IN對象數 / IN筆數 | 每對象1-2次 | 38.2%案例 |
|  | `avg_txn_per_in_source` | 每來源平均交易數 | - | - |
|  | `single_txn_in_sources_ratio` | 只交易1次的來源占比 | - | - |
|  | `out_dispersion_score` | OUT對象數 / OUT筆數 | - | - |
|  | `avg_txn_per_out_target` | 每目標平均交易數 | - | - |
|  | `single_txn_out_targets_ratio` | 只交易1次的目標占比 | - | - |
| **小額匯入** | `small_in_ratio_dynamic` | 小於200元的IN占比 | 大量小額收款 | 21.1%案例>60% |
|  | `small_in_count` | 小額IN數量 | - | - |
|  | `small_in_density` | 小額IN數 / 活躍天數 | - | - |

---

#### **V14 特徵組：異常交易模式（26 個）**

| 特徵組 | 特徵名稱 | 計算方式 | 檢測模式 |
|-------|---------|---------|---------|
| **同時多筆交易** | `max_concurrent_txns_in` | 同一時間點最多IN筆數 | 批量操作 |
|  | `max_concurrent_txns_out` | 同一時間點最多OUT筆數 | - |
|  | `concurrent_time_points_in` | 有≥2筆IN的時間點數 | - |
|  | `concurrent_time_points_out` | 有≥2筆OUT的時間點數 | - |
|  | `max_concurrent_txns_all` | 同一時間點最多總筆數 | - |
|  | `concurrent_time_points_all` | 有≥2筆交易的時間點數 | - |
| **異常時間交易** | `abnormal_hour_ratio` | 0-6點或22-24點交易占比 | 夜間操作 |
|  | `abnormal_hour_count` | 異常時段交易數 | - |
|  | `late_night_ratio` | 22-24點占比 | - |
|  | `early_morning_ratio` | 0-6點占比 | - |
| **固定金額重複** | `top_amount_count` | 最常出現金額的次數 | 固定金額模式 |
|  | `repeated_amounts_5plus` | 重複≥5次的金額種類數 | - |
|  | `repeated_amounts_10plus` | 重複≥10次的金額種類數 | - |
|  | `amount_diversity_score` | 重複金額種類占比 | - |
| **連續交易天數** | `max_consecutive_txn_days` | 最長連續有交易天數 | 持續活躍 |
|  | `consecutive_in_days` | 最長連續有IN天數 | - |
|  | `consecutive_out_days` | 最長連續有OUT天數 | - |
|  | `consecutive_days_ratio` | 連續天數 / 活躍天數 | - |
| **轉出集中度** | `top1_out_ratio` | 第1大OUT對象占比 | 資金集中流向 |
|  | `top3_out_ratio` | 前3大OUT對象占比 | - |
|  | `top5_out_ratio` | 前5大OUT對象占比 | - |
|  | `out_concentration_gini` | OUT的Gini係數 | - |
| **單日高頻交易** | `max_daily_txns` | 單日最多交易數 | 爆量交易 |
|  | `avg_daily_txns` | 平均每日交易數 | - |
|  | `daily_txn_std` | 每日交易數標準差 | - |
|  | `high_freq_days_10` | 單日≥10筆的天數 | - |
|  | `high_freq_days_20` | 單日≥20筆的天數 | - |
| **完全單向交易** | `is_only_in` | 是否只有IN（0/1）| 只收不支 |
|  | `is_only_out` | 是否只有OUT（0/1）| 只支不收 |
|  | `in_out_imbalance` | \|IN數-OUT數\| / 總數 | 方向不平衡度 |
| **交易時間方差** | `hour_std` | 交易小時的標準差 | 時間固定度 |
|  | `time_concentration_score` | 最常出現小時的占比 | - |

---

#### 2.3 特徵過濾

**過濾策略**：
- 移除相關係數 > 0.95 的高度相關特徵
- 保留每組相關特徵中的第一個

**最終特徵數**：約 **130-140 個**（經過過濾後）

**輸出檔案**：`features.csv`（5,784 個帳戶 × 130+ 特徵）

---

### Step 3: AutoGluon 模型訓練 (`step3_train_autogluon.py`)

#### 3.1 資料準備

**特徵篩選流程**：
1. 移除常數特徵（variance = 0）
2. 移除高缺失特徵（>50% 缺失）
3. 移除高相關特徵（>0.95）

**樣本不平衡處理**：
- 原始分布：
  - Label = 1: 1,046 個
  - Label = 0: 4,738 個
  - 不平衡比例：1:4.5

- **過採樣策略**：
  - 對 `label=1` 且交易筆數 ≥ 10 的帳戶複製一份（實現 2 倍權重）
  - **理由**：交易筆數較多的警示帳戶特徵更可靠，模式更清晰

#### 3.2 AutoGluon 模型配置

| 參數 | 設定值 | 說明 |
|------|-------|------|
| **問題類型** | `binary` | 二元分類 |
| **評估指標** | `f1` | F1 Score（符合競賽要求）|
| **訓練時間** | `3600` 秒（1小時）| - |
| **預設模式** | `best_quality` | 最高質量預設 |
| **Bagging** | 5-fold bagging | 5折交叉驗證，減少過擬合 |
| **Stacking** | 1 層 | 模型堆疊增強泛化能力 |

#### 3.3 超參數配置

```python
hyperparameters = {
    'GBM': [  # LightGBM
        {'num_boost_round': 10000, 'learning_rate': 0.01},
        {'num_boost_round': 5000, 'learning_rate': 0.03},
    ],
    'CAT': {},   # CatBoost
    'XGB': {},   # XGBoost
    'NN_TORCH': {},  # PyTorch 神經網路
    'FASTAI': {},    # FastAI 深度學習
}
```

**超參數調優設定**：
- 試驗次數：10
- 搜尋器：自動（Auto）
- 排程器：本地（Local）

#### 3.4 模型選擇策略

AutoGluon 會自動嘗試以下模型類型：
1. **GBM（LightGBM）**：梯度提升樹，速度快、效果好
2. **XGBoost**：另一種梯度提升實現
3. **CatBoost**：處理類別特徵強
4. **NN_TORCH**：PyTorch 神經網路
5. **FASTAI**：深度學習框架

透過 **Bagging（5-fold）** 和 **Stacking（1層）** 組合多個模型，提升預測穩定性。

#### 3.5 模型輸出

**輸出目錄**：`./autogluon_models/`
- 包含訓練好的模型權重
- 模型排行榜（Leaderboard）
- 特徵重要性分析

---

### Step 4: 預測所有帳戶 (`step4_predict_all_accounts.py`)

#### 預測流程

1. **載入模型**：從 `./autogluon_models/` 載入訓練好的 AutoGluon 模型
2. **載入特徵**：讀取 `features.csv`（5,784 個帳戶）
3. **執行預測**：
   - 預測機率（`proba`）：label=1 的機率值（0-1）
   - 預測標籤（`label`）：使用 0.5 閾值產生二元標籤
4. **儲存結果**：

#### 輸出檔案

**檔名**：`all_predictions.csv`

**欄位說明**：
- `acct`: 帳戶編號（簡化版）
- `label`: 預測標籤（0 或 1）
- `proba`: 預測為警示帳戶的機率（0-1）
- `true_label`: 真實標籤（用於驗證模型效果）

---

### Step 5: 產生最終提交檔案 (`step5_answer.py`)

#### 提交策略

**Top-240 選擇策略**：
1. 讀取 `all_predictions.csv`
2. 從原始 `data/acct_predict.csv` 載入待預測帳戶（4,781 個）
3. **選擇標準**：按預測機率（`proba`）降序排序，選出**最高的 240 個帳戶**標記為 1
4. 其餘 4,541 個帳戶標記為 0
5. **恢復原始格式**：將簡化的帳戶編號還原為原始完整格式

#### 輸出檔案

**檔名**：`submit.csv`

**格式**：
```csv
acct,label
原始完整帳號1,1
原始完整帳號2,0
...
```

**統計**：
- 總筆數：4,781
- Label=1：240 個（機率最高的前240個）
- Label=0：4,541 個

---

## 資料流向圖

```
原始資料 (data/)
├── acct_transaction.csv (443萬筆交易)
├── acct_alert.csv (1,004個警示帳戶)
└── acct_predict.csv (4,781個待預測帳戶)
    ↓
    ↓ Step 0: 簡化帳號（截取前11字元）
    ↓
簡化資料 (datas/)
├── acct_transaction.csv
├── acct_alert.csv
└── acct_predict.csv
    ↓
    ↓ Step 1: 合併交易 + 標記方向（IN/OUT）
    ↓
訓練資料 (train/)
├── all_acct_transactions.csv (5,784個帳戶的所有交易)
└── new_label_1.csv (42個高機率警示帳戶)
    ↓
    ↓ Step 2: 特徵工程（V8-V14）+ Label修正
    ↓
特徵檔案
└── features.csv (5,784 × 130+ 特徵)
    ↓
    ↓ Step 3: AutoGluon訓練（F1優化，5-fold bagging）
    ↓
模型檔案
└── autogluon_models/ (訓練好的模型集合)
    ↓
    ↓ Step 4: 預測所有帳戶
    ↓
預測結果
└── all_predictions.csv (5,784 × [acct, label, proba, true_label])
    ↓
    ↓ Step 5: Top-240策略 + 恢復原始格式
    ↓
最終提交
└── submit.csv (4,781 × [acct, label]，其中240個為1)
```

---

## 關鍵技術亮點

### 1. 半監督學習策略

**new_label_1.csv 的生成與應用**：
- **來源**：透過多次模型迭代，篩選出預測機率 ≥ 0.9 的 42 個帳戶
- **作用**：從待預測集中挖掘隱藏的正樣本，擴充訓練集
- **影響**：正樣本從 1,004 → 1,046 個（+4.2%），提升模型對邊界案例的識別能力

### 2. 時序特徵設計

**倒數天數（days_to_end）**：
- 從每個帳戶的最後交易日往回計算
- 允許模型捕捉「警示前夕」的異常行為
- 符合業務邏輯：警示帳戶在 event_date 後無交易記錄

### 3. 多層次特徵工程

**三個層次的特徵設計**：
1. **交易層面**：單筆交易特徵（金額、時間、對象、類型）
2. **單日層面**：每日聚合特徵（最大值、平均值、標準差）
3. **帳戶層面**：整體行為模式（連續天數、總體趨勢、分散度）

**六版本迭代**：
- V8: 基礎統計（收支比、最近活躍度）
- V9: 短時聚集（60/120/240分鐘爆發）
- V10: 詐騙模式（Type2集中、異常大額）
- V12: 多樣性與活躍度（Gini係數、連續天數）
- V13: 專家特徵（93.4%單向、43.4%快速轉出）
- V14: 異常模式（同時交易、夜間操作、固定金額）

### 4. 不平衡處理

**交易量加權過採樣**：
- 對 label=1 且交易筆數 ≥ 10 的帳戶複製樣本
- **理由**：高交易量帳戶的特徵模式更清晰可靠
- 避免低交易量噪音樣本影響模型學習

### 5. 模型集成策略

**AutoGluon 多模型集成**：
- 自動嘗試 5 種模型類型（GBM, XGB, CAT, NN, FASTAI）
- **5-fold Bagging**：減少過擬合，提升泛化能力
- **1層 Stacking**：組合不同模型的優勢
- **F1 優化**：直接針對競賽評估指標訓練

### 6. Top-240 選擇策略

**基於機率排序的選擇**：
- 不使用固定閾值（如 0.5），而是選出機率最高的 240 個
- **優勢**：適應模型校準問題，確保提交正確數量的正樣本
- **符合競賽設定**：已知待預測集中約有 240 個警示帳戶

---

## 特徵統計摘要

| 特徵組 | 數量 | 核心目標 | 關鍵創新 |
|-------|------|---------|---------|
| V8 基礎統計 | 8 | 收支比例、最近活躍度 | 時間窗口（7天、最後5/10/20筆）|
| V9 短時聚集 | 45 | 爆發式交易、來源/目標集中 | 多時間窗口（60/120/240分鐘）|
| V10 詐騙模式 | 20 | Type2集中、異常大額、跨天模式 | Z-score、IQR離群值檢測 |
| V12 多樣性 | 18 | 帳戶分散度、活躍天數 | Gini係數、變異係數 |
| V13 行為模式 | 23 | 單向交易、快速轉出、資金不對稱 | 基於93.4%/43.4%等統計設計 |
| V14 異常模式 | 26 | 同時交易、異常時間、固定金額 | 批量操作、夜間交易檢測 |
| **總計** | **約140** | **全面捕捉詐騙行為特徵** | **多層次、多時間尺度設計** |

---

## 建模策略總結

### 核心假設
1. 警示帳戶在被警示前會有異常交易模式
2. 這些模式可透過時序、金額、對象、頻率等多維度特徵捕捉
3. 短時間聚集、快速清空、單向交易是關鍵區分特徵

### 創新點
1. **V13-V14 專家特徵**：針對詐騙行為統計設計（93.4%單向交易、43.4%快速轉出、42%資金不對稱）
2. **半監督學習**：從待預測集中挖掘 42 個高機率樣本擴充訓練集
3. **交易量加權**：對高交易量正樣本提升權重，增強模式學習
4. **Top-240 策略**：依機率排序選出最可疑的 240 個帳戶，適應模型校準問題

### 潛在挑戰
1. **Domain Shift**：待預測帳戶的交易量分布可能與訓練集不同
2. **負樣本污染**：4,738 個 label=0 中可能包含未被發現的警示帳戶
3. **過擬合風險**：42 個修正樣本可能來自特定子群體，泛化能力待驗證
4. **閾值選擇**：Top-240 假設待預測集正好有 240 個，實際可能有偏差

---

## 執行指令

按順序執行以下程式：

```bash
# Step 0: 簡化帳戶編號
python step0_short_acct.py

# Step 1: 處理交易資料
python step1_process_acct_transactions.py

# Step 2: 特徵工程
python step2_feature_engineering.py

# Step 3: 訓練模型
python step3_train_autogluon.py

# Step 4: 預測所有帳戶
python step4_predict_all_accounts.py

# Step 5: 產生提交檔案
python step5_answer.py
```

**最終輸出**：`submit.csv`（可直接提交至競賽平台）

---

## 專案結構

```
/home/rapids/notebooks/sk/TEAM_8688/
├── data/                          # 原始資料（勿修改）
│   ├── acct_transaction.csv       # 443萬筆交易記錄
│   ├── acct_alert.csv             # 1,004個警示帳戶
│   └── acct_predict.csv           # 4,781個待預測帳戶
│
├── datas/                         # 簡化後資料（Step 0輸出）
│   ├── acct_transaction.csv
│   ├── acct_alert.csv
│   └── acct_predict.csv
│
├── train/                         # 訓練相關檔案
│   ├── all_acct_transactions.csv  # 所有帳戶交易（Step 1輸出）
│   └── new_label_1.csv            # 42個高機率警示帳戶（label修正用）
│
├── features.csv                   # 特徵檔案（Step 2輸出）
├── all_predictions.csv            # 預測結果（Step 4輸出）
├── submit.csv                     # 最終提交檔案（Step 5輸出）
│
├── autogluon_models/              # AutoGluon模型目錄（Step 3輸出）
│
├── step0_short_acct.py
├── step1_process_acct_transactions.py
├── step2_feature_engineering.py
├── step3_train_autogluon.py
├── step4_predict_all_accounts.py
├── step5_answer.py
│
├── info/
│   └── datainfo.md                # 資料詳細說明
│
├── CLAUDE.md                      # Claude Code 專案指引
└── README.md                      # 本說明文件
```

---

## 結論

本專案採用**精細化特徵工程 + 強大AutoML + 半監督學習**的完整策略，透過 6 個版本迭代累積超過 140 個特徵，全面覆蓋詐騙帳戶的行為模式。

### 技術優勢
1. **多層次特徵設計**：從交易、單日到帳戶層面全方位捕捉異常
2. **專家知識融入**：V13特徵基於真實詐騙案例統計（93.4%、43.4%等）
3. **半監督學習**：從待預測集挖掘42個高機率樣本提升訓練效果
4. **集成學習**：AutoGluon自動嘗試5種模型+Bagging+Stacking
5. **評估指標優化**：直接針對F1 Score訓練，符合競賽要求

### 預期效果
透過系統化的特徵工程和強大的模型集成，能夠有效識別隱藏的警示帳戶，在高度不平衡和負樣本不確定的環境下實現良好的F1 Score。
