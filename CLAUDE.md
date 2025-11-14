# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 語言與建模原則
- **使用繁體中文**回應與溝通
- 不要用data/裡面的資料 改用datas/ 的資料
- 採用 **Kaggle 建模專家**視角，以**準確度優先**的策略建模
- 請自己運行程式了解資訊後繼續運行下一階段的分析，最後寫出你的結論跟我討論
- 如果在Linux環境 畫圖要用這個顯示中文 plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
- 編碼用 utf-8

## 專案概述

本專案為 **T-Brain 2025 玉山銀行警示帳戶預測競賽**，目標是預測金融帳戶是否為警示帳戶（詐騙相關）。

- **任務類型**：二元分類（Binary Classification）
- **評估指標**：F1
- **資料規模**：443萬+ 筆交易記錄（703 MB）
- **預測資料**：4,781筆要預測的資料其中240為1其餘為0

### 資料集
- **詳細資料說明** 請些閱讀文件 info/datainfo.md
- 正樣本：`acct_alert.csv` 共 1,004 筆（警示後無後續交易 都為正樣本，Y=1）。
  - 欄位：acct, event_date
  - **關鍵特性**：警示帳戶在 event_date 後無交易記錄
  - 因為acct_predict沒有辦法知道 event_date 所以每個acct都是最大的txn_date都當event_date
  
- 待預測：`acct_predict.csv` 4,781 筆（目前標記 0，約 240 真實為 1）。
  - 是待預測的帳戶，標籤都是0 需要預測 acct,label (實際上有240個應該是1但我們不知道) 這個待預測會有主辦單位的抽樣問題
  
- 交易：`acct_transaction.csv` 443 萬+ 筆，含 `from_acct`, `to_acct`, `txn_date`（相對日序）。
  - from_acct 和 to_acct 都是需要預測的 acct
  - txn_date：相對日期（第一日 = 1）


## 建模方式
- 警示帳戶（label=1）：1,004 個 acct_alert.csv
- 待預測帳戶（label=0）：4,780 個（實際有240個為警示帳戶）acct_predict.csv 為負樣本


## 關鍵挑戰與限制

### 1. 時間問題
- ⚠️ **警示帳戶在 event_date 後無交易記錄**
- 每筆可預測的acct_alert來自acct_transaction.csv的from_acct或to_acct
- acct_transaction.csv的最大的txn_date為acct的event_date

### 2. 類別不平衡
- 正樣本僅 1,004 個，負樣本無定義
- 需使用 但不確定其中 負樣本有沒有正樣本在裡面

### 3. 負樣本定義模糊
- 負樣本 = 有交易但不在警示清單中的帳戶
- 可能包含未被偵測的警示帳戶


## 專案結構

```
/home/rapids/notebooks/sk/TEAM_8688/
├── data/              # 原始資料（勿修改）
│   ├── acct_transaction.csv
│   ├── acct_alert.csv
│   └── acct_predict.csv
│
├── info/              # 提示
│   └── datainfo.md    # 資料資訊
│
│

```

