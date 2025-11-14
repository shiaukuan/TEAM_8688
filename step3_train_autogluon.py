# -*- coding: utf-8 -*-
"""
V15 AutoGluon 建模與預測
1. 載入 train/all_acct_transactions.csv
2. 使用 特徵工程（過濾無用特徵）
3. 修正 42 筆 label
4. 使用 AutoGluon 訓練模型
5. 預測 label=0，選出機率最高的 240 個
6. 輸出預測結果到 predictions.csv
"""

import pandas as pd
import numpy as np

from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V15 AutoGluon 建模與預測")
print("=" * 80)

# ============================================================================
# 1. 載入資料與特徵
# ============================================================================
print("\n[1/6] 載入 特徵...")
features_df = pd.read_csv('features.csv')

print(f"特徵數: {len(features_df.columns) - 2}")
print(f"帳戶數: {len(features_df)}")
print(f"  - label=1: {(features_df['label']==1).sum()}")
print(f"  - label=0: {(features_df['label']==0).sum()}")

# 載入原始交易資料以計算交易筆數
print("\n[2/6] 計算每個帳戶的交易筆數...")
df_train = pd.read_csv('../train/all_acct_transactions.csv')
txn_counts = df_train.groupby('acct').size().reset_index(name='txn_count')
print(f"交易資料筆數: {len(df_train):,}")

# 合併交易筆數到特徵
features_df = features_df.merge(txn_counts, on='acct', how='left')
features_df['txn_count'] = features_df['txn_count'].fillna(0).astype(int)

print(f"交易筆數統計:")
print(f"  - 最小: {features_df['txn_count'].min()}")
print(f"  - 最大: {features_df['txn_count'].max()}")
print(f"  - 平均: {features_df['txn_count'].mean():.1f}")
print(f"  - 中位數: {features_df['txn_count'].median():.0f}")

# 統計 label=1 且交易 >= 10 筆的帳戶
label1_high_txn = features_df[(features_df['label'] == 1) & (features_df['txn_count'] >= 10)]
print(f"\nlabel=1 且交易 >= 10 筆的帳戶: {len(label1_high_txn)} 個")
print(f"label=1 總共: {(features_df['label']==1).sum()} 個")

# ============================================================================
# 3. 特徵篩選
# ============================================================================
print("\n[3/6] 特徵篩選...")

# 分離特徵和標籤（txn_count 不作為特徵）
feature_cols = [col for col in features_df.columns if col not in ['acct', 'label', 'txn_count']]
X = features_df[feature_cols].copy()
y = features_df['label'].copy()
acct = features_df['acct'].copy()
txn_count = features_df['txn_count'].copy()  # 保存用於權重計算

print(f"原始特徵數: {len(feature_cols)}")

# 移除常數特徵（方差為 0）
variance = X.var()
constant_features = variance[variance == 0].index.tolist()
print(f"移除常數特徵: {len(constant_features)} 個")

# 移除缺失值過多的特徵（>50%）
missing_ratio = X.isnull().sum() / len(X)
high_missing_features = missing_ratio[missing_ratio > 0.5].index.tolist()
print(f"移除高缺失特徵 (>50%): {len(high_missing_features)} 個")

# 移除高度相關的特徵（>0.95）
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_features = [column for column in upper_triangle.columns
                      if any(upper_triangle[column] > 0.95)]
print(f"移除高相關特徵 (>0.95): {len(high_corr_features)} 個")

# 合併要移除的特徵
features_to_remove = set(constant_features + high_missing_features + high_corr_features)
print(f"\n總共移除: {len(features_to_remove)} 個特徵")

# 保留的特徵
selected_features = [f for f in feature_cols if f not in features_to_remove]
print(f"保留特徵數: {len(selected_features)}")

# 更新特徵集
X = X[selected_features].fillna(0)

# ============================================================================
# 4. 準備訓練資料與樣本權重
# ============================================================================
print("\n[4/6] 準備訓練資料與樣本權重...")

# 分離正樣本和負樣本
label_1_mask = (y == 1)
label_0_mask = (y == 0)

X_label_1 = X[label_1_mask].copy()
y_label_1 = y[label_1_mask].copy()
acct_label_1 = acct[label_1_mask].copy()
txn_count_label_1 = txn_count[label_1_mask].copy()

X_label_0 = X[label_0_mask].copy()
y_label_0 = y[label_0_mask].copy()
acct_label_0 = acct[label_0_mask].copy()

print(f"正樣本 (label=1): {len(X_label_1)}")
print(f"負樣本 (label=0): {len(X_label_0)}")
print(f"樣本不平衡比例: 1:{len(X_label_0)/len(X_label_1):.1f}")

# 合併訓練資料
train_df = pd.concat([X_label_1, X_label_0], axis=0)
train_df['label'] = pd.concat([y_label_1, y_label_0], axis=0)
train_acct = pd.concat([acct_label_1, acct_label_0], axis=0)
train_txn_count = pd.concat([txn_count_label_1, pd.Series([0]*len(X_label_0), index=X_label_0.index)], axis=0)

print(f"\n訓練集大小: {len(train_df)}")
print(f"  - label=1: {(train_df['label']==1).sum()} ({(train_df['label']==1).sum()/len(train_df)*100:.2f}%)")
print(f"  - label=0: {(train_df['label']==0).sum()} ({(train_df['label']==0).sum()/len(train_df)*100:.2f}%)")

# 對 label=1 且交易 >= 10 筆的帳戶進行過採樣
label1_high_txn = (train_df['label'] == 1) & (train_txn_count >= 10)
label1_low_txn = (train_df['label'] == 1) & (train_txn_count < 10)

# 統計數量
label1_high_count = label1_high_txn.sum()
label1_low_count = label1_low_txn.sum()

print(f"\n過採樣策略:")
print(f"  - label=1 (交易 >= 10 筆): {label1_high_count} 個")
print(f"  - label=1 (交易 < 10 筆): {label1_low_count} 個")
print(f"  - label=0: {len(X_label_0)} 個")

# 對高交易筆數的 label=1 樣本複製一份（實現 2 倍權重效果）
high_txn_samples = train_df[label1_high_txn].copy()
train_df = pd.concat([train_df, high_txn_samples], axis=0, ignore_index=True)

print(f"\n過採樣後訓練集大小: {len(train_df)}")
print(f"  - label=1 (總計): {(train_df['label']==1).sum()}")
print(f"  - label=0: {(train_df['label']==0).sum()}")
print(f"  - 其中 label=1 (交易 >= 10 筆) 已複製，實際權重提升 2 倍")

# ============================================================================
# 5. AutoGluon 訓練
# ============================================================================
print("\n[5/6] 開始 AutoGluon 訓練...")
print("這可能需要較長時間，請耐心等待...")

# AutoGluon 配置
predictor = TabularPredictor(
    label='label',
    problem_type='binary',
    eval_metric='f1',  # 使用 F1 作為評估指標
    path='./autogluon_models',
    verbosity=2
)

# 訓練參數
train_params = {
    'time_limit': 3600,  # 1小時訓練時間
    'presets': 'best_quality',  # 使用最高質量預設（會訓練更多模型）
    'num_bag_folds': 5,  # 5-fold bagging
    'num_bag_sets': 1,
    'num_stack_levels': 1,  # 使用 stacking
    'hyperparameters': {
        'GBM': [
            {'num_boost_round': 10000, 'learning_rate': 0.01},
            {'num_boost_round': 5000, 'learning_rate': 0.03},
        ],
        'CAT': {},
        'XGB': {},
        'NN_TORCH': {},
        'FASTAI': {},
    },
    'hyperparameter_tune_kwargs': {
        'num_trials': 10,
        'searcher': 'auto',
        'scheduler': 'local',
    }
}

# 訓練模型
predictor.fit(
    train_data=train_df,
    **train_params
)

print("\n[OK] 模型訓練完成！")