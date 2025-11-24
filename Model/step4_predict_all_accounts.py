# -*- coding: utf-8 -*-
"""
預測所有帳戶
載入訓練好的模型，預測所有帳戶（label=0 和 label=1）
輸出格式：acct, label, proba, true_label
"""

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

print("=" * 80)
print("預測所有帳戶")
print("=" * 80)

# ============================================================================
# 1. 載入特徵資料
# ============================================================================
print("\n[1/4] 載入特徵資料...")
features_df = pd.read_csv('../features.csv')
print(f"總帳戶數: {len(features_df)}")
print(f"  - label=1: {(features_df['label']==1).sum()}")
print(f"  - label=0: {(features_df['label']==0).sum()}")

# 保存真實標籤
true_labels = features_df[['acct', 'label']].copy()
true_labels.columns = ['acct', 'true_label']

# ============================================================================
# 2. 載入訓練好的模型
# ============================================================================
print("\n[2/4] 載入訓練好的模型...")
predictor = TabularPredictor.load('../autogluon_models')
print(f"模型路徑: ../autogluon_models")

# ============================================================================
# 3. 準備預測資料
# ============================================================================
print("\n[3/4] 準備預測資料...")

# 移除 acct 和 label 列
feature_cols = [col for col in features_df.columns if col not in ['acct', 'label']]
X_all = features_df[feature_cols].copy().fillna(0)
acct_all = features_df['acct'].copy()

print(f"預測樣本數: {len(X_all)}")
print(f"使用特徵數: {len(feature_cols)}")

# ============================================================================
# 4. 預測所有帳戶
# ============================================================================
print("\n[4/4] 預測所有帳戶...")

# 預測機率（預測為 label=1 的機率）
pred_proba = predictor.predict_proba(X_all, as_multiclass=False)

# 預測類別（使用 0.5 閾值）
pred_label = (pred_proba >= 0.5).astype(int)

# 整理結果
results = pd.DataFrame({
    'acct': acct_all.values,
    'label': pred_label,  # 預測標籤
    'proba': pred_proba   # 預測為 1 的機率
})

# 合併真實標籤
results = results.merge(true_labels, on='acct', how='left')

# 重新排序列
results = results[['acct', 'label', 'proba', 'true_label']]

# 按機率降序排序
results = results.sort_values('proba', ascending=False).reset_index(drop=True)

print(f"\n預測完成！")
print(f"  總樣本數: {len(results)}")
print(f"  預測為 1: {(results['label']==1).sum()}")
print(f"  預測為 0: {(results['label']==0).sum()}")

# 統計真實標籤分布
print(f"\n真實標籤分布:")
print(f"  真實 label=1: {(results['true_label']==1).sum()}")
print(f"  真實 label=0: {(results['true_label']==0).sum()}")

# 計算準確度（針對所有樣本）
accuracy = (results['label'] == results['true_label']).sum() / len(results)
print(f"\n整體準確度: {accuracy:.4f}")

# 針對真實 label=1 的樣本
true_label_1 = results[results['true_label'] == 1]
if len(true_label_1) > 0:
    recall = (true_label_1['label'] == 1).sum() / len(true_label_1)
    print(f"真實 label=1 的召回率: {recall:.4f} ({(true_label_1['label']==1).sum()}/{len(true_label_1)})")

# 針對真實 label=0 的樣本
true_label_0 = results[results['true_label'] == 0]
if len(true_label_0) > 0:
    specificity = (true_label_0['label'] == 0).sum() / len(true_label_0)
    print(f"真實 label=0 的特異度: {specificity:.4f} ({(true_label_0['label']==0).sum()}/{len(true_label_0)})")

# 機率分布統計
print(f"\n機率分布:")
print(f"  最高: {results['proba'].max():.4f}")
print(f"  最低: {results['proba'].min():.4f}")
print(f"  平均: {results['proba'].mean():.4f}")
print(f"  中位數: {results['proba'].median():.4f}")

# ============================================================================
# 5. 儲存結果
# ============================================================================
print("\n儲存預測結果...")
results.to_csv('../all_predictions.csv', index=False, encoding='utf-8-sig')
print("[OK] 已保存: ../all_predictions.csv")