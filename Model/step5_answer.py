import pandas as pd

print("=" * 80)
print("Step 5: 生成提交檔案")
print("=" * 80)

# 讀取預測結果
print("\n[1/4] 讀取預測結果...")
perd = pd.read_csv('../all_predictions.csv')
print(f"  預測結果: {len(perd)} 筆")

# 讀取原始待預測帳戶（含完整帳號）
print("\n[2/4] 讀取原始待預測帳戶...")
predict = pd.read_csv('../data/acct_predict.csv')
print(f"  待預測帳戶: {len(predict)} 筆")

# 保留原始完整帳號
predict['oacct'] = predict['acct']
predict['acct'] = predict['acct'].apply(lambda x: x[:11])
del predict['label']

# 合併預測結果
print("\n[3/4] 合併預測結果...")
data = pd.merge(predict, perd[['acct', 'label', 'proba', 'true_label']], on='acct', how='left')

# 選出機率最高的 240 個
top_240 = data.nlargest(240, 'proba')
print(f"  機率最高的 240 個帳戶已選出")
print(f"  最高機率: {top_240['proba'].max():.4f}")
print(f"  最低機率: {top_240['proba'].min():.4f}")

# 生成提交檔案
data['label'] = 0  # 先將所有標籤設為 0
data.loc[data['acct'].isin(top_240['acct']), 'label'] = 1  # 將選出的 240 個設為 1

data['label'] = data['label'].astype(int)
del data['acct']
data.rename(columns={'oacct': 'acct'}, inplace=True)

# 儲存提交檔案
print("\n[4/4] 儲存提交檔案...")
data[['acct', 'label']].to_csv('../submit.csv', index=False)

print("\n" + "=" * 80)
print("[完成] 提交檔案已生成")
print("=" * 80)
print(f"  檔案位置: submit.csv")
print(f"  總筆數: {len(data)}")
print(f"  label=1: {(data['label']==1).sum()} 筆")
print(f"  label=0: {(data['label']==0).sum()} 筆")