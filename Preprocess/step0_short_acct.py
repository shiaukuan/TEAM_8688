import pandas as pd

print("=" * 80)
print("Step 0: 帳戶編號簡化處理")
print("=" * 80)

# 讀取原始資料（從上層目錄）
print("\n[1/3] 處理交易資料...")
trans = pd.read_csv('../data/acct_transaction.csv')
trans['from_acct'] = trans['from_acct'].apply(lambda x: x[:11])
trans['to_acct'] = trans['to_acct'].apply(lambda x: x[:11])
trans.to_csv('../datas/acct_transaction.csv', index=False)
print(f"  完成！處理 {len(trans):,} 筆交易記錄")

# 處理警示帳戶
print("\n[2/3] 處理警示帳戶...")
alert = pd.read_csv('../data/acct_alert.csv')
alert['acct'] = alert['acct'].apply(lambda x: x[:11])
alert.to_csv('../datas/acct_alert.csv', index=False)
print(f"  完成！處理 {len(alert):,} 個警示帳戶")

# 處理待預測帳戶
print("\n[3/3] 處理待預測帳戶...")
predict = pd.read_csv('../data/acct_predict.csv')
predict['acct'] = predict['acct'].apply(lambda x: x[:11])
predict.to_csv('../datas/acct_predict.csv', index=False)
print(f"  完成！處理 {len(predict):,} 個待預測帳戶")

print("\n" + "=" * 80)
print("[完成] 所有帳戶編號已簡化至前11碼")
print("=" * 80)