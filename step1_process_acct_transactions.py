"""
處理每個帳戶的交易資料，直接輸出統計檔案
"""
import pandas as pd
from pathlib import Path

# 定義路徑
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'datas'

# 讀取資料
acct_alert = pd.read_csv(DATA_DIR / 'acct_alert.csv')
acct_predict = pd.read_csv(DATA_DIR / 'acct_predict.csv')
acct_transaction = pd.read_csv(DATA_DIR / 'acct_transaction.csv')

# 合併所有需要處理的帳戶
all_accts = pd.concat([
    acct_alert[['acct']].assign(label=1),
    acct_predict[['acct', 'label']]
], ignore_index=True)

# 添加 hour 欄位到交易資料
acct_transaction['hour'] = acct_transaction['txn_time'].str.split(':').str[0].astype(int)

# 計算每個帳戶的倒數天數
def calculate_days_to_end(group):
    max_date = group['txn_date'].max()
    group['days_to_end'] = max_date - group['txn_date']
    return group

# 統計每個帳戶的交易數量
all_transactions = []

processed_count = 0
for idx, row in all_accts.iterrows():
    acct = row['acct']
    label = row['label']

    # 找出該帳戶的所有交易（作為 from_acct 或 to_acct）
    # IN: 其他帳戶轉入 (to_acct = 該帳戶)
    txn_in = acct_transaction[acct_transaction['to_acct'] == acct].copy()
    txn_in['direction'] = 'IN'

    # OUT: 該帳戶轉出 (from_acct = 該帳戶)
    txn_out = acct_transaction[acct_transaction['from_acct'] == acct].copy()
    txn_out['direction'] = 'OUT'

    # 合併 IN 和 OUT
    txn_all = pd.concat([txn_in, txn_out], ignore_index=True)

    if len(txn_all) == 0:
        continue

    # 排序：依日期和時間
    txn_all = txn_all.sort_values(['txn_date', 'txn_time']).reset_index(drop=True)

    # 添加 acct 和 label 欄位到最前面
    txn_all.insert(0, 'label', label)
    txn_all.insert(0, 'acct', str(acct))

    # 計算倒數天數
    txn_all = calculate_days_to_end(txn_all)

    # 將交易記錄添加到總列表
    all_transactions.append(txn_all)

    processed_count += 1
    if processed_count % 100 == 0:
        print(f"  已處理: {processed_count}/{len(all_accts)} 個帳戶")


# 合併所有交易記錄為一個大檔案
if all_transactions:
    merged_transactions = pd.concat(all_transactions, ignore_index=True)
    merged_transactions.to_csv(BASE_DIR / 'train/all_acct_transactions.csv', index=False)
