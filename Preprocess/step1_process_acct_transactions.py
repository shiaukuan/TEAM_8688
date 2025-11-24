"""
處理每個帳戶的交易資料，直接輸出統計檔案
"""
import pandas as pd
from pathlib import Path

# 定義路徑（從 Preprocess 目錄出發）
BASE_DIR = Path(__file__).parent.parent  # 回到專案根目錄
DATA_DIR = BASE_DIR / 'datas'
TRAIN_DIR = BASE_DIR / 'train'

# 確保 train 目錄存在
TRAIN_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Step 1: 處理帳戶交易資料")
print("=" * 80)

# 讀取資料
print("\n讀取資料中...")
acct_alert = pd.read_csv(DATA_DIR / 'acct_alert.csv')
acct_predict = pd.read_csv(DATA_DIR / 'acct_predict.csv')
acct_transaction = pd.read_csv(DATA_DIR / 'acct_transaction.csv')

print(f"  - 警示帳戶: {len(acct_alert):,} 筆")
print(f"  - 待預測帳戶: {len(acct_predict):,} 筆")
print(f"  - 交易記錄: {len(acct_transaction):,} 筆")

# 合併所有需要處理的帳戶
all_accts = pd.concat([
    acct_alert[['acct']].assign(label=1),
    acct_predict[['acct', 'label']]
], ignore_index=True)

print(f"\n總共需要處理 {len(all_accts):,} 個帳戶")

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
    merged_transactions.to_csv(TRAIN_DIR / 'all_acct_transactions.csv', index=False)
    print(f"\n" + "=" * 80)
    print("[完成] 交易資料處理完成")
    print("=" * 80)
    print(f"  總交易筆數: {len(merged_transactions):,}")
    print(f"  總帳戶數: {merged_transactions['acct'].nunique()}")
    print(f"  儲存位置: train/all_acct_transactions.csv")
else:
    print("\n[錯誤] 沒有交易記錄可以處理")
