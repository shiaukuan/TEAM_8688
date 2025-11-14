# UV 套件管理設定說明

本文件說明如何使用 `uv` 來管理本專案的 Python 依賴套件。

## 專案使用的套件

根據程式碼分析，本專案使用以下套件：

- **pandas**: 資料處理與分析
- **numpy**: 數值計算
- **autogluon**: 自動機器學習框架（包含 autogluon.tabular）
- **matplotlib**: 圖表繪製（用於資料視覺化）

## 安裝 UV

如果還沒安裝 `uv`，請先執行：

```bash
# 使用 pip 安裝
pip install uv

# 或使用 curl 安裝（推薦）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 初始化專案

### 1. 初始化 uv 專案

在專案根目錄下執行：

```bash
cd /home/rapids/notebooks/sk/TEAM_8688
uv init
```

這會創建 `pyproject.toml` 和 `.python-version` 檔案。

### 2. 指定 Python 版本

```bash
uv python install 3.10
uv python pin 3.10
```

## 安裝依賴套件

### 方法一：逐一安裝套件

```bash
# 安裝核心資料處理套件
uv add pandas numpy

# 安裝 AutoGluon（機器學習框架）
uv add "autogluon.tabular[all]"

# 安裝視覺化套件
uv add matplotlib

# 安裝中文字型支援（Linux 環境）
uv add fonts-noto-cjk
```

### 方法二：使用 pyproject.toml（推薦）

創建或編輯 `pyproject.toml`，添加以下內容：

```toml
[project]
name = "team_8688"
version = "0.1.0"
description = "T-Brain 2025 玉山銀行警示帳戶預測競賽"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "autogluon.tabular[all]>=1.0.0",
    "matplotlib>=3.7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

然後執行：

```bash
uv sync
```

## 運行專案

### 1. 使用 uv run 執行 Python 腳本

```bash
# Step 0: 前處理帳戶資料
uv run python step0_short_acct.py

# Step 1: 處理交易資料
uv run python step1_process_acct_transactions.py

# Step 2: 特徵工程
uv run python step2_feature_engineering.py

# Step 3: 訓練 AutoGluon 模型
uv run python step3_train_autogluon.py

# Step 4: 預測所有帳戶
uv run python step4_predict_all_accounts.py

# Step 5: 產生最終答案
uv run python step5_answer.py
```

### 2. 進入虛擬環境（可選）

```bash
# 啟動 shell
uv venv
source .venv/bin/activate

# 然後可以直接執行 python
python step0_short_acct.py
```

## 常用 UV 指令

```bash
# 查看已安裝的套件
uv pip list

# 添加新套件
uv add <package_name>

# 移除套件
uv remove <package_name>

# 更新所有套件
uv sync --upgrade

# 鎖定依賴版本
uv lock

# 查看專案資訊
uv tree
```

## 注意事項

1. **AutoGluon 需要較多系統資源**：建議至少 8GB RAM
2. **中文顯示**：如果需要在圖表中顯示中文，請確保已安裝 `fonts-noto-cjk` 或在程式中設定：
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
   ```
3. **編碼設定**：所有檔案使用 UTF-8 編碼
4. **資料路徑**：使用 `datas/` 資料夾而非 `data/` 資料夾

## 專案結構

```
/home/rapids/notebooks/sk/TEAM_8688/
├── pyproject.toml          # UV 專案配置檔（由 uv init 生成）
├── uv.lock                 # 依賴鎖定檔（由 uv lock 生成）
├── .python-version         # Python 版本指定
├── .venv/                  # 虛擬環境目錄（自動生成）
├── datas/                  # 處理後的資料
├── train/                  # 訓練資料
├── step0_short_acct.py     # 步驟 0
├── step1_process_acct_transactions.py  # 步驟 1
├── step2_feature_engineering.py        # 步驟 2
├── step3_train_autogluon.py           # 步驟 3
├── step4_predict_all_accounts.py      # 步驟 4
└── step5_answer.py                    # 步驟 5
```

## 完整工作流程範例

```bash
# 1. 初始化專案
cd /home/rapids/notebooks/sk/TEAM_8688
uv init
uv python pin 3.10

# 2. 安裝依賴
uv add pandas numpy "autogluon.tabular[all]" matplotlib

# 3. 執行所有步驟
uv run python step0_short_acct.py
uv run python step1_process_acct_transactions.py
uv run python step2_feature_engineering.py
uv run python step3_train_autogluon.py
uv run python step4_predict_all_accounts.py
uv run python step5_answer.py
```

## 參考資源

- [UV 官方文檔](https://docs.astral.sh/uv/)
- [AutoGluon 文檔](https://auto.gluon.ai/)
- [Pandas 文檔](https://pandas.pydata.org/)
