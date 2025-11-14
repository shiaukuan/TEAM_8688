# UV 套件管理設定說明

本文件說明如何使用 `uv` 來管理本專案的 Python 依賴套件。

## 專案使用的套件

根據程式碼分析，本專案使用以下套件：

### 核心套件
- **pandas**: 資料處理與分析
- **numpy**: 數值計算
- **autogluon.tabular**: 自動機器學習框架（表格資料專用）
- **matplotlib**: 圖表繪製（用於資料視覺化）
- **typing-extensions**: 類型提示擴展（AutoGluon 依賴）

### 機器學習模型訓練器
- **lightgbm**: LightGBM 梯度提升框架
- **xgboost**: XGBoost 梯度提升框架
- **catboost**: CatBoost 梯度提升框架

### 可選套件（深度學習，已安裝 GPU 版本）
- **torch**: PyTorch 深度學習框架 (v2.5.1+cu121)
- **torchvision**: PyTorch 視覺工具 (v0.20.1+cu121)
- **torchaudio**: PyTorch 音訊工具 (v2.5.1+cu121)
- **fastai**: FastAI 深度學習庫 (v2.8.5)
- **spacy**: 自然語言處理庫（FastAI 依賴）

## 安裝 UV

如果還沒安裝 `uv`，請先執行：

### Windows (PowerShell)
```powershell
# 使用 pip 安裝
pip install uv

# 或使用官方安裝腳本（推薦）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux / macOS
```bash
# 使用 pip 安裝
pip install uv

# 或使用 curl 安裝（推薦）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 快速開始（已設定好的專案）

本專案已經配置好 `pyproject.toml` 和 `uv.lock`，只需執行：

```bash
# 1. 確認 Python 版本（專案使用 Python 3.12）
uv python install 3.12

# 2. 同步安裝所有依賴
uv sync --no-install-project

```

## 安裝依賴套件

### 方法一：使用 uv sync（推薦，適用於已有 pyproject.toml）

```bash
# 安裝所有依賴套件（不安裝專案本身）
uv sync --no-install-project
```

### 方法二：逐一安裝套件

```bash
# 安裝核心資料處理套件
uv add pandas numpy matplotlib

# 安裝 AutoGluon（機器學習框架）
# 注意：不使用 [all] 避免在 Windows 環境下的編譯問題
uv add "autogluon.tabular"

# 安裝必要的依賴
uv add typing-extensions

# 安裝機器學習模型訓練器
uv add lightgbm xgboost catboost
```

### 可選：安裝深度學習模型（進階使用）

如果需要使用 PyTorch 或 FastAI 模型，可以額外安裝：

#### 方法 A：使用 pip 直接安裝（推薦，適用於 GPU 版本）

由於 PyTorch GPU 版本需要從特定索引安裝，建議使用以下方法：

```bash
# 1. 先安裝 pip 到虛擬環境
uv pip install pip

# 2. 安裝 PyTorch GPU 版本（CUDA 12.1）
.venv/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安裝 FastAI
.venv/Scripts/python.exe -m pip install fastai

# 4. 驗證安裝
.venv/Scripts/python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**CUDA 版本選擇**：
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`（推薦，適用於大多數現代 GPU）
- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CPU 版本: `--index-url https://download.pytorch.org/whl/cpu`

#### 方法 B：使用 uv add（適用於 CPU 版本）

```bash
# 安裝 PyTorch（CPU 版本）
uv add torch torchvision torchaudio

# 安裝 FastAI
uv add fastai
```

**注意**：
- 深度學習模型非必需，AutoGluon 會自動使用可用的模型訓練器
- GPU 版本需要 NVIDIA GPU 和對應的 CUDA 驅動程式
- PyTorch GPU 版本檔案較大（約 2.5GB），請確保有足夠的硬碟空間
- 已測試環境：RTX 4080 SUPER + CUDA 13.0 驅動（使用 CUDA 12.1 PyTorch）

### 方法三：從頭建立專案

```bash
# 1. 初始化專案
uv init

# 2. 指定 Python 版本
uv python pin 3.12

# 3. 編輯 pyproject.toml（參考下方配置）

# 4. 同步安裝依賴
uv sync --no-install-project
```

## pyproject.toml 設定

本專案的 `pyproject.toml` 配置如下：

```toml
[project]
name = "team-8688"
version = "0.1.0"
description = "T-Brain 2025 玉山銀行警示帳戶預測競賽"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "autogluon.tabular>=1.0.0",
    "matplotlib>=3.7.0",
    "typing-extensions>=4.15.0",
    "lightgbm>=4.0.0",
    "xgboost>=3.0.0",
    "catboost>=1.2.0",
]
```

**重要說明**：
- 使用 `autogluon.tabular>=1.0.0`（不含 `[all]` extras），避免在 Windows 環境下編譯 Rust 依賴（tokenizers）的問題
- 包含三大梯度提升框架（LightGBM、XGBoost、CatBoost）以支援完整的模型訓練
- `typing-extensions` 是 AutoGluon 必需的依賴
- 此配置為腳本型專案，不需要 `[build-system]` 配置
- 使用 `--no-install-project` 參數跳過安裝專案本身

## 運行專案

### 1. 使用 uv run 執行 Python 腳本（推薦）

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

**Windows (PowerShell/CMD):**
```bash
# uv 會自動管理虛擬環境，也可以手動啟動
.venv\Scripts\activate

# 然後可以直接執行 python
python step0_short_acct.py
```

**Linux / macOS:**
```bash
# 啟動虛擬環境
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
uv sync --upgrade --no-install-project

# 鎖定依賴版本
uv lock

# 查看專案資訊
uv tree

# 執行 Python 腳本（自動使用虛擬環境）
uv run python <script.py>

# 清理快取
uv cache clean
```

## 注意事項

### 1. Windows 環境特別注意
- **不要使用 `autogluon.tabular[all]`**：會嘗試安裝需要 Rust 編譯器的依賴（如 tokenizers）
- **使用 `autogluon.tabular`（無 extras）**：只安裝核心功能，適用於大多數表格資料建模任務
- **如需完整功能**：請安裝 Rust 編譯器或在 Linux/macOS 環境下使用

### 2. AutoGluon 系統需求
- 建議至少 **8GB RAM**
- 建議至少 **10GB 可用硬碟空間**（用於模型訓練和快取）
- 訓練時間可能較長（取決於資料集大小和 `time_limit` 設定）

### 3. 中文顯示設定
如果需要在圖表中顯示中文，請在程式中設定：

**Linux 環境：**
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
```

**Windows 環境：**
```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']  # 微軟正黑體或黑體
plt.rcParams['axes.unicode_minus'] = False
```

### 4. 其他注意事項
- **編碼設定**：所有檔案使用 UTF-8 編碼
- **資料路徑**：使用 `datas/` 資料夾而非 `data/` 資料夾（根據 CLAUDE.md 指示）
- **Python 版本**：專案使用 Python 3.12，確保版本一致性

## 專案結構

```
C:\Users\User\work\TEAM_8688\  (Windows)
或
/home/rapids/notebooks/sk/TEAM_8688/  (Linux)
├── pyproject.toml          # UV 專案配置檔
├── uv.lock                 # 依賴鎖定檔（確保可重現的環境）
├── .python-version         # Python 版本指定（3.12）
├── .venv/                  # 虛擬環境目錄（自動生成）
├── data/                   # 原始資料（勿修改）
├── datas/                  # 處理後的資料
├── train/                  # 訓練資料和模型
├── step0_short_acct.py     # 步驟 0: 前處理帳戶資料
├── step1_process_acct_transactions.py  # 步驟 1: 處理交易資料
├── step2_feature_engineering.py        # 步驟 2: 特徵工程
├── step3_train_autogluon.py           # 步驟 3: 訓練 AutoGluon 模型
├── step4_predict_all_accounts.py      # 步驟 4: 預測所有帳戶
└── step5_answer.py                    # 步驟 5: 產生最終答案
```

## 完整工作流程範例

### 初次設定（新環境）

```bash
# 1. 切換到專案目錄
cd C:\Users\User\work\TEAM_8688  # Windows
# 或
cd /home/rapids/notebooks/sk/TEAM_8688  # Linux

# 2. 確認 Python 版本
uv python install 3.12

# 3. 同步安裝所有依賴
uv sync --no-install-project

```

### 執行完整建模流程

```bash
# 依序執行所有步驟
uv run python step0_short_acct.py
uv run python step1_process_acct_transactions.py
uv run python step2_feature_engineering.py
uv run python step3_train_autogluon.py
uv run python step4_predict_all_accounts.py
uv run python step5_answer.py
```

## 疑難排解

### 問題 1: 安裝失敗（tokenizers 編譯錯誤）
**解決方案**：確保 `pyproject.toml` 中使用 `autogluon.tabular>=1.0.0`（不含 `[all]`）

### 問題 2: 找不到模組
**解決方案**：確認是否使用 `uv run` 或已啟動虛擬環境

### 問題 3: 記憶體不足
**解決方案**：
- 減少 AutoGluon 的 `num_bag_folds` 或 `time_limit`
- 關閉其他佔用記憶體的程式
- 考慮使用雲端運算資源（如 Google Colab、Kaggle Notebooks）

### 問題 4: uv 指令找不到
**解決方案**：
- Windows: 重新啟動 PowerShell/CMD
- Linux/macOS: 執行 `source ~/.bashrc` 或 `source ~/.zshrc`

## 參考資源

- [UV 官方文檔](https://docs.astral.sh/uv/)
- [AutoGluon 官方文檔](https://auto.gluon.ai/)
- [AutoGluon Tabular 教學](https://auto.gluon.ai/stable/tutorials/tabular/index.html)
- [Pandas 官方文檔](https://pandas.pydata.org/)
- [UV GitHub Repository](https://github.com/astral-sh/uv)
