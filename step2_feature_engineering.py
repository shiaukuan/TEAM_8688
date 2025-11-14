# -*- coding: utf-8 -*-
"""
修正 42 筆 label（new_label_1.csv）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("V14 特徵工程 - V13 + 異常交易模式特徵")
print("=" * 80)

# ============================================================================
# 輔助函數（沿用 V13）
# ============================================================================

def extract_time_minutes(time_str):
    """將時間字串轉換為當日分鐘數"""
    try:
        if pd.isna(time_str):
            return 0
        h, m, s = map(int, str(time_str).split(':'))
        return h * 60 + m
    except:
        return 0

def calculate_time_window_features(daily_txns, windows=[60, 120, 240]):
    """計算單日內不同時間窗口的交易集中度（V9）"""
    features = {}

    if len(daily_txns) == 0:
        for window in windows:
            features[f'in_burst_{window}m'] = 0
            features[f'out_burst_{window}m'] = 0
            features[f'in_amt_{window}m'] = 0
        return features

    in_txns = daily_txns[daily_txns['direction'] == 'IN'].copy()
    out_txns = daily_txns[daily_txns['direction'] == 'OUT'].copy()

    for window in windows:
        max_in_count = 0
        max_out_count = 0
        max_in_amt = 0

        if len(in_txns) > 0:
            in_times = sorted(in_txns['time_minutes'].values)
            for start_time in in_times:
                count = sum((in_times >= start_time) & (in_times < start_time + window))
                amt = in_txns[(in_txns['time_minutes'] >= start_time) &
                             (in_txns['time_minutes'] < start_time + window)]['txn_amt'].sum()
                max_in_count = max(max_in_count, count)
                max_in_amt = max(max_in_amt, amt)

        if len(out_txns) > 0:
            out_times = sorted(out_txns['time_minutes'].values)
            for start_time in out_times:
                count = sum((out_times >= start_time) & (out_times < start_time + window))
                max_out_count = max(max_out_count, count)

        features[f'in_burst_{window}m'] = max_in_count
        features[f'out_burst_{window}m'] = max_out_count
        features[f'in_amt_{window}m'] = max_in_amt

    return features

def calculate_source_concentration(in_txns, windows=[60, 120, 240]):
    """計算IN交易的來源集中度（V9）"""
    features = {}

    if len(in_txns) == 0:
        for window in windows:
            features[f'same_source_ratio_{window}m'] = 0
            features[f'same_source_count_{window}m'] = 0
        return features

    for window in windows:
        max_ratio = 0
        max_count = 0

        in_times = sorted(in_txns['time_minutes'].unique())
        for start_time in in_times:
            window_txns = in_txns[(in_txns['time_minutes'] >= start_time) &
                                 (in_txns['time_minutes'] < start_time + window)]

            if len(window_txns) > 1:
                from_counts = window_txns['from_acct'].value_counts()
                top_ratio = from_counts.iloc[0] / len(window_txns)
                top_count = from_counts.iloc[0]

                max_ratio = max(max_ratio, top_ratio)
                max_count = max(max_count, top_count)

        features[f'same_source_ratio_{window}m'] = max_ratio
        features[f'same_source_count_{window}m'] = max_count

    return features

def calculate_target_concentration(out_txns, windows=[60, 120, 240]):
    """計算OUT交易的目標集中度（V9）"""
    features = {}

    if len(out_txns) == 0:
        for window in windows:
            features[f'same_target_ratio_{window}m'] = 0
            features[f'same_target_count_{window}m'] = 0
        return features

    for window in windows:
        max_ratio = 0
        max_count = 0

        out_times = sorted(out_txns['time_minutes'].unique())
        for start_time in out_times:
            window_txns = out_txns[(out_txns['time_minutes'] >= start_time) &
                                  (out_txns['time_minutes'] < start_time + window)]

            if len(window_txns) > 1:
                to_counts = window_txns['to_acct'].value_counts()
                top_ratio = to_counts.iloc[0] / len(window_txns)
                top_count = to_counts.iloc[0]

                max_ratio = max(max_ratio, top_ratio)
                max_count = max(max_count, top_count)

        features[f'same_target_ratio_{window}m'] = max_ratio
        features[f'same_target_count_{window}m'] = max_count

    return features

def calculate_small_amount_patterns(txns, threshold=None):
    """識別小額和重複金額模式（V9，改進：動態閾值）"""
    if len(txns) == 0:
        return {
            'small_amt_count': 0,
            'small_amt_ratio': 0,
            'repeated_amt_max_count': 0,
            'repeated_amt_types': 0
        }

    # 使用動態閾值（中位數）
    if threshold is None:
        threshold = txns['txn_amt'].median()

    small_txns = txns[txns['txn_amt'] < threshold]
    features = {
        'small_amt_count': len(small_txns),
        'small_amt_ratio': len(small_txns) / len(txns)
    }

    amt_counts = txns['txn_amt'].value_counts()
    features['repeated_amt_max_count'] = amt_counts.iloc[0] if len(amt_counts) > 0 else 0
    features['repeated_amt_types'] = (amt_counts >= 2).sum()

    return features

def calculate_time_gaps(txns):
    """計算交易間隔統計（V9）"""
    if len(txns) <= 1:
        return {
            'min_time_gap': 0,
            'short_gap_count': 0,
            'burst_session_count': 0
        }

    txns_sorted = txns.sort_values('time_minutes')
    gaps = txns_sorted['time_minutes'].diff().dropna()

    features = {
        'min_time_gap': gaps.min() if len(gaps) > 0 else 0,
        'short_gap_count': (gaps < 10).sum(),
    }

    # 爆發會話
    burst_sessions = 0
    current_session = 1
    for gap in gaps:
        if gap < 30:
            current_session += 1
        else:
            if current_session >= 3:
                burst_sessions += 1
            current_session = 1
    if current_session >= 3:
        burst_sessions += 1

    features['burst_session_count'] = burst_sessions

    return features

def calculate_same_type2_concentration(in_txns, windows=[60, 120]):
    """
    V10：計算短時間內來自同一 type2 帳戶的 IN 交易集中度
    """
    features = {}

    # 篩選 type2 IN交易
    type2_in = in_txns[in_txns['from_acct_type'] == 2].copy()

    if len(type2_in) == 0:
        for window in windows:
            features[f'same_type2_in_{window}m'] = 0
            features[f'same_type2_in_ratio_{window}m'] = 0
            features[f'same_type2_amt_{window}m'] = 0
        return features

    for window in windows:
        max_count = 0
        max_ratio = 0
        max_amt = 0

        type2_times = sorted(type2_in['time_minutes'].unique())
        for start_time in type2_times:
            window_txns = type2_in[(type2_in['time_minutes'] >= start_time) &
                                   (type2_in['time_minutes'] < start_time + window)]

            if len(window_txns) > 0:
                from_counts = window_txns['from_acct'].value_counts()
                top_count = from_counts.iloc[0]
                top_acct = from_counts.index[0]
                same_acct_amt = window_txns[window_txns['from_acct'] == top_acct]['txn_amt'].sum()
                total_in_count = len(in_txns)
                ratio = top_count / max(total_in_count, 1)

                max_count = max(max_count, top_count)
                max_ratio = max(max_ratio, ratio)
                max_amt = max(max_amt, same_acct_amt)

        features[f'same_type2_in_{window}m'] = max_count
        features[f'same_type2_in_ratio_{window}m'] = max_ratio
        features[f'same_type2_amt_{window}m'] = max_amt

    return features

def calculate_abnormal_large_in(in_txns, all_txns):
    """
    V10：異常大額匯入檢測
    """
    features = {}

    if len(in_txns) == 0:
        return {
            'max_in_amt_vs_history': 1.0,
            'top3_in_amt_ratio': 0,
            'sudden_large_in_7d': 1.0,
            'large_in_count_ratio': 0,
            'max_in_amt_std_score': 0,
            'recent_large_in_spike': 0,
            'in_amt_iqr_outlier': 0,
            'max_single_day_in_amt': 1.0
        }

    in_amts = in_txns['txn_amt'].values

    # 1. 最大IN金額 vs 歷史均值
    max_in = in_amts.max()
    mean_in = in_amts.mean()
    features['max_in_amt_vs_history'] = max_in / max(mean_in, 1e-6)

    # 2. 前3大IN金額占比
    top3_sum = np.sort(in_amts)[-3:].sum() if len(in_amts) >= 3 else in_amts.sum()
    features['top3_in_amt_ratio'] = top3_sum / max(in_amts.sum(), 1e-6)

    # 3. 最後7天 vs 前期的最大IN
    last7_in = in_txns[in_txns['days_to_end'] <= 7]
    early_in = in_txns[in_txns['days_to_end'] > 7]

    if len(last7_in) > 0 and len(early_in) > 0:
        features['sudden_large_in_7d'] = last7_in['txn_amt'].max() / max(early_in['txn_amt'].max(), 1e-6)
    else:
        features['sudden_large_in_7d'] = 1.0

    # 4. 大額IN占比
    large_threshold = mean_in * 3
    large_count = (in_amts > large_threshold).sum()
    features['large_in_count_ratio'] = large_count / len(in_amts)

    # 5. 最大IN的Z-score
    if len(in_amts) > 1:
        std_in = in_amts.std()
        features['max_in_amt_std_score'] = (max_in - mean_in) / max(std_in, 1e-6)
    else:
        features['max_in_amt_std_score'] = 0

    # 6. 最近5筆中的大額IN數量
    recent_5 = in_txns.sort_values('days_to_end').head(5)
    median_in = np.median(in_amts)
    large_recent = (recent_5['txn_amt'] > median_in * 2).sum()
    features['recent_large_in_spike'] = large_recent

    # 7. IQR離群值
    q1 = np.percentile(in_amts, 25)
    q3 = np.percentile(in_amts, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outlier_count = (in_amts > upper_bound).sum()
    features['in_amt_iqr_outlier'] = outlier_count

    # 8. 單日最大IN金額 vs 日均
    if 'txn_date' in in_txns.columns:
        daily_in = in_txns.groupby('txn_date')['txn_amt'].sum()
        if len(daily_in) > 0:
            max_daily = daily_in.max()
            mean_daily = daily_in.mean()
            features['max_single_day_in_amt'] = max_daily / max(mean_daily, 1e-6)
        else:
            features['max_single_day_in_amt'] = 1.0
    else:
        features['max_single_day_in_amt'] = 1.0

    return features

def calculate_cross_day_patterns(acct_df):
    """
    V10：跨天連續模式
    """
    features = {}

    last7 = acct_df[acct_df['days_to_end'] <= 7].copy()

    if len(last7) == 0:
        return {
            'consecutive_in_days': 0,
            'daily_in_burst_count': 0,
            'cross_day_same_source': 0,
            'rapid_clear_pattern': 0
        }

    # 1. 連續有IN交易的最長天數
    in_txns = last7[last7['direction'] == 'IN']
    if len(in_txns) > 0:
        in_dates = sorted(in_txns['txn_date'].unique())
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(in_dates)):
            if in_dates[i] == in_dates[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        features['consecutive_in_days'] = max_consecutive
    else:
        features['consecutive_in_days'] = 0

    # 2. 單日IN交易數 > 10 的天數
    daily_in_counts = last7[last7['direction'] == 'IN'].groupby('txn_date').size()
    features['daily_in_burst_count'] = (daily_in_counts > 10).sum()

    # 3. 不同天出現的相同type2來源數
    type2_in = last7[(last7['direction'] == 'IN') & (last7['from_acct_type'] == 2)]
    if len(type2_in) > 0:
        acct_days = type2_in.groupby('from_acct')['txn_date'].nunique()
        features['cross_day_same_source'] = (acct_days >= 2).sum()
    else:
        features['cross_day_same_source'] = 0

    # 4. 快速清空模式
    rapid_clear = 0
    for date, daily_txns in last7.groupby('txn_date'):
        daily_txns = daily_txns.sort_values('time_minutes')
        has_in = (daily_txns['direction'] == 'IN').any()
        has_out = (daily_txns['direction'] == 'OUT').any()

        if has_in and has_out:
            first_in_time = daily_txns[daily_txns['direction'] == 'IN']['time_minutes'].min()
            first_out_time = daily_txns[daily_txns['direction'] == 'OUT']['time_minutes'].min()

            if first_in_time < first_out_time:
                rapid_clear += 1

    features['rapid_clear_pattern'] = rapid_clear

    return features

# ============================================================================
# V12 函數（移除 0/1 判斷特徵）
# ============================================================================

def calculate_gini_coefficient(values):
    """計算 Gini 係數（0=完全平等, 1=完全不平等）"""
    if len(values) == 0:
        return 0
    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

def calculate_account_diversity_features(acct_df):
    """
    特徵組 1：異常帳戶多樣性特徵（V12，移除 0/1 判斷）
    """
    features = {}

    in_txns = acct_df[acct_df['direction'] == 'IN']
    out_txns = acct_df[acct_df['direction'] == 'OUT']

    # IN 交易的帳戶多樣性
    if len(in_txns) > 0:
        unique_from = in_txns['from_acct'].nunique()
        total_in = len(in_txns)
        features['unique_from_acct_ratio'] = unique_from / max(total_in, 1)

        # Gini 係數（衡量集中度）
        from_counts = in_txns['from_acct'].value_counts().values
        features['in_diversity_score'] = calculate_gini_coefficient(from_counts)
    else:
        features['unique_from_acct_ratio'] = 0
        features['in_diversity_score'] = 0

    # OUT 交易的帳戶多樣性
    if len(out_txns) > 0:
        unique_to = out_txns['to_acct'].nunique()
        total_out = len(out_txns)
        features['unique_to_acct_ratio'] = unique_to / max(total_out, 1)

        # Gini 係數
        to_counts = out_txns['to_acct'].value_counts().values
        features['out_diversity_score'] = calculate_gini_coefficient(to_counts)
    else:
        features['unique_to_acct_ratio'] = 0
        features['out_diversity_score'] = 0

    return features

def calculate_rapid_multi_account_out(acct_df, windows=[60, 120, 240]):
    """
    特徵組 2：短時間多帳號匯出特徵（V12）
    """
    features = {}

    out_txns = acct_df[acct_df['direction'] == 'OUT']

    if len(out_txns) == 0:
        features['max_out_accounts_1day'] = 0
        for window in windows:
            features[f'max_out_accounts_{window}m'] = 0
            features[f'out_burst_diversity_{window}m'] = 0
        features['rapid_multi_out_pattern'] = 0
        return features

    # 1. 單日最多匯出到幾個不同帳戶
    daily_out_accounts = out_txns.groupby('txn_date')['to_acct'].nunique()
    features['max_out_accounts_1day'] = daily_out_accounts.max()

    # 2. 時間窗口內匯出的不同帳戶數
    for window in windows:
        max_accounts = 0
        max_diversity = 0

        out_times = sorted(out_txns['time_minutes'].unique())
        for start_time in out_times:
            window_txns = out_txns[(out_txns['time_minutes'] >= start_time) &
                                   (out_txns['time_minutes'] < start_time + window)]

            if len(window_txns) > 0:
                unique_accounts = window_txns['to_acct'].nunique()
                max_accounts = max(max_accounts, unique_accounts)

                # 計算多樣性（不同帳戶數 / 總交易數）
                diversity = unique_accounts / len(window_txns)
                max_diversity = max(max_diversity, diversity)

        features[f'max_out_accounts_{window}m'] = max_accounts
        features[f'out_burst_diversity_{window}m'] = max_diversity

    # 3. 快速多帳號匯出模式計數（10分鐘內匯出>=3個不同帳戶）
    rapid_pattern_count = 0
    out_sorted = out_txns.sort_values('time_minutes')

    for idx in range(len(out_sorted)):
        current_time = out_sorted.iloc[idx]['time_minutes']
        window_txns = out_sorted[(out_sorted['time_minutes'] >= current_time) &
                                 (out_sorted['time_minutes'] < current_time + 10)]

        if window_txns['to_acct'].nunique() >= 3:
            rapid_pattern_count += 1

    features['rapid_multi_out_pattern'] = rapid_pattern_count

    return features


def calculate_active_days_features(acct_df, high_freq_thresholds=[5, 10, 20]):
    """
    特徵組 4：活躍天數與高頻交易特徵（只使用比例，不用絕對值）（V12）
    """
    features = {}

    # 計算總天數範圍
    if len(acct_df) == 0:
        features['active_days_ratio'] = 0
        features['consecutive_days_ratio'] = 0
        for threshold in high_freq_thresholds:
            features[f'high_freq_ratio_{threshold}'] = 0
            features[f'consecutive_high_freq_ratio_{threshold}'] = 0
        features['daily_txn_variability'] = 0
        return features

    # 1. 活躍天數統計
    active_dates = acct_df['txn_date'].unique()
    total_active_days = len(active_dates)

    # 計算總天數（從最早到最晚）
    min_date = acct_df['txn_date'].min()
    max_date = acct_df['txn_date'].max()
    total_days_span = max_date - min_date + 1

    features['active_days_ratio'] = total_active_days / max(total_days_span, 1)

    # 2. 每日交易數統計（使用變異係數，不用絕對值）
    daily_txn_counts = acct_df.groupby('txn_date').size()
    mean_daily = daily_txn_counts.mean()
    std_daily = daily_txn_counts.std()
    # 變異係數：標準差/平均值，衡量交易量的波動程度
    features['daily_txn_variability'] = std_daily / max(mean_daily, 1e-6) if std_daily > 0 else 0

    # 3. 高頻交易天數比例
    for threshold in high_freq_thresholds:
        high_freq_days = (daily_txn_counts > threshold).sum()
        features[f'high_freq_ratio_{threshold}'] = high_freq_days / total_active_days

    # 4. 連續活躍模式（使用比例）
    sorted_dates = sorted(active_dates)

    # 最長連續有交易的天數（轉換為比例）
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] == sorted_dates[i-1] + 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    # 轉換為比例：最長連續天數 / 總活躍天數
    features['consecutive_days_ratio'] = max_consecutive / total_active_days

    # 5. 連續高頻天數比例
    for threshold in high_freq_thresholds:
        # 找出高頻的日期
        high_freq_dates = sorted(daily_txn_counts[daily_txn_counts > threshold].index)

        if len(high_freq_dates) > 0:
            max_consecutive_high = 1
            current_consecutive_high = 1
            for i in range(1, len(high_freq_dates)):
                if high_freq_dates[i] == high_freq_dates[i-1] + 1:
                    current_consecutive_high += 1
                    max_consecutive_high = max(max_consecutive_high, current_consecutive_high)
                else:
                    current_consecutive_high = 1

            # 轉換為比例：最長連續高頻天數 / 總活躍天數
            features[f'consecutive_high_freq_ratio_{threshold}'] = max_consecutive_high / total_active_days
        else:
            features[f'consecutive_high_freq_ratio_{threshold}'] = 0

    return features

# ============================================================================
# V13 函數：詐騙模式特徵
# ============================================================================

def calculate_one_way_counterparty_features(acct_df):
    """
    特徵組 5：單向交易對象分析（93.4% 案例特徵）
    問題：大部分交易對象都只有單向往來（只轉入或只轉出）
    """
    features = {}

    in_txns = acct_df[acct_df['direction'] == 'IN']
    out_txns = acct_df[acct_df['direction'] == 'OUT']

    # 統計所有交易對象
    in_counterparties = set(in_txns['from_acct'].unique()) if len(in_txns) > 0 else set()
    out_counterparties = set(out_txns['to_acct'].unique()) if len(out_txns) > 0 else set()

    all_counterparties = in_counterparties | out_counterparties
    bidirectional = in_counterparties & out_counterparties
    one_way = all_counterparties - bidirectional

    total_counterparties = len(all_counterparties)

    if total_counterparties > 0:
        features['one_way_counterparty_ratio'] = len(one_way) / total_counterparties
        features['bidirectional_counterparty_ratio'] = len(bidirectional) / total_counterparties
    else:
        features['one_way_counterparty_ratio'] = 0
        features['bidirectional_counterparty_ratio'] = 0

    # 計數
    features['in_counterparties_count'] = len(in_counterparties)
    features['out_counterparties_count'] = len(out_counterparties)
    features['one_way_counterparties_count'] = len(one_way)
    features['bidirectional_counterparties_count'] = len(bidirectional)

    return features

def calculate_in_to_out_time_features(acct_df):
    """
    特徵組 6：收款後轉出時間分析（43.4% 案例特徵）
    問題：收款後快速轉出（平均3天內）
    """
    features = {}

    in_txns = acct_df[acct_df['direction'] == 'IN'].copy()
    out_txns = acct_df[acct_df['direction'] == 'OUT'].copy()

    if len(in_txns) == 0 or len(out_txns) == 0:
        return {
            'avg_days_in_to_out': 0,
            'median_days_in_to_out': 0,
            'min_days_in_to_out': 0,
            'quick_turnover_ratio': 0,
            'quick_turnover_3d_ratio': 0,
            'first_in_to_first_out_days': 0
        }

    # 計算每筆 IN 後的最近一筆 OUT 時間差（以天數計）
    in_txns_sorted = in_txns.sort_values('txn_date').copy()
    out_txns_sorted = out_txns.sort_values('txn_date').copy()

    time_diffs = []
    quick_count = 0
    quick_3d_count = 0

    for _, in_row in in_txns_sorted.iterrows():
        in_date = in_row['txn_date']
        # 找出該 IN 後的第一筆 OUT
        future_outs = out_txns_sorted[out_txns_sorted['txn_date'] >= in_date]

        if len(future_outs) > 0:
            first_out_date = future_outs.iloc[0]['txn_date']
            diff_days = first_out_date - in_date
            time_diffs.append(diff_days)

            if diff_days <= 1:
                quick_count += 1
            if diff_days <= 3:
                quick_3d_count += 1

    if len(time_diffs) > 0:
        features['avg_days_in_to_out'] = np.mean(time_diffs)
        features['median_days_in_to_out'] = np.median(time_diffs)
        features['min_days_in_to_out'] = np.min(time_diffs)
        features['quick_turnover_ratio'] = quick_count / len(time_diffs)
        features['quick_turnover_3d_ratio'] = quick_3d_count / len(time_diffs)
    else:
        features['avg_days_in_to_out'] = 0
        features['median_days_in_to_out'] = 0
        features['min_days_in_to_out'] = 0
        features['quick_turnover_ratio'] = 0
        features['quick_turnover_3d_ratio'] = 0

    # 第一筆 IN 到第一筆 OUT 的天數
    first_in_date = in_txns_sorted.iloc[0]['txn_date']
    first_out_date = out_txns_sorted.iloc[0]['txn_date']
    features['first_in_to_first_out_days'] = abs(first_out_date - first_in_date)

    return features

def calculate_flow_asymmetry_features(acct_df):
    """
    特徵組 7：資金流向不對稱分析（42% 案例特徵）
    問題：轉出遠大於轉入 OR 轉入遠大於轉出
    """
    features = {}

    in_txns = acct_df[acct_df['direction'] == 'IN']
    out_txns = acct_df[acct_df['direction'] == 'OUT']

    total_in_amt = in_txns['txn_amt'].sum() if len(in_txns) > 0 else 0
    total_out_amt = out_txns['txn_amt'].sum() if len(out_txns) > 0 else 0

    total_in_count = len(in_txns)
    total_out_count = len(out_txns)

    # 1. 金額不對稱
    if total_in_amt > 0:
        features['flow_asymmetry_ratio'] = total_out_amt / total_in_amt
    else:
        features['flow_asymmetry_ratio'] = 0

    # 2. 淨流量占比
    total_amt = total_in_amt + total_out_amt
    if total_amt > 0:
        features['net_flow_ratio'] = abs(total_in_amt - total_out_amt) / total_amt
    else:
        features['net_flow_ratio'] = 0

    # 3. 筆數不對稱
    if total_in_count > 0:
        features['flow_count_asymmetry'] = total_out_count / total_in_count
    else:
        features['flow_count_asymmetry'] = 0

    # 4. 平均單筆金額比
    avg_in = total_in_amt / max(total_in_count, 1)
    avg_out = total_out_amt / max(total_out_count, 1)

    if avg_in > 0:
        features['avg_amt_in_out_ratio'] = avg_out / avg_in
    else:
        features['avg_amt_in_out_ratio'] = 0

    return features

def calculate_dispersion_features(acct_df):
    """
    特徵組 8：轉入/轉出分散度改進（38.2% 案例特徵）
    問題：每個對象只交易1-2次，用於分散資金
    """
    features = {}

    in_txns = acct_df[acct_df['direction'] == 'IN']
    out_txns = acct_df[acct_df['direction'] == 'OUT']

    # 轉入分散度
    if len(in_txns) > 0:
        in_counterparties = in_txns['from_acct'].nunique()
        features['in_dispersion_score'] = in_counterparties / len(in_txns)

        # 每個來源平均交易次數
        in_counts = in_txns['from_acct'].value_counts()
        features['avg_txn_per_in_source'] = in_counts.mean()
        features['single_txn_in_sources_ratio'] = (in_counts == 1).sum() / len(in_counts)
    else:
        features['in_dispersion_score'] = 0
        features['avg_txn_per_in_source'] = 0
        features['single_txn_in_sources_ratio'] = 0

    # 轉出分散度
    if len(out_txns) > 0:
        out_counterparties = out_txns['to_acct'].nunique()
        features['out_dispersion_score'] = out_counterparties / len(out_txns)

        # 每個目標平均交易次數
        out_counts = out_txns['to_acct'].value_counts()
        features['avg_txn_per_out_target'] = out_counts.mean()
        features['single_txn_out_targets_ratio'] = (out_counts == 1).sum() / len(out_counts)
    else:
        features['out_dispersion_score'] = 0
        features['avg_txn_per_out_target'] = 0
        features['single_txn_out_targets_ratio'] = 0

    return features

def calculate_small_in_features(acct_df):
    """
    特徵組 9：小額匯入特徵改進（21.1% 案例特徵）
    問題：超過60%的轉入交易為小額
    """
    features = {}

    in_txns = acct_df[acct_df['direction'] == 'IN']

    if len(in_txns) == 0:
        return {
            'small_in_ratio_dynamic': 0,
            'small_in_density': 0,
            'small_in_count': 0
        }

    # 小額閾值
    # median_in = in_txns['txn_amt'].median()
    median_in = 200  # 固定為200元
    small_in = in_txns[in_txns['txn_amt'] < median_in]

    features['small_in_ratio_dynamic'] = len(small_in) / len(in_txns)
    features['small_in_count'] = len(small_in)

    # 小額轉入密度（小額轉入數 / 活躍天數）
    active_days = acct_df['txn_date'].nunique()
    features['small_in_density'] = len(small_in) / max(active_days, 1)

    return features

# ============================================================================
# V14 新函數：異常交易模式特徵
# ============================================================================

def calculate_concurrent_txns_features(acct_df):
    """
    【V14 新特徵 1】同時多筆交易特徵
    定義：同一天同一時間點（精確到分鐘）有多筆相同方向的交易
    """
    features = {}

    if len(acct_df) == 0:
        return {
            'max_concurrent_txns_in': 0,
            'max_concurrent_txns_out': 0,
            'concurrent_time_points_in': 0,
            'concurrent_time_points_out': 0,
            'max_concurrent_txns_all': 0,
            'concurrent_time_points_all': 0
        }

    # 分析 IN 和 OUT 分別的同時交易
    for direction in ['IN', 'OUT']:
        dir_txns = acct_df[acct_df['direction'] == direction]

        if len(dir_txns) > 0:
            # 按日期+時間分組，計算每個時間點的交易數
            time_groups = dir_txns.groupby(['txn_date', 'txn_time']).size()

            # 最大同時交易筆數
            max_concurrent = time_groups.max() if len(time_groups) > 0 else 0
            # 有多筆同時交易的時間點數量（>=2筆）
            concurrent_points = (time_groups >= 2).sum()

            suffix = '_in' if direction == 'IN' else '_out'
            features[f'max_concurrent_txns{suffix}'] = max_concurrent
            features[f'concurrent_time_points{suffix}'] = concurrent_points
        else:
            suffix = '_in' if direction == 'IN' else '_out'
            features[f'max_concurrent_txns{suffix}'] = 0
            features[f'concurrent_time_points{suffix}'] = 0

    # 不分方向的總體同時交易
    time_groups_all = acct_df.groupby(['txn_date', 'txn_time']).size()
    features['max_concurrent_txns_all'] = time_groups_all.max() if len(time_groups_all) > 0 else 0
    features['concurrent_time_points_all'] = (time_groups_all >= 2).sum()

    return features

def calculate_abnormal_hour_features(acct_df):
    """
    【V14 新特徵 2】異常時間交易特徵
    定義：凌晨0-6點或深夜22-24點的交易比例過高
    """
    features = {}

    if len(acct_df) == 0:
        return {
            'abnormal_hour_ratio': 0,
            'abnormal_hour_count': 0,
            'late_night_ratio': 0,
            'early_morning_ratio': 0
        }

    # 異常時間：0-6點 或 22-24點
    abnormal_txns = acct_df[(acct_df['hour'] < 6) | (acct_df['hour'] >= 22)]
    late_night = acct_df[acct_df['hour'] >= 22]
    early_morning = acct_df[acct_df['hour'] < 6]

    features['abnormal_hour_ratio'] = len(abnormal_txns) / len(acct_df)
    features['abnormal_hour_count'] = len(abnormal_txns)
    features['late_night_ratio'] = len(late_night) / len(acct_df)
    features['early_morning_ratio'] = len(early_morning) / len(acct_df)

    return features

def calculate_repeated_amount_features(acct_df):
    """
    【V14 新特徵 3】固定金額重複度特徵
    定義：某個特定金額重複出現≥5次
    """
    features = {}

    if len(acct_df) == 0:
        return {
            'top_amount_count': 0,
            'repeated_amounts_5plus': 0,
            'repeated_amounts_10plus': 0,
            'amount_diversity_score': 0
        }

    amount_counts = acct_df['txn_amt'].value_counts()

    # 最常出現金額的重複次數
    features['top_amount_count'] = amount_counts.iloc[0] if len(amount_counts) > 0 else 0

    # 重複出現≥5次的金額種類數
    features['repeated_amounts_5plus'] = (amount_counts >= 5).sum()
    features['repeated_amounts_10plus'] = (amount_counts >= 10).sum()

    # 金額多樣性分數（重複金額占比）
    if len(amount_counts) > 0:
        features['amount_diversity_score'] = (amount_counts >= 2).sum() / len(amount_counts)
    else:
        features['amount_diversity_score'] = 0

    return features

def calculate_consecutive_days_features(acct_df):
    """
    【V14 新特徵 4】連續交易天數特徵
    定義：連續N天每天都有至少一筆交易
    """
    features = {}

    if len(acct_df) == 0:
        return {
            'max_consecutive_txn_days': 0,
            'consecutive_in_days': 0,
            'consecutive_out_days': 0,
            'consecutive_days_ratio': 0
        }

    # 所有交易的連續天數
    all_dates = sorted(acct_df['txn_date'].unique())
    max_consecutive_all = 1
    current_consecutive = 1
    for i in range(1, len(all_dates)):
        if all_dates[i] == all_dates[i-1] + 1:
            current_consecutive += 1
            max_consecutive_all = max(max_consecutive_all, current_consecutive)
        else:
            current_consecutive = 1

    features['max_consecutive_txn_days'] = max_consecutive_all

    # IN 交易的連續天數
    in_dates = sorted(acct_df[acct_df['direction'] == 'IN']['txn_date'].unique())
    max_consecutive_in = 1
    current_consecutive = 1
    for i in range(1, len(in_dates)):
        if in_dates[i] == in_dates[i-1] + 1:
            current_consecutive += 1
            max_consecutive_in = max(max_consecutive_in, current_consecutive)
        else:
            current_consecutive = 1
    features['consecutive_in_days'] = max_consecutive_in if len(in_dates) > 0 else 0

    # OUT 交易的連續天數
    out_dates = sorted(acct_df[acct_df['direction'] == 'OUT']['txn_date'].unique())
    max_consecutive_out = 1
    current_consecutive = 1
    for i in range(1, len(out_dates)):
        if out_dates[i] == out_dates[i-1] + 1:
            current_consecutive += 1
            max_consecutive_out = max(max_consecutive_out, current_consecutive)
        else:
            current_consecutive = 1
    features['consecutive_out_days'] = max_consecutive_out if len(out_dates) > 0 else 0

    # 連續天數占活躍天數比例
    total_active_days = len(all_dates)
    features['consecutive_days_ratio'] = max_consecutive_all / max(total_active_days, 1)

    return features

def calculate_out_concentration_features(acct_df):
    """
    【V14 新特徵 5】轉出集中度特徵
    定義：轉出交易中，集中給少數帳戶的比例
    """
    features = {}

    out_txns = acct_df[acct_df['direction'] == 'OUT']

    if len(out_txns) == 0:
        return {
            'top1_out_ratio': 0,
            'top3_out_ratio': 0,
            'top5_out_ratio': 0,
            'out_concentration_gini': 0
        }

    to_acct_counts = out_txns['to_acct'].value_counts()

    # Top 1, 3, 5 集中度
    features['top1_out_ratio'] = to_acct_counts.iloc[0] / len(out_txns) if len(to_acct_counts) > 0 else 0
    features['top3_out_ratio'] = to_acct_counts.head(3).sum() / len(out_txns) if len(to_acct_counts) > 0 else 0
    features['top5_out_ratio'] = to_acct_counts.head(5).sum() / len(out_txns) if len(to_acct_counts) > 0 else 0

    # Gini 係數（集中度）
    features['out_concentration_gini'] = calculate_gini_coefficient(to_acct_counts.values)

    return features

def calculate_high_freq_day_features(acct_df):
    """
    【V14 新特徵 6】單日高頻交易特徵
    定義：某一天的交易筆數異常高
    """
    features = {}

    if len(acct_df) == 0:
        return {
            'max_daily_txns': 0,
            'avg_daily_txns': 0,
            'high_freq_days_10': 0,
            'high_freq_days_20': 0,
            'daily_txn_std': 0
        }

    daily_counts = acct_df.groupby('txn_date').size()

    features['max_daily_txns'] = daily_counts.max()
    features['avg_daily_txns'] = daily_counts.mean()
    features['daily_txn_std'] = daily_counts.std()

    # 高頻天數（單日>=10筆、>=20筆）
    features['high_freq_days_10'] = (daily_counts >= 10).sum()
    features['high_freq_days_20'] = (daily_counts >= 20).sum()

    return features

def calculate_single_direction_features(acct_df):
    """
    【V14 新特徵 7】完全單向交易特徵
    定義：只有轉入或只有轉出
    """
    features = {}

    in_count = len(acct_df[acct_df['direction'] == 'IN'])
    out_count = len(acct_df[acct_df['direction'] == 'OUT'])

    features['is_only_in'] = 1 if in_count > 0 and out_count == 0 else 0
    features['is_only_out'] = 1 if out_count > 0 and in_count == 0 else 0
    features['in_out_imbalance'] = abs(in_count - out_count) / max(in_count + out_count, 1)

    return features

def calculate_time_variance_features(acct_df):
    """
    【V14 新特徵 8】交易時間方差特徵
    定義：交易時間的標準差（低方差=固定時段操作）
    """
    features = {}

    if len(acct_df) == 0 or 'hour' not in acct_df.columns:
        return {
            'hour_std': 0,
            'time_concentration_score': 0
        }

    # 小時標準差
    features['hour_std'] = acct_df['hour'].std()

    # 時間集中度（最常出現小時的占比）
    hour_counts = acct_df['hour'].value_counts()
    features['time_concentration_score'] = hour_counts.iloc[0] / len(acct_df) if len(hour_counts) > 0 else 0

    return features



# ============================================================================
# 載入資料
# ============================================================================
print("\n[1/8] 載入資料...")
df = pd.read_csv('../train/all_acct_transactions.csv')
print(f"總筆數: {len(df):,}")
print(f"總帳戶數: {df['acct'].nunique()}")

# 載入需要修正的 label
print("\n[2/8] 修正 label...")
new_label_1 = pd.read_csv('../train/new_label_1.csv')
new_label_1_accts = set(new_label_1['acct'].values)

print(f"需要修正的帳戶數: {len(new_label_1_accts)}")

# 修正 label
df.loc[df['acct'].isin(new_label_1_accts), 'label'] = 1

print(f"修正後 label=1 數量: {(df['label']==1).sum()}")
print(f"修正後 label=0 數量: {(df['label']==0).sum()}")

# 轉換時間為分鐘數
print("\n[3/8] 轉換時間格式...")
df['time_minutes'] = df['txn_time'].apply(extract_time_minutes)

# ============================================================================
# 特徵計算
# ============================================================================
print("\n[4/8] 計算特徵...")
print("  - V8 基礎特徵")
print("  - V9 短時間聚集特徵")
print("  - V10 詐騙模式特徵")
print("  - V12 新特徵組 1-4（移除 0/1 判斷）")
print("  - V13 新特徵組 5-9（詐騙模式）")
print("  - V14 新特徵組（異常交易模式）")

features_dict = {}

LAST_N_TXNS = [5, 10, 20]  # V8
WINDOWS = [60, 120, 240]  # V9
TYPE2_WINDOWS = [60, 120]  # V10

for idx, acct in enumerate(df['acct'].unique()):
    if (idx + 1) % 500 == 0:
        print(f"  處理進度: {idx+1}/{df['acct'].nunique()} 帳戶...")

    acct_df = df[df['acct'] == acct].copy()
    acct_df = acct_df.sort_values('days_to_end')
    label = acct_df['label'].iloc[0]

    feat = {
        'acct': acct,
        'label': label
    }

    # ========== V8 特徵 ==========
    total_in = len(acct_df[acct_df['direction'] == 'IN'])
    total_out = len(acct_df[acct_df['direction'] == 'OUT'])
    feat['in_out_ratio'] = total_in / (total_out + 1e-6)

    last7_df = acct_df[acct_df['days_to_end'] <= 7]
    if len(last7_df) > 0:
        feat['txn_concentration_7d'] = len(last7_df) / len(acct_df)
        last7_in = last7_df[last7_df['direction'] == 'IN']
        feat['in_concentration_7d'] = len(last7_in) / max(len(last7_df), 1)
    else:
        feat['txn_concentration_7d'] = 0
        feat['in_concentration_7d'] = 0

    for n in LAST_N_TXNS:
        if len(acct_df) >= n:
            last_n = acct_df.head(n)
            early_n = acct_df.iloc[n:]

            last_n_in = last_n[last_n['direction'] == 'IN']
            feat[f'last{n}_in_pct'] = len(last_n_in) / len(last_n)

            if len(last_n_in) > 0:
                feat[f'last{n}_type2_pct'] = (last_n_in['from_acct_type'] == 2).sum() / len(last_n_in)
            else:
                feat[f'last{n}_type2_pct'] = 0

            if len(early_n) > 0:
                feat[f'last{n}_amt_ratio'] = last_n['txn_amt'].mean() / max(early_n['txn_amt'].mean(), 1e-6)
            else:
                feat[f'last{n}_amt_ratio'] = 1.0
        else:
            feat[f'last{n}_in_pct'] = 0
            feat[f'last{n}_type2_pct'] = 0
            feat[f'last{n}_amt_ratio'] = 1.0

    early_df_7 = acct_df[acct_df['days_to_end'] > 7]
    if len(last7_df) > 0 and len(early_df_7) > 0:
        feat['max_amt_ratio_7d'] = last7_df['txn_amt'].max() / max(early_df_7['txn_amt'].max(), 1e-6)
    else:
        feat['max_amt_ratio_7d'] = 1.0

    # ========== V9 特徵 ==========
    daily_features = []

    for date, daily_txns in acct_df.groupby('txn_date'):
        daily_feat = {}

        window_feat = calculate_time_window_features(daily_txns, WINDOWS)
        daily_feat.update(window_feat)

        in_txns = daily_txns[daily_txns['direction'] == 'IN']
        source_feat = calculate_source_concentration(in_txns, WINDOWS)
        daily_feat.update(source_feat)

        out_txns = daily_txns[daily_txns['direction'] == 'OUT']
        target_feat = calculate_target_concentration(out_txns, WINDOWS)
        daily_feat.update(target_feat)

        small_feat = calculate_small_amount_patterns(daily_txns)
        daily_feat.update(small_feat)

        gap_feat = calculate_time_gaps(daily_txns)
        daily_feat.update(gap_feat)

        daily_features.append(daily_feat)

    if len(daily_features) > 0:
        daily_df = pd.DataFrame(daily_features)
        for col in daily_df.columns:
            feat[f'max_{col}'] = daily_df[col].max()
    else:
        for window in WINDOWS:
            feat[f'max_in_burst_{window}m'] = 0
            feat[f'max_out_burst_{window}m'] = 0
            feat[f'max_in_amt_{window}m'] = 0
            feat[f'max_same_source_ratio_{window}m'] = 0
            feat[f'max_same_source_count_{window}m'] = 0
            feat[f'max_same_target_ratio_{window}m'] = 0
            feat[f'max_same_target_count_{window}m'] = 0
        feat['max_small_amt_count'] = 0
        feat['max_small_amt_ratio'] = 0
        feat['max_repeated_amt_max_count'] = 0
        feat['max_repeated_amt_types'] = 0
        feat['max_min_time_gap'] = 0
        feat['max_short_gap_count'] = 0
        feat['max_burst_session_count'] = 0

    # ========== V10 特徵 ==========
    all_in = acct_df[acct_df['direction'] == 'IN']

    type2_daily_features = []
    for date, daily_txns in acct_df.groupby('txn_date'):
        daily_in = daily_txns[daily_txns['direction'] == 'IN']
        type2_feat = calculate_same_type2_concentration(daily_in, TYPE2_WINDOWS)
        type2_daily_features.append(type2_feat)

    if len(type2_daily_features) > 0:
        type2_df = pd.DataFrame(type2_daily_features)
        for col in type2_df.columns:
            feat[f'max_{col}'] = type2_df[col].max()
    else:
        for window in TYPE2_WINDOWS:
            feat[f'max_same_type2_in_{window}m'] = 0
            feat[f'max_same_type2_in_ratio_{window}m'] = 0
            feat[f'max_same_type2_amt_{window}m'] = 0

    large_in_feat = calculate_abnormal_large_in(all_in, acct_df)
    feat.update(large_in_feat)

    cross_day_feat = calculate_cross_day_patterns(acct_df)
    feat.update(cross_day_feat)

    # ========== V12 新特徵（移除 0/1 判斷）==========
    # 特徵組 1: 帳戶多樣性
    diversity_feat = calculate_account_diversity_features(acct_df)
    feat.update(diversity_feat)

    # 特徵組 2: 多帳號匯出
    multi_out_feat = calculate_rapid_multi_account_out(acct_df, WINDOWS)
    feat.update(multi_out_feat)

    # 特徵組 4: 活躍天數與高頻交易
    active_days_feat = calculate_active_days_features(acct_df, [5, 10, 20])
    feat.update(active_days_feat)

    # ========== V13 新特徵 ==========
    # 特徵組 5: 單向交易對象
    one_way_feat = calculate_one_way_counterparty_features(acct_df)
    feat.update(one_way_feat)

    # 特徵組 6: 收款後轉出時間
    in_to_out_feat = calculate_in_to_out_time_features(acct_df)
    feat.update(in_to_out_feat)

    # 特徵組 7: 資金流向不對稱
    flow_feat = calculate_flow_asymmetry_features(acct_df)
    feat.update(flow_feat)

    # 特徵組 8: 分散度改進
    dispersion_feat = calculate_dispersion_features(acct_df)
    feat.update(dispersion_feat)

    # 特徵組 9: 小額匯入改進
    small_in_feat = calculate_small_in_features(acct_df)
    feat.update(small_in_feat)

    # ========== V14 新特徵：異常交易模式 ==========
    # 特徵 1: 同時多筆交易
    concurrent_feat = calculate_concurrent_txns_features(acct_df)
    feat.update(concurrent_feat)

    # 特徵 2: 異常時間交易
    abnormal_hour_feat = calculate_abnormal_hour_features(acct_df)
    feat.update(abnormal_hour_feat)

    # 特徵 3: 固定金額重複
    repeated_amt_feat = calculate_repeated_amount_features(acct_df)
    feat.update(repeated_amt_feat)

    # 特徵 4: 連續交易天數
    consecutive_feat = calculate_consecutive_days_features(acct_df)
    feat.update(consecutive_feat)

    # 特徵 5: 轉出集中度
    out_concentration_feat = calculate_out_concentration_features(acct_df)
    feat.update(out_concentration_feat)

    # 特徵 6: 單日高頻交易
    high_freq_feat = calculate_high_freq_day_features(acct_df)
    feat.update(high_freq_feat)

    # 特徵 7: 完全單向交易
    single_dir_feat = calculate_single_direction_features(acct_df)
    feat.update(single_dir_feat)

    # 特徵 8: 交易時間方差
    time_var_feat = calculate_time_variance_features(acct_df)
    feat.update(time_var_feat)


    features_dict[acct] = feat

# ============================================================================
# 轉換為 DataFrame
# ============================================================================
print("\n[5/8] 轉換為 DataFrame...")
features_df = pd.DataFrame.from_dict(features_dict, orient='index').reset_index(drop=True)

print(f"特徵數量: {len(features_df.columns) - 2} 個（不含 acct, label）")
print(f"帳戶數量: {len(features_df)}")
print(f"  - label=1: {(features_df['label']==1).sum()} 個")
print(f"  - label=0: {(features_df['label']==0).sum()} 個")

# ============================================================================
# 過濾相似特徵
# ============================================================================
print("\n[6/8] 過濾相似特徵...")

# 計算特徵相關矩陣
feature_cols = [col for col in features_df.columns if col not in ['acct', 'label']]
corr_matrix = features_df[feature_cols].corr().abs()

# 找出高度相關的特徵對（相關係數 > 0.95）
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = set()
for column in upper_triangle.columns:
    correlated_features = upper_triangle[column][upper_triangle[column] > 0.95].index.tolist()
    if correlated_features:
        # 保留第一個，移除其他相似特徵
        to_drop.update(correlated_features)

print(f"  發現 {len(to_drop)} 個高度相關特徵（相關係數 > 0.95）")
if len(to_drop) > 0:
    print(f"  移除特徵: {', '.join(list(to_drop)[:10])}{'...' if len(to_drop) > 10 else ''}")
    features_df = features_df.drop(columns=list(to_drop))
    print(f"  過濾後特徵數量: {len(features_df.columns) - 2} 個")

# ============================================================================
# 儲存特徵
# ============================================================================
print("\n[7/8] 儲存特徵檔案...")
features_df.to_csv('features.csv', index=False)
print("[OK] 已儲存: features.csv")

# ============================================================================
# 統計報告
# ============================================================================
print("\n[8/8] 產生統計報告...")
print("=" * 80)
print("特徵工程完成")
print("=" * 80)
print(f"\n最終特徵數量: {len(features_df.columns) - 2} 個")
print(f"帳戶數量: {len(features_df)}")
print(f"  - label=1: {(features_df['label']==1).sum()} 個（{(features_df['label']==1).sum()/len(features_df)*100:.2f}%）")
print(f"  - label=0: {(features_df['label']==0).sum()} 個（{(features_df['label']==0).sum()/len(features_df)*100:.2f}%）")
