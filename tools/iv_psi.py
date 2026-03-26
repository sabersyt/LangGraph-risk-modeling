"""
本地 IV / PSI / KS 计算工具
当外部 Metric API 不可用时作为 fallback。

IV  = sum[(P(event_i) - P(non_event_i)) * WOE_i]
WOE = ln(P(event_i) / P(non_event_i))
PSI = sum[(actual_i% - expected_i%) * ln(actual_i% / expected_i%)]
KS  = max|F(event) - F(non_event)|
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


# ─── 分箱 ─────────────────────────────────────────────────────────────────────

def optimal_bins(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
    min_bin_pct: float = 0.05,
) -> pd.Series:
    """
    基于等频分箱 + 合并小箱的简化最优分箱。
    返回分箱后的 pd.Series（int 类型 bin 标签）。
    """
    # 过滤 NaN
    mask = x.notna() & y.notna()
    x_valid, y_valid = x[mask], y[mask]

    if x_valid.empty:
        return pd.Series(np.nan, index=x.index)

    # 等频分箱
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            binned = pd.qcut(x_valid, q=n_bins, duplicates="drop", labels=False)
    except Exception:
        try:
            binned = pd.cut(x_valid, bins=n_bins, labels=False)
        except Exception:
            binned = pd.Series(0, index=x_valid.index)

    result = pd.Series(np.nan, index=x.index)
    result[mask] = binned
    return result


# ─── WOE / IV ─────────────────────────────────────────────────────────────────

def compute_woe_iv(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    计算单变量 IV 值及每箱的 WOE 详情。

    Returns:
        iv (float): 信息价值
        bins (List[dict]): 每箱统计：[{bin, count, event_rate, woe, iv_i}]
    """
    binned = optimal_bins(x, y, n_bins)

    df = pd.DataFrame({"bin": binned, "y": y})
    total_event = max(y.sum(), EPS)
    total_non_event = max((1 - y).sum(), EPS)

    bins_info = []
    total_iv = 0.0

    for bin_val in sorted(df["bin"].dropna().unique()):
        mask = df["bin"] == bin_val
        cnt = mask.sum()
        ev = df.loc[mask, "y"].sum()
        non_ev = cnt - ev

        p_ev = max(ev, EPS) / total_event
        p_non = max(non_ev, EPS) / total_non_event
        woe = np.log(p_ev / p_non)
        iv_i = (p_ev - p_non) * woe
        total_iv += iv_i

        bins_info.append({
            "bin": int(bin_val),
            "count": int(cnt),
            "event": int(ev),
            "non_event": int(non_ev),
            "event_rate": round(ev / max(cnt, 1), 4),
            "woe": round(woe, 4),
            "iv": round(iv_i, 4),
        })

    # NULL 箱
    null_mask = df["bin"].isna()
    if null_mask.sum() > 0:
        cnt = null_mask.sum()
        ev = df.loc[null_mask, "y"].sum()
        bins_info.append({
            "bin": "NULL",
            "count": int(cnt),
            "event": int(ev),
            "non_event": int(cnt - ev),
            "event_rate": round(ev / max(cnt, 1), 4),
            "woe": None,
            "iv": None,
        })

    return round(total_iv, 4), bins_info


# ─── PSI ──────────────────────────────────────────────────────────────────────

def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    计算 PSI（Population Stability Index）。
    expected: 基期（训练期）数据
    actual:   对比期（验证期/近期）数据
    """
    # 共同使用 expected 的分位数作为分箱边界
    try:
        quantiles = expected.dropna().quantile(np.linspace(0, 1, n_bins + 1)).unique()
        if len(quantiles) < 2:
            return 0.0

        def get_bin_dist(series: pd.Series) -> np.ndarray:
            binned = pd.cut(series.dropna(), bins=quantiles, include_lowest=True, labels=False)
            counts = binned.value_counts(sort=False).sort_index()
            dist = counts.values / max(counts.sum(), 1)
            return dist

        exp_dist = get_bin_dist(expected)
        act_dist = get_bin_dist(actual)

        # 对齐长度
        n = min(len(exp_dist), len(act_dist))
        exp_dist = exp_dist[:n]
        act_dist = act_dist[:n]

        # 防零
        exp_dist = np.where(exp_dist == 0, EPS, exp_dist)
        act_dist = np.where(act_dist == 0, EPS, act_dist)

        psi = np.sum((act_dist - exp_dist) * np.log(act_dist / exp_dist))
        return round(float(psi), 4)
    except Exception:
        return 0.0


# ─── KS ───────────────────────────────────────────────────────────────────────

def compute_ks(y_true: pd.Series, y_score: pd.Series) -> float:
    """计算 KS 统计量"""
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return round(float(np.max(tpr - fpr)), 4)
    except Exception:
        return 0.0


# ─── 批量评估 ────────────────────────────────────────────────────────────────

def evaluate_features(
    df: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    time_col: Optional[str] = None,
    quality_thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    对 df 中的多个特征批量计算 IV/PSI/缺失率并判断可用性。

    Args:
        df: 含特征列和目标 Y 的 DataFrame
        feature_names: 要评估的特征列表
        target_col: 目标 Y 列名
        time_col: 时间列（用于 PSI 计算，按时间二分）
        quality_thresholds: 质量阈值配置

    Returns:
        每个特征的评估结果列表
    """
    qt = quality_thresholds or {}
    avail = qt.get("availability", {})
    exc_cfg = avail.get("excellent", {"iv_min": 0.10, "missing_rate_max": 0.20, "psi_max": 0.10})
    use_cfg = avail.get("usable", {"iv_min": 0.02, "missing_rate_max": 0.90, "psi_max": 0.25})

    y = df[target_col].map(lambda x: 1 if str(x).strip() in ("1", "是", "Y", "y", "True", "true", "yes") else 0)
    results = []

    # 时间分期（用于 PSI）
    if time_col and time_col in df.columns:
        df_sorted = df.sort_values(time_col)
        split_idx = int(len(df_sorted) * 0.6)
        expected_idx = df_sorted.index[:split_idx]
        actual_idx = df_sorted.index[split_idx:]
    else:
        split_idx = int(len(df) * 0.6)
        expected_idx = df.index[:split_idx]
        actual_idx = df.index[split_idx:]

    for feat in feature_names:
        if feat not in df.columns:
            continue

        col = df[feat]
        missing_rate = round(col.isna().mean(), 4)

        # IV + WOE 分箱
        try:
            iv, binning = compute_woe_iv(col, y, n_bins=10)
        except Exception:
            iv, binning = 0.0, []

        # PSI（时间分期）
        try:
            psi = compute_psi(col[expected_idx], col[actual_idx], n_bins=10)
        except Exception:
            psi = 0.0

        # 月度指标（按月计算缺失率/均值）
        monthly_metrics: Dict[str, Any] = {}
        if time_col and time_col in df.columns:
            try:
                df_tmp = df[[time_col, feat]].copy()
                df_tmp["ym"] = pd.to_datetime(df_tmp[time_col]).dt.to_period("M").astype(str)
                for ym, grp in df_tmp.groupby("ym"):
                    monthly_metrics[str(ym)] = {
                        "missing_rate": round(grp[feat].isna().mean(), 4),
                        "mean": round(grp[feat].mean(), 4) if grp[feat].notna().any() else None,
                    }
            except Exception:
                pass

        # 可用性判断
        if (iv >= exc_cfg["iv_min"]
                and missing_rate <= exc_cfg["missing_rate_max"]
                and psi <= exc_cfg["psi_max"]):
            availability = "优质"
        elif (iv >= use_cfg["iv_min"]
              and missing_rate <= use_cfg["missing_rate_max"]
              and psi <= use_cfg["psi_max"]):
            availability = "可用"
        else:
            availability = "废弃"

        results.append({
            "name": feat,
            "iv": iv,
            "psi": psi,
            "missing_rate": missing_rate,
            "availability": availability,
            "binning": binning,
            "monthly_metrics": monthly_metrics,
        })

    return results
