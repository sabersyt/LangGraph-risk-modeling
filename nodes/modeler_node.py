"""
节点 5：ML 建模
- 从衍生特征表加载高质量特征（优质 + 可用）
- 时间序列分割训练/验证/测试集
- 训练多个模型：LightGBM（主）、Logistic Regression（基线）
- 评估 AUC、KS、Gini、特征重要性
- 保存模型文件
- 写入 model_results 到 state
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from langchain_core.messages import AIMessage

from langgraph_pipeline.config import get_models_dir
from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.tools.doris import load_feature_matrix

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """计算 AUC、KS、Gini"""
    from sklearn.metrics import roc_auc_score, roc_curve
    try:
        auc = round(roc_auc_score(y_true, y_score), 4)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ks = round(float(np.max(np.abs(tpr - fpr))), 4)
        gini = round(2 * auc - 1, 4)
        return {"auc": auc, "ks": ks, "gini": gini}
    except Exception as e:
        logger.warning(f"指标计算失败: {e}")
        return {"auc": 0.0, "ks": 0.0, "gini": 0.0}


# ─── 数据预处理 ──────────────────────────────────────────────────────────────

def _preprocess(df: pd.DataFrame, feature_names: List[str], target_col: str) -> Tuple[
    pd.DataFrame, pd.Series
]:
    """
    1. 过滤目标列为 NULL 的行
    2. 用中位数填充特征列的 NaN（保留风险信号，与本地评估不同——这里是为了模型训练）
    3. clip 极端值（±5σ）
    """
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].map(lambda x: 1 if str(x).strip() in ("1", "是", "Y", "y", "True", "true", "yes") else 0)

    X = df[feature_names].copy()
    for col in X.columns:
        if X[col].dtype in [np.float64, np.float32, float]:
            median = X[col].median()
            X[col] = X[col].fillna(median)
            # clip 5-sigma
            mu, sigma = X[col].mean(), X[col].std()
            if sigma > 0:
                X[col] = X[col].clip(mu - 5 * sigma, mu + 5 * sigma)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    return X, y


def _time_split(
    df: pd.DataFrame,
    time_col: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """按时间顺序分割 train/val/test"""
    df_sorted = df.sort_values(time_col)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df_sorted.index[:train_end],
        df_sorted.index[train_end:val_end],
        df_sorted.index[val_end:],
    )


# ─── 训练 LightGBM ────────────────────────────────────────────────────────────

def _train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, Dict[str, float], Dict[str, float]]:
    """返回: (model, val_metrics, feature_importance)"""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "scale_pos_weight": max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1),
        "n_estimators": 500,
        "verbose": -1,
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    metrics = _compute_metrics(y_val.values, y_pred)

    importance = dict(zip(
        X_train.columns.tolist(),
        model.feature_importances_.tolist(),
    ))
    # 归一化
    total = max(sum(importance.values()), 1)
    importance = {k: round(v / total, 4) for k, v in
                  sorted(importance.items(), key=lambda x: -x[1])[:30]}

    return model, metrics, importance


# ─── 训练 Logistic Regression ─────────────────────────────────────────────────

def _train_lr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    metrics = _compute_metrics(y_val.values, y_pred)
    return model, metrics


# ─── 主节点 ──────────────────────────────────────────────────────────────────

def model_train_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph 节点：ML 建模

    输入: state.eval_results, state.derived_table, state.target_y_column
    输出: state.model_results
    """
    eval_results = state.get("eval_results", {})
    derived_table = state.get("derived_table", "")
    target_y = state.get("target_y_column", "")
    session_id = state["session_id"]
    session_dir = Path(state["session_dir"])

    if not eval_results or not derived_table:
        return {
            "errors": state.get("errors", []) + ["modeler: eval_results 或 derived_table 为空"],
            "current_step": "model_skipped",
            "messages": [AIMessage(content="建模跳过：缺少评估结果或衍生表")],
        }

    # ── 1. 筛选可用特征 ───────────────────────────────────────────
    usable_features = [
        f["name"] for f in eval_results.get("features", [])
        if f.get("availability") in ("优质", "可用")
    ]

    if len(usable_features) < 2:
        msg = f"可用特征不足（{len(usable_features)} 个），跳过建模"
        logger.warning(f"[modeler] {msg}")
        return {
            "current_step": "model_skipped",
            "messages": [AIMessage(content=msg)],
        }

    # 优先使用优质特征，最多取 50 个
    excellent = [
        f["name"] for f in eval_results.get("features", [])
        if f.get("availability") == "优质"
    ]
    feature_names = (excellent + [f for f in usable_features if f not in excellent])[:50]

    logger.info(f"[modeler] 建模特征数: {len(feature_names)}（优质: {len(excellent)}）")

    # ── 2. 加载数据 ───────────────────────────────────────────────
    try:
        df = load_feature_matrix(
            derived_table,
            feature_names,
            target_col=target_y,
            time_col="lend_tm",
            limit=100_000,
        )
    except Exception as e:
        logger.error(f"[modeler] 数据加载失败: {e}")
        return {
            "errors": state.get("errors", []) + [f"modeler load: {e}"],
            "current_step": "model_failed",
            "messages": [AIMessage(content=f"建模数据加载失败: {e}")],
        }

    if df.empty or target_y not in df.columns:
        return {
            "current_step": "model_skipped",
            "messages": [AIMessage(content="建模跳过：数据为空或缺少目标列")],
        }

    # ── 3. 时间分割 ───────────────────────────────────────────────
    has_time = "lend_tm" in df.columns
    if has_time:
        train_idx, val_idx, test_idx = _time_split(df, "lend_tm")
    else:
        n = len(df)
        train_idx = df.index[: int(n * 0.6)]
        val_idx = df.index[int(n * 0.6): int(n * 0.8)]
        test_idx = df.index[int(n * 0.8):]

    valid_features = [f for f in feature_names if f in df.columns]
    X, y = _preprocess(df, valid_features, target_y)

    X_train, y_train = X.loc[X.index.isin(train_idx)], y.loc[y.index.isin(train_idx)]
    X_val, y_val = X.loc[X.index.isin(val_idx)], y.loc[y.index.isin(val_idx)]
    X_test, y_test = X.loc[X.index.isin(test_idx)], y.loc[y.index.isin(test_idx)]

    logger.info(
        f"[modeler] 样本分布: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}"
        f" | Y 分布: {y_train.mean():.2%} 逾期率"
    )

    if len(X_train) < 100 or y_train.nunique() < 2:
        return {
            "current_step": "model_skipped",
            "messages": [AIMessage(content="建模跳过：训练样本不足或目标变量单值")],
        }

    models_results: Dict[str, Any] = {}
    best_model_name = ""
    best_auc = 0.0
    best_model_obj = None

    # ── 4. LightGBM ───────────────────────────────────────────────
    try:
        lgbm_model, lgbm_val_metrics, lgbm_importance = _train_lgbm(
            X_train, y_train, X_val, y_val
        )
        # 测试集评估
        lgbm_test_pred = lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_test_metrics = _compute_metrics(y_test.values, lgbm_test_pred)

        models_results["lgbm"] = {
            "val_metrics": lgbm_val_metrics,
            "test_metrics": lgbm_test_metrics,
            "feature_importance": lgbm_importance,
            "n_estimators": lgbm_model.best_iteration_ if hasattr(lgbm_model, "best_iteration_") else 500,
        }
        if lgbm_test_metrics["auc"] > best_auc:
            best_auc = lgbm_test_metrics["auc"]
            best_model_name = "lgbm"
            best_model_obj = lgbm_model

        logger.info(f"[modeler] LightGBM — val AUC: {lgbm_val_metrics['auc']}, test AUC: {lgbm_test_metrics['auc']}")
    except ImportError:
        logger.warning("[modeler] lightgbm 未安装，跳过 LightGBM")
    except Exception as e:
        logger.warning(f"[modeler] LightGBM 训练失败: {e}")

    # ── 5. Logistic Regression ────────────────────────────────────
    try:
        lr_model, lr_val_metrics = _train_lr(X_train, y_train, X_val, y_val)
        lr_test_pred = lr_model.predict_proba(X_test)[:, 1]
        lr_test_metrics = _compute_metrics(y_test.values, lr_test_pred)

        models_results["lr"] = {
            "val_metrics": lr_val_metrics,
            "test_metrics": lr_test_metrics,
        }
        if lr_test_metrics["auc"] > best_auc:
            best_auc = lr_test_metrics["auc"]
            best_model_name = "lr"
            best_model_obj = lr_model

        logger.info(f"[modeler] LogReg — val AUC: {lr_val_metrics['auc']}, test AUC: {lr_test_metrics['auc']}")
    except Exception as e:
        logger.warning(f"[modeler] LogReg 训练失败: {e}")

    if not models_results:
        return {
            "current_step": "model_failed",
            "errors": state.get("errors", []) + ["modeler: 所有模型训练失败"],
            "messages": [AIMessage(content="所有模型训练均失败")],
        }

    # ── 6. 保存最优模型 ───────────────────────────────────────────
    models_dir = get_models_dir()
    model_path = str(models_dir / f"{session_id}_{best_model_name}.joblib")
    if best_model_obj is not None:
        joblib.dump(best_model_obj, model_path)
        logger.info(f"[modeler] 最优模型已保存: {model_path}")

    # 同时保存特征列表（推理时需要）
    feature_meta_path = str(models_dir / f"{session_id}_features.json")
    with open(feature_meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": valid_features, "target_y": target_y}, f, ensure_ascii=False)

    model_results = {
        "best_model": best_model_name,
        "best_test_auc": round(best_auc, 4),
        "models": models_results,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "positive_rate": round(float(y_train.mean()), 4),
        "feature_names": valid_features,
        "model_path": model_path,
    }

    # 写入 session 目录
    (session_dir / "model_results.json").write_text(
        json.dumps(model_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # ── 7. 汇总消息 ───────────────────────────────────────────────
    lines = [
        f"建模完成 | 最优模型: {best_model_name.upper()} | 测试 AUC: {best_auc:.4f}",
        f"样本: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}",
        f"逾期率: {float(y_train.mean()):.2%}",
        "",
    ]
    for model_name, res in models_results.items():
        tm = res["test_metrics"]
        lines.append(
            f"[{model_name.upper()}] AUC={tm['auc']:.4f}, KS={tm['ks']:.4f}, Gini={tm['gini']:.4f}"
        )
    if "lgbm" in models_results and "feature_importance" in models_results["lgbm"]:
        top_feats = list(models_results["lgbm"]["feature_importance"].items())[:5]
        lines.append("\nTop 5 重要特征 (LightGBM):")
        for feat, score in top_feats:
            lines.append(f"  {feat}: {score:.4f}")

    lines.append(f"\n模型文件: {model_path}")

    return {
        "model_results": model_results,
        "current_step": "model_done",
        "messages": [AIMessage(content="\n".join(lines))],
    }
