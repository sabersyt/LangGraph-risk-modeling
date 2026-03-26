"""
节点 4：特征评估
- 从衍生特征表加载数据
- 本地计算 IV / PSI / 缺失率 / 分箱
- 判断可用性（优质/可用/废弃）
- 优先调用外部 Metric API，失败则本地计算
- 写入 eval_results.json
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from langchain_core.messages import AIMessage

from langgraph_pipeline.config import get_metric_api_url, get_quality_thresholds
from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.tools.doris import load_feature_matrix, query
from langgraph_pipeline.tools.iv_psi import evaluate_features

logger = logging.getLogger(__name__)

METRIC_API_TIMEOUT = 300  # 5 分钟超时
POLL_INTERVAL = 10


# ─── 外部 Metric API ──────────────────────────────────────────────────────────

def _try_metric_api(
    derived_table: str,
    feature_names: List[str],
    target_y: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """
    尝试调用外部 Metric Compute API。
    返回 None 表示 API 不可用或超时。
    """
    api_url = get_metric_api_url()
    try:
        # 提交任务
        payload = {
            "tableName": derived_table,
            "featureNms": feature_names,
            "targetY": target_y,
            "pkColumn": "pid",
            "partitionColumn": "lend_tm",
            "ivMethod": "optimal",
            "monotonic": False,
            "ifIncludeNull": False,
            "enableMonthlyMetrics": True,
            "jobId": f"{session_id}_eval",
        }
        resp = requests.post(f"{api_url}/submit", json=payload, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[evaluator] Metric API submit 失败: {resp.status_code}")
            return None

        job_id = resp.json().get("job_id") or resp.json().get("jobId")
        if not job_id:
            return None

        # 轮询等待
        deadline = time.time() + METRIC_API_TIMEOUT
        while time.time() < deadline:
            status_resp = requests.get(f"{api_url}/status/{job_id}", timeout=10)
            status = status_resp.json().get("status", "")
            if status in ("done", "success", "completed"):
                break
            if status in ("failed", "error"):
                logger.warning(f"[evaluator] Metric API 任务失败: {status_resp.json()}")
                return None
            time.sleep(POLL_INTERVAL)
        else:
            logger.warning("[evaluator] Metric API 超时")
            return None

        # 获取结果
        results_resp = requests.get(f"{api_url}/results/{job_id}", timeout=15)
        return results_resp.json()

    except requests.exceptions.ConnectionError:
        logger.info("[evaluator] Metric API 不可达，使用本地计算")
        return None
    except Exception as e:
        logger.warning(f"[evaluator] Metric API 异常: {e}")
        return None


# ─── 主节点 ──────────────────────────────────────────────────────────────────

def feature_evaluate_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph 节点：特征质量评估

    输入: state.derived_table, state.feature_design, state.target_y_column
    输出: state.eval_results
    """
    derived_table = state.get("derived_table", "")
    feature_design = state.get("feature_design", [])
    target_y = state.get("target_y_column", "")
    session_dir = Path(state["session_dir"])
    session_id = state["session_id"]

    if not derived_table or not feature_design:
        return {
            "errors": state.get("errors", []) + ["evaluator: 衍生表或特征设计为空"],
            "should_stop": True,
            "messages": [AIMessage(content="特征评估失败：衍生表或特征设计为空")],
        }

    feature_names = [f["name"] for f in feature_design]
    quality_thresholds = state.get("quality_thresholds") or get_quality_thresholds()

    # ── 1. 尝试外部 Metric API ────────────────────────────────────
    api_result = _try_metric_api(derived_table, feature_names, target_y, session_id)

    if api_result:
        logger.info("[evaluator] 使用外部 Metric API 结果")
        features_eval = api_result.get("features", [])
        # 标准化可用性字段（API 可能返回英文）
        avail_map = {
            "excellent": "优质", "usable": "可用", "discard": "废弃",
            "优质": "优质", "可用": "可用", "废弃": "废弃",
        }
        for feat in features_eval:
            feat["availability"] = avail_map.get(feat.get("availability", ""), "废弃")
        source_fields = api_result.get("source_fields", {})
    else:
        # ── 2. 本地计算 ───────────────────────────────────────────
        logger.info("[evaluator] 本地计算 IV/PSI...")

        # 加载数据：衍生特征 + 来源字段
        all_cols = list(set(
            feature_names
            + [s for f in feature_design for s in f.get("sources", [])]
        ))
        try:
            df = load_feature_matrix(
                derived_table,
                all_cols,
                target_col=target_y,
                time_col="lend_tm",
                limit=50_000,
            )
        except Exception as e:
            logger.error(f"[evaluator] 加载数据失败: {e}")
            return {
                "errors": state.get("errors", []) + [f"evaluator load_data: {e}"],
                "should_stop": True,
                "messages": [AIMessage(content=f"特征评估失败（数据加载）: {e}")],
            }

        if df.empty or target_y not in df.columns:
            return {
                "errors": state.get("errors", []) + ["evaluator: 数据为空或缺少目标 Y"],
                "should_stop": True,
                "messages": [AIMessage(content="特征评估失败：数据为空或缺少目标列")],
            }

        # 评估衍生特征
        features_eval = evaluate_features(
            df,
            feature_names=[f for f in feature_names if f in df.columns],
            target_col=target_y,
            time_col="lend_tm" if "lend_tm" in df.columns else None,
            quality_thresholds=quality_thresholds,
        )

        # 评估来源字段（用于 IV 增益对比）
        source_col_names = list(set(
            s for f in feature_design
            for s in f.get("sources", [])
            if s in df.columns
        ))
        source_eval = evaluate_features(
            df,
            feature_names=source_col_names,
            target_col=target_y,
            time_col="lend_tm" if "lend_tm" in df.columns else None,
            quality_thresholds=quality_thresholds,
        )
        source_fields = {e["name"]: {"iv": e["iv"], "psi": e["psi"], "missing_rate": e["missing_rate"]}
                        for e in source_eval}

    # ── 3. 汇总统计 ───────────────────────────────────────────────
    avail_counts = {"优质": 0, "可用": 0, "废弃": 0}
    for feat in features_eval:
        av = feat.get("availability", "废弃")
        avail_counts[av] = avail_counts.get(av, 0) + 1

    summary = {
        "total": len(features_eval),
        "excellent_cnt": avail_counts["优质"],
        "usable_cnt": avail_counts["可用"],
        "discard_cnt": avail_counts["废弃"],
        "usable_rate": round((avail_counts["优质"] + avail_counts["可用"]) / max(len(features_eval), 1), 3),
    }

    eval_results = {
        "job_id": f"{session_id}_local",
        "features": features_eval,
        "source_fields": source_fields,
        "summary": summary,
    }

    # ── 4. 写入文件 ───────────────────────────────────────────────
    out_path = session_dir / "eval_results.json"
    out_path.write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 来源字段趋势
    trends_path = session_dir / "source_field_trends.json"
    trends_path.write_text(
        json.dumps(source_fields, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(f"[evaluator] 评估完成: {summary}")

    # ── 5. 构建摘要消息 ───────────────────────────────────────────
    top_feats = sorted(features_eval, key=lambda x: x.get("iv", 0), reverse=True)[:5]
    feat_lines = "\n".join(
        f"  - {f['name']}: IV={f['iv']:.4f}, PSI={f['psi']:.4f}, 缺失率={f['missing_rate']:.2%} [{f['availability']}]"
        for f in top_feats
    )
    msg = (
        f"特征评估完成（共 {summary['total']} 个）\n"
        f"优质: {summary['excellent_cnt']} | 可用: {summary['usable_cnt']} | 废弃: {summary['discard_cnt']}\n"
        f"可用率: {summary['usable_rate']:.1%}\n"
        f"\nTop 5 特征（按 IV）:\n{feat_lines}\n"
        f"详情: {out_path}"
    )

    return {
        "eval_results": eval_results,
        "current_step": "eval_done",
        "messages": [AIMessage(content=msg)],
    }
