"""
Pipeline State Definition — LangGraph 风控特征工程流水线状态
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PipelineState(TypedDict):
    """
    整个 Pipeline 的共享状态。每个节点接收此状态并返回局部更新。

    LangGraph 会合并返回值：普通字段覆盖，messages 字段追加。
    """

    # ── 会话标识 ──────────────────────────────────────────────────
    session_id: str                         # fe_YYYYMMDD_HHMMSS
    session_dir: str                        # sessions/{session_id}/
    source_table: str                       # schema.table 全名
    target_y_column: str                    # 目标变量列名
    derivation_mode: Optional[str]          # ratio/diff/log_transform/... 或 null=自动
    scene: str                              # 业务场景描述

    # ── 流程控制 ──────────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str                       # schema/feature/sql/evaluate/model/report
    retry_count: int                        # 当前节点重试次数
    errors: List[str]                       # 累计错误列表
    should_stop: bool                       # True → 终止流程

    # ── Schema 探索结果 ──────────────────────────────────────────
    schema_analysis: Dict[str, Any]
    # {
    #   "columns": [{name, type, comment, missing_rate_estimate, is_numeric, is_categorical}],
    #   "numeric_fields": [str],
    #   "categorical_fields": [str],
    #   "time_fields": [str],
    #   "total_rows": int,
    #   "summary": str
    # }

    # ── 特征设计结果 ─────────────────────────────────────────────
    feature_design: List[Dict[str, Any]]
    # [{
    #   "name": str,
    #   "formula": str,
    #   "sources": [str],
    #   "template": str,
    #   "description": str,
    #   "expected_iv": float
    # }]

    ddl_sql: str                            # 完整建表 SQL
    ddl_path: str                           # DDL 文件路径
    derived_table: str                      # 已创建的衍生特征表名
    derived_row_count: int                  # 衍生表行数

    # ── 特征评估结果 ─────────────────────────────────────────────
    eval_results: Dict[str, Any]
    # {
    #   "features": [{name, iv, psi, missing_rate, availability, binning, monthly_metrics}],
    #   "source_fields": {name: {iv, psi, missing_rate}},
    #   "summary": {excellent_cnt, usable_cnt, discard_cnt}
    # }

    quality_thresholds: Dict[str, Any]     # 来自 feature_templates.yaml

    # ── ML 建模结果 ──────────────────────────────────────────────
    model_results: Dict[str, Any]
    # {
    #   "best_model": str,
    #   "models": {
    #     "lgbm": {auc, ks, gini, feature_importance: {name: score}},
    #     "lr":   {auc, ks, gini},
    #   },
    #   "train_samples": int,
    #   "test_samples": int,
    #   "model_path": str,
    #   "feature_names": [str]
    # }

    # ── 输出 ─────────────────────────────────────────────────────
    report_path: str
