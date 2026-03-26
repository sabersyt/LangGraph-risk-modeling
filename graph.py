"""
LangGraph 主图定义
=============================
节点流程：

  init_session
      ↓
  schema_explore
      ↓
  feature_generate
      ↓
  sql_execute ─── retry ───┐
      ↓                    │ (retry_count < 3)
  feature_evaluate         │
      ↓                    │
  model_train   ←──────────┘
      ↓
  report_generate
      ↓
    END

条件边：
  - sql_execute → retry / evaluate / stop（根据 current_step）
  - feature_evaluate → model / stop（根据可用率）
  - 任何节点 should_stop=True → END
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.nodes.schema_node import schema_explore_node
from langgraph_pipeline.nodes.feature_node import feature_generate_node
from langgraph_pipeline.nodes.executor_node import sql_execute_node
from langgraph_pipeline.nodes.evaluator_node import feature_evaluate_node
from langgraph_pipeline.nodes.modeler_node import model_train_node
from langgraph_pipeline.nodes.reporter_node import report_generate_node

logger = logging.getLogger(__name__)


# ─── 初始化节点（轻量，仅处理会话目录创建） ────────────────────────────────

def init_session_node(state: PipelineState) -> dict:
    """创建 session 目录，写入初始 context.json"""
    import json
    from pathlib import Path
    from datetime import datetime

    from langgraph_pipeline.config import get_sessions_dir, get_quality_thresholds
    from langchain_core.messages import AIMessage

    session_id = state.get("session_id") or f"fe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = get_sessions_dir() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    quality_thresholds = get_quality_thresholds()

    # 写 context.json（兼容现有 generate_report.py）
    ctx = {
        "session_id": session_id,
        "module": "feature_engineering",
        "scene": state.get("scene", "信贷风控"),
        "source_table": state["source_table"],
        "target_y_column": state.get("target_y_column", ""),
        "derivation_mode": state.get("derivation_mode"),
        "quality_thresholds": quality_thresholds,
        "features": {},
    }
    (session_dir / "context.json").write_text(
        json.dumps(ctx, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"[init] 会话初始化: {session_id}，目录: {session_dir}")

    return {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "quality_thresholds": quality_thresholds,
        "current_step": "initialized",
        "retry_count": 0,
        "errors": [],
        "should_stop": False,
        "messages": [AIMessage(
            content=f"会话初始化完成: {session_id}\n宽表: {state['source_table']}"
        )],
    }


# ─── 条件路由函数 ─────────────────────────────────────────────────────────────

def route_after_sql(state: PipelineState) -> Literal["feature_evaluate", "sql_execute", "__end__"]:
    """SQL 执行后的路由"""
    if state.get("should_stop"):
        return "__end__"
    step = state.get("current_step", "")
    if step == "sql_retry":
        return "sql_execute"  # 重试
    if step in ("sql_done", "sql_done_empty"):
        return "feature_evaluate"
    return "__end__"


def route_after_eval(state: PipelineState) -> Literal["model_train", "report_generate", "__end__"]:
    """特征评估后的路由"""
    if state.get("should_stop"):
        return "__end__"

    eval_results = state.get("eval_results", {})
    summary = eval_results.get("summary", {})
    usable_rate = summary.get("usable_rate", 0)

    # 可用率极低 → 仍生成报告，但跳过建模
    if usable_rate == 0 or (summary.get("excellent_cnt", 0) + summary.get("usable_cnt", 0)) < 2:
        logger.warning(f"[route] 可用特征不足（可用率 {usable_rate:.1%}），跳过建模直接出报告")
        return "report_generate"

    return "model_train"


def route_after_model(state: PipelineState) -> Literal["report_generate", "__end__"]:
    if state.get("should_stop"):
        return "__end__"
    return "report_generate"


def route_global(state: PipelineState) -> Literal["__end__", "continue"]:
    """全局 should_stop 检查（用于各节点之后）"""
    return "__end__" if state.get("should_stop") else "continue"


# ─── 图组装 ──────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """构建并编译 LangGraph Pipeline"""
    graph = StateGraph(PipelineState)

    # 注册节点
    graph.add_node("init_session", init_session_node)
    graph.add_node("schema_explore", schema_explore_node)
    graph.add_node("feature_generate", feature_generate_node)
    graph.add_node("sql_execute", sql_execute_node)
    graph.add_node("feature_evaluate", feature_evaluate_node)
    graph.add_node("model_train", model_train_node)
    graph.add_node("report_generate", report_generate_node)

    # 起始边
    graph.add_edge(START, "init_session")

    # 串行前段（无条件）
    graph.add_edge("init_session", "schema_explore")
    graph.add_edge("schema_explore", "feature_generate")
    graph.add_edge("feature_generate", "sql_execute")

    # SQL 执行后：条件路由（重试 / 评估 / 终止）
    graph.add_conditional_edges(
        "sql_execute",
        route_after_sql,
        {
            "sql_execute": "sql_execute",
            "feature_evaluate": "feature_evaluate",
            "__end__": END,
        },
    )

    # 评估后：条件路由（建模 / 直接报告 / 终止）
    graph.add_conditional_edges(
        "feature_evaluate",
        route_after_eval,
        {
            "model_train": "model_train",
            "report_generate": "report_generate",
            "__end__": END,
        },
    )

    # 建模后：条件路由（报告 / 终止）
    graph.add_conditional_edges(
        "model_train",
        route_after_model,
        {
            "report_generate": "report_generate",
            "__end__": END,
        },
    )

    # 报告完成 → 结束
    graph.add_edge("report_generate", END)

    return graph.compile()


# ─── 公开 app 实例（可直接 import） ──────────────────────────────────────────

app = build_graph()
