"""
节点 1：Schema 探索
- 查询 Doris information_schema 获取宽表字段信息
- 估算数值/类别/时间字段分类
- 写入 schema_analysis.json
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage

from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.tools.doris import get_table_schema, get_table_row_count, sample_column_values, query

logger = logging.getLogger(__name__)

# 常见数值类型
NUMERIC_TYPES = {"int", "tinyint", "smallint", "bigint", "float", "double", "decimal", "numeric", "real"}
# 常见文本类型
TEXT_TYPES = {"varchar", "char", "text", "string"}
# 时间类型
TIME_TYPES = {"date", "datetime", "timestamp"}

# 排除的系统列
EXCLUDE_COLS = {"pid", "lend_tm", "t_5", "t_10", "m2_mob4", "created_time", "id", "update_time"}


def _classify_column(col: Dict[str, Any]) -> str:
    """根据 data_type 判断字段大类"""
    dt = str(col.get("data_type", "")).lower()
    if dt in NUMERIC_TYPES:
        return "numeric"
    if dt in TEXT_TYPES:
        return "categorical"
    if dt in TIME_TYPES:
        return "time"
    return "other"


def _estimate_missing_rate_fast(table: str, col_name: str, total: int) -> float:
    """快速估算缺失率（采样 5000 条）"""
    try:
        sql = f"""
            SELECT SUM(CASE WHEN `{col_name}` IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS mr
            FROM (
                SELECT `{col_name}` FROM {table}
                ORDER BY lend_tm DESC
                LIMIT 5000
            ) t
        """
        df = query(sql, max_rows=1)
        if not df.empty and "mr" in df.columns:
            return round(float(df["mr"].iloc[0] or 0), 4)
    except Exception:
        pass
    return 0.0


def schema_explore_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph 节点：探索宽表 Schema

    输入: state.source_table
    输出: state.schema_analysis, state.messages
    """
    table = state["source_table"]
    session_dir = Path(state["session_dir"])

    logger.info(f"[schema_node] 探索表: {table}")

    # ── 1. 获取字段列表 ───────────────────────────────────────────
    try:
        columns = get_table_schema(table)
    except Exception as e:
        logger.error(f"[schema_node] 获取 Schema 失败: {e}")
        return {
            "errors": state.get("errors", []) + [f"schema_explore: {e}"],
            "should_stop": True,
            "current_step": "schema_failed",
            "messages": [AIMessage(content=f"Schema 探索失败: {e}")],
        }

    if not columns:
        return {
            "errors": state.get("errors", []) + ["schema_explore: 表字段为空"],
            "should_stop": True,
            "current_step": "schema_failed",
            "messages": [AIMessage(content=f"表 {table} 无字段，请检查表名是否正确")],
        }

    # ── 2. 获取总行数 ─────────────────────────────────────────────
    try:
        total_rows = get_table_row_count(table)
    except Exception:
        total_rows = -1

    # ── 3. 字段分类 & 缺失率估算 ──────────────────────────────────
    schema_detail: List[Dict[str, Any]] = []
    numeric_fields: List[str] = []
    categorical_fields: List[str] = []
    time_fields: List[str] = []
    target_candidates: List[str] = []  # 可能是 Y 列的候选

    for col in columns:
        name = col["column_name"]
        if name in EXCLUDE_COLS:
            continue

        col_type = _classify_column(col)
        comment = col.get("column_comment", "") or ""

        # 估算缺失率
        missing_rate = _estimate_missing_rate_fast(table, name, total_rows)

        # 判断是否是目标 Y 列
        is_target = bool(re.search(r"\bt[0-9]|m[0-9]|mob|逾期|违约|default", name.lower()))

        entry = {
            "name": name,
            "data_type": col.get("data_type", ""),
            "column_type": col.get("column_type", ""),
            "comment": comment,
            "col_type": col_type,
            "missing_rate": missing_rate,
            "is_target_candidate": is_target,
        }

        # 类别型字段额外采样
        if col_type == "categorical":
            sample_vals = sample_column_values(table, name, limit=30)
            entry["sample_values"] = sample_vals
            entry["cardinality_estimate"] = len(sample_vals)
            if missing_rate < 0.90:
                categorical_fields.append(name)
        elif col_type == "numeric":
            if missing_rate < 0.90:
                numeric_fields.append(name)
        elif col_type == "time":
            time_fields.append(name)

        if is_target:
            target_candidates.append(name)

        schema_detail.append(entry)

    # ── 4. 确定目标 Y（优先使用 state 中已指定的） ────────────────
    target_y = state.get("target_y_column", "")
    if not target_y and target_candidates:
        target_y = target_candidates[0]

    # ── 5. 组装结果 ───────────────────────────────────────────────
    schema_analysis = {
        "table": table,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "schema_columns": schema_detail,
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "time_fields": time_fields,
        "target_candidates": target_candidates,
        "summary": (
            f"表 {table} 共 {len(columns)} 列，{total_rows} 行。"
            f"数值字段 {len(numeric_fields)} 个，类别字段 {len(categorical_fields)} 个，"
            f"时间字段 {len(time_fields)} 个。"
            f"目标候选列: {target_candidates[:5]}"
        ),
    }

    # ── 6. 写入文件 ───────────────────────────────────────────────
    out_path = session_dir / "schema_analysis.json"
    out_path.write_text(
        json.dumps(schema_analysis, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"[schema_node] Schema 分析写入: {out_path}")

    summary_msg = (
        f"Schema 探索完成。\n"
        f"- 表: {table}，约 {total_rows} 行\n"
        f"- 数值特征: {len(numeric_fields)} 个\n"
        f"- 类别特征: {len(categorical_fields)} 个\n"
        f"- 目标 Y 候选: {target_candidates[:3]}\n"
        f"- 详情已写入: {out_path}"
    )

    return {
        "schema_analysis": schema_analysis,
        "target_y_column": target_y,
        "current_step": "schema_done",
        "messages": [AIMessage(content=summary_msg)],
    }
