"""
节点 3：SQL 执行
- 执行 DDL，在 Doris 中创建衍生特征表
- 验证行数
- 最多重试 3 次
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage

from langgraph_pipeline.state import PipelineState
from langgraph_pipeline.tools.doris import execute, get_table_row_count

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def sql_execute_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph 节点：执行建表 DDL

    输入: state.ddl_sql, state.derived_table
    输出: state.derived_row_count, state.current_step
    """
    ddl_sql = state.get("ddl_sql", "")
    derived_table = state.get("derived_table", "")
    retry_count = state.get("retry_count", 0)

    if not ddl_sql:
        return {
            "errors": state.get("errors", []) + ["sql_execute: ddl_sql 为空"],
            "should_stop": True,
            "messages": [AIMessage(content="SQL 执行失败：DDL 为空")],
        }

    logger.info(f"[executor_node] 执行 DDL，衍生表: {derived_table}（第 {retry_count + 1} 次）")

    try:
        result = execute(ddl_sql)
        row_count = result.get("row_count", -1)

        if row_count == -1:
            # 二次确认行数
            try:
                row_count = get_table_row_count(derived_table)
            except Exception:
                row_count = 0

        if row_count == 0:
            msg = f"衍生表 {derived_table} 建立成功但行数为 0，请检查 WHERE 条件或目标 Y 列"
            logger.warning(f"[executor_node] {msg}")
            return {
                "derived_row_count": 0,
                "errors": state.get("errors", []) + [msg],
                "current_step": "sql_done_empty",
                "messages": [AIMessage(content=f"警告：{msg}")],
            }

        logger.info(f"[executor_node] 建表成功，行数: {row_count}")
        return {
            "derived_row_count": row_count,
            "current_step": "sql_done",
            "retry_count": 0,
            "messages": [AIMessage(content=f"建表成功：{derived_table}，共 {row_count} 行")],
        }

    except PermissionError as e:
        # 安全限制，不重试
        logger.error(f"[executor_node] 安全限制: {e}")
        return {
            "errors": state.get("errors", []) + [f"sql_execute security: {e}"],
            "should_stop": True,
            "current_step": "sql_failed",
            "messages": [AIMessage(content=f"SQL 安全限制：{e}")],
        }

    except Exception as e:
        logger.error(f"[executor_node] SQL 执行失败: {e}")
        new_retry = retry_count + 1
        errors = state.get("errors", []) + [f"sql_execute attempt {new_retry}: {e}"]

        if new_retry >= MAX_RETRIES:
            return {
                "errors": errors,
                "retry_count": new_retry,
                "should_stop": True,
                "current_step": "sql_failed",
                "messages": [AIMessage(content=f"SQL 执行失败（已重试 {new_retry} 次）: {e}")],
            }

        return {
            "errors": errors,
            "retry_count": new_retry,
            "current_step": "sql_retry",
            "messages": [AIMessage(content=f"SQL 执行失败，准备重试（{new_retry}/{MAX_RETRIES}）: {e}")],
        }
