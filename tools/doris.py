"""
Doris 数据库工具函数
- 直接封装 pymysql，供各节点调用
- 安全层：阻断危险 SQL，强制表前缀
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pymysql

from langgraph_pipeline.config import get_doris_config

logger = logging.getLogger(__name__)

BLOCKED_KEYWORDS = {"DROP ", "TRUNCATE ", "DELETE ", "ALTER ", "GRANT ", "REVOKE "}
ALLOWED_WRITE_PREFIXES = ("CREATE TABLE", "INSERT INTO", "INSERT OVERWRITE", "SELECT", "SHOW", "DESC", "DESCRIBE", "EXPLAIN")
SAFE_TABLE_PREFIXES = ("feat_derived_", "derived_", "sample_pool.derived_")
MAX_ROWS = 50_000


# ─── 连接 ─────────────────────────────────────────────────────────────────────

def _get_conn(database: Optional[str] = None) -> pymysql.Connection:
    cfg = {
        "host": "****",
        "port": ***,
        "user": "***",
        "password": "***",
        "database": "表名",
        "charset": "utf8mb4"
    }
    if database:
        cfg["database"] = database
    return pymysql.connect(**cfg)


# ─── 安全校验 ─────────────────────────────────────────────────────────────────

def _validate_readonly(sql: str) -> Tuple[bool, str]:
    upper = sql.upper().strip()
    for kw in BLOCKED_KEYWORDS:
        if kw in upper:
            return False, f"安全限制：只读模式不允许 {kw.strip()}"
    if not any(upper.startswith(p) for p in ("SELECT", "SHOW", "DESC", "DESCRIBE", "EXPLAIN")):
        return False, "只读模式仅允许 SELECT/SHOW/DESC/EXPLAIN"
    return True, ""


def _validate_write(sql: str) -> Tuple[bool, str]:
    upper = sql.upper().strip()
    for kw in BLOCKED_KEYWORDS:
        if kw in upper:
            return False, f"安全限制：不允许 {kw.strip()}"
    if not any(upper.startswith(p) for p in ALLOWED_WRITE_PREFIXES):
        return False, "只允许 CREATE TABLE / INSERT / SELECT 操作"
    if upper.startswith("CREATE"):
        lower = sql.lower()
        if not any(p in lower for p in SAFE_TABLE_PREFIXES):
            return False, f"建表必须使用前缀: {SAFE_TABLE_PREFIXES}"
    return True, ""


# ─── 公共 API ─────────────────────────────────────────────────────────────────

def query(sql: str, database: Optional[str] = None, max_rows: int = 500) -> pd.DataFrame:
    """执行只读查询，返回 DataFrame"""
    ok, err = _validate_readonly(sql)
    if not ok:
        raise PermissionError(err)
    conn = _get_conn(database)
    try:
        df = pd.read_sql(sql, conn)
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df
    finally:
        conn.close()


def execute(sql: str, database: Optional[str] = None) -> Dict[str, Any]:
    """执行写操作（CREATE/INSERT），返回执行结果摘要"""
    ok, err = _validate_write(sql)
    if not ok:
        raise PermissionError(err)
    conn = _get_conn(database)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        affected = cursor.rowcount
        cursor.close()

        result: Dict[str, Any] = {"status": "success", "affected_rows": affected}

        # CREATE TABLE 后自动查询行数
        if sql.upper().strip().startswith("CREATE"):
            m = re.search(
                r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\S+)[`\"]?",
                sql, re.IGNORECASE
            )
            if m:
                table = m.group(1).strip("`\"")
                result["created_table"] = table
                try:
                    cur2 = conn.cursor()
                    cur2.execute(f"SELECT COUNT(*) FROM {table}")
                    result["row_count"] = cur2.fetchone()[0]
                    cur2.close()
                except Exception:
                    result["row_count"] = -1
        return result
    finally:
        conn.close()


def get_table_schema(table_name: str) -> List[Dict[str, Any]]:
    """
    获取表的列定义：column_name, data_type, column_comment, is_nullable
    返回列表，每项对应一列。
    """
    if "." in table_name:
        schema, table = table_name.split(".", 1)
    else:
        schema = get_doris_config()["database"]
        table = table_name

    sql = f"""
        SELECT column_name, data_type, column_type, is_nullable, column_comment, ordinal_position
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        ORDER BY ordinal_position
        LIMIT 500
    """
    conn = _get_conn()
    try:
        df = pd.read_sql(sql, conn)
        return df.to_dict(orient="records")
    finally:
        conn.close()


def get_table_row_count(table_name: str) -> int:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception:
        return -1
    finally:
        conn.close()


def load_feature_matrix(
    derived_table: str,
    feature_names: List[str],
    target_col: str,
    id_col: str = "pid",
    time_col: str = "lend_tm",
    limit: int = MAX_ROWS,
) -> pd.DataFrame:
    """
    从衍生特征表加载建模用特征矩阵（含目标 Y）。
    采用时间降序 LIMIT 避免 TABLESAMPLE（Doris 不支持）。
    """
    all_cols = list(dict.fromkeys([id_col, time_col, target_col] + feature_names))  # 去重保序
    cols = ", ".join([f"`{c}`" for c in all_cols])
    sql = f"""
        SELECT {cols}
        FROM (
            SELECT {cols}
            FROM {derived_table}
            WHERE `{target_col}` IS NOT NULL
            ORDER BY `{time_col}` DESC
            LIMIT {limit}
        ) t
    """
    return query(sql, max_rows=limit)


def sample_column_values(table_name: str, col: str, limit: int = 50) -> List[Any]:
    """抽样查询某列的非 NULL 值，用于类别字段分析"""
    sql = f"""
        SELECT DISTINCT `{col}` FROM {table_name}
        WHERE `{col}` IS NOT NULL
        LIMIT {limit}
    """
    try:
        df = query(sql, max_rows=limit)
        return df[col].tolist() if not df.empty else []
    except Exception:
        return []


def estimate_missing_rate(table_name: str, col: str, limit: int = 10000) -> float:
    """估算列缺失率（采样）"""
    sql = f"""
        SELECT
            SUM(CASE WHEN `{col}` IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS missing_rate
        FROM (
            SELECT `{col}` FROM {table_name}
            ORDER BY rand()
            LIMIT {limit}
        ) t
    """
    try:
        df = query(sql, max_rows=1)
        return float(df["missing_rate"].iloc[0]) if not df.empty else 0.0
    except Exception:
        return 0.0
