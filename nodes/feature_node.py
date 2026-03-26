"""
节点 2：特征生成
- 调用 LLM（Claude）根据 schema 分析设计衍生特征公式
- 输出结构化的特征列表 + DDL SQL
- 写入 feature_design.json 和 ddl/batch_1_create.sql
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from langgraph_pipeline.config import get_anthropic_api_key, get_llm_model, load_feature_templates
from langgraph_pipeline.state import PipelineState

logger = logging.getLogger(__name__)


# ─── LLM 初始化 ──────────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=get_llm_model(),
        api_key=get_anthropic_api_key(),
        temperature=0.1,
        max_tokens=8192,
    )


# ─── Prompt 构建 ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = dedent("""
你是一名资深风控特征工程师，专注于信贷风控场景的特征衍生。

你的任务是：根据宽表的字段信息，设计有风险区分力的衍生特征。

**设计原则：**
1. 每个衍生特征必须有明确的业务含义和风险解释
2. 数值型特征优先使用：ratio（比率）、diff（差值）、log_transform（对数变换）、time_window_diff（时间窗口差值）
3. 类别型特征使用：binary_flag（二值标志）、ordinal_encode（有序编码）、frequency_encode（频率编码）
4. 除法必须使用 `A / NULLIF(B, 0)` 防零，禁止 COALESCE 填充
5. NULL 值保持 NULL，不得人为填充
6. 禁止使用：STR_TO_DATE、DATE_FORMAT、IF()、TABLESAMPLE
7. 不要给整个表达式加外层括号
                       
**Doris SQL 规则：**
- 自然对数：LN(x)
- 平方根：SQRT(x)
- 条件：CASE WHEN ... THEN ... ELSE ... END
- 窗口函数：COUNT(...) OVER (PARTITION BY ...)

**输出格式（严格 JSON，不加任何说明文字）：**
```json
{
  "features": [
    {
      "name": "feat_xxx",
      "formula": "A / NULLIF(B, 0)",
      "sources": ["A", "B"],
      "template": "ratio",
      "description": "业务含义解释",
      "expected_iv_range": "0.05~0.15"
    }
  ]
}
```
""").strip()


def _build_user_prompt(
    schema_analysis: Dict[str, Any],
    derivation_mode: Optional[str],
    templates_config: Dict[str, Any],
    max_features: int = 20,
) -> str:
    numeric_fields = schema_analysis.get("numeric_fields", [])[:40]
    categorical_fields = schema_analysis.get("categorical_fields", [])[:20]
    time_fields = schema_analysis.get("time_fields", [])

    # 附带字段注释
    col_map = {
        c["name"]: c.get("comment", "") or c.get("description", "")
        for c in schema_analysis.get("schema_columns", [])
    }
    numeric_with_comments = [
        f"{f}（{col_map.get(f, '')}）" for f in numeric_fields
    ]
    cat_with_comments = [
        f"{f}（{col_map.get(f, '')}）" for f in categorical_fields
    ]

    mode_hint = ""
    if derivation_mode and derivation_mode in templates_config.get("templates", {}):
        tpl = templates_config["templates"][derivation_mode]
        mode_hint = (
            f"\n**指定衍生方式：{derivation_mode}**\n"
            f"公式模板：{tpl.get('formula_pattern', '')}\n"
            f"描述：{tpl.get('description', '')}\n"
            f"约束：{tpl.get('constraints', [])}\n"
        )
    else:
        mode_hint = "\n**衍生方式：综合（ratio + diff + log_transform + time_window_diff + 类别编码）**\n"

    return dedent(f"""
宽表：{schema_analysis.get("table", "")}
总行数：{schema_analysis.get("total_rows", "未知")}
{mode_hint}

**数值型字段（{len(numeric_fields)} 个，选取前 40）：**
{chr(10).join(f"  - {f}" for f in numeric_with_comments)}

**类别型字段（{len(categorical_fields)} 个）：**
{chr(10).join(f"  - {f}" for f in cat_with_comments)}

**时间字段：** {', '.join(time_fields)}

**要求：**
1. 设计 {min(max_features, 20)} 个有风险区分力的衍生特征
2. 优先选择缺失率低、业务含义清晰的字段组合
3. 避免两个几乎相同的字段组合（例如都是近7天申请次数的变体）
4. 每个特征 name 必须以 `feat_` 开头且唯一
5. 严格按 JSON 格式输出，不加任何额外说明

请直接输出 JSON，不要有任何前缀或后缀文字。
""").strip()


# ─── DDL 生成 ────────────────────────────────────────────────────────────────

def _build_ddl(
    features: List[Dict[str, Any]],
    source_table: str,
    derived_table: str,
    target_y: str,
    include_sources: bool = True,
) -> str:
    """根据特征列表生成 Doris CREATE TABLE AS SELECT DDL"""

    # 收集所有 source 列（去重）
    all_sources: List[str] = []
    if include_sources:
        seen = set()
        for feat in features:
            for s in feat.get("sources", []):
                if s not in seen and s != target_y:
                    seen.add(s)
                    all_sources.append(s)

    # 构造 SELECT 表达式
    select_parts = ["    pid", "    lend_tm", f"    `{target_y}`"]

    # 来源字段（用于 IV 增益对比）
    for s in all_sources:
        select_parts.append(f"    `{s}`")

    # 衍生特征
    for feat in features:
        formula = feat["formula"]
        name = feat["name"]
        desc = feat.get("description", "")
        comment = f"  -- {desc}" if desc else ""
        select_parts.append(f"    ({formula}) AS `{name}`")

    select_clause = ",\n".join(select_parts)

    ddl = dedent(f"""
        CREATE TABLE IF NOT EXISTS {derived_table}
        PROPERTIES("replication_num" = "1")
        AS
        SELECT
        {select_clause}
        FROM {source_table}
        WHERE `{target_y}` IS NOT NULL
    """).strip()

    return ddl


# ─── 主节点函数 ──────────────────────────────────────────────────────────────

def feature_generate_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph 节点：LLM 驱动特征设计

    输入: state.schema_analysis, state.derivation_mode
    输出: state.feature_design, state.ddl_sql, state.ddl_path, state.derived_table
    """
    session_dir = Path(state["session_dir"])
    schema_analysis = state.get("schema_analysis", {})
    derivation_mode = state.get("derivation_mode")
    source_table = state["source_table"]
    target_y = state.get("target_y_column", "")
    session_id = state["session_id"]

    if not schema_analysis:
        return {
            "errors": state.get("errors", []) + ["feature_generate: schema_analysis 为空"],
            "should_stop": True,
            "messages": [AIMessage(content="特征生成失败：schema_analysis 为空")],
        }

    templates_config = load_feature_templates()

    # ── 1. LLM 生成特征公式 ───────────────────────────────────────
    logger.info("[feature_node] 调用 LLM 生成特征公式...")
    llm = _get_llm()
    user_prompt = _build_user_prompt(schema_analysis, derivation_mode, templates_config)

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw_text = response.content
        logger.debug(f"[feature_node] LLM 原始回复长度: {len(raw_text)}")
    except Exception as e:
        logger.error(f"[feature_node] LLM 调用失败: {e}")
        return {
            "errors": state.get("errors", []) + [f"feature_generate LLM: {e}"],
            "should_stop": True,
            "messages": [AIMessage(content=f"LLM 调用失败: {e}")],
        }

    # ── 2. 解析 JSON ──────────────────────────────────────────────
    features: List[Dict[str, Any]] = []
    try:
        # 提取 JSON 块（可能有前缀/后缀文字）
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            parsed = json.loads(json_match.group())
            features = parsed.get("features", [])
        else:
            raise ValueError("LLM 输出中未找到 JSON")
    except Exception as e:
        logger.warning(f"[feature_node] JSON 解析失败，尝试补救: {e}")
        # 退路：手工构造几个基础特征
        features = _fallback_features(schema_analysis)

    if not features:
        features = _fallback_features(schema_analysis)

    # ── 3. 确保特征名合法 & 唯一 ─────────────────────────────────
    seen_names = set()
    clean_features = []
    for feat in features:
        name = re.sub(r"[^\w]", "_", feat.get("name", "feat_unknown"))
        if not name.startswith("feat_"):
            name = f"feat_{name}"
        if name in seen_names:
            name = f"{name}_{len(seen_names)}"
        seen_names.add(name)
        feat["name"] = name
        clean_features.append(feat)

    features = clean_features

    # ── 4. 衍生表命名 ─────────────────────────────────────────────
    naming = templates_config.get("naming", {})
    schema = naming.get("schema", "sample_pool")
    prefix = naming.get("table_prefix", "feat_derived")
    derived_table = f"{schema}.{prefix}_{session_id}"

    # ── 5. 生成 DDL ───────────────────────────────────────────────
    include_sources = templates_config.get("include_source_fields_for_comparison", True)
    ddl_sql = _build_ddl(features, source_table, derived_table, target_y, include_sources)

    # ── 6. 写入文件 ───────────────────────────────────────────────
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "ddl").mkdir(exist_ok=True)

    feature_design_path = session_dir / "feature_design.json"
    feature_design_path.write_text(
        json.dumps({"features": features, "derived_table": derived_table}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ddl_path = session_dir / "ddl" / "batch_1_create.sql"
    ddl_path.write_text(ddl_sql, encoding="utf-8")

    logger.info(f"[feature_node] 设计了 {len(features)} 个特征，DDL 写入: {ddl_path}")

    summary = (
        f"特征设计完成，共 {len(features)} 个衍生特征：\n"
        + "\n".join(
            f"  - {f['name']}（{f.get('template', '')}）: {f.get('description', '')[:50]}"
            for f in features[:10]
        )
        + (f"\n  ... 共 {len(features)} 个" if len(features) > 10 else "")
        + f"\n衍生表: {derived_table}\nDDL: {ddl_path}"
    )

    return {
        "feature_design": features,
        "ddl_sql": ddl_sql,
        "ddl_path": str(ddl_path),
        "derived_table": derived_table,
        "current_step": "feature_done",
        "messages": [AIMessage(content=summary)],
    }


# ─── Fallback 特征（LLM 失败时） ─────────────────────────────────────────────

def _fallback_features(schema_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """当 LLM 失败时，基于启发式规则生成基础特征"""
    numeric = schema_analysis.get("numeric_fields", [])[:10]
    features = []

    # 对数变换
    for f in numeric[:5]:
        features.append({
            "name": f"feat_log_{f}",
            "formula": f"LN(`{f}` + 1)",
            "sources": [f],
            "template": "log_transform",
            "description": f"{f} 对数变换，缓解偏态",
            "expected_iv_range": "0.01~0.10",
        })

    # 比率（相邻两个数值字段）
    for i in range(0, min(len(numeric) - 1, 5), 2):
        a, b = numeric[i], numeric[i + 1]
        features.append({
            "name": f"feat_ratio_{a}_div_{b}",
            "formula": f"`{a}` / NULLIF(`{b}`, 0)",
            "sources": [a, b],
            "template": "ratio",
            "description": f"{a} 与 {b} 的比率",
            "expected_iv_range": "0.02~0.15",
        })

    return features
