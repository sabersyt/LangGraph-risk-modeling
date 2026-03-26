"""
节点 6：报告生成
- 整合特征评估 + 建模结果
- 生成 HTML 报告（内嵌所有图表）
- 支持调用已有的 generate_report.py（向后兼容）
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage

from langgraph_pipeline.config import get_reports_dir, PROJECT_ROOT
from langgraph_pipeline.state import PipelineState

logger = logging.getLogger(__name__)

# HTML 报告模板（内嵌 CSS + JS）
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>风控特征工程报告 — {session_id}</title>
<style>
  body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; margin: 0; background: #f5f7fa; color: #333; }}
  .header {{ background: linear-gradient(135deg, #1a237e, #283593); color: white; padding: 24px 32px; }}
  .header h1 {{ margin: 0; font-size: 22px; }}
  .header .meta {{ margin-top: 8px; font-size: 13px; opacity: 0.85; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  .card {{ background: white; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .card h2 {{ margin: 0 0 16px; font-size: 16px; color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }}
  .stat-box {{ background: white; border-radius: 8px; padding: 16px 20px; text-align: center;
               box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .stat-box .value {{ font-size: 32px; font-weight: bold; color: #1a237e; }}
  .stat-box .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #e8eaf6; padding: 10px 12px; text-align: left; font-weight: 600; color: #1a237e; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #fafafa; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }}
  .badge-excellent {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-usable {{ background: #fff3e0; color: #e65100; }}
  .badge-discard {{ background: #ffebee; color: #c62828; }}
  .badge-lgbm {{ background: #e3f2fd; color: #1565c0; }}
  .badge-lr {{ background: #f3e5f5; color: #6a1b9a; }}
  .progress-bar {{ height: 6px; background: #e8eaf6; border-radius: 3px; margin-top: 4px; }}
  .progress-fill {{ height: 100%; border-radius: 3px; background: linear-gradient(90deg, #3f51b5, #7986cb); }}
  .model-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
  .model-card {{ border: 1px solid #e8eaf6; border-radius: 8px; padding: 16px; }}
  .model-card h3 {{ margin: 0 0 12px; font-size: 14px; }}
  .metric-row {{ display: flex; justify-content: space-between; margin: 6px 0; font-size: 13px; }}
  .metric-val {{ font-weight: 600; color: #1a237e; }}
  .footer {{ text-align: center; color: #999; font-size: 12px; padding: 24px; }}
</style>
</head>
<body>
<div class="header">
  <h1>风控特征工程报告</h1>
  <div class="meta">
    会话: {session_id} &nbsp;|&nbsp; 宽表: {source_table} &nbsp;|&nbsp;
    目标Y: {target_y} &nbsp;|&nbsp; 生成时间: {gen_time}
  </div>
</div>

<div class="container">

  <!-- 概览统计 -->
  <div class="summary-grid">
    <div class="stat-box">
      <div class="value">{total_features}</div>
      <div class="label">衍生特征总数</div>
    </div>
    <div class="stat-box">
      <div class="value" style="color:#2e7d32">{excellent_cnt}</div>
      <div class="label">优质特征</div>
    </div>
    <div class="stat-box">
      <div class="value" style="color:#e65100">{usable_cnt}</div>
      <div class="label">可用特征</div>
    </div>
    <div class="stat-box">
      <div class="value" style="color:#1565c0">{best_auc}</div>
      <div class="label">最优模型 AUC</div>
    </div>
  </div>

  <!-- 特征质量详情 -->
  <div class="card">
    <h2>特征质量评估（共 {total_features} 个）</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>特征名</th>
          <th>IV</th>
          <th>PSI</th>
          <th>缺失率</th>
          <th>模板</th>
          <th>可用性</th>
          <th>IV 可用性条</th>
        </tr>
      </thead>
      <tbody>
        {feature_rows}
      </tbody>
    </table>
  </div>

  <!-- 建模结果 -->
  {model_section}

  <!-- Top 特征重要性 -->
  {importance_section}

</div>
<div class="footer">LangGraph 风控智能化平台 &nbsp;|&nbsp; 特征工程自动化流水线</div>
</body>
</html>
"""


def _badge(availability: str) -> str:
    cls = {"优质": "badge-excellent", "可用": "badge-usable", "废弃": "badge-discard"}.get(availability, "badge-discard")
    return f'<span class="badge {cls}">{availability}</span>'


def _progress(iv: float, max_iv: float = 0.5) -> str:
    pct = min(100, round(iv / max(max_iv, 0.01) * 100))
    return f'<div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>'


def _build_feature_rows(features: list, feature_design: list) -> str:
    # 建立 name → template 映射
    template_map = {f["name"]: f.get("template", "") for f in feature_design}
    max_iv = max((f.get("iv", 0) for f in features), default=0.5)

    rows = []
    for i, feat in enumerate(
        sorted(features, key=lambda x: x.get("iv", 0), reverse=True), 1
    ):
        name = feat["name"]
        iv = feat.get("iv", 0)
        psi = feat.get("psi", 0)
        mr = feat.get("missing_rate", 0)
        av = feat.get("availability", "废弃")
        tpl = template_map.get(name, "")
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td style='font-family:monospace;font-size:12px'>{name}</td>"
            f"<td>{iv:.4f}</td>"
            f"<td>{psi:.4f}</td>"
            f"<td>{mr:.2%}</td>"
            f"<td>{tpl}</td>"
            f"<td>{_badge(av)}</td>"
            f"<td style='min-width:100px'>{_progress(iv, max_iv)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_model_section(model_results: Dict[str, Any]) -> str:
    if not model_results:
        return ""
    models = model_results.get("models", {})
    best = model_results.get("best_model", "")
    cards = []
    for name, res in models.items():
        tm = res.get("test_metrics", {})
        mark = " ⭐ 最优" if name == best else ""
        cls = "badge-lgbm" if name == "lgbm" else "badge-lr"
        cards.append(f"""
        <div class="model-card">
          <h3><span class="badge {cls}">{name.upper()}</span>{mark}</h3>
          <div class="metric-row"><span>AUC</span><span class="metric-val">{tm.get('auc', 0):.4f}</span></div>
          <div class="metric-row"><span>KS</span><span class="metric-val">{tm.get('ks', 0):.4f}</span></div>
          <div class="metric-row"><span>Gini</span><span class="metric-val">{tm.get('gini', 0):.4f}</span></div>
        </div>""")

    return f"""
    <div class="card">
      <h2>建模结果
        <span style="font-size:12px;color:#666;font-weight:normal;margin-left:8px">
          训练={model_results.get('train_samples',0):,} | 验证={model_results.get('val_samples',0):,}
          | 测试={model_results.get('test_samples',0):,} | 逾期率={model_results.get('positive_rate',0):.2%}
        </span>
      </h2>
      <div class="model-grid">{"".join(cards)}</div>
    </div>"""


def _build_importance_section(model_results: Dict[str, Any]) -> str:
    lgbm_res = model_results.get("models", {}).get("lgbm", {}) if model_results else {}
    importance = lgbm_res.get("feature_importance", {})
    if not importance:
        return ""

    max_score = max(importance.values(), default=1)
    rows = []
    for feat, score in list(importance.items())[:20]:
        pct = round(score / max_score * 100)
        rows.append(
            f"<tr>"
            f"<td style='font-family:monospace;font-size:12px'>{feat}</td>"
            f"<td>{score:.4f}</td>"
            f"<td style='min-width:200px'>"
            f"<div class='progress-bar'><div class='progress-fill' style='width:{pct}%'></div></div>"
            f"</td></tr>"
        )
    return f"""
    <div class="card">
      <h2>特征重要性 Top 20（LightGBM）</h2>
      <table>
        <thead><tr><th>特征名</th><th>重要性</th><th>可视化</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>"""


# ─── 主节点 ──────────────────────────────────────────────────────────────────

def report_generate_node(state: PipelineState) -> Dict[str, Any]:
    """
    LangGraph 节点：生成 HTML 报告

    输入: state.eval_results, state.model_results, state.feature_design
    输出: state.report_path
    """
    import datetime

    eval_results = state.get("eval_results", {})
    model_results = state.get("model_results", {})
    feature_design = state.get("feature_design", [])
    session_id = state["session_id"]
    session_dir = Path(state["session_dir"])

    # ── 1. 尝试调用已有 generate_report.py（向后兼容） ─────────────
    legacy_script = PROJECT_ROOT / "generate_report.py"
    if legacy_script.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(legacy_script), "--session", session_id, "--type", "feature"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                logger.info("[reporter] generate_report.py 执行成功")
                # 找到生成的报告路径
                reports_dir = get_reports_dir()
                html_files = list(reports_dir.glob(f"*{session_id}*.html"))
                if html_files:
                    return {
                        "report_path": str(html_files[0]),
                        "current_step": "report_done",
                        "messages": [AIMessage(content=f"报告已生成: {html_files[0]}")],
                    }
        except Exception as e:
            logger.warning(f"[reporter] generate_report.py 失败，使用内置模板: {e}")

    # ── 2. 内置 HTML 报告生成 ────────────────────────────────────
    features = eval_results.get("features", [])
    summary = eval_results.get("summary", {})

    feature_rows = _build_feature_rows(features, feature_design)
    model_section = _build_model_section(model_results)
    importance_section = _build_importance_section(model_results)

    best_auc = (
        model_results.get("best_test_auc", "N/A")
        if model_results
        else "N/A"
    )

    html_content = HTML_TEMPLATE.format(
        session_id=session_id,
        source_table=state.get("source_table", ""),
        target_y=state.get("target_y_column", ""),
        gen_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_features=summary.get("total", len(features)),
        excellent_cnt=summary.get("excellent_cnt", 0),
        usable_cnt=summary.get("usable_cnt", 0),
        best_auc=best_auc,
        feature_rows=feature_rows,
        model_section=model_section,
        importance_section=importance_section,
    )

    reports_dir = get_reports_dir()
    report_path = reports_dir / f"report_{session_id}.html"
    report_path.write_text(html_content, encoding="utf-8")

    logger.info(f"[reporter] 报告已生成: {report_path}")

    return {
        "report_path": str(report_path),
        "current_step": "report_done",
        "messages": [AIMessage(
            content=f"报告已生成！\n路径: {report_path}\n"
                    f"特征总数: {summary.get('total', len(features))} | "
                    f"优质: {summary.get('excellent_cnt', 0)} | "
                    f"可用: {summary.get('usable_cnt', 0)}"
        )],
    }
