"""
CLI 入口 — LangGraph 风控特征工程 Pipeline

用法：
  python -m langgraph_pipeline.main \\
    --table sample_pool.wide_credit_user_behavior \\
    --target t_5 \\
    --mode ratio \\
    --scene "信贷用户行为风控"

交互模式（不指定表名）：
  python -m langgraph_pipeline.main
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph_pipeline.graph import build_graph
from langgraph_pipeline.state import PipelineState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline.main")


# ─── 打印工具 ─────────────────────────────────────────────────────────────────

DIVIDER = "─" * 60


def _print_step(step: str, message: str):
    print(f"\n{DIVIDER}")
    print(f"  步骤: {step}")
    print(DIVIDER)
    print(message)


def _print_final_summary(final_state: dict):
    print(f"\n{'═' * 60}")
    print("  Pipeline 完成")
    print(f"{'═' * 60}")

    session_id = final_state.get("session_id", "")
    report_path = final_state.get("report_path", "")
    eval_summary = final_state.get("eval_results", {}).get("summary", {})
    model_results = final_state.get("model_results", {})
    errors = final_state.get("errors", [])

    print(f"会话 ID:     {session_id}")
    print(f"宽表:        {final_state.get('source_table', '')}")
    print(f"目标 Y:      {final_state.get('target_y_column', '')}")
    print(f"衍生特征:    {eval_summary.get('total', 0)} 个（"
          f"优质 {eval_summary.get('excellent_cnt', 0)} | "
          f"可用 {eval_summary.get('usable_cnt', 0)} | "
          f"废弃 {eval_summary.get('discard_cnt', 0)}）")

    if model_results:
        best = model_results.get("best_model", "")
        auc = model_results.get("best_test_auc", 0)
        model_path = model_results.get("model_path", "")
        print(f"最优模型:    {best.upper()}  AUC={auc:.4f}")
        print(f"模型文件:    {model_path}")

    if report_path:
        print(f"HTML 报告:   {report_path}")

    if errors:
        print(f"\n警告/错误 ({len(errors)} 条):")
        for e in errors[:5]:
            print(f"  ⚠ {e}")

    print(f"{'═' * 60}\n")


# ─── 流式运行 ────────────────────────────────────────────────────────────────

def run_pipeline(
    source_table: str,
    target_y: str = "",
    derivation_mode: Optional[str] = None,
    scene: str = "信贷风控",
    session_id: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    运行完整 Pipeline，返回最终 state。

    Args:
        source_table:     宽表全名，如 sample_pool.wide_credit_user_behavior
        target_y:         目标 Y 列名，如 t_5；为空则自动探测
        derivation_mode:  衍生方式（ratio/diff/log_transform 等），None=自动综合
        scene:            业务场景描述
        session_id:       指定会话 ID（None=自动生成）
        verbose:          是否打印流程日志

    Returns:
        最终 PipelineState 字典
    """
    sid = session_id or f"fe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    initial_state: PipelineState = {
        "session_id": sid,
        "session_dir": "",           # init_session 节点填充
        "source_table": source_table,
        "target_y_column": target_y,
        "derivation_mode": derivation_mode,
        "scene": scene,
        "messages": [],
        "current_step": "start",
        "retry_count": 0,
        "errors": [],
        "should_stop": False,
        "schema_analysis": {},
        "feature_design": [],
        "ddl_sql": "",
        "ddl_path": "",
        "derived_table": "",
        "derived_row_count": 0,
        "eval_results": {},
        "quality_thresholds": {},
        "model_results": {},
        "report_path": "",
    }

    app = build_graph()

    final_state = {}
    if verbose:
        print(f"\n{'═' * 60}")
        print(f"  LangGraph 风控特征工程 Pipeline")
        print(f"  宽表: {source_table}")
        print(f"  目标Y: {target_y or '自动探测'}")
        print(f"  衍生方式: {derivation_mode or '综合'}")
        print(f"  会话: {sid}")
        print(f"{'═' * 60}")

    # 流式执行，逐步打印
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_state in event.items():
            if verbose:
                messages = node_state.get("messages", [])
                for msg in messages:
                    _print_step(node_name, msg.content)

            # 合并到 final_state
            final_state.update(node_state)

    if verbose:
        _print_final_summary(final_state)

    return final_state


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LangGraph 风控特征工程 Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--table", "-t", required=True,
                        help="宽表全名，如 sample_pool.wide_credit_user_behavior")
    parser.add_argument("--target", "-y", default="",
                        help="目标 Y 列名，如 t_5（默认自动探测）")
    parser.add_argument("--mode", "-m", default=None,
                        choices=["ratio", "diff", "log_transform", "time_window_diff",
                                 "normalized", "window_compare", "binary_flag",
                                 "ordinal_encode", "frequency_encode", None],
                        help="衍生方式（默认综合）")
    parser.add_argument("--scene", "-s", default="信贷风控",
                        help="业务场景描述")
    parser.add_argument("--session", default=None,
                        help="指定会话 ID（默认自动生成）")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式（不打印流程日志）")
    parser.add_argument("--output-json", default=None,
                        help="将最终 state 导出为 JSON 文件")

    args = parser.parse_args()

    final_state = run_pipeline(
        source_table=args.table,
        target_y=args.target,
        derivation_mode=args.mode,
        scene=args.scene,
        session_id=args.session,
        verbose=not args.quiet,
    )

    if args.output_json:
        out_path = Path(args.output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            # 过滤掉不可序列化的 messages
            exportable = {k: v for k, v in final_state.items() if k != "messages"}
            json.dump(exportable, f, ensure_ascii=False, indent=2, default=str)
        print(f"State 已导出: {out_path}")

    # 返回码：有 should_stop=True 且无报告 → 失败
    if final_state.get("should_stop") and not final_state.get("report_path"):
        sys.exit(1)


if __name__ == "__main__":
    main()
