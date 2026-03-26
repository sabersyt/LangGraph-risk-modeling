"""
配置加载 — 读取 .env 和 feature_templates.yaml
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# 项目根目录（langgraph_pipeline 的上级目录）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


# ─── Doris ────────────────────────────────────────────────────────────────────

def get_doris_config() -> Dict[str, Any]:
    return {
        "host": os.getenv("DORIS_HOST", ""),
        "port": int(os.getenv("DORIS_PORT", "9030")),
        "user": os.getenv("DORIS_USER", ""),
        "password": os.getenv("DORIS_PASSWORD", ""),
        "database": os.getenv("DORIS_DB", "sample_pool"),
        "charset": "utf8mb4",
        "connect_timeout": 30,
    }


# ─── MySQL 元数据库 ────────────────────────────────────────────────────────────

def get_mysql_config() -> Dict[str, Any]:
    return {
        "host": os.getenv("IV_META_HOST", ""),
        "port": int(os.getenv("IV_META_PORT", "3306")),
        "user": os.getenv("IV_META_USER", ""),
        "password": os.getenv("IV_META_PASSWORD", ""),
        "database": os.getenv("IV_META_DB", "iv_meta"),
        "charset": "utf8mb4",
        "connect_timeout": 15,
    }


# ─── Metric API ───────────────────────────────────────────────────────────────

def get_metric_api_url() -> str:
    return os.getenv("METRIC_API_URL", "http://10.1.20.243:8200")


# ─── Feature Templates ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_feature_templates() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "configs" / "feature_templates.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_quality_thresholds() -> Dict[str, Any]:
    return load_feature_templates().get("quality_thresholds", {})


def get_template_by_mode(mode: str) -> Dict[str, Any]:
    templates = load_feature_templates().get("templates", {})
    return templates.get(mode, {})


# ─── LLM ──────────────────────────────────────────────────────────────────────

def get_llm_model() -> str:
    """返回 Claude 模型 ID，优先从环境变量读取"""
    return os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")


def get_anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY 未设置。请在 .env 文件或环境变量中配置。"
        )
    return key


# ─── Session 目录 ─────────────────────────────────────────────────────────────

def get_sessions_dir() -> Path:
    d = PROJECT_ROOT / "sessions"
    d.mkdir(exist_ok=True)
    return d


def get_models_dir() -> Path:
    d = PROJECT_ROOT / "models"
    d.mkdir(exist_ok=True)
    return d


def get_reports_dir() -> Path:
    d = PROJECT_ROOT / "reports" / "feature"
    d.mkdir(parents=True, exist_ok=True)
    return d
