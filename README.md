# 风控特征工程 AI Pipeline

> 基于 LangGraph + Claude API 的风控特征自动衍生系统

## 项目简介

将 AI Agent 引入金融风控特征工程，通过 LangGraph 编排多个节点，自动完成宽表分析、特征公式设计、Doris 建表、IV/PSI 评估、LightGBM 建模的全流程，无需人工编写 SQL 和特征代码。

## 流程

```
宽表输入
  → Schema 探索（字段分类 + 缺失率估算）
  → LLM 特征设计（Claude API 生成衍生公式）
  → SQL 执行（Doris 建表，失败自动重试）
  → 特征评估（IV/PSI/缺失率，本地计算）
  → 自动建模（LightGBM/LR，输出 AUC/KS）
  → HTML 报告生成
```

## 技术栈

| 模块 | 技术 |
|---|---|
| Agent 编排 | LangGraph |
| LLM | Claude API (claude-sonnet-4-6) |
| 数据仓库 | Apache Doris |
| 建模 | LightGBM / Logistic Regression |

## 架构

```
init_session
     ↓
schema_explore    → 字段分类、缺失率估算、识别 Y 列
     ↓
feature_generate  → LLM 设计衍生公式、生成建表 DDL
     ↓
sql_execute       → Doris 建表（失败自动重试 3 次）
     ↓
feature_evaluate  → IV/PSI/缺失率评估，标记优质/可用/废弃
     ↓
model_train       → LightGBM/LR 自动建模，输出 AUC/KS
     ↓
report_generate   → HTML 可视化报告
```

## 亮点设计

- **LangGraph 状态机驱动**：节点职责隔离，通过共享 State 传递数据，条件边控制重试和分支
- **SQL 失败自动重试**：executor_node 最多重试 3 次，超限自动终止并报告错误
- **LLM 兜底机制**：Claude API 输出解析失败自动 fallback 到规则生成，流程不中断
- **SQL 安全层**：硬编码禁止 DROP/DELETE/TRUNCATE，建表强制使用 `feat_derived_` 前缀隔离
- **断点追溯**：所有中间结果持久化到 `sessions/` 目录，包含 schema 分析、特征设计、DDL、评估结果

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -r langgraph_pipeline/requirements_lg.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下配置：

```
ANTHROPIC_API_KEY=sk-ant-你的key

DORIS_HOST=你的Doris地址
DORIS_PORT=9030
DORIS_USER=用户名
DORIS_PASSWORD=密码
DORIS_DB=sample_pool

IV_META_HOST=你的MySQL地址
IV_META_USER=用户名
IV_META_PASSWORD=密码
```

### 3. 运行

```bash
python -m langgraph_pipeline.main \
  --table your_schema.your_table \
  --target your_y_column \
  --scene 信贷风控
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--table` | 宽表全名，如 `sample_pool.wide_credit` | 必填 |
| `--target` | Y 列名，如 `is_m2_mob4` | 自动识别 |
| `--mode` | 衍生方式：`ratio`/`diff`/`log_transform` 等 | 综合 |
| `--scene` | 业务场景描述 | 信贷风控 |
| `--quiet` | 静默模式，不打印流程日志 | 关闭 |

## 会话数据

每次运行在 `sessions/{session_id}/` 下生成：

```
sessions/fe_20260326_135224/
├── context.json          # 会话状态
├── schema_analysis.json  # 字段分析结果
├── feature_design.json   # 特征设计方案
└── ddl/
    └── batch_1_create.sql  # 建表 DDL
```

## 注意事项

- 宽表需包含 `pid`（ID列）、`lend_tm`（时间列）、Y列 三个基础字段
- Y 列建议为 0/1 整型，中文"是/否"也支持
- Doris 账号需要有目标库的建表和查询权限
