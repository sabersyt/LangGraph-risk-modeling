# LangGraph-risk-modeling
基于langgraph的特征衍生加建模pipeline 比较节省token
风控特征+建模+输出报告的pipeline工作流(互金场景)
风控特征工程 AI Pipeline

基于 LangGraph + Claude API 的风控特征自动衍生系统

项目简介
将 AI Agent 引入金融风控特征工程，通过 LangGraph 编排多个节点，自动完成宽表分析、特征公式设计、Doris 建表、IV/PSI 评估、LightGBM 建模的全流程，无需人工编写 SQL 和特征代码。
流程
宽表输入
  → Schema 探索（字段分类 + 缺失率估算）
  → LLM 特征设计（Claude API 生成衍生公式）
  → SQL 执行（Doris 建表，失败自动重试）
  → 特征评估（IV/PSI/缺失率，本地计算）
  → 自动建模（LightGBM/LR，输出 AUC/KS）
  → HTML 报告生成
技术栈

Agent 编排：LangGraph
LLM：Claude API (claude-sonnet-4-6)
数据仓库：Apache Doris
建模：LightGBM / Logistic Regression

亮点

LangGraph 状态机驱动，节点职责隔离，SQL 失败自动重试 3 次
LLM 输出解析失败自动 fallback 到规则兜底，流程不中断
SQL 安全层硬编码禁止 DROP/DELETE，建表强制前缀隔离
中间结果持久化到 sessions/ 目录，支持断点追溯
