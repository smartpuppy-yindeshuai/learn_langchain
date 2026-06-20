# LangChain 链式任务跟踪与调试规范
# Chain Tracing & Debugging Specification

## 1. 目标 / Objective

构建一个完整的 LangChain 链式任务跟踪与调试系统，实现：

- **可观测性 (Observability)**：记录链中每个组件的输入、输出及中间状态
- **性能分析 (Performance Profiling)**：采集每个步骤的执行耗时，识别瓶颈
- **错误定位 (Error Localization)**：在链断裂时快速定位失败节点及根因
- **审计回溯 (Audit Trail)**：持久化执行日志，支持事后审查与复现

Build a comprehensive tracing and debugging system for LangChain chains that provides:

- **Observability**: Record inputs, outputs, and intermediate state for every component
- **Performance profiling**: Capture execution time per step, identify bottlenecks
- **Error localization**: Quickly locate the failing node and root cause when a chain breaks
- **Audit trail**: Persist execution logs for post-mortem review and reproducibility

---

## 2. 架构总览 / Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Chain Execution                    │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Step 1   │──▶│  Step 2   │──▶│  Step 3   │──▶ … │
│  │ (LLM)    │   │ (Parser) │   │ (Tool)   │        │
│  └──────────┘   └──────────┘   └──────────┘        │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌──────────────────────────────────────────┐       │
│  │         Trace / Span Collector           │       │
│  └──────────────────────────────────────────┘       │
│       │                                              │
│       ▼                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐    │
│  │ Console  │  │  In-Memory   │  │  Custom    │    │
│  │ Handler  │  │  SpanStore   │  │  Handler   │    │
│  └──────────┘  └──────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 3. 核心概念 / Core Concepts

### 3.1 Run & Span

- **Run**: 一次完整的链执行实例，拥有唯一 `run_id`
- **Span**: 链中单个组件的一次调用，通过 `parent_run_id` 嵌套形成树状结构

- **Run**: A complete chain execution instance with a unique `run_id`
- **Span**: A single component invocation within a chain, nested via `parent_run_id` to form a tree

### 3.2 Callback 事件模型 / Callback Event Model

| 事件 / Event | 触发时机 / Trigger | 携带数据 / Data |
|------|---------|---------|
| `on_chain_start` | 链/步骤开始 / Step begins | `run_id`, `parent_run_id`, `inputs`, `name` |
| `on_chain_end` | 链/步骤结束 / Step completes | `run_id`, `outputs`, `duration_ms` |
| `on_chain_error` | 链/步骤异常 / Step fails | `run_id`, `error`, `stack_trace` |
| `on_llm_start` | LLM 调用开始 / LLM call begins | `run_id`, `prompts`, `invocation_params` |
| `on_llm_end` | LLM 调用结束 / LLM call ends | `run_id`, `response`, `token_usage` |
| `on_tool_start` | 工具调用开始 / Tool call begins | `run_id`, `tool_name`, `tool_input` |
| `on_tool_end` | 工具调用结束 / Tool call ends | `run_id`, `tool_output` |

### 3.3 Span 数据结构 / Span Data Structure

```python
@dataclass
class SpanRecord:
    run_id: str                  # UUID, unique execution identifier
    parent_run_id: Optional[str] # parent run_id for building the call tree
    span_type: str               # "chain" | "llm" | "tool" | "retriever"
    name: str                    # component name
    inputs: dict                 # input parameters
    outputs: Optional[dict]      # output (filled upon completion)
    start_time: datetime         # start timestamp
    end_time: Optional[datetime] # end timestamp
    duration_ms: Optional[float] # execution time in milliseconds
    tags: List[str]              # user-defined tags
    metadata: dict               # extra metadata (model params, token usage, etc.)
    error: Optional[str]         # error message if failed
    status: str                  # "running" | "success" | "error"
```

---

## 4. 调试机制 / Debugging Mechanisms

### 4.1 日志级别 / Log Levels

| 级别 / Level | 用途 / Purpose | 输出 / Output |
|------|------|---------|
| `INFO` | 生产监控 / Production | 链名、步骤数、总耗时 / chain name, step count, total time |
| `DEBUG` | 开发调试 / Development | 每个 Span 的完整输入/输出 / full I/O per span |
| `TRACE` | 深度排查 / Deep inspection | HTTP 原文、模板渲染过程 / raw HTTP, template rendering |

### 4.2 控制台输出格式 / Console Output Format

```
[chain/start] [qa_chain] ▶ input: {"question": "What is quantum computing?"}
  [llm/start] [ChatOpenAI] ▶ prompts: ["You are a helpful assistant..."]
  [llm/end]   [ChatOpenAI] ◀ tokens: 128 in / 256 out | 1.23s
  [tool/start] [search] ▶ input: "quantum computing definition"
  [tool/end]   [search] ◀ output: "Quantum computing leverages..." | 0.45s
[chain/end]   [qa_chain] ◀ total: 1.68s | status: success
```

---

## 5. 性能监控 / Performance Monitoring

### 5.1 关键指标 / Key Metrics

| 指标 / Metric | 计算方式 / Calculation | 告警阈值 / Alert Threshold |
|------|---------|----------------|
| 端到端延迟 / E2E Latency | `end_time - start_time` | > 30s |
| LLM 延迟 / LLM Latency | sum of LLM span durations | > 20s |
| Token 消耗 / Token Usage | `prompt_tokens + completion_tokens` | > 4000/req |
| 错误率 / Error Rate | `error_spans / total_spans` | > 5% |

---

## 6. 数据安全 / Data Security

| 规则 / Rule | 说明 / Description |
|------|------|
| PII 脱敏 / PII Masking | 个人信息必须在记录前脱敏 |
| 密钥过滤 / Secret Filtering | 回调中不得记录 API Key 等凭证 |
| 日志保留 / Log Retention | 生产环境跟踪日志保留 90 天 |

---

## 7. 实现步骤 / Implementation Steps

1. **实现 SpanRecord 数据结构** — 用 `@dataclass` 定义每个执行跨度
2. **实现 TracingCallbackHandler** — 自定义 `BaseCallbackHandler`，收集所有事件
3. **实现 SpanStore** — 内存存储，支持按 run_id 查询和树状构建
4. **实现 ConsoleTraceHandler** — 格式化控制台输出
5. **实现 PerformanceMonitor** — 性能指标采集与告警
6. **构建示例链** — 多步骤链（LLM + Tool + Parser）演示跟踪效果
7. **测试验证** — 正常执行、异常执行、性能告警三种场景

---

## 8. 依赖 / Dependencies

```
langchain >= 1.3
langchain-openai >= 1.3
langchain-core >= 1.4
openai (DashScope compatible mode)
```
