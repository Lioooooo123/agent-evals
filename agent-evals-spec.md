# Agent Evals MVP Spec

## 1. 背景

我们要做一个 local-first 的 Agent Evals 工具，用来评估 agent 的任务完成质量、工具调用过程、执行轨迹、成本、延迟和失败模式。目标不是复刻 LangSmith、Braintrust 这类完整 SaaS，而是先做一个能放进 GitHub、能本地跑、能做回归对比、能产出可读报告的 MVP。

调研结论很明确：Agent Evals 不能只看最终回答。OpenAI 的 agent evals 文档把 traces、graders、datasets 和 eval runs 放在同一个工作流里，并建议先用 trace grading 定位工作流问题，再沉淀为可重复的数据集。LangChain AgentEvals 也把 execution trajectory 定义为 messages 和 tool calls 的序列，并提供 deterministic trajectory match 和 LLM-as-judge 两条路径。Braintrust 的 scorer 设计则强调 scorer 可以接收 input、output、expected、metadata 和 trace，并区分 span scope 与 trace scope。近期的 agentevals 项目进一步体现了 trace-first 趋势：从 OpenTelemetry traces 评分，避免重复跑 agent。

## 2. 产品定位

一句话定位：

> 一个 local-first 的 Agent Evals CLI，用 JSONL 定义任务，用结构化 trace 记录 agent 过程，用规则 scorer 和 LLM judge 评分，用 Markdown/JSONL/CSV 报告解释失败和版本回归。

第一版主要服务三类用户：

- 想评估自己 agent 项目的个人开发者。
- 想把 agent 质量评测写进作品集或简历的人。
- 小团队在没有 SaaS 平台预算或不想上传 trace 的情况下做本地回归。

第一版优先支持这三类 agent：

- Tool Agent：重点评估工具选择、参数、顺序和结果 groundedness。
- RAG Tool Agent：额外评估检索上下文、答案忠实性和 citation。
- Code/Research Agent：重点评估任务完成、artifact 是否真实产生、步骤是否可解释。

## 3. 非目标

MVP 不做以下能力：

- 不做 Web dashboard。
- 不做生产 trace 实时采样和在线监控。
- 不做多租户、权限、团队协作。
- 不做完整 OpenTelemetry ingest。MVP 只保留一个 trace schema，后续再加 OTel importer。
- 不做自动生成 eval case。第一版先手写或从失败样例导入。
- 不做复杂失败聚类模型。第一版按 `failure_type` 汇总。
- 不做 full chain-of-thought 存储。只保存可审计的 summary、tool call、observation 和 error。

## 4. 成功标准

MVP 完成后应满足：

- 能读取 `cases/*.eval_cases.jsonl`。
- 能通过 adapter 执行一个被测 agent 或一个 mock/custom function。
- 每条 case 都生成结构化 trace。
- 至少支持 deterministic scorer、tool trajectory scorer、LLM judge scorer 和 aggregate scorer。
- 输出 `eval_results.jsonl`、`summary.csv`、`eval_report.md`、`failed_cases.md`。
- 支持 baseline vs candidate 对比，并能按阈值返回失败退出码。
- 不看原始 JSON，也能从报告判断：整体是否通过、哪些指标下降、哪些 case 失败、失败原因是什么。

## 5. 核心工作流

```text
eval_cases.jsonl
  -> load dataset
  -> run agent through adapter
  -> record trace
  -> run scorers
  -> aggregate results
  -> generate reports
  -> compare with baseline
```

CLI 入口：

```bash
agent-evals run \
  --cases cases/sample.eval_cases.jsonl \
  --config configs/agent.yaml \
  --rubric configs/judge_rubric.yaml \
  --out runs/2026-06-08T21-00-00
```

```bash
agent-evals compare \
  --baseline runs/baseline/eval_results.jsonl \
  --candidate runs/candidate/eval_results.jsonl \
  --fail-if-drop task_success_rate=0.03 \
  --fail-if-drop tool_call_accuracy=0.05
```

```bash
agent-evals inspect \
  --run runs/candidate \
  --case-id case_001
```

## 6. 目录结构

```text
agent-evals/
  cases/
    sample.eval_cases.jsonl
  configs/
    agent.yaml
    judge_rubric.yaml
    tools_schema.json
  src/
    agent_evals/
      cli.py
      datasets.py
      runners/
        base.py
        custom_function.py
      traces/
        recorder.py
        schema.py
      scorers/
        base.py
        rules.py
        tool_trajectory.py
        judge.py
        aggregate.py
      reports/
        markdown.py
        csv_export.py
      compare.py
  runs/
    .gitkeep
  pyproject.toml
  README.md
```

## 7. 数据模型

### 7.1 EvalCase

每条 case 是一行 JSONL。必填字段应尽量少，扩展信息放进 `expected` 和 `metadata`。

```json
{
  "id": "case_001",
  "input": {
    "messages": [
      {"role": "user", "content": "我的订单 A123 到哪了？"}
    ]
  },
  "expected": {
    "answer_contains": ["订单", "状态"],
    "answer_must_not_contain": ["编造物流单号"],
    "tool_calls": [
      {
        "tool_name": "lookup_order",
        "arguments": {"order_id": "A123"},
        "match_mode": "exact"
      }
    ],
    "outcome": {
      "task_success": true,
      "handoff_required": false
    }
  },
  "metadata": {
    "category": "customer_support",
    "difficulty": "easy",
    "tags": ["tool_call", "order_lookup"]
  }
}
```

设计要求：

- `id` 必须唯一。
- `input.messages` 兼容 OpenAI/LangChain 常见 message shape。
- `expected.tool_calls` 支持 `strict`、`unordered`、`subset`、`superset` 四种 trajectory match。
- `metadata.tags` 用于报告切片。

### 7.2 Trace

Trace 是评分的一等数据，不只是日志。

```json
{
  "trace_id": "trace_001",
  "run_id": "run_2026_06_08_210000",
  "case_id": "case_001",
  "agent_version": "agent_v0.1.0",
  "status": "completed",
  "started_at": "2026-06-08T21:00:00+08:00",
  "ended_at": "2026-06-08T21:00:04+08:00",
  "final_answer": "订单 A123 当前正在配送中。",
  "metrics": {
    "latency_ms": 4200,
    "input_tokens": 1200,
    "output_tokens": 220,
    "cost_usd": 0.0031
  },
  "steps": []
}
```

### 7.3 TraceStep

```json
{
  "step_id": "step_002",
  "index": 2,
  "type": "tool_call",
  "timestamp": "2026-06-08T21:00:02+08:00",
  "summary": "Agent 查询订单状态。",
  "tool_call": {
    "tool_name": "lookup_order",
    "arguments": {"order_id": "A123"}
  },
  "observation": {
    "status": "out_for_delivery"
  },
  "error": null,
  "metrics": {
    "latency_ms": 310,
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  }
}
```

`type` 枚举：

- `llm`
- `tool_call`
- `retriever`
- `observation`
- `retry`
- `handoff`
- `final`
- `error`

### 7.4 EvalResult

```json
{
  "case_id": "case_001",
  "run_id": "run_2026_06_08_210000",
  "pass": true,
  "scores": {
    "task_success": 1.0,
    "final_answer_correctness": 0.9,
    "tool_call_accuracy": 1.0,
    "trajectory_score": 1.0,
    "format_compliance": 1.0,
    "safety": 1.0,
    "efficiency": 0.8
  },
  "failure_type": "none",
  "reason": "工具调用正确，最终答案由 observation 支持。",
  "trace_path": "traces/case_001.trace.json"
}
```

`failure_type` 枚举：

- `none`
- `tool_selection`
- `tool_arguments`
- `tool_order`
- `unsupported_answer`
- `incomplete_task`
- `format_error`
- `unsafe`
- `inefficient`
- `timeout`
- `runtime_error`
- `judge_error`

## 8. Scorer 设计

### 8.1 Scorer 接口

```python
class Scorer(Protocol):
    name: str

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        ...
```

`ScoreResult`：

```json
{
  "name": "tool_call_accuracy",
  "score": 1.0,
  "pass": true,
  "reason": "actual tool calls match expected calls",
  "failure_type": "none",
  "metadata": {}
}
```

### 8.2 MVP Scorers

必须实现：

- `AnswerRuleScorer`：检查 `answer_contains`、`answer_must_not_contain`、regex、JSON schema。
- `ToolTrajectoryScorer`：检查工具名、参数、顺序，支持 strict/unordered/subset/superset。
- `TaskSuccessJudgeScorer`：用 LLM-as-a-judge 评估目标完成、grounding、tool use、safety。
- `ExecutionMetricsScorer`：检查 latency、cost、step count、timeout。
- `AggregateScorer`：按权重聚合 pass/fail。

推荐默认权重：

```yaml
weights:
  task_success: 0.35
  tool_call_accuracy: 0.25
  final_answer_correctness: 0.20
  grounding: 0.10
  efficiency: 0.05
  safety: 0.05
pass_threshold: 0.80
hard_fail:
  safety: 0
  format_compliance: 0
```

### 8.3 Judge Rubric

Judge 只处理规则无法稳定判断的语义问题。工具参数、格式、文件存在、外部状态这类确定性检查必须优先走 rule scorer。

Judge 输入：

- user task
- expected behavior
- final answer
- trace summary
- tool calls and observations
- rubric

Judge 输出必须是 JSON：

```json
{
  "goal_completion": 0,
  "tool_use": 0,
  "grounding": 0,
  "efficiency": 0,
  "safety": 0,
  "overall_score": 0,
  "pass": false,
  "failure_type": "unsupported_answer",
  "reason": "最终答案没有被工具返回结果支持。"
}
```

## 9. Runner 与 Adapter

MVP 用 adapter 隔离具体 agent 框架：

```python
class AgentAdapter(Protocol):
    def run(self, case: EvalCase, recorder: TraceRecorder) -> AgentOutput:
        ...
```

第一版实现：

- `CustomFunctionAdapter`：加载用户提供的 Python 函数。
- `MockAgentAdapter`：用于 demo 和测试，返回预设步骤。

后续实现：

- `OpenAIAgentsAdapter`
- `LangGraphAdapter`
- `CrewAIAdapter`
- `LlamaIndexAdapter`
- `OTelTraceAdapter`

## 10. PiAgentAdapter 设计

Pi 是第一个推荐接入的被测 agent。它当前没有独立的产品化 evals 目录，但已有可复用的观测入口：

- CLI/print mode：可黑盒执行任务，收集 stdout、stderr、exit code、耗时和 workspace diff。
- Session JSONL：Pi session 是 JSONL，可解析 user、assistant、toolResult、bashExecution、custom、compactionSummary 等消息。
- Extension events：Pi extension/runtime 可观测 `message_end`、`tool_call`、`tool_result` 等事件。
- Test suite harness：`packages/coding-agent/test/suite/harness.ts` 支持 faux provider 和事件收集，适合 deterministic regression eval，不消耗真实模型 token。

### 10.1 接入层级

| 层级 | 入口 | 能观测什么 | 适用场景 |
| --- | --- | --- | --- |
| 黑盒 | Pi CLI / print mode | 最终输出、退出码、文件变化、测试结果 | 真实 agent 端到端任务 |
| 半白盒 | Session JSONL | 消息树、assistant tool calls、toolResult、bashExecution、token/cost | 离线 trace scoring |
| 白盒 | Extension events / test harness | `message_end`、`tool_call`、`tool_result`、runtime events | 工具轨迹、回归测试、无 token 测试 |

第一版先实现黑盒 + session JSONL 解析；如果后续要测 Pi 内部工具轨迹，再实现 extension event recorder。

### 10.2 PiEvalCase 扩展字段

Pi 任务 case 在通用 `EvalCase` 上扩展 `metadata.pi`：

```json
{
  "id": "pi_fix_auth_bug",
  "input": {
    "messages": [
      {"role": "user", "content": "修复登录接口空密码时崩溃的问题，并运行相关测试。"}
    ]
  },
  "expected": {
    "files_changed": ["src/auth.py"],
    "commands_pass": ["pytest tests/test_auth.py"],
    "answer_contains": ["修复", "测试"],
    "outcome": {
      "task_success": true
    }
  },
  "metadata": {
    "agent": "pi",
    "tags": ["code_agent", "bugfix", "tool_call"],
    "pi": {
      "mode": "cli",
      "workspace": "/tmp/agent-evals/pi/case_001",
      "timeout_s": 600,
      "session_jsonl": null,
      "post_run_commands": ["pytest tests/test_auth.py"]
    }
  }
}
```

### 10.3 PiAgentAdapter 契约

```python
class PiAgentAdapter:
    def run(self, case: EvalCase, recorder: TraceRecorder) -> AgentOutput:
        ...
```

执行步骤：

1. 准备隔离 workspace。真实任务使用 git worktree 或临时 copy，避免污染原仓库。
2. 记录初始状态：`git status --short`、`git diff --stat`、关键文件 hash。
3. 启动 Pi，传入用户任务。
4. 收集 stdout、stderr、exit code、latency。
5. 查找本次 Pi session JSONL，如果 case 指定了 `session_jsonl` 则使用指定文件。
6. 解析 session JSONL 为 trace steps。
7. 运行 `post_run_commands`，例如测试命令。
8. 记录最终状态：git diff、文件变化、测试结果。
9. 返回 `AgentOutput`。

### 10.4 Pi Trace 映射

Pi session / event 到标准 trace 的映射：

| Pi 来源 | 标准 TraceStep |
| --- | --- |
| user message | `type=llm_input` 或保存在 trace input |
| assistant message text | `type=llm`，`summary` 为文本摘要 |
| assistant content `toolCall` | `type=tool_call`，记录 tool name 和 arguments |
| toolResult message | `type=observation`，记录 result content、details、isError |
| bashExecution message | `type=tool_call` + `observation`，tool name 固定为 `bash` |
| message usage | trace metrics 的 token/cost |
| extension `tool_call` event | `type=tool_call` |
| extension `tool_result` event | `type=observation` |

工具调用 scorer 优先使用 extension events；没有 events 时退回 session JSONL；再没有 session 时只做黑盒 outcome scoring。

### 10.5 Pi Outcome Scorers

Pi code-agent eval 不能只看最终回答，必须检查真实 outcome：

- `WorkspaceDiffScorer`：检查是否改了预期文件、是否改了禁止文件、diff 是否为空。
- `CommandPassScorer`：运行 `expected.commands_pass` 或 `metadata.pi.post_run_commands`。
- `FinalAnswerGroundingScorer`：检查最终回答是否与测试和文件状态一致。
- `PiToolTrajectoryScorer`：基于 session/event 检查工具名、参数和顺序。
- `NoUncommittedNoiseScorer`：检查是否产生无关临时文件、大文件或敏感文件。

### 10.6 Faux Provider Regression Mode

Pi 自带 `packages/coding-agent/test/suite/harness.ts` 和 faux provider，适合测 deterministic agent runtime 行为：

- 不使用真实 provider API。
- 不消耗 token。
- 可以预设 assistant responses 和 tool calls。
- 可以断言 `AgentSessionEvent[]`。

这个模式不用于评估真实模型能力，而用于验证 Pi adapter、trace recorder、tool trajectory scorer 和 regression case 是否稳定。

### 10.7 限制

- 黑盒 CLI 无法可靠还原真实工具调用，只能观察 outcome。
- Session JSONL 是离线记录，可能缺少工具执行前的参数变更细节。
- Extension events 最准确，但需要 Pi runtime 支持加载 recorder extension。
- 真实模型 eval 成本和不确定性较高，需要和 faux provider regression 分开。
- Pi 默认不提供强权限隔离；真实 eval 应在临时目录、容器、sandbox 或 git worktree 中运行。

## 11. 报告

### 11.1 eval_report.md

必须包含：

- run metadata：run id、时间、agent version、case 数。
- 总体指标：pass rate、task success、tool call accuracy、avg latency、avg cost。
- 按 tag 切片：例如 `tool_call`、`rag`、`safety`。
- 失败类型分布。
- top regressions。
- 建议下一步。

### 11.2 failed_cases.md

每个失败 case 展示：

```text
Case: case_001
Result: FAIL
Failure: tool_arguments

Input:
我的订单 A123 到哪了？

Trace:
1. [LLM] Agent 决定查询订单。
2. [Tool] lookup_order({"order_id": "A132"}) -> not_found
3. [Final] 订单 A123 正在配送中。

Reason:
工具参数把 A123 写成 A132，最终答案没有 observation 支持。
```

### 11.3 summary.csv

字段：

- `case_id`
- `pass`
- `aggregate_score`
- `task_success`
- `tool_call_accuracy`
- `trajectory_score`
- `final_answer_correctness`
- `failure_type`
- `latency_ms`
- `cost_usd`
- `tags`

## 12. Baseline 对比

`agent-evals compare` 输出：

- candidate 相比 baseline 的指标变化。
- 新增失败 case。
- 修复的失败 case。
- 按 tag 的退化。
- 是否触发 gate。

Gate 示例：

```yaml
gates:
  task_success_rate:
    min: 0.85
    fail_if_drop_greater_than: 0.03
  tool_call_accuracy:
    min: 0.90
    fail_if_drop_greater_than: 0.05
  safety_violation_rate:
    max: 0.00
```

## 13. 验证计划

### 13.1 单元测试

- JSONL dataset loader 能识别重复 id 和非法 schema。
- trace recorder 能按顺序记录 step。
- tool trajectory scorer 覆盖 strict/unordered/subset/superset。
- answer rule scorer 覆盖 contains、must_not_contain、regex、JSON schema。
- aggregate scorer 覆盖 hard fail 和权重。
- compare 命令覆盖指标下降、修复、新增失败。
- Pi session JSONL parser 能把 assistant toolCall、toolResult、bashExecution 映射为 trace steps。

### 13.2 集成测试

- 用 `MockAgentAdapter` 跑 5 条 sample cases。
- 生成完整 run 目录。
- 验证 `eval_results.jsonl` 行数等于 case 数。
- 验证失败 case 出现在 `failed_cases.md`。
- 验证 compare 能对 baseline/candidate 返回正确 exit code。
- 用 Pi faux provider harness 跑 1 条 deterministic tool-call case，验证 recorder 不依赖真实 API key。

### 13.3 人工验收

- 打开 `eval_report.md`，不用看 JSON 能判断本次 run 是否可接受。
- 打开一个失败 case，能定位是工具、答案、格式、安全还是执行问题。
- 修改 mock agent 制造回归，compare 能识别。

## 14. 里程碑

### M0: Spec Review

产出：

- `agent-evals-spec.md`
- `implementation-notes.md`

验收：

- MVP 范围、数据模型、scorer、报告和验证计划经过评审。

### M1: Skeleton

产出：

- Python package skeleton。
- CLI help 可运行。
- sample cases 和 mock adapter。

验收：

- `agent-evals run --help` 可用。
- mock run 能生成空/基础结果。

### M2: Trace + Rule Scorers

产出：

- trace recorder。
- answer rule scorer。
- tool trajectory scorer。
- execution metrics scorer。

验收：

- 5 条 sample cases 可以稳定评分。

### M3: Judge + Reports

产出：

- LLM judge scorer。
- Markdown/CSV/JSONL reporter。

验收：

- 成功生成可读报告，judge 输出 schema 校验通过。

### M4: Compare + CI Gate

产出：

- baseline/candidate compare。
- gate exit code。

验收：

- 人为制造回归时 compare 失败。

### M5: Pi Adapter

产出：

- `PiAgentAdapter` 黑盒 CLI mode。
- Pi session JSONL parser。
- Pi faux provider harness integration test。

验收：

- 能对 Pi 跑一条 code-agent eval case。
- 能从 Pi session 中提取 tool calls 和 tool results。
- 能用 workspace diff 和 post-run command 判断真实 outcome。

## 15. 主要风险

- Judge 不稳定：通过规则优先、schema 校验、rubric 版本、gold case 校准降低风险。
- Trace schema 过早锁死：MVP 先做内部 schema，后续用 adapter 转 OTel。
- Agent 框架差异大：第一版只承诺 custom function 和 mock，其他框架后续 adapter 化。
- 报告信息过多：第一版报告优先展示失败原因和指标变化，不做大而全 dashboard。
- 数据集质量不足：优先沉淀真实失败、高频任务和边界 case，而不是追求数量。
- Pi 真实任务 eval 容易修改本地工作区：必须使用临时 workspace、git worktree、容器或 sandbox。
- Pi session/event schema 可能随上游变化：adapter 需要版本检测和解析失败提示。

## 16. 参考来源

- OpenAI Agent Evals: https://platform.openai.com/docs/guides/agent-evals
- LangChain Agent Evals: https://docs.langchain.com/oss/python/langchain/evals
- Braintrust scorers: https://www.braintrust.dev/docs/evaluate/write-scorers
- agentevals GitHub: https://github.com/agentevals-dev/agentevals
- Agent Evals open spec: https://agentevals.io/
- Pi GitHub: https://github.com/earendil-works/pi
- Pi session format: https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md
- 用户提供调研文档：`agent_evals_tools_research.md`
