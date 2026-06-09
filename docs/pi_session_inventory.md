# Pi Session JSONL Inventory

调查日期：2026-06-09  
上游格式文档：<https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md>  
样本来源：HuggingFace dataset `badlogicgames/pi-mono`，`repo_type="dataset"`  
样本文件：`2026-03-02T20-53-11-495Z_9084660c-d6dd-42f7-8892-91d565b75da5.jsonl`  
固定 fixture：`cases/fixtures/pi_session_sample.jsonl`

选择过程：主 dataset `badlogicgames/pi-mono` 有 627 个 `.jsonl` 文件，备份 dataset `aaaaliou/pi-mono` 有 146 个 `.jsonl` 文件。按文件大小取主 dataset 前 5 个候选后统计行数，最终选择行数最多的样本：495 行，约 3.4 MB。

可复现命令：

```bash
python3 scripts/inspect_pi_session.py cases/fixtures/pi_session_sample.jsonl
```

## 1. 消息类型清单

总行数：495。

顶层 `type` 计数：

| type | count |
| --- | ---: |
| `message` | 489 |
| `session_info` | 2 |
| `session` | 1 |
| `model_change` | 1 |
| `thinking_level_change` | 1 |
| `compaction` | 1 |

`message.role` 计数：

| role | count |
| --- | ---: |
| `assistant` | 235 |
| `toolResult` | 234 |
| `user` | 19 |
| `bashExecution` | 1 |

内容块 `content[].type` 计数：

| content type | count |
| --- | ---: |
| `text` | 265 |
| `toolCall` | 234 |
| `thinking` | 225 |
| `image` | 1 |

assistant 内容块计数：

| assistant content type | count |
| --- | ---: |
| `toolCall` | 234 |
| `thinking` | 225 |
| `text` | 12 |

assistant 工具名分布：

| tool name | count |
| --- | ---: |
| `bash` | 97 |
| `read` | 80 |
| `edit` | 55 |
| `write` | 2 |

## 2. Assistant 工具调用

确认：assistant 工具调用是 `message.role="assistant"` 记录的 `message.content[]` 里的 `type="toolCall"` 块。真实字段包括：

- `id`
- `name`
- `arguments`
- `partialJson`

所有 234 个 `toolCall` 都可从 `content[]` 中枚举出来。

真实例子 1，`bash`，line 7：

```json
{
  "type": "toolCall",
  "id": "call_mitCOadp...fa5078",
  "name": "bash",
  "arguments": {
    "command": "gh issue view 1720 --json title,body,comments,..."
  },
  "partialJson": "{\"command\":\"gh issue view 1720 --json ...\"}"
}
```

真实例子 2，`read`，line 15：

```json
{
  "type": "toolCall",
  "id": "call_10vgDMUT...ca61",
  "name": "read",
  "arguments": {
    "path": "/Users/badlogic/workspaces/pi-mono/packages/coding-agent/src/core/extensions/loader.ts"
  },
  "partialJson": "{\"path\":\"/Users/badlogic/workspaces/.../loader.ts\"}"
}
```

## 3. ToolResult

确认：`toolResult` 是独立的 `type="message"` 记录，角色为 `message.role="toolResult"`。

本样本中 `toolResult` 字段出现次数：

| field | count |
| --- | ---: |
| `role` | 234 |
| `toolCallId` | 234 |
| `toolName` | 234 |
| `content` | 234 |
| `isError` | 234 |
| `timestamp` | 234 |
| `details` | 69 |

结论：

- `toolCallId` 是和 assistant `toolCall.id` 对齐的主键。
- `toolName` 可用于 sanity check，但不是唯一主键。
- `details` 是可选字段；这份样本里 69/234 条有，常见用途是截断信息，例如 `truncation`、`fullOutputPath`。
- 234/234 个 `toolResult.toolCallId` 都能匹配到某个 assistant `toolCall.id`。

真实例子，line 8：

```json
{
  "role": "toolResult",
  "toolCallId": "call_mitCOadp...fa5078",
  "toolName": "bash",
  "content": [{"type": "text", "text": "{\"assignees\":[],\"author\":{..."}],
  "isError": false,
  "timestamp": 1772484797322
}
```

## 4. Bash 记录方式

### 是否有独立 bashExecution

有。样本中出现 1 条独立 `bashExecution`，它也是 `type="message"`，但 `message.role="bashExecution"`。

真实字段：

| field | observed |
| --- | --- |
| `role` | `bashExecution` |
| `command` | yes |
| `output` | yes |
| `exitCode` | yes |
| `cancelled` | yes |
| `truncated` | yes |
| `timestamp` | yes |
| `excludeFromContext` | yes |
| `toolCallId` | no |

真实例子，line 265：

```json
{
  "role": "bashExecution",
  "command": "rm packages/coding-agent/examples/extensions/template-accelerator.ts",
  "output": "",
  "exitCode": 0,
  "cancelled": false,
  "truncated": false,
  "excludeFromContext": true
}
```

### bash 是否同时记为 toolCall + toolResult

模型调用的 bash 是标准 assistant `toolCall(name="bash")` + `toolResult(toolName="bash")`：

- `bash` toolCall：97 条
- `bash` toolResult：97 条
- 97/97 个 bash toolResult 都能用 `toolResult.toolCallId == toolCall.id` 对上

独立 `bashExecution` 只有 1 条，且没有 `toolCallId`，也没有附近可对应的相同 command 的 `toolCall`。它的上下文是：

| line | role | note |
| ---: | --- | --- |
| 262 | `assistant` | `toolCall(name="bash")`: `git status --short` |
| 263 | `toolResult` | 对应 line 262 的 bash toolCall |
| 264 | `assistant` | 最终答复文本，无 toolCall |
| 265 | `bashExecution` | `rm packages/.../template-accelerator.ts` |
| 266 | `user` | 用户后续指令 |

明确结论：

- 在这份真实 session 中，同一次模型发起的 bash 执行没有被同时记录成 `bashExecution`。
- `bashExecution` 是另一类独立记录，没有观察到可和 assistant bash `toolCall` 对应的字段。
- 因此当前 parser 不应默认把 `bashExecution` 当作 assistant `name="bash"` toolCall 的权威副本来去重。
- 对模型工具调用，可靠配对字段是 `toolCall.id == toolResult.toolCallId`。
- 对独立 `bashExecution`，本样本没有可用于跨记录去重的 `toolCallId`；如未来样本出现重复，只能在看到显式关联字段后再做去重，单靠 `command` 字符串匹配不够安全。

## 5. Token / Cost

确认：assistant 消息上有 `message.usage`。本样本 235/235 条 assistant 消息都有 usage。

`usage` 字段：

| usage field | count |
| --- | ---: |
| `input` | 235 |
| `output` | 235 |
| `cacheRead` | 235 |
| `cacheWrite` | 235 |
| `totalTokens` | 235 |
| `cost` | 235 |

`usage.cost` 字段：

| cost field | count |
| --- | ---: |
| `input` | 235 |
| `output` | 235 |
| `cacheRead` | 235 |
| `cacheWrite` | 235 |
| `total` | 235 |

## 6. 树结构

结论：这份 session 的有效对话链是单链，没有观察到分叉。

细节：

- 每条非 header 记录都有 `id`。
- 除第一行 `session` header 外，其余 494 条都有 `parentId`。
- 非根节点中没有发现“同一父节点多个子节点”的分叉点。
- `parentId=null` 下有 2 个 root-level 记录：`session` header 和 `model_change`。这更像文件级元数据 + 对话链起点，不代表对话分叉。

## 7. 映射核对

| Pi 来源 | 我假设的 TraceStep | 核对 | 说明 / 调整 |
| --- | --- | --- | --- |
| user message | 存进 trace input，不生成 step | ✅确认 | `message.role="user"` 共 19 条。content 可能含 `text`，样本中还出现 1 个 `image` content type；trace input 要保留多 content block。 |
| assistant message text | `type=llm` | ⚠️需调整 | 只有 12 个 assistant `text` block，但有 225 个 `thinking` block 和 234 个 `toolCall` block。建议只把 assistant `text` block 映射为可见 `llm` 文本；`thinking` 作为可选 debug/reasoning metadata，避免把无文本的 tool-call assistant 消息强行记成空 `llm` step。 |
| assistant toolCall | `type=tool_call(name + arguments)` | ✅确认 | `toolCall` 块在 assistant `content[]` 中，字段为 `id/name/arguments/partialJson`。工具名分布为 bash/read/edit/write。 |
| toolResult message | `type=observation(content / details / isError)` | ✅确认 | `toolResult.toolCallId` 对 `toolCall.id`，`details` 可选。建议 observation 保留 `toolName`、`content`、`isError`、`details`、`timestamp`。 |
| bashExecution message | `type=tool_call + observation`，tool name 固定 bash | ⚠️需调整 | 样本有独立 `role="bashExecution"`，但它没有 `toolCallId`，也不是模型 bash toolCall 的重复副本。可将它映射为独立 local shell execution step，生成 synthetic id；不要用它覆盖或去重 assistant `name="bash"` toolCall。 |
| message usage | trace metrics 的 token / cost | ✅确认 | assistant `usage` 全量存在，含 token 和 cost breakdown。建议按 assistant message 汇总到 trace metrics，也保留 per-step usage。 |

## Parser 前置建议

1. 第一版 parser 先按 `message.role` 分流，而不是只看顶层 `type`；真实消息类型都包在 `type="message"` 下。
2. assistant `content[]` 要按 block 遍历；一个 assistant message 可以同时有 `thinking` 和 `toolCall`。
3. 工具调用主链用 `toolCall.id` / `toolResult.toolCallId` 配对。
4. `toolResult.details` 和 `bashExecution` 都应按可选字段处理。
5. `bashExecution` 暂时作为独立执行事件处理；不要实现“bashExecution 覆盖 bash toolCall”的去重假设，除非后续 fixture 观察到显式关联字段。
