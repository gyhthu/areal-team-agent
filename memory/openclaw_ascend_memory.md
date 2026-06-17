# OpenClaw on AReaL Ascend Memory

Last updated: 2026-06-17
Branch: `ascend-v1.0.1-openclaw`
Remote: `gyhthu/areal-team-agent`

## Goal

Port the core OpenClaw-RL online training experience onto the official AReaL
Ascend branch:

1. Start AReaL online RL on Ascend NPU.
2. Expose an OpenAI-compatible gateway URL.
3. Let OpenCode/OpenClaw use that URL as a normal model provider.
4. Automatically collect multi-turn trajectories.
5. Automatically end idle sessions.
6. Assign trajectory-level reward with an LLM judge, or a simple rule fallback.
7. Feed completed trajectories back into AReaL PPO training.

The intended user experience is: configure only `baseUrl` and model/provider in
OpenCode/OpenClaw, then use it normally. Users should not manually start
sessions, rotate session keys, or send explicit end-session signals.

## Main Files Changed

Core OpenAI/proxy path:

- `areal/experimental/openai/types.py`
- `areal/experimental/openai/client.py`
- `areal/experimental/openai/tool_call_parser.py`
- `areal/experimental/openai/proxy/server.py`
- `areal/experimental/openai/proxy/proxy_rollout_server.py`
- `areal/experimental/openai/proxy/proxy_gateway.py`
- `areal/utils/stats_logger.py`
- `areal/infra/utils/http.py`

OpenClaw examples and scripts:

- `examples/openclaw/train.py`
- `examples/openclaw/config_qwen25_1_5b_npu.yaml`
- `examples/openclaw/run_qwen25_1_5b_npu_online.sh`
- `examples/openclaw/config_qwen3_8b_npu.yaml`
- `examples/openclaw/run_qwen3_8b_npu_online.sh`
- `examples/openclaw/OPENCODE_PROVIDER.md`

## Implemented Behavior

### Automatic next_state tracking

`SessionData.attach_next_state_from_request(...)` in
`areal/experimental/openai/proxy/server.py` extracts state-like payloads from the
next request and attaches them to the previous pending interaction. This lets
the training data capture transition-style information without the client
manually managing trajectory objects.

### Direct provider gateway

`areal/experimental/openai/proxy/proxy_gateway.py` supports direct provider use:

- On first OpenAI-compatible request, the gateway takes a ready worker and calls
  `/rl/start_session`.
- The route key is inferred from request/auth/provider context.
- Subsequent requests for the same route are forwarded to the same worker and
  AReaL session.
- The gateway rewrites worker authorization internally, so external clients only
  see a normal provider URL.

### Idle timeout session ending

Since OpenCode/OpenClaw users will not explicitly send a session-done signal,
the gateway auto-ends inactive sessions:

- `AREAL_PROVIDER_IDLE_TIMEOUT`, default `300`
- `AREAL_PROVIDER_IDLE_CHECK_INTERVAL`, default auto-derived or overridden

When idle time expires, the gateway calls worker `/rl/end_session`, which
notifies `_OnlineAgent` and makes the trajectory available for training/reward.

### LLM judge and rule reward fallback

`areal/experimental/openai/proxy/proxy_rollout_server.py` now supports
trajectory-level reward:

- If `AREAL_PRM_JUDGE_BASE_URL` is configured, it calls an OpenAI-compatible LLM
  judge with a simple trajectory prompt.
- If judge URL is empty or judge fails, it assigns a basic rule reward.
- Logs include messages such as `Rule reward assigned trajectory reward ...`.

This is intentionally simple: the priority is to reproduce the OpenClaw-RL
feedback loop of trajectory collection -> scoring -> training.

### OpenCode/OpenClaw provider docs

`examples/openclaw/OPENCODE_PROVIDER.md` documents:

- Start the Qwen3 NPU online script.
- Use the printed gateway URL as provider `baseUrl`.
- Configure model as `default`.
- Use any placeholder API key if the client requires one.
- How to debug idle timeout, tool parser, and token budget issues.

### Ascend compatibility fixes

Important Ascend-branch compatibility fixes:

- Config schema uses `allocation_mode` and `rollout.openai`.
- Missing `trackio` no longer crashes import eagerly.
- Pydantic v1/v2 differences are handled by helper wrappers.
- `validate_admin_api_key` helper was added for proxy compatibility.
- `lora_name` is not passed to `ArealOpenAI` in this branch.

## Qwen3-8B NPU Default Script

Primary script:

```bash
examples/openclaw/run_qwen3_8b_npu_online.sh
```

Current conservative defaults:

```bash
MODEL_PATH=Qwen/Qwen3-8B
ALLOCATION_MODE=vllm:d1p1t4+d4p1t1
TRAIN_BATCH_SIZE=4
MAX_CONCURRENT_ROLLOUTS=4
MAX_MODEL_LEN=32768
MAX_NEW_TOKENS=2048
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_GPU_MEMORY_UTILIZATION=0.7
TOOL_CALL_PARSER=qwen25
AREAL_PROVIDER_IDLE_TIMEOUT=300
AREAL_PRM_RULE_REWARD_MODE=basic
```

Why `TRAIN_BATCH_SIZE=4`:

- `vllm:d1p1t4+d4p1t1` means generation uses 4-way TP and training uses 4-way
  data parallelism.
- AReaL's DP dispatcher requires at least one item per DP shard.
- If batch size is 2, training fails with:

```text
ValueError: Number of items (2) must be >= K (4).
```

## Tool Calling Notes

The inference backend is vLLM, but AReaL's OpenAI client reuses SGLang parser
classes for tool-call and reasoning parsing:

- `FunctionCallParser`
- `ReasoningParser`

For Qwen3-8B, official Qwen SGLang deployment examples use:

```bash
--tool-call-parser qwen25
--reasoning-parser qwen3
```

So Qwen3 config uses:

```yaml
rollout:
  openai:
    tool_call_parser: qwen25
    reasoning_parser: qwen3
```

### Qwen XML fallback

OpenCode showed raw text like:

```text
<tool_call>...</tool_call>
```

instead of actually executing tools. This means the model tried to call a tool,
but the server returned it as content instead of structured OpenAI `tool_calls`.

To harden this, `areal/experimental/openai/tool_call_parser.py` now:

1. First tries the SGLang parser.
2. If that fails or does not detect a call, checks for Qwen XML-style
   `<tool_call>...</tool_call>`.
3. Parses the JSON payload.
4. Converts it to OpenAI `tool_calls`.
5. Removes the raw XML block from assistant content.

This should not affect ordinary chat responses.

## Debugging Tool Calls

Set:

```bash
export AREAL_OPENAI_DEBUG=1
```

Then restart:

```bash
bash examples/openclaw/run_qwen3_8b_npu_online.sh
```

After reproducing in OpenCode, extract relevant logs:

```bash
grep -E "OpenAI proxy request|generated output|parser result|Qwen XML fallback|SGLang parser|Failed to parse Qwen" your_log_file.log
```

Interpretation:

- `has_tools=False`
  - OpenCode did not send tool definitions to the provider.
  - Check OpenCode provider/model configuration.

- `has_tools=True, has_xml_tool_call=True, Qwen XML fallback parsed N tool call(s)`
  - AReaL parsed the tool call.
  - If OpenCode still displays raw XML, inspect streaming response format or
    OpenCode compatibility.

- `Failed to parse Qwen tool call payload`
  - Model emitted malformed or truncated JSON inside `<tool_call>`.
  - Check max token budget, stop reason, and prompt/tool template.

- `has_tools=True, has_xml_tool_call=False`
  - Model did not produce a tool call in that turn.
  - Check chat template/tool prompt effectiveness.

## Common Runtime Issues Already Seen

### `ModuleNotFoundError: No module named 'trackio'`

Fixed by lazy/missing-config handling in `areal/utils/stats_logger.py`.

### Ascend schema rejects config fields

Old fields like `rollout.backend` and `rollout.agent` did not fit the official
Ascend branch schema. Fixed by using `allocation_mode` and `rollout.openai`.

### Qwen3 OOM or bad process layout

Using multiple single-card vLLM instances was too aggressive. Qwen3-8B default
now uses:

```text
vllm:d1p1t4+d4p1t1
```

This gives one 4-card TP vLLM generation group and 4-card DP training.

### OpenCode `max_tokens` too large

OpenCode may request a large budget. Config caps:

```yaml
rollout.openai.engine_max_tokens: ${vllm.max_model_len}
```

Qwen3 default context was raised to 32k:

```yaml
vllm.max_model_len: 32768
sglang.context_length: 32768
```

### `Unsupported tool_call_parser: qwen3`

Do not use `qwen3` as tool parser. Use:

```bash
TOOL_CALL_PARSER=qwen25
```

Keep:

```yaml
reasoning_parser: qwen3
```

### `Number of items (2) must be >= K (4)`

Training DP size is 4, but PPO batch had only 2 trajectories. Fixed by default:

```bash
TRAIN_BATCH_SIZE=4
```

If changing allocation mode, keep `TRAIN_BATCH_SIZE >= train DP size`.

## Recent Commit Trail

Most relevant commits on `ascend-v1.0.1-openclaw`:

```text
e2b6eaa Add OpenAI tool call diagnostics
5b2cf91 Match Qwen3 online batch to train DP
66bef82 Parse Qwen XML tool call fallback
414e4c0 Use Qwen25 tool parser for Qwen3
2ac77b0 Raise Qwen3 8B NPU context to 32k
2f15a53 Cap OpenCode token budget by engine context
fac7bf8 Document OpenCode provider setup for OpenClaw
e205136 Avoid passing lora name to OpenAI client
707bf85 Harden proxy pydantic compatibility
b95911b Support Ascend openai proxy config field
cac1536 Add admin key validation helper for proxy
7688bbc Tune Qwen3 8B NPU script for 4-way vLLM TP
cdce716 Handle missing trackio config on Ascend branch
266923b Align OpenClaw NPU configs with Ascend schema
2eff42a Port OpenClaw online RL support to Ascend branch
```

## Current Open Question

As of the last debugging round, OpenCode still displayed raw
`<tool_call>...</tool_call>` in the UI. The next step is to rerun with:

```bash
export AREAL_OPENAI_DEBUG=1
```

and inspect whether:

1. OpenCode actually sends `tools`.
2. AReaL parses tool calls.
3. The streaming response sends structured `tool_calls` in a format OpenCode
   accepts.

Do not guess this part from normal logs; use the debug lines above.
