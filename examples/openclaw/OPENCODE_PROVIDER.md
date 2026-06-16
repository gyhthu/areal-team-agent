# OpenCode Provider Setup for OpenClaw Online RL

This guide starts after `examples/openclaw/run_qwen3_8b_npu_online.sh` is already
running and the AReaL log shows the proxy gateway address.

## 1. Confirm AReaL Is Ready

Look for these log lines:

```text
Proxy servers initialized. Addresses: ['http://<host>:<proxy_worker_port>']
Proxy gateway started on <host>:<gateway_port>
Proxy gateway available at http://<host>:<gateway_port>
[wait_for_session] Worker http://<host>:<proxy_worker_port> registered in readiness queue
```

Use the **proxy gateway** address, not the raw vLLM address.

Correct:

```text
http://<host>:<gateway_port>/v1
```

Do not use the vLLM server address printed by lines like:

```text
Starting vLLM API server 0 on http://<host>:<vllm_port>
```

That address bypasses the OpenClaw RL gateway and will not record trajectories,
manage sessions, assign rewards, or feed data into training.

## 2. Configure OpenCode

Create or edit your OpenCode config. A global config usually lives at:

```bash
~/.config/opencode/opencode.json
```

Use the gateway URL from the AReaL log:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "openclaw/default",
  "provider": {
    "openclaw": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenClaw RL",
      "options": {
        "baseURL": "http://<host>:<gateway_port>/v1"
      },
      "models": {
        "default": {
          "name": "AReaL Qwen3-8B"
        }
      }
    }
  }
}
```

Example:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "openclaw/default",
  "provider": {
    "openclaw": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenClaw RL",
      "options": {
        "baseURL": "http://7.242.108.93:3630/v1"
      },
      "models": {
        "default": {
          "name": "AReaL Qwen3-8B"
        }
      }
    }
  }
}
```

## 3. Add the Provider Credential

Run:

```bash
opencode auth login
```

Choose `Other`, then use:

```text
provider id: openclaw
api key: any-placeholder-key
```

The key only needs to be non-empty. The gateway automatically creates and routes
backend sessions, so users do not need to call `/rl/start_session` or manually
rotate session keys.

## 4. Use OpenCode Normally

Start OpenCode and select:

```text
openclaw/default
```

Then use OpenCode as usual. The gateway will:

1. Auto-start a session for the incoming provider key.
1. Route requests to a ready AReaL proxy worker.
1. Record model interactions as trajectories.
1. End idle sessions automatically after `AREAL_PROVIDER_IDLE_TIMEOUT`.
1. Score the trajectory using the configured judge, or rule reward fallback when
   no judge URL is configured.
1. Feed completed trajectories into the training loop.

The default Qwen3-8B script sets:

```text
AREAL_PROVIDER_IDLE_TIMEOUT=300
AREAL_PRM_RULE_REWARD_MODE=basic
```

So if the user stops interacting, the session is finalized after about five
minutes and receives a basic rule-based reward unless an LLM judge is configured.

For Qwen3-8B, the launch script also sets `rollout.openai.engine_max_tokens` to
the same value as `vllm.max_model_len`. This protects the gateway when clients
request a very large output budget, such as `max_tokens=32000`, while the served
model context is smaller, such as `16384`.

## 5. Quick Smoke Test

Before opening OpenCode, you can test the gateway directly:

```bash
curl -sS http://<host>:<gateway_port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer any-placeholder-key' \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "max_tokens": 32
  }'
```

After the first request, the AReaL log should show gateway/session activity.

## 6. Troubleshooting

If OpenCode cannot find the model:

1. Make sure the provider id in `opencode auth login` is exactly `openclaw`.
1. Make sure the config provider key is also exactly `openclaw`.
1. Make sure the selected model is `openclaw/default`.
1. Restart OpenCode after editing `opencode.json`.

If requests reach vLLM but AReaL does not record trajectories, you are probably
using the raw vLLM URL. Switch back to:

```text
http://<host>:<gateway_port>/v1
```

If the session ends too early or too late, adjust before launching training:

```bash
export AREAL_PROVIDER_IDLE_TIMEOUT=300
export AREAL_PROVIDER_IDLE_CHECK_INTERVAL=30
bash examples/openclaw/run_qwen3_8b_npu_online.sh
```
