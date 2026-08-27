# ComfyUI-YCYY-API
[**English**](README.md) | [**中文**](README_zh_CN.md)

Freely call APIs in ComfyUI

## Instructions for use
* In ComfyUI, go to your custom_nodes folder, open a terminal and run the following command:
```
git clone https://github.com/ycyy/ComfyUI-YCYY-API.git
```
* Copy the `config.json.example` file and rename it to `config.json`. Then modify the corresponding `base_url` and `api_key`.

* Start `ComfyUI` and find the `YCYY/API` node directory to start your experience.

## Configuration file description

### gemini-image

`base_url` supports both official and third-party compatible interfaces, which can be configured according to actual conditions. The path ends with `v1beta/models`. `models` supports `gemini-2.5-flash-image` and `gemini-3-pro-image-preview` models. If using a third-party interface, modify the model name according to actual conditions.

### ollama-vlm and ollama-llm
`base_url` supports both local and `ollama` cloud service interfaces, which can be configured according to actual conditions. The interface is in `OpenAI` compatible format. The models in the configuration file are those supported by the official cloud service and can be modified as needed. If calling a local interface without configuring `api_key`, this option can be left blank.

### modelscope-image

The ModelScope image generation interface only requires you to fill in the corresponding `api_key`. Other parameters remain unchanged.

### openai-text

`openai-text` is an array of OpenAI or compatible API configurations. Each item contains `api-name`, `base_url`, `api_key`, `timeout`, `api_protocol`, and `models`. Supported protocols are `openai-completions` and `openai-responses`.

`base_url` may be a root URL such as `https://api.openai.com/v1`; the node appends `/chat/completions` or `/responses` according to the protocol. It may also be a complete matching endpoint. A complete endpoint that does not match the selected protocol is rejected. `API Config Options` overrides only explicitly supplied values: empty `base_url` and `api_key`, `api_protocol=inherit`, and `timeout=0` use the selected API's original configuration.

The node supports text, image, and video inputs. Video is sent directly using the current protocol's content format; if the target API or protocol does not support video, its error is surfaced as an explicit video-unsupported message. File input is not implemented in this version. The `OpenAI Text Advanced Options` node accepts a JSON object for protocol/API-specific parameters, for example `{"temperature":0.7,"max_output_tokens":4096}`. JSON options cannot override request fields such as `model`, `messages`, `input`, `instructions`, `stream`, `api_key`, `base_url`, or `timeout`.

### skills

The `OpenAI Text Skill Options` node loads standard local `SKILL.md` directories and connects to `OpenAI Text API` through `skill_options`. Configure Skill roots with `skills.paths`; relative paths resolve from this plugin directory, and the default is `skills/`. Skill calls are disabled unless `skills.allow_call` is explicitly set to `true`; discovery and read-only loading remain available by default.

Skills use progressive disclosure with a mandatory first load. In a new Skill session the model must read and validate the complete `SKILL.md` before final text is accepted; after that, allowlisted reference files are read only when needed. A persistent session may reuse the same loaded Skill hash, which is reported explicitly in `Skill Trace`. Files in `scripts/` are never executed, and shell, subprocess, network, and file-write capabilities are not provided.

Both protocols use the same internal `pi_skill_agent` loop and the same read-only tools. A new session exposes only `load_skill`; after loading it exposes only `read_skill_file`. Like Pi Agent, the loop omits `tool_choice` and uses the provider default; the host-side `skill_not_loaded` gate rejects final text until the initial load succeeds. `openai-responses` transports calls as custom `function_call`/`function_call_output` items, while `openai-completions` uses assistant function tool calls and `role: tool` messages. This is a local plugin protocol, not an official OpenAI Skills or shell attachment. Responses failures never fall back to Completions.

Supported reference extensions are `.md`, `.txt`, `.json`, `.yaml`, and `.yml`. Paths, file sizes, tool rounds, call counts, and disclosed bytes have fixed internal limits; the public `skills` configuration contains only `paths` and `allow_call`. In Skill mode, advanced options cannot override Skill tools, tool choice, state IDs, containers, or Skill attachments. The public Skill Options protocol remains `schema_version: 1`.

For Skill calls, `Conversation` is a complete, deterministically redacted session ledger, not a summary. Each `model_request.payload` and `model_response.response` retains the actual protocol-native structure, fields, item ordering, IDs, reasoning, messages, function calls, and provider extensions. Tool lifecycle and continuation items are recorded alongside them. The committed `provider_context` used for Session continuation is derived from the same `turn_commit` ledger event used by the UI, preventing a second divergent history. `Skill Trace` is a separate JSON audit summary containing the selected Skill name/hash, load state and source, actual tool-choice strategy, compatibility retry count, tool rounds, tool call count, successfully read relative paths, and errors. Trace never includes file contents or real absolute paths and is not added to future model context. If execution fails, ComfyUI cannot produce node outputs, so the same trace summary is appended to the raised error message.

## Advanced usage instructions

API nodes support `Config Options` and `Proxy Options`. Both can be used to override configuration file parameters by configuring parameters through the front-end node.
