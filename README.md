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

When `stream` is enabled, the model response is streamed to the client as it is generated using server-sent events (SSE). This also works with `skill_options`. In Skill mode, text from rounds that request a tool is treated as a candidate and is discarded after the tool call; only the first round that completes without tool calls is promoted to the final answer. The final `end` event is authoritative and reconciles any difference between streamed deltas and the returned node text. Terminal states distinguish normal completion, truncation, and errors.

### skills

The `OpenAI Text Skill Options` node discovers local skills organized around `SKILL.md`. Select a Skill and connect the node to `OpenAI Text API` through `skill_options`. Configure Skill locations with `skills.paths`; relative paths resolve from this plugin directory, and the default location is `skills/`. Set `skills.allow_call` to `true` to enable Skill calls.

When a Skill is selected, the model follows its instructions and reads bundled text resources as needed. Script files can be read as text but are never executed. Skill mode cannot run commands, edit or write files, or independently access the network. The target model service must support tool calls.

Skill streaming is deliberately conservative: candidate text is buffered per model round, tool-call rounds are cleared, and final text is displayed only after a no-tool round completes. This avoids showing an intermediate answer that the model later revises after reading a Skill resource.

## Advanced usage instructions

API nodes support `Config Options` and `Proxy Options`. Both can be used to override configuration file parameters by configuring parameters through the front-end node.
