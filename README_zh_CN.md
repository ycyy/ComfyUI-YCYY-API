# ComfyUI-YCYY-API
[**English**](README.md) | [**中文**](README_zh_CN.md)

在 ComfyUI 中自由的调用 API

## 使用说明
* 在 ComfyUI 中进入你的 custom_nodes 文件夹，打开终端并运行以下命令：
```
git clone https://github.com/ycyy/ComfyUI-YCYY-API.git
```
* 复制 `config.json.example` 文件并重命名为 `config.json`。然后修改对应的 `base_url` 和 `api_key`

* 启动 `ComfyUI` 找到 `YCYY/API` 节点目录开始体验吧

## 配置文件说明

### gemini-image

`base_url` 同时支持官方和第三方兼容接口根据实际情况配置，路径以 `v1beta/models` 结尾。`models` 支持 `gemini-2.5-flash-image`、`gemini-3-pro-image-preview` 模型。如果使用第三方接口，模型名称根据实际情况修改。

### ollama-vlm 和 ollama-llm
`base_url` 同时支持本地和 `ollama` 云服务接口，根据实际情况配置，接口为 `OpenAI` 兼容格式。配置文件的模型为官方云服务支持的模型，可以根据需要进行修改。如果调用本地接口没有配置 `api_key` 则可以不填写该选项。

### modelscope-image

魔搭图片生成接口只需要填写对应的 `api_key` 其他参数保持不变即可

### openai-text

`openai-text` 使用数组配置多个 OpenAI 或兼容接口。每项包含 `api-name`、`base_url`、`api_key`、`timeout`、`api_protocol` 和 `models`。`api_protocol` 支持 `openai-completions` 与 `openai-responses`。

`base_url` 可以填写根地址（例如 `https://api.openai.com/v1`），节点会按协议追加 `/chat/completions` 或 `/responses`；也可以直接填写对应完整端点。完整端点与协议不匹配时会报错。`API Config Options` 只覆盖显式提供的值：`base_url` 和 `api_key` 留空、`api_protocol=inherit`、`timeout=0` 时，使用所选 API 的原配置。

节点支持文本、图像和视频输入。视频会按当前协议的格式直接发送；如果目标 API 或协议不支持视频，接口错误会转换为明确的视频不支持提示。`files` 输入当前版本暂不支持。`OpenAI 文本高级选项（JSON）` 节点接受协议或 API 特有参数，例如 `{"temperature":0.7,"max_output_tokens":4096}`。JSON 参数不能覆盖 `model`、`messages`、`input`、`instructions`、`stream`、`api_key`、`base_url` 或 `timeout` 等请求字段。

启用 `stream` 后，模型响应会在生成过程中通过服务器发送事件（SSE）流式传输到客户端。当前流式模式不能与 `skill_options` 同时启用。

### skills

`OpenAI 文本 Skill 选项` 节点可以发现以 `SKILL.md` 组织的本地 Skill。选择 Skill 后，通过 `skill_options` 连接到 `OpenAI 文本 API` 节点。使用 `skills.paths` 配置 Skill 位置；相对路径以本插件目录为基准，默认位置为 `skills/`。将 `skills.allow_call` 设置为 `true` 后即可启用 Skill 调用。

运行时，模型会遵循所选 Skill 的说明，并按需读取其中的文本资源。脚本文件只能作为文本读取，不会被执行。Skill 模式不能运行命令、编辑或写入文件，也不能自行访问网络。目标模型服务需要支持工具调用。

### proxy

`proxy` 支持配置http代理，适用于特殊网络环境

## 高级使用说明

API节点支持 `Config Options` 和 `Proxy Options`。均可以通过前台节点配置参数实现覆盖配置文件参数的功能。

