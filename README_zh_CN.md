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

### skills

`OpenAI 文本 Skill 选项` 节点可以加载服务器本地的标准 `SKILL.md` 目录，并通过 `skill_options` 连接到 `OpenAI 文本 API`。`skills.paths` 配置 Skill 根目录；相对路径以本插件目录为基准，未配置时默认扫描 `skills/`。只有显式设置 `skills.allow_call=true` 才允许进入 Skill 调用链；文件发现和只读加载默认可用。

Skill 使用带首次必载门的渐进式披露。新 Skill Session 中，模型必须先完整读取并校验 `SKILL.md`，否则最终文本会被拒绝；加载后才按需读取允许的 reference。持久 Session 可以复用相同 Skill hash 的已加载上下文，复用来源会明确记录在 `Skill Trace` 中。`scripts/` 永远不会执行，也不提供 shell、子进程、网络或文件写入能力。

两种协议共用同一个内部 `pi_skill_agent` 循环和同一组只读工具。新 Session 只暴露 `load_skill`，加载后只暴露 `read_skill_file`。与 Pi Agent 一致，循环省略 `tool_choice` 并使用 provider 默认策略；在首次加载成功前，宿主侧 `skill_not_loaded` gate 会拒绝最终文本。`openai-responses` 使用自定义 `function_call`/`function_call_output` Items 传输，`openai-completions` 使用 assistant function tool call 与 `role: tool` message。这是插件本地协议，不是 OpenAI 官方 Skills 或 shell attachment；Responses 失败时不会回退到 Completions。

支持的 reference 扩展名为 `.md`、`.txt`、`.json`、`.yaml` 和 `.yml`。路径、文件大小、工具轮数、调用次数和累计披露量使用固定的内部限制；公开的 `skills` 配置只包含 `paths` 与 `allow_call`。Skill 模式下，高级参数不能覆盖 Skill 工具、tool choice、状态 ID、container 或 Skill attachment。公共 Skill Options 协议版本保持为 `schema_version: 1`。

Skill 调用时，`Conversation` 是完整且确定性脱敏的 Session ledger，不是摘要。每个 `model_request.payload` 和 `model_response.response` 都保留实际协议的原始结构、字段、Item 顺序、ID、reasoning、message、function call 以及 provider 扩展字段；工具生命周期和实际 continuation Item 与它们一起记录。Session 续接使用的已提交 `provider_context` 与 UI 输出均派生自同一个 `turn_commit` ledger 事件，避免出现第二份不一致的历史。`Skill 调用记录` 是独立的 JSON 审计摘要，包含所选 Skill 名称/hash、加载状态与来源、实际 tool-choice 策略、兼容重试次数、工具轮数、工具调用次数、成功读取的相对路径和错误，但不包含文件正文或真实绝对路径，也不会加入后续模型上下文。执行失败时 ComfyUI 无法生成节点输出，因此同一份 trace 摘要会附加到异常信息中。

### proxy

`proxy` 支持配置http代理，适用于特殊网络环境

## 高级使用说明

API节点支持 `Config Options` 和 `Proxy Options`。均可以通过前台节点配置参数实现覆盖配置文件参数的功能。

