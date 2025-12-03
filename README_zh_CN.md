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

### proxy

`proxy` 支持配置http代理，适用于特殊网络环境

## 高级使用说明

API节点支持 `Config Options` 和 `Proxy Options`。均可以通过前台节点配置参数实现覆盖配置文件参数的功能。

