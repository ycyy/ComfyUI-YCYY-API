from comfy_api.latest import ComfyExtension, io

class OllamaConfigOptions(io.ComfyNode):
    """
    这个节点用于覆盖 Ollama API 的配置参数（base_url、api_key、timeout）
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Ollama_Config_Options",
            display_name="Ollama Config Options",
            category="YCYY/API/utils",
            inputs=[
                io.String.Input(
                    id="base_url",
                    default="http://localhost:11434/api/chat",
                    multiline=True,
                    tooltip="Override the base URL for Ollama API"
                ),
                io.String.Input(
                    id="api_key",
                    default="",
                    multiline=True,
                    tooltip="Override the API key for Ollama API (optional)"
                ),
                io.Int.Input(
                    id="timeout",
                    default=120,
                    min=1,
                    max=600,
                    tooltip="Override the request timeout in seconds"
                )
            ],
            outputs=[
                io.AnyType.Output(
                    id="config_options",
                    display_name="config_options",
                    tooltip="Configuration override for Ollama API"
                )
            ],
            description="This node provides configuration override options for Ollama API (base_url, api_key, timeout)."
        )

    @classmethod
    def execute(cls, base_url, api_key, timeout) -> io.NodeOutput:
        # 验证和清理输入
        base_url = base_url.strip() if base_url else ""
        api_key = api_key.strip() if api_key else ""

        config_options = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout
        }

        return io.NodeOutput(config_options)