from comfy_api.latest import ComfyExtension, io

class ConfigOptions(io.ComfyNode):
    """
    这个节点用于覆盖 API 的配置参数（base_url、api_key、timeout）
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Config_Options",
            display_name="Config Options",
            category="YCYY/API/utils",
            inputs=[
                io.String.Input(
                    id="base_url",
                    multiline=True,
                    tooltip="Override the API base URL"
                ),
                io.String.Input(
                    id="api_key",
                    default="",
                    multiline=True,
                    tooltip="Override the API key"
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
                    tooltip="Override configuration options"
                )
            ],
            description="This node provides configuration override options (base_url, api_key, timeout)."
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