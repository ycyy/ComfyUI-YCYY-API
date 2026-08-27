from comfy_api.latest import ComfyExtension, io


def build_config_options(base_url, api_key, api_protocol, timeout):
    """Return only values that explicitly override a target API config."""
    base_url = base_url.strip() if isinstance(base_url, str) else ""
    api_key = api_key.strip() if isinstance(api_key, str) else ""
    options = {}
    if base_url:
        options["base_url"] = base_url
    if api_key:
        options["api_key"] = api_key
    if api_protocol and api_protocol != "inherit":
        options["api_protocol"] = api_protocol
    if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout > 0:
        options["timeout"] = timeout
    return options


class ConfigOptions(io.ComfyNode):
    """
    这个节点用于覆盖 API 的配置参数（base_url、api_key、api_protocol、timeout）
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_API_Config_Options",
            display_name="API Config Options",
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
                io.Combo.Input(
                    id="api_protocol",
                    options=["inherit", "openai-completions", "openai-responses", "anthropic-messages"],
                    default="inherit",
                    tooltip="Override the API protocol, or inherit the selected API configuration"
                ),
                io.Int.Input(
                    id="timeout",
                    default=0,
                    min=0,
                    max=600,
                    tooltip="Override the request timeout in seconds; 0 inherits the selected API configuration"
                )
            ],
            outputs=[
                io.AnyType.Output(
                    id="config_options",
                    display_name="config_options",
                    tooltip="Override configuration options"
                )
            ],
            description="This node provides override API config options."
        )

    @classmethod
    def execute(cls, base_url, api_key, api_protocol, timeout) -> io.NodeOutput:
        return io.NodeOutput(build_config_options(base_url, api_key, api_protocol, timeout))
