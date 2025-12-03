from comfy_api.latest import ComfyExtension, io

class ProxyOptions(io.ComfyNode):
    """
    这个节点用于覆盖代理配置参数（enable、http、https）
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Proxy_Options",
            display_name="Proxy Options",
            category="YCYY/API/utils",
            inputs=[
                io.Boolean.Input(
                    id="enable",
                    default=False,
                    tooltip="Enable or disable proxy"
                ),
                io.String.Input(
                    id="http",
                    default="",
                    multiline=True,
                    tooltip="HTTP proxy address (e.g., http://127.0.0.1:7890)"
                ),
                io.String.Input(
                    id="https",
                    default="",
                    multiline=True,
                    tooltip="HTTPS proxy address (e.g., http://127.0.0.1:7890)"
                )
            ],
            outputs=[
                io.AnyType.Output(
                    id="proxy_options",
                    display_name="proxy_options",
                    tooltip="Proxy configuration override"
                )
            ],
            description="This node provides proxy configuration override options (enable, http, https)."
        )

    @classmethod
    def execute(cls, enable, http, https) -> io.NodeOutput:
        # 验证和清理输入
        http = http.strip() if http else ""
        https = https.strip() if https else ""

        proxy_options = {
            "enable": enable,
            "http": http,
            "https": https
        }

        return io.NodeOutput(proxy_options)