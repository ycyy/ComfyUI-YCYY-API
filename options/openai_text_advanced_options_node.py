from comfy_api.latest import io
from ..utils.request_utils import parse_json_options


class OpenAITextAdvancedOptions(io.ComfyNode):
    """Parse user-supplied protocol/API-specific JSON request parameters."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_OpenAI_Text_Advanced_Options",
            display_name="OpenAI Text Advanced Options",
            category="YCYY/API/utils",
            inputs=[
                io.String.Input(
                    id="options_json",
                    multiline=True,
                    default="{}",
                    tooltip=(
                        "JSON object containing protocol/API-specific request parameters. "
                        "Example:\n{\n  \"temperature\": 0.7,\n  \"reasoning\": {\n"
                        "    \"effort\": \"medium\"\n  }\n}"
                    ),
                ),
            ],
            outputs=[io.AnyType.Output(id="advanced_options", display_name="advanced_options")],
            description="Pass custom JSON parameters to the OpenAI Text API.",
        )

    @classmethod
    def execute(cls, options_json) -> io.NodeOutput:
        return io.NodeOutput(parse_json_options(options_json))
