from comfy_api.latest import io, ui


class PreviewAPIResult(io.ComfyNode):
    """Render a string in the frontend Markdown preview widget."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Preview_API_Result",
            display_name="Preview API Result",
            category="YCYY/API/utils",
            inputs=[
                io.String.Input(
                    id="source",
                    force_input=True,
                    multiline=True,
                    tooltip="Markdown or plain text to preview",
                ),
            ],
            outputs=[io.String.Output(id="text", display_name="text")],
            is_output_node=True,
            description="Render text as a Markdown preview with copy support.",
        )

    @classmethod
    def execute(cls, source) -> io.NodeOutput:
        text = source if isinstance(source, str) else ("" if source is None else str(source))
        return io.NodeOutput(text, ui=ui.PreviewText(text))
