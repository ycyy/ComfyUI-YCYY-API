import nodes

from comfy_api.latest import io


class ImageCompare(io.ComfyNode):
    """Compares two images with a slider interface."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Image_Compare",
            display_name="Compare Images",
            description="Compares two images side by side with a slider.",
            category="YCYY/API/image",
            is_output_node=True,
            inputs=[
                io.Image.Input("image_a", optional=True),
                io.Image.Input("image_b", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, image_a=None, image_b=None) -> io.NodeOutput:
        result = {"a_images": [], "b_images": []}
        preview_node = nodes.PreviewImage()

        if image_a is not None and len(image_a) > 0:
            saved = preview_node.save_images(image_a, "ycyy.compare.a")
            result["a_images"] = saved["ui"]["images"]

        if image_b is not None and len(image_b) > 0:
            saved = preview_node.save_images(image_b, "ycyy.compare.b")
            result["b_images"] = saved["ui"]["images"]

        return io.NodeOutput(ui=result)
