from comfy_api.latest import io


class GeminiBatchSpeakersOptions(io.ComfyNode):
    """
    这个节点用于合并两个 Gemini 说话人配置数组
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Gemini_Batch_Speakers_Options",
            display_name="Gemini Batch Speakers Options",
            category="YCYY/API/utils",
            inputs=[
                io.AnyType.Input(
                    id="speaker_options1",
                    tooltip="The first speaker options array"
                ),
                io.AnyType.Input(
                    id="speaker_options2",
                    tooltip="The second speaker options array"
                )
            ],
            outputs=[
                io.AnyType.Output(
                    id="speaker_options",
                    display_name="speaker_options",
                    tooltip="Merged speaker options array"
                )
            ],
            description="This node merges two Gemini multi-speaker option arrays."
        )

    @classmethod
    def execute(cls, speaker_options1, speaker_options2) -> io.NodeOutput:
        merged_options = cls._normalize_options(speaker_options1) + cls._normalize_options(speaker_options2)
        if not merged_options:
            raise ValueError("speaker_options cannot be empty")

        return io.NodeOutput(merged_options)

    @staticmethod
    def _normalize_options(value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("speaker_options input must be a list")

        normalized = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each speaker option must be an object")

            speaker = str(item.get("speaker", "")).strip()
            voice_name = str(item.get("voiceName", "")).strip()

            if not speaker:
                raise ValueError("speaker cannot be empty")
            if not voice_name:
                raise ValueError("voiceName cannot be empty")

            normalized.append({
                "speaker": speaker,
                "voiceName": voice_name,
            })

        return normalized
