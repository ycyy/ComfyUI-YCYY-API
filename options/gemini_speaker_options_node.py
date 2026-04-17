from comfy_api.latest import io

from ..gemini.gemini_tts_node import VOICE_OPTIONS


class GeminiSpeakerOptions(io.ComfyNode):
    """
    这个节点用于构造 Gemini 多说话人 TTS 的单个说话人配置
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YCYY_Gemini_Speaker_Options",
            display_name="Gemini Speaker Options",
            category="YCYY/API/utils",
            inputs=[
                io.String.Input(
                    id="speaker",
                    default="Speaker 1",
                    tooltip="Speaker name used in the conversation text"
                ),
                io.Combo.Input(
                    id="voiceName",
                    options=VOICE_OPTIONS,
                    default="Zephyr",
                    tooltip="The voice assigned to this speaker"
                )
            ],
            outputs=[
                io.AnyType.Output(
                    id="speaker_options",
                    display_name="speaker_options",
                    tooltip="Single speaker option item for Gemini multi-speaker TTS"
                )
            ],
            description="This node builds a single speaker option for Gemini multi-speaker TTS."
        )

    @classmethod
    def execute(cls, speaker, voiceName) -> io.NodeOutput:
        speaker_name = speaker.strip() if speaker else ""
        if not speaker_name:
            raise ValueError("speaker cannot be empty")

        return io.NodeOutput([
            {
                "speaker": speaker_name,
                "voiceName": voiceName,
            }
        ])
