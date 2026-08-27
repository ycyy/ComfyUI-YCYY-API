from comfy_api.latest import io

from ..utils.skill_utils import create_skill_options, discover_skills


class OpenAITextSkillOptions(io.ComfyNode):
    """Select and snapshot a local SKILL.md package."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        skills, _ = discover_skills(strict=True)
        names = [item["name"] for item in skills] or [""]
        return io.Schema(
            node_id="YCYY_OpenAI_Text_Skill_Options",
            display_name="OpenAI Text Skill Options",
            category="YCYY/API/utils",
            inputs=[
                io.Combo.Input(
                    id="skill_name",
                    options=names,
                    default=names[0],
                    tooltip="Select a configured local SKILL.md package.",
                )
            ],
            outputs=[
                io.AnyType.Output(
                    id="skill_options",
                    display_name="skill_options",
                    tooltip="Progressive local Skill configuration for OpenAI Text API.",
                )
            ],
            description="Load a local SKILL.md package for progressive use by OpenAI Text API.",
        )

    @classmethod
    def execute(cls, skill_name) -> io.NodeOutput:
        if not skill_name:
            raise ValueError("No valid local Skills were found")
        return io.NodeOutput(create_skill_options(skill_name))
