from aiohttp import web
from server import PromptServer
from comfy_api.latest import io

from ..utils.skill_utils import create_skill_options, get_skill_summaries


@PromptServer.instance.routes.get("/ycyy/openai/skills/all")
async def get_all_openai_skills(request):
    try:
        return web.json_response(get_skill_summaries())
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


class OpenAITextSkillOptions(io.ComfyNode):
    """Select and snapshot a local SKILL.md package."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        skills = get_skill_summaries()
        names = [item["name"] for item in skills] or [""]
        description = skills[0]["description"] if skills else ""
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
                ),
                io.String.Input(
                    id="description",
                    default=description,
                    multiline=True,
                    socketless=True,
                    tooltip="Description of the selected Skill (read-only).",
                ),
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
    def execute(cls, skill_name, description="") -> io.NodeOutput:
        if not skill_name:
            raise ValueError("No valid local Skills were found")
        # Description is display-only. Always rebuild options from the selected Skill.
        return io.NodeOutput(create_skill_options(skill_name))
