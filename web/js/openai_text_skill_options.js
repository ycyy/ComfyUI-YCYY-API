import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "YCYY_OpenAI_Text_Skill_Options";
let skillMap = new Map();

function refreshSkillNodes() {
    for (const node of app.graph?._nodes ?? []) {
        if (node.comfyClass === NODE_CLASS || node.type === NODE_CLASS) syncNode(node);
    }
}

async function loadSkills(retries = 1) {
    try {
        const response = await api.fetchApi("/ycyy/openai/skills/all");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        skillMap = new Map(
            (Array.isArray(data) ? data : []).map(skill => [skill.name, skill])
        );
        refreshSkillNodes();
    } catch (error) {
        console.error("[YCYY] Failed to load Skill descriptions:", error);
        if (retries > 0) setTimeout(() => loadSkills(retries - 1), 2000);
    }
}

function makeReadOnly(widget) {
    if (!widget) return;
    widget.options = { ...(widget.options ?? {}), readOnly: true };
    if (widget.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.setAttribute("aria-readonly", "true");
        widget.inputEl.spellcheck = false;
    }
}

function applyDescription(node, skillName) {
    const descriptionWidget = node.widgets?.find(widget => widget.name === "description");
    if (!descriptionWidget) return;
    const skill = skillMap.get(skillName);
    descriptionWidget.value = typeof skill?.description === "string"
        ? skill.description
        : "";
    makeReadOnly(descriptionWidget);
    app.canvas?.draw(true, true);
}

function syncNode(node) {
    const skillWidget = node.widgets?.find(widget => widget.name === "skill_name");
    const descriptionWidget = node.widgets?.find(widget => widget.name === "description");
    if (!skillWidget || !descriptionWidget) return;

    const values = skillWidget.options?.values ?? [];
    if (values.length && !values.includes(skillWidget.value)) {
        const fallback = values[0] ?? "";
        console.warn(
            `[YCYY] Skill "${skillWidget.value}" no longer exists; using "${fallback}"`
        );
        skillWidget.value = fallback;
    }
    makeReadOnly(descriptionWidget);
    applyDescription(node, skillWidget.value);
}

app.registerExtension({
    name: "YCYY.OpenAI.Text.SkillOptions",
    async setup() {
        await loadSkills();
    },
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const skillWidget = this.widgets?.find(widget => widget.name === "skill_name");
            const descriptionWidget = this.widgets?.find(widget => widget.name === "description");
            if (skillWidget && descriptionWidget) {
                makeReadOnly(descriptionWidget);
                const originalCallback = skillWidget.callback;
                skillWidget.callback = value => {
                    applyDescription(this, value);
                    originalCallback?.call(this, value);
                };
                setTimeout(() => syncNode(this), 0);
            }
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            setTimeout(() => syncNode(this), 0);
            return result;
        };
    },
});
