import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "./marked.umd.js";

const NODE_CLASS = "YCYY_Preview_API_Result";
const OPENAI_NODE_CLASS = "YCYY_OpenAI_Text_API";
const STREAM_EVENT = "ycyy_openai_text_stream";
const LOCALE_SETTING = "Comfy.Locale";
const MIN_WIDTH = 360;
const MIN_HEIGHT = 260;
const MAX_TYPING_UNITS_PER_FRAME = 18;
const RENDER_INTERVAL_MS = 40;

const MESSAGES = {
    en: {
        empty: "Connect an API result and run the workflow",
        ready: "Ready",
        waiting: "Waiting for response…",
        loading_skill: "Loading Skill…",
        reading_skill: "Reading Skill files…",
        reasoning: "Thinking…",
        generating: "Generating…",
        drafting: "Model note…",
        toolRunning: "Tool running…",
        tool_running: "Tool running…",
        promoting: "Preparing answer…",
        activityExpand: "Show activity",
        activityCollapse: "Hide activity",
        processTitle: "Process & status",
        intermediate: "Model note",
        displaying: "Displaying result…",
        complete: "Complete",
        copy: "Copy",
        copied: "Copied",
        copyFailed: "Copy failed",
        success: "Success",
        copiedDetail: "Text copied to clipboard",
        error: "Error",
    },
    zh: {
        empty: "连接 API 结果并运行工作流",
        ready: "就绪",
        waiting: "等待响应…",
        loading_skill: "正在加载 Skill…",
        reading_skill: "正在读取 Skill 文件…",
        reasoning: "正在思考…",
        generating: "正在生成…",
        drafting: "模型中间说明…",
        toolRunning: "工具执行中…",
        tool_running: "工具执行中…",
        promoting: "正在整理答案…",
        activityExpand: "展开过程",
        activityCollapse: "收起过程",
        processTitle: "过程与状态",
        intermediate: "模型中间说明",
        displaying: "正在显示结果…",
        complete: "完成",
        copy: "复制",
        copied: "已复制",
        copyFailed: "复制失败",
        success: "成功",
        copiedDetail: "文本已复制到剪贴板",
        error: "错误",
    },
};

const ALLOWED_TAGS = new Set([
    "P", "BR", "H1", "H2", "H3", "H4", "H5", "H6",
    "UL", "OL", "LI", "BLOCKQUOTE", "PRE", "CODE", "HR",
    "STRONG", "EM", "DEL", "A", "TABLE", "THEAD", "TBODY",
    "TR", "TH", "TD",
]);
const DROP_CONTENT_TAGS = new Set([
    "SCRIPT", "STYLE", "IFRAME", "OBJECT", "EMBED", "FORM",
    "SVG", "MATH", "META", "LINK", "BASE", "IMG", "VIDEO", "AUDIO",
]);

let currentLocale = "en";
const graphemeSegmenter = globalThis.Intl?.Segmenter
    ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
    : null;
const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)");

function normalizeLocale(locale) {
    return String(locale || "en").toLowerCase().startsWith("zh") ? "zh" : "en";
}

function message(key) {
    return MESSAGES[normalizeLocale(currentLocale)]?.[key] ?? MESSAGES.en[key] ?? key;
}

function nodeClass(node) {
    return node?.constructor?.comfyClass || node?.comfyClass || node?.type;
}

function getLink(graph, linkId) {
    return graph?.links?.get?.(linkId) ?? graph?.links?.[linkId] ?? null;
}

function safeHref(value) {
    try {
        const url = new URL(value, window.location.href);
        return ["http:", "https:", "mailto:"].includes(url.protocol) ? value : null;
    } catch {
        return null;
    }
}

function sanitizeMarkedHtml(html) {
    const template = document.createElement("template");
    template.innerHTML = html;

    for (const element of [...template.content.querySelectorAll("*")]) {
        if (DROP_CONTENT_TAGS.has(element.tagName)) {
            element.remove();
            continue;
        }
        if (!ALLOWED_TAGS.has(element.tagName)) {
            element.replaceWith(...element.childNodes);
            continue;
        }

        const href = element.tagName === "A"
            ? safeHref(element.getAttribute("href") || "")
            : null;
        const title = element.getAttribute("title");
        for (const attribute of [...element.attributes]) {
            element.removeAttribute(attribute.name);
        }
        if (href) {
            element.setAttribute("href", href);
            element.setAttribute("target", "_blank");
            element.setAttribute("rel", "noopener noreferrer");
        }
        if (title) element.setAttribute("title", title);
    }
    return template.innerHTML;
}

function markedApi() {
    return globalThis.marked;
}

function isNearBottom(element) {
    return element.scrollHeight - element.scrollTop - element.clientHeight < 48;
}

function splitTextUnits(value) {
    const text = String(value ?? "");
    if (!text) return [];
    if (graphemeSegmenter) {
        return [...graphemeSegmenter.segment(text)].map(part => part.segment);
    }
    return Array.from(text);
}

function typingRate(state) {
    const queueLength = state.pendingUnits.length;
    if (state.sourceEnded) {
        if (queueLength > 500) return 720;
        if (queueLength > 200) return 480;
        return 260;
    }
    if (queueLength > 500) return 260;
    if (queueLength > 200) return 180;
    if (queueLength > 80) return 110;
    return 48;
}

function renderNow(state) {
    const followTail = isNearBottom(state.content);
    const raw = state.displayedText || "";
    if (!raw) {
        state.content.replaceChildren();
        if (!["waiting", "loading_skill", "reading_skill", "reasoning", "drafting", "tool_running", "promoting", "generating", "displaying", "error"].includes(state.statusKey)) {
            const placeholder = document.createElement("div");
            placeholder.className = "empty";
            placeholder.textContent = message("empty");
            state.content.append(placeholder);
        }
    } else {
        const parser = markedApi();
        if (!parser?.parse) {
            state.content.textContent = raw;
            setStatus(state, "error", "marked.umd.js unavailable");
            renderActivity(state);
        } else {
            const html = parser.parse(raw, { gfm: true, breaks: true });
            state.content.innerHTML = sanitizeMarkedHtml(html);
        }
    }
    if (followTail) state.content.scrollTop = state.content.scrollHeight;
}

function scheduleRender(state, immediate = false) {
    if (immediate) {
        clearTimeout(state.renderTimer);
        state.renderTimer = null;
        state.lastRenderTime = performance.now();
        renderNow(state);
        return;
    }
    if (state.renderTimer !== null) return;
    const elapsed = performance.now() - state.lastRenderTime;
    const delay = Math.max(0, RENDER_INTERVAL_MS - elapsed);
    state.renderTimer = setTimeout(() => {
        state.renderTimer = null;
        state.lastRenderTime = performance.now();
        renderNow(state);
    }, delay);
}

function setStatus(state, key, detail = "") {
    state.statusKey = key;
    state.statusDetail = detail;
    const visible = [
        "waiting", "loading_skill", "reading_skill", "reasoning",
        "drafting", "tool_running", "promoting", "generating", "displaying", "complete", "error",
    ].includes(key);
    state.host.dataset.state = key;
    state.status.textContent = key === "error" && detail
        ? `${message("error")}: ${detail}`
        : message(key);
    state.status.hidden = !visible;
}

function appendActivity(state, entry) {
    state.activityLog.push(entry);
    if (state.activityLog.length > 20) state.activityLog.splice(0, state.activityLog.length - 20);
}

function renderActivity(state) {
    if (!state.activityPanel) return;
    const followTail = isNearBottom(state.activityWrap);
    state.activityPanel.replaceChildren();
    const latest = state.currentActivity;
    if (latest) {
        const current = document.createElement("div");
        current.className = "activity-current";
        current.textContent = latest.detail ? `${latest.label} · ${latest.detail}` : latest.label;
        state.activityPanel.append(current);
    }
    const log = document.createElement("div");
    log.className = "activity-log";
    const entries = state.activityExpanded ? state.activityLog : state.activityLog.slice(-2);
    for (const entry of entries) {
        const row = document.createElement("div");
        row.className = `activity-entry activity-${entry.type || "info"}`;
        row.textContent = entry.text || "";
        log.append(row);
    }
    if (state.candidateText) {
        const candidate = document.createElement("div");
        candidate.className = "activity-candidate";
        candidate.textContent = `${message("intermediate")}: ${state.candidateText}`;
        log.append(candidate);
    }
    state.activityPanel.append(log);
    // Keep the process box visible for the complete state as well, so the
    // final status and activity history remain available after rendering.
    const statusVisible = !state.status.hidden && state.statusKey !== "ready";
    state.activityWrap.hidden = !latest && !state.activityLog.length && !state.candidateText && !statusVisible;
    // Keep the toggle available after collapsing a short log; otherwise a
    // two-entry process history could be hidden permanently.
    state.activityToggle.hidden = !state.activityLog.length && !state.candidateText;
    state.activityToggle.textContent = state.activityExpanded
        ? message("activityCollapse") : message("activityExpand");
    if (followTail) state.activityWrap.scrollTop = state.activityWrap.scrollHeight;
}

function updateCopyAvailability(state) {
    state.copyButton.disabled = !state.finalText;
}

function setCopyState(state, copyState) {
    state.copyButton.dataset.copyState = copyState;
    const label = copyState === "copied"
        ? message("copied")
        : (copyState === "failed" ? message("copyFailed") : message("copy"));
    state.copyButton.title = label;
    state.copyButton.setAttribute("aria-label", label);
}

function showCopyToast(success) {
    const toast = app.extensionManager?.toast;
    if (!toast?.add) return;
    toast.add(success
        ? {
            severity: "success",
            summary: message("success"),
            detail: message("copiedDetail"),
            life: 3000,
        }
        : {
            severity: "error",
            summary: message("error"),
            detail: message("copyFailed"),
        });
}

async function copyRawText(state) {
    const text = state.finalText;
    if (!text) return;
    try {
        if (navigator.clipboard?.writeText && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            document.body.append(textarea);
            try {
                textarea.select();
                if (!document.execCommand("copy")) throw new Error("copy command failed");
            } finally {
                textarea.remove();
            }
        }
        setCopyState(state, "copied");
        showCopyToast(true);
    } catch (error) {
        console.error("[YCYY] Failed to copy preview text:", error);
        setCopyState(state, "failed");
        showCopyToast(false);
    } finally {
        clearTimeout(state.copyTimer);
        state.copyTimer = setTimeout(() => {
            setCopyState(state, "idle");
        }, 1200);
    }
}

function stopTyping(state) {
    if (state.typingFrame !== null) cancelAnimationFrame(state.typingFrame);
    state.typingFrame = null;
    state.lastTypingTime = 0;
    state.typingBudget = 0;
}

function finishTypingIfReady(state) {
    if (!state.sourceEnded || state.pendingUnits.length > 0) return false;

    // Normally the queue already produced finalText. This assignment also
    // reconciles unusual providers whose final response differs from deltas.
    if (state.displayedText !== state.finalText) {
        state.displayedText = state.finalText;
    }
    state.receivedText = state.finalText;
    state.streaming = false;
    state.activeRunId = null;
    if (state.terminalError) {
        setStatus(state, "error", state.terminalError);
    } else {
        state.currentActivity = null;
        setStatus(state, "complete");
    }
    renderActivity(state);
    updateCopyAvailability(state);
    scheduleRender(state, true);
    return true;
}

function typingTick(state, timestamp) {
    state.typingFrame = null;
    if (!state.streaming) return;

    if (!state.lastTypingTime) state.lastTypingTime = timestamp;
    const elapsed = Math.min(250, Math.max(0, timestamp - state.lastTypingTime));
    state.lastTypingTime = timestamp;
    state.typingBudget += elapsed * typingRate(state) / 1000;

    const available = Math.floor(state.typingBudget);
    const count = Math.min(
        available,
        MAX_TYPING_UNITS_PER_FRAME,
        state.pendingUnits.length,
    );
    if (count > 0) {
        state.displayedText += state.pendingUnits.splice(0, count).join("");
        state.typingBudget -= count;
        scheduleRender(state);
    }

    if (state.pendingUnits.length > 0) {
        state.typingFrame = requestAnimationFrame(time => typingTick(state, time));
    } else {
        state.lastTypingTime = 0;
        state.typingBudget = 0;
        finishTypingIfReady(state);
    }
}

function startTyping(state) {
    if (state.typingFrame !== null) return;
    if (!state.pendingUnits.length) {
        finishTypingIfReady(state);
        return;
    }
    state.typingFrame = requestAnimationFrame(time => typingTick(state, time));
}

function enqueueDelta(state, value) {
    const delta = String(value ?? "");
    if (!delta) return;
    state.receivedText += delta;
    updateCopyAvailability(state);
    if (reducedMotion?.matches) {
        state.displayedText = state.receivedText;
        state.pendingUnits = [];
        scheduleRender(state);
        return;
    }
    state.pendingUnits.push(...splitTextUnits(delta));
    startTyping(state);
}

function enqueueFinalText(state, value) {
    const finalText = String(value ?? state.receivedText);
    state.sourceEnded = true;
    state.finalText = finalText;
    state.receivedText = finalText;
    updateCopyAvailability(state);

    if (reducedMotion?.matches) {
        state.displayedText = finalText;
        state.pendingUnits = [];
        finishTypingIfReady(state);
        return;
    }

    // Rebuild the unplayed tail from the authoritative final response. This
    // also prevents duplicate characters when both `end` and `onExecuted`
    // deliver the same full text before the animation has caught up.
    if (finalText.startsWith(state.displayedText)) {
        state.pendingUnits = splitTextUnits(finalText.slice(state.displayedText.length));
    } else {
        const shown = splitTextUnits(state.displayedText);
        const final = splitTextUnits(finalText);
        let common = 0;
        while (common < shown.length && common < final.length && shown[common] === final[common]) {
            common += 1;
        }
        state.displayedText = shown.slice(0, common).join("");
        state.pendingUnits = final.slice(common);
    }
    startTyping(state);
}

function createPreview(node) {
    if (node._ycyyApiResultPreview) return node._ycyyApiResultPreview;

    const root = document.createElement("div");
    root.style.width = "100%";
    root.style.height = "100%";
    root.style.minHeight = "210px";
    root.style.boxSizing = "border-box";
    const shadow = root.attachShadow({ mode: "open" });
    shadow.innerHTML = `
        <style>
            :host { display: block; width: 100%; height: 100%; }
            .preview {
                position: relative; display: flex; flex-direction: column; gap: 7px; height: 100%; min-height: 200px; overflow: hidden;
                box-sizing: border-box; border: 0; border-radius: 8px;
                color: var(--input-text, var(--fg-color, #ddd));
                font: 13px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
            }
            .result-wrap {
                order: 1; position: relative; flex: 1 1 auto; min-height: 0; width: 100%; overflow: hidden;
                box-sizing: border-box; border: 1px solid var(--border-color, #484848); border-radius: 8px;
                background: var(--comfy-input-bg, #181818);
            }
            .content {
                width: 100%; height: 100%; overflow: auto; box-sizing: border-box;
                padding: 12px 14px 22px; overflow-wrap: anywhere;
                scrollbar-color: color-mix(in srgb, currentColor 35%, transparent) transparent;
            }
            .activity-wrap { order: 2; position: relative; z-index: 1; flex: 0 0 auto; width: 100%; max-height: 140px; min-height: 34px; overflow: auto; box-sizing: border-box; border: 1px solid color-mix(in srgb, var(--border-color, #484848) 80%, transparent); border-radius: 8px; background: color-mix(in srgb, var(--comfy-input-bg, #181818) 92%, #000 8%); font-size: 12px; opacity: .88; }
            .activity-wrap > .status { position: static; display: block; max-width: none; margin: 5px 8px 0; padding: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; background: transparent; border-radius: 0; font-size: 12px; opacity: .85; }
            .activity-title { padding: 6px 8px 0; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; opacity: .55; }
            .activity-panel { padding: 2px 8px 2px; }
            .activity-current { color: var(--input-text, var(--fg-color, #ddd)); font-weight: 600; }
            .activity-log { margin-top: 2px; }
            .activity-entry, .activity-candidate { white-space: pre-wrap; overflow-wrap: anywhere; opacity: .72; }
            .activity-candidate { color: color-mix(in srgb, currentColor 78%, #7c8cf8 22%); }
            .activity-toggle {
                position: sticky; bottom: 5px; z-index: 2; display: block;
                margin: 3px 8px 5px auto; padding: 3px 8px;
                border: 1px solid color-mix(in srgb, currentColor 28%, transparent);
                border-radius: 5px;
                background: color-mix(in srgb, var(--comfy-input-bg, #181818) 94%, currentColor 6%);
                box-shadow: 0 1px 5px color-mix(in srgb, #000 45%, transparent);
                color: inherit; opacity: .78; cursor: pointer; font-size: 11px;
            }
            .activity-toggle:hover { opacity: 1; background: color-mix(in srgb, var(--comfy-input-bg, #181818) 84%, currentColor 16%); }
            .empty { height: 100%; display: grid; place-items: center; text-align: center; opacity: .42; }
            .copy {
                position: absolute; z-index: 2; top: 6px; right: 8px; width: 28px; height: 28px;
                display: grid; place-items: center; padding: 0; border: 0; border-radius: 6px;
                background: color-mix(in srgb, var(--comfy-input-bg, #181818) 88%, currentColor 12%);
                color: inherit; cursor: pointer; opacity: 0; transition: opacity .12s, background .12s;
            }
            .preview:hover .copy, .copy:focus-visible { opacity: .82; }
            .copy:hover { opacity: 1; background: color-mix(in srgb, var(--comfy-input-bg, #181818) 76%, currentColor 24%); }
            .copy:focus-visible { outline: 1px solid #60a5fa; outline-offset: 1px; }
            .copy svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
            .copy .check-icon { display: none; }
            .copy[data-copy-state="copied"] { color: #69d69b; opacity: 1; }
            .copy[data-copy-state="copied"] .copy-icon { display: none; }
            .copy[data-copy-state="copied"] .check-icon { display: block; }
            .copy[data-copy-state="failed"] { color: #f87171; opacity: 1; }
            .copy:disabled { display: none; }
            .status { color: inherit; pointer-events: none; }
            .preview[data-state="waiting"] .status,
            .preview[data-state="loading_skill"] .status,
            .preview[data-state="reading_skill"] .status,
            .preview[data-state="reasoning"] .status,
            .preview[data-state="error"] .status { transform: none; }
            .preview[data-state="error"] .status { color: #fca5a5; opacity: .95; }
            h1, h2, h3, h4, h5, h6 { margin: 1.1em 0 .55em; line-height: 1.25; }
            h1 { font-size: 1.55em; } h2 { font-size: 1.35em; } h3 { font-size: 1.18em; }
            p, ul, ol, blockquote, pre, table { margin: .65em 0; }
            ul, ol { padding-left: 1.7em; } li + li { margin-top: .2em; }
            blockquote { margin-left: 0; padding: .1em 1em; border-left: 3px solid #7c8cf8; opacity: .82; }
            code { border-radius: 4px; padding: .12em .35em; background: color-mix(in srgb, currentColor 10%, transparent); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
            pre { overflow: auto; padding: 12px; border-radius: 7px; background: color-mix(in srgb, currentColor 8%, transparent); }
            pre code { padding: 0; background: transparent; white-space: pre; }
            table { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
            th, td { padding: 6px 9px; border: 1px solid color-mix(in srgb, currentColor 18%, transparent); }
            th { background: color-mix(in srgb, currentColor 7%, transparent); }
            a { color: #7c8cf8; } hr { border: 0; border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent); }
            .preview[data-state="generating"] .content::after,
            .preview[data-state="displaying"] .content::after { content: ""; display: inline-block; width: 6px; height: 1em; margin-left: 3px; vertical-align: -.12em; background: currentColor; animation: blink .85s steps(1) infinite; }
            @keyframes blink { 50% { opacity: 0; } }
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after { animation: none !important; transition: none !important; }
                .preview[data-state="generating"] .content::after,
                .preview[data-state="displaying"] .content::after { display: none; }
            }
        </style>
        <div class="preview" data-state="idle">
            <section class="result-wrap">
                <div class="content"></div>
                <button class="copy" type="button" data-copy-state="idle">
                <svg class="copy-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"></rect><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"></path></svg>
                <svg class="check-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"></path></svg>
                </button>
            </section>
            <section class="activity-wrap" hidden>
                <div class="activity-title"></div>
                <span class="status" hidden aria-live="polite"></span>
                <div class="activity-panel"></div>
                <button class="activity-toggle" type="button" hidden></button>
            </section>
        </div>`;

    const state = {
        displayedText: "",
        receivedText: "",
        finalText: "",
        pendingUnits: [],
        sourceEnded: false,
        terminalError: "",
        activeRunId: null,
        lastSeq: -1,
        streaming: false,
        typingFrame: null,
        lastTypingTime: 0,
        typingBudget: 0,
        renderTimer: null,
        lastRenderTime: 0,
        copyTimer: null,
        statusKey: "ready",
        statusDetail: "",
        candidateText: "",
        activityLog: [],
        currentActivity: null,
        activityExpanded: false,
        root,
        host: shadow.querySelector(".preview"),
        content: shadow.querySelector(".content"),
        status: shadow.querySelector(".activity-wrap .status"),
        copyButton: shadow.querySelector(".copy"),
        activityWrap: shadow.querySelector(".activity-wrap"),
        activityTitle: shadow.querySelector(".activity-title"),
        activityPanel: shadow.querySelector(".activity-panel"),
        activityToggle: shadow.querySelector(".activity-toggle"),
    };
    setStatus(state, "ready");
    state.activityTitle.textContent = message("processTitle");
    setCopyState(state, "idle");
    updateCopyAvailability(state);
    state.copyButton.addEventListener("click", () => copyRawText(state));
    state.activityToggle.addEventListener("click", () => {
        state.activityExpanded = !state.activityExpanded;
        renderActivity(state);
    });
    renderActivity(state);
    root.addEventListener("pointerdown", event => event.stopPropagation());
    root.addEventListener("wheel", event => event.stopPropagation(), { passive: true });

    node.addDOMWidget("preview_api_result", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 210,
    });
    node._ycyyApiResultPreview = state;
    node.size ||= [MIN_WIDTH, MIN_HEIGHT];
    node.size[0] = Math.max(node.size[0] || 0, MIN_WIDTH);
    node.size[1] = Math.max(node.size[1] || 0, MIN_HEIGHT);
    scheduleRender(state);
    return state;
}

function disposePreview(node) {
    const state = node?._ycyyApiResultPreview;
    if (!state) return;
    stopTyping(state);
    clearTimeout(state.renderTimer);
    clearTimeout(state.copyTimer);
    node._ycyyApiResultPreview = null;
}

function resetStreamState(state, data, sequence) {
    stopTyping(state);
    state.activeRunId = data.run_id;
    state.lastSeq = Number.isFinite(sequence) ? sequence : -1;
    state.displayedText = "";
    state.receivedText = "";
    state.finalText = "";
    state.candidateText = "";
    state.activityLog = [];
    state.currentActivity = null;
    state.activityExpanded = true;
    state.currentRound = 0;
    state.pendingUnits = [];
    state.sourceEnded = false;
    state.terminalError = "";
    state.streaming = true;
    setStatus(state, "waiting");
    renderActivity(state);
    updateCopyAvailability(state);
    scheduleRender(state, true);
}

function setFinalText(node, value) {
    const state = createPreview(node);
    const text = String(value ?? "");
    if (state.streaming || state.typingFrame !== null || state.pendingUnits.length > 0) {
        enqueueFinalText(state, text);
        return;
    }
    const unchanged = state.displayedText === text && !state.streaming;
    stopTyping(state);
    state.displayedText = text;
    state.receivedText = text;
    state.finalText = text;
    state.pendingUnits = [];
    state.sourceEnded = true;
    state.terminalError = "";
    state.streaming = false;
    state.activeRunId = null;
    setStatus(state, "complete");
    renderActivity(state);
    updateCopyAvailability(state);
    if (!unchanged) scheduleRender(state, true);
}

function connectedPreviewNodes(sourceNodeId) {
    const graph = app.graph;
    const source = graph?.getNodeById?.(sourceNodeId)
        ?? graph?.getNodeById?.(Number(sourceNodeId));
    if (!source || nodeClass(source) !== OPENAI_NODE_CLASS) return [];

    const resultOutput = source.outputs?.[0];
    const targets = [];
    for (const linkId of resultOutput?.links || []) {
        const link = getLink(graph, linkId);
        if (!link || link.origin_slot !== 0 || link.target_slot !== 0) continue;
        const target = graph.getNodeById(link.target_id);
        if (target && nodeClass(target) === NODE_CLASS) targets.push(target);
    }
    return targets;
}

function receiveStreamEvent(event) {
    const data = event.detail || {};
    for (const node of connectedPreviewNodes(data.node_id)) {
        const state = createPreview(node);
        const sequence = Number(data.seq);
        if (data.phase === "start") {
            resetStreamState(state, data, sequence);
            continue;
        }
        if (state.activeRunId !== data.run_id) continue;
        if (Number.isFinite(sequence) && sequence <= state.lastSeq) continue;
        if (Number.isFinite(sequence)) state.lastSeq = sequence;

        if (data.phase === "round_start") {
            state.currentRound = Number(data.round) || 0;
            state.candidateText = "";
            state.currentActivity = { label: message("waiting"), detail: `Round ${state.currentRound + 1}` };
            appendActivity(state, { type: "round", text: `Round ${state.currentRound + 1}` });
            setStatus(state, "waiting");
            renderActivity(state);
        } else if (data.phase === "candidate_delta") {
            state.candidateText += String(data.delta || "");
            state.currentActivity = { label: message("drafting"), detail: `Round ${state.currentRound + 1}` };
            setStatus(state, "drafting");
            renderActivity(state);
        } else if (data.phase === "tool_call_start") {
            const detail = data.path ? `${data.tool} · ${data.path}` : data.tool;
            state.currentActivity = { label: message("toolRunning"), detail };
            appendActivity(state, { type: "tool", text: `${message("toolRunning")} ${detail}` });
            setStatus(state, "tool_running");
            renderActivity(state);
        } else if (data.phase === "tool_call_end") {
            const detail = data.path ? `${data.tool} · ${data.path}` : data.tool;
            appendActivity(state, { type: "tool", text: `${detail} · ${data.status || "success"}` });
            renderActivity(state);
        } else if (data.phase === "round_end") {
            const hasTools = Boolean(data.has_tool_calls);
            if (hasTools) {
                state.candidateText = "";
                state.currentActivity = { label: message("toolRunning"), detail: `Round ${state.currentRound + 1} complete` };
            } else {
                const promotedText = state.candidateText;
                state.receivedText = promotedText;
                state.finalText = promotedText;
                state.candidateText = "";
                state.currentActivity = { label: message("generating"), detail: "" };
                setStatus(state, promotedText ? "promoting" : "generating");
                if (promotedText) {
                    state.pendingUnits = splitTextUnits(promotedText);
                    state.displayedText = "";
                    state.sourceEnded = false;
                    startTyping(state);
                }
            }
            renderActivity(state);
            scheduleRender(state, true);
        } else if (data.phase === "activity") {
            const activity = [
                "loading_skill", "reading_skill", "reasoning",
            ].includes(data.activity) ? data.activity : null;
            if (activity && !state.receivedText) {
                setStatus(state, activity);
                renderActivity(state);
                scheduleRender(state, true);
            }
        } else if (data.phase === "delta") {
            if (!state.receivedText) {
                setStatus(state, "generating");
                renderActivity(state);
            }
            enqueueDelta(state, data.delta);
        } else if (data.phase === "end") {
            if (state.pendingUnits.length || state.displayedText !== String(data.text ?? state.receivedText)) {
                setStatus(state, "displaying");
            }
            enqueueFinalText(state, data.text ?? state.receivedText);
            if (data.stop_reason && data.stop_reason !== "stop") {
                state.terminalError = `stop_reason: ${data.stop_reason}`;
            }
            state.candidateText = "";
            state.currentActivity = null;
            renderActivity(state);
        } else if (data.phase === "error") {
            state.sourceEnded = true;
            state.finalText = state.receivedText;
            state.terminalError = data.message || "unknown";
            setStatus(state, "error", state.terminalError);
            renderActivity(state);
            updateCopyAvailability(state);
            scheduleRender(state, true);
            startTyping(state);
        }
    }
}

function refreshLocale() {
    currentLocale = app.ui?.settings?.getSettingValue?.(LOCALE_SETTING) || "en";
    for (const node of app.graph?._nodes || []) {
        const state = node?._ycyyApiResultPreview;
        if (!state) continue;
        if (state.activityTitle) state.activityTitle.textContent = message("processTitle");
        setCopyState(state, state.copyButton.dataset.copyState || "idle");
        setStatus(state, state.statusKey, state.statusDetail);
        renderActivity(state);
        scheduleRender(state);
    }
}

app.registerExtension({
    name: "YCYY.Preview.API.Result",

    async setup() {
        refreshLocale();
        app.ui?.settings?.addEventListener?.(`${LOCALE_SETTING}.change`, refreshLocale);
        api.addEventListener(STREAM_EVENT, receiveStreamEvent);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS && nodeType.comfyClass !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            createPreview(this);
            return result;
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigured?.apply(this, arguments);
            setTimeout(() => createPreview(this), 0);
            return result;
        };

        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (executionMessage) {
            originalExecuted?.apply(this, arguments);
            const value = Array.isArray(executionMessage?.text)
                ? executionMessage.text[0]
                : (executionMessage?.text ?? "");
            setFinalText(this, value);
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            disposePreview(this);
            return originalRemoved?.apply(this, arguments);
        };
    },

    async nodeCreated(node) {
        if (nodeClass(node) === NODE_CLASS) createPreview(node);
    },

    async onNodeOutputsUpdated(nodeOutputs) {
        for (const [nodeId, output] of Object.entries(nodeOutputs || {})) {
            const node = app.graph?.getNodeById?.(nodeId)
                ?? app.graph?.getNodeById?.(Number(nodeId));
            if (node && nodeClass(node) === NODE_CLASS) {
                const value = Array.isArray(output?.text)
                    ? output.text[0]
                    : (output?.text ?? "");
                setFinalText(node, value);
            }
        }
    },
});
