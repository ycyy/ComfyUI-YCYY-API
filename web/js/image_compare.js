import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "YCYY_Image_Compare";
const MIN_WIDTH = 320;
const MIN_HEIGHT = 320;
const PADDING = 10;
const FOOTER_HEIGHT = 30;
const EMPTY_IMAGE = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";
const NATIVE_PREVIEW_WIDGET = "$$canvas-image-preview";
const LOCALE_SETTING = "Comfy.Locale";

const MESSAGES = {
    en: {
        empty: "Connect images and run the workflow",
    },
    zh: {
        empty: "连接图像并运行工作流",
    },
};

let currentLocale = "en";

function normalizeLocale(locale) {
    const value = String(locale || "en").toLowerCase();
    return value.startsWith("zh") ? "zh" : "en";
}

function translate(key) {
    const locale = normalizeLocale(currentLocale);
    return MESSAGES[locale]?.[key] ?? MESSAGES.en[key] ?? key;
}

function isTargetNode(node) {
    const type = node?.constructor?.comfyClass || node?.comfyClass || node?.type;
    return type === NODE_NAME;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function pointInRect(pos, rect) {
    return Boolean(rect)
        && pos[0] >= rect[0]
        && pos[0] <= rect[0] + rect[2]
        && pos[1] >= rect[1]
        && pos[1] <= rect[1] + rect[3];
}

function requestDraw(node, foreground = false) {
    node?._ycyyCompareWidget?.triggerDraw?.();
    app.graph?.setDirtyCanvas?.(true, foreground);
}

function redrawCompareNodes() {
    for (const node of app.graph?._nodes || []) {
        if (isTargetNode(node)) requestDraw(node, true);
    }
}

function handleLocaleChange(event) {
    currentLocale = event.detail?.value || "en";
    redrawCompareNodes();
}

function imageRefs(value) {
    return Array.isArray(value) ? value.filter((ref) => ref?.filename) : [];
}

function imageRefsFromMessage(message, key, seen = new Set()) {
    if (!message || typeof message !== "object" || seen.has(message)) return [];
    seen.add(message);

    const direct = imageRefs(message[key]);
    if (direct.length) return direct;

    for (const value of Object.values(message)) {
        const nested = imageRefsFromMessage(value, key, seen);
        if (nested.length) return nested;
    }
    return [];
}

function imageKey(ref) {
    return `${ref?.type || ""}/${ref?.subfolder || ""}/${ref?.filename || ""}`;
}

function imageUrl(ref) {
    return api.apiURL(`/view?${new URLSearchParams(ref || {}).toString()}`);
}

function releaseImage(img) {
    if (!img) return;
    img.onload = null;
    img.onerror = null;
    try {
        img.src = EMPTY_IMAGE;
    } catch (_) {
        // The browser may reject changing the source while an image is decoding.
    }
}

function suppressNativePreview(node) {
    if (!node) return;
    node.preview = undefined;
    const index = node.widgets?.findIndex((widget) => widget?.name === NATIVE_PREVIEW_WIDGET) ?? -1;
    if (index >= 0) {
        node.widgets[index].onRemove?.();
        node.widgets.splice(index, 1);
    }
    node.imgs = undefined;
    node.images = undefined;
}

function loadImage(node, side, ref) {
    node._ycyyCompareImages ||= {};
    const previous = node._ycyyCompareImages[side];
    const key = ref?.filename ? imageKey(ref) : "";

    if (!key) {
        node._ycyyCompareImages[side] = null;
        if (previous?.img) releaseImage(previous.img);
        requestDraw(node, true);
        return;
    }
    if (previous?.key === key) return;

    const img = new Image();
    const entry = { key, ref: { ...ref }, img };
    node._ycyyCompareImages[side] = entry;
    img.onload = () => {
        if (node._ycyyCompareImages?.[side] !== entry) return;
        requestDraw(node, true);
    };
    img.onerror = () => {
        if (node._ycyyCompareImages?.[side] === entry) {
            node._ycyyCompareImages[side] = null;
        }
        requestDraw(node, true);
    };
    img.src = imageUrl(ref);

    if (previous?.img && previous.img !== img) releaseImage(previous.img);
}

function refAt(refs, index) {
    if (!refs?.length) return null;
    return refs[Math.min(index, refs.length - 1)];
}

function batchLength(node) {
    const refs = node?._ycyyCompareRefs || {};
    return Math.max(refs.a?.length || 0, refs.b?.length || 0);
}

function selectBatch(node, index) {
    const total = batchLength(node);
    if (!total) return;
    node._ycyyCompareIndex = clamp(index, 0, total - 1);
    loadImage(node, "a", refAt(node._ycyyCompareRefs?.a, node._ycyyCompareIndex));
    loadImage(node, "b", refAt(node._ycyyCompareRefs?.b, node._ycyyCompareIndex));
    requestDraw(node, true);
}

function receiveImages(node, message) {
    const a = imageRefsFromMessage(message, "a_images");
    const b = imageRefsFromMessage(message, "b_images");
    node._ycyyCompareRefs = {
        a: a.map((ref) => ({ ...ref })),
        b: b.map((ref) => ({ ...ref })),
    };
    if (!a.length && !b.length) {
        loadImage(node, "a", null);
        loadImage(node, "b", null);
        requestDraw(node, true);
        return;
    }
    selectBatch(node, 0);
}

function fitRect(img, rect) {
    const width = img?.naturalWidth || 0;
    const height = img?.naturalHeight || 0;
    if (!width || !height) return null;
    const scale = Math.min(rect[2] / width, rect[3] / height);
    const fittedWidth = Math.max(1, width * scale);
    const fittedHeight = Math.max(1, height * scale);
    return [
        rect[0] + (rect[2] - fittedWidth) / 2,
        rect[1] + (rect[3] - fittedHeight) / 2,
        fittedWidth,
        fittedHeight,
    ];
}

function drawImageContained(ctx, img, rect) {
    const fitted = fitRect(img, rect);
    if (fitted) ctx.drawImage(img, ...fitted);
    return fitted;
}

function drawBadge(ctx, text, x, y, align = "left") {
    ctx.font = "12px sans-serif";
    const width = ctx.measureText(text).width + 14;
    const left = align === "right" ? x - width : x;
    ctx.fillStyle = "rgba(0, 0, 0, 0.68)";
    ctx.beginPath();
    ctx.roundRect?.(left, y, width, 22, 5);
    if (!ctx.roundRect) ctx.rect(left, y, width, 22);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + width / 2, y + 11);
    return [left, y, width, 22];
}

class CompareWidget {
    constructor(node) {
        this.type = "custom";
        this.name = "ycyy_image_compare";
        this.value = "";
        this.options = {};
        this.node = node;
        this.rect = [0, 0, MIN_WIDTH, MIN_HEIGHT];
        this.imageRect = null;
        this.pageRect = null;
        this.dragging = false;
    }

    computeSize(width) {
        return [Math.max(MIN_WIDTH, width), MIN_HEIGHT];
    }

    normalizePosition(pos) {
        if (pointInRect(pos, this.rect)) return pos;
        return [pos[0] + this.rect[0], pos[1] + this.rect[1]];
    }

    setSplit(pos) {
        const rect = this.imageRect || this.rect;
        if (!rect[2]) return;
        this.node._ycyyCompareSplit = clamp((pos[0] - rect[0]) / rect[2] * 100, 0, 100);
        requestDraw(this.node, false);
    }

    mouse(event, rawPos) {
        const pos = this.normalizePosition(rawPos);
        const type = String(event?.type || "");

        if (type.includes("up") || type.includes("cancel")) {
            const wasDragging = this.dragging;
            this.dragging = false;
            return wasDragging;
        }
        if (type.includes("down") && event?.button === 0 && pointInRect(pos, this.pageRect)) {
            const total = batchLength(this.node);
            if (total > 1) selectBatch(this.node, (this.node._ycyyCompareIndex + 1) % total);
            return true;
        }
        if (type.includes("down") && event?.button === 0 && pointInRect(pos, this.rect)) {
            this.dragging = true;
            this.setSplit(pos);
            return true;
        }
        if (type.includes("move") && this.dragging) {
            if (!((event?.buttons ?? 1) & 1)) {
                this.dragging = false;
                return false;
            }
            this.setSplit(pos);
            return true;
        }
        return false;
    }

    draw(ctx, node, width, y, widgetHeight) {
        // Node 2.0 renders legacy widgets inside a separate canvas whose width
        // is independent from the LiteGraph node's stored size.
        const nodeWidth = Number(width) || node.size?.[0] || MIN_WIDTH;
        const isolatedCanvasHeight = Number(widgetHeight) >= 100
            ? Number(widgetHeight)
            : Number(node.canvasHeight) >= 100
                ? Number(node.canvasHeight)
                : 0;
        const nodeHeight = isolatedCanvasHeight || node.size?.[1] || MIN_HEIGHT;
        this.rect = [
            PADDING,
            y + PADDING,
            Math.max(1, nodeWidth - PADDING * 2),
            Math.max(1, nodeHeight - y - PADDING * 2),
        ];

        const images = node._ycyyCompareImages || {};
        const a = images.a?.img;
        const b = images.b?.img;
        const hasA = Boolean(a?.naturalWidth);
        const hasB = Boolean(b?.naturalWidth);

        ctx.save();
        ctx.fillStyle = "#101010";
        ctx.fillRect(...this.rect);

        if (!hasA && !hasB) {
            ctx.fillStyle = "#ccc";
            ctx.font = "13px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
                translate("empty"),
                this.rect[0] + this.rect[2] / 2,
                this.rect[1] + this.rect[3] / 2,
            );
            ctx.restore();
            return;
        }

        const imageArea = [this.rect[0], this.rect[1], this.rect[2], Math.max(1, this.rect[3] - FOOTER_HEIGHT)];
        const baseRect = fitRect(a || b, imageArea) || imageArea;
        this.imageRect = baseRect;
        const splitX = baseRect[0] + baseRect[2] * (node._ycyyCompareSplit ?? 50) / 100;

        if (hasB) drawImageContained(ctx, b, baseRect);
        if (hasA && !hasB) {
            drawImageContained(ctx, a, baseRect);
        } else if (hasA) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(baseRect[0], baseRect[1], Math.max(0, splitX - baseRect[0]), baseRect[3]);
            ctx.clip();
            drawImageContained(ctx, a, baseRect);
            ctx.restore();
        }

        if (hasA && hasB) {
            ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(splitX, baseRect[1]);
            ctx.lineTo(splitX, baseRect[1] + baseRect[3]);
            ctx.stroke();
            ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
            ctx.beginPath();
            ctx.arc(splitX, baseRect[1] + baseRect[3] / 2, 8, 0, Math.PI * 2);
            ctx.fill();
        }

        const footerY = this.rect[1] + this.rect[3] - FOOTER_HEIGHT + 4;
        if (hasA) drawBadge(ctx, `A · ${a.naturalWidth} × ${a.naturalHeight}`, this.rect[0] + 8, footerY);
        if (hasB) drawBadge(ctx, `B · ${b.naturalWidth} × ${b.naturalHeight}`, this.rect[0] + this.rect[2] - 8, footerY, "right");

        const total = batchLength(node);
        if (total > 1) {
            const text = `${(node._ycyyCompareIndex || 0) + 1}/${total}`;
            ctx.font = "12px sans-serif";
            const badgeWidth = ctx.measureText(text).width + 14;
            this.pageRect = drawBadge(ctx, text, this.rect[0] + (this.rect[2] - badgeWidth) / 2, footerY);
        } else {
            this.pageRect = null;
        }
        ctx.restore();
    }
}

function activateNode(node) {
    if (!isTargetNode(node)) return;
    suppressNativePreview(node);
    if (!node._ycyyCompareWidget && typeof node.addCustomWidget === "function") {
        node._ycyyCompareSplit ??= 50;
        node._ycyyCompareIndex ??= 0;
        node._ycyyCompareImages ||= {};
        node._ycyyCompareWidget = node.addCustomWidget(new CompareWidget(node));
    }
    node.size ||= [MIN_WIDTH, MIN_HEIGHT];
    node.size[0] = Math.max(node.size[0] || 0, MIN_WIDTH);
    node.size[1] = Math.max(node.size[1] || 0, MIN_HEIGHT);
    requestDraw(node, true);
}

function disposeNode(node) {
    for (const entry of Object.values(node?._ycyyCompareImages || {})) {
        releaseImage(entry?.img);
    }
    node._ycyyCompareImages = {};
    node._ycyyCompareRefs = {};
    node._ycyyCompareWidget = null;
}

app.registerExtension({
    name: "YCYY.Image.Compare",

    async setup() {
        currentLocale = app.ui?.settings?.getSettingValue?.(LOCALE_SETTING) || "en";
        app.ui?.settings?.addEventListener?.(`${LOCALE_SETTING}.change`, handleLocaleChange);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            activateNode(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => activateNode(this), 0);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            activateNode(this);
            receiveImages(this, message);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            disposeNode(this);
            return onRemoved?.apply(this, arguments);
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function () {
            suppressNativePreview(this);
            return onDrawBackground?.apply(this, arguments);
        };
    },

    async nodeCreated(node) {
        activateNode(node);
    },
});
