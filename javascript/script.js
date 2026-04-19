// based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/script.js
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
    initStylePreviewOverlay();
});

var onAppend = function(elem, f) {
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.addedNodes.length) {
                f(m.addedNodes);
            }
        });
    });
    observer.observe(elem, {childList: true});
}

function addObserverIfDesiredNodeAvailable(querySelector, callback) {
    var elem = document.querySelector(querySelector);
    if (!elem) {
        window.setTimeout(() => addObserverIfDesiredNodeAvailable(querySelector, callback), 1000);
        return;
    }

    onAppend(elem, callback);
}

/**
 * Show reset button on toast "Connection errored out."
 */
addObserverIfDesiredNodeAvailable(".toast-wrap", function(added) {
    added.forEach(function(element) {
         if (element.innerText.includes("Connection errored out.")) {
             window.setTimeout(function() {
                document.getElementById("reset_button").classList.remove("hidden");
                document.getElementById("generate_button").classList.add("hidden");
                document.getElementById("skip_button").classList.add("hidden");
                document.getElementById("stop_button").classList.add("hidden");
            });
         }
    });
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
    const isEnterKey = (e.key == "Enter" || e.keyCode == 13);

    if(isModifierKey && isEnterKey) {
        const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
        if (generateButton) {
            generateButton.click();
            e.preventDefault();
            return;
        }

        const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]')
        if(stopButton) {
            stopButton.click();
            e.preventDefault();
            return;
        }
    }
});

function initStylePreviewOverlay() {
    let overlayVisible = false;
    const samplesPath = document.querySelector("meta[name='samples-path']").getAttribute("content")
    const overlay = document.createElement('div');
    const tooltip = document.createElement('div');
    tooltip.className = 'preview-tooltip';
    overlay.appendChild(tooltip);
    overlay.id = 'stylePreviewOverlay';
    document.body.appendChild(overlay);
    document.addEventListener('mouseover', function (e) {
        const label = e.target.closest('.style_selections label');
        if (!label) return;
        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);
        overlayVisible = true;
        overlay.style.opacity = "1";
        const originalText = label.querySelector("span").getAttribute("data-original-text");
        const name = originalText || label.querySelector("span").textContent;
        overlay.style.backgroundImage = `url("${samplesPath.replace(
            "fooocus_v2",
            name.toLowerCase().replaceAll(" ", "_")
        ).replaceAll("\\", "\\\\")}")`;

        tooltip.textContent = name;

        function onMouseLeave() {
            overlayVisible = false;
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    });
    document.addEventListener('mousemove', function (e) {
        if (!overlayVisible) return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    });
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}

function playNotification() {
    gradioApp().querySelector('#audio_notification audio')?.play();
}

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function htmlDecode(input) {
  var doc = new DOMParser().parseFromString(input, "text/html");
  return doc.documentElement.textContent;
}

function setGradioTextValue(elemId, value) {
    const root = gradioApp();
    const container = root.querySelector('#' + elemId);
    if (!container) return;
    const input = container.querySelector('textarea, input');
    if (!input) return;
    if (input.value === value) return;

    const prototype = input.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(prototype, 'value');
    descriptor.set.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
}

function getGradioTextValue(elemId) {
    const root = gradioApp();
    const container = root.querySelector('#' + elemId);
    if (!container) return '';
    const input = container.querySelector('textarea, input');
    return input ? input.value || '' : '';
}

function createNoobaiRegionEditor() {
    const editorRoot = gradioApp().querySelector('#noobai_inpaint_region_controls');
    const status = gradioApp().querySelector('#noobai_region_status');
    const sourceRoot = gradioApp().querySelector('#noobai_inpaint_canvas');

    if (!editorRoot || !status || !sourceRoot) {
        return null;
    }

    if (editorRoot.dataset.initialized === 'true') {
        return editorRoot._noobaiRegionEditor || null;
    }

    editorRoot.dataset.initialized = 'true';
    const state = {
        rects: [],
        selectedIndex: -1,
        mode: 'brush',
        dragMode: null,
        dragRectIndex: -1,
        dragStart: null,
        draftRect: null,
        sourceElement: null,
        overlayCanvas: null,
        overlayHost: null,
        overlayParent: null,
        ctx: null,
        sourceWidth: 1,
        sourceHeight: 1,
        lastSignature: '',
    };

    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

    const serialize = () => {
        if (state.rects.length === 0) return '';
        const payload = {
            normalized: true,
            regions: state.rects.map((rect) => ({
                x: Number(rect.x.toFixed(6)),
                y: Number(rect.y.toFixed(6)),
                width: Number(rect.width.toFixed(6)),
                height: Number(rect.height.toFixed(6)),
            })),
        };
        return JSON.stringify(payload);
    };

    const syncToTextbox = () => {
        setGradioTextValue('noobai_inpaint_regions_data', serialize());
        const modeLabel = state.mode === 'rect' ? 'Rectangle Mask' : 'Brush Mask';
        status.textContent = `${modeLabel} | Rectangles: ${state.rects.length}`;
    };

    const parseTextboxValue = () => {
        const raw = getGradioTextValue('noobai_inpaint_regions_data').trim();
        if (raw === '') {
            state.rects = [];
            state.selectedIndex = -1;
            return;
        }
        try {
            const parsed = JSON.parse(raw);
            const regions = Array.isArray(parsed) ? parsed : (Array.isArray(parsed.regions) ? parsed.regions : []);
            const normalized = !Array.isArray(parsed) && Boolean(parsed.normalized);
            state.rects = regions.map((region) => {
                if (normalized) {
                    return {
                        x: Number(region.x ?? 0),
                        y: Number(region.y ?? 0),
                        width: Number(region.width ?? region.w ?? 0),
                        height: Number(region.height ?? region.h ?? 0),
                    };
                }
                const width = Number(region.width ?? region.w ?? 0);
                const height = Number(region.height ?? region.h ?? 0);
                return {
                    x: Number(region.x ?? 0) / Math.max(state.sourceWidth, 1),
                    y: Number(region.y ?? 0) / Math.max(state.sourceHeight, 1),
                    width: width / Math.max(state.sourceWidth, 1),
                    height: height / Math.max(state.sourceHeight, 1),
                };
            }).filter((rect) => rect.width > 0 && rect.height > 0);
            state.selectedIndex = state.rects.length === 0 ? -1 : Math.min(state.selectedIndex, state.rects.length - 1);
        } catch (error) {
            console.warn('Failed to parse NoobAI region state:', error);
            state.rects = [];
            state.selectedIndex = -1;
        }
    };

    const ensureOverlay = () => {
        if (state.overlayCanvas && state.overlayCanvas.isConnected) {
            return;
        }

        const overlay = document.createElement('canvas');
        overlay.id = 'noobai_region_overlay_canvas';
        overlay.className = 'noobai-region-overlay-canvas';
        overlay.tabIndex = 0;
        overlay.setAttribute('aria-label', 'NoobAI rectangle region editor');
        state.overlayCanvas = overlay;
        state.ctx = overlay.getContext('2d');
        overlay.addEventListener('mousedown', handleMouseDown);
    };

    const getSourceElement = () => {
        const candidates = Array.from(sourceRoot.querySelectorAll('canvas, img')).filter((element) => {
            const rect = element.getBoundingClientRect();
            return rect.width > 20 && rect.height > 20;
        });
        if (candidates.length === 0) return null;
        candidates.sort((a, b) => {
            const areaA = a.getBoundingClientRect().width * a.getBoundingClientRect().height;
            const areaB = b.getBoundingClientRect().width * b.getBoundingClientRect().height;
            return areaB - areaA;
        });
        return candidates[0];
    };

    const updateSource = () => {
        ensureOverlay();
        const source = getSourceElement();
        if (!source) return false;

        const parent = source.parentElement;
        if (!parent) return false;

        if (getComputedStyle(parent).position === 'static') {
            parent.style.position = 'relative';
        }

        if (state.overlayParent !== parent) {
            if (state.overlayCanvas.parentElement) {
                state.overlayCanvas.parentElement.removeChild(state.overlayCanvas);
            }
            parent.appendChild(state.overlayCanvas);
            state.overlayParent = parent;
        }

        const width = source.tagName === 'IMG'
            ? (source.naturalWidth || source.width || 1)
            : (source.width || source.clientWidth || 1);
        const height = source.tagName === 'IMG'
            ? (source.naturalHeight || source.height || 1)
            : (source.height || source.clientHeight || 1);

        const rect = source.getBoundingClientRect();
        const parentRect = parent.getBoundingClientRect();

        state.overlayCanvas.style.left = `${rect.left - parentRect.left}px`;
        state.overlayCanvas.style.top = `${rect.top - parentRect.top}px`;
        state.overlayCanvas.style.width = `${rect.width}px`;
        state.overlayCanvas.style.height = `${rect.height}px`;

        const dpr = window.devicePixelRatio || 1;
        const canvasWidth = Math.max(1, Math.round(rect.width * dpr));
        const canvasHeight = Math.max(1, Math.round(rect.height * dpr));
        if (state.overlayCanvas.width !== canvasWidth || state.overlayCanvas.height !== canvasHeight) {
            state.overlayCanvas.width = canvasWidth;
            state.overlayCanvas.height = canvasHeight;
            state.ctx.setTransform(1, 0, 0, 1, 0, 0);
            state.ctx.scale(dpr, dpr);
        }

        const signature = `${width}x${height}:${rect.width}x${rect.height}:${source.src || ''}`;
        state.sourceElement = source;
        state.overlayHost = source;
        state.sourceWidth = width;
        state.sourceHeight = height;
        if (state.lastSignature === signature) return false;
        state.lastSignature = signature;
        return true;
    };

    const normalizedToCanvasRect = (rect) => {
        const overlayRect = state.overlayCanvas.getBoundingClientRect();
        return {
            x: rect.x * overlayRect.width,
            y: rect.y * overlayRect.height,
            width: rect.width * overlayRect.width,
            height: rect.height * overlayRect.height,
        };
    };

    const canvasPointToNormalized = (event) => {
        const rect = state.overlayCanvas.getBoundingClientRect();
        const x = clamp((event.clientX - rect.left) / rect.width, 0, 1);
        const y = clamp((event.clientY - rect.top) / rect.height, 0, 1);
        return {
            x,
            y,
        };
    };

    const hitTest = (point) => {
        for (let index = state.rects.length - 1; index >= 0; index -= 1) {
            const rect = state.rects[index];
            if (
                point.x >= rect.x &&
                point.x <= rect.x + rect.width &&
                point.y >= rect.y &&
                point.y <= rect.y + rect.height
            ) {
                return index;
            }
        }
        return -1;
    };

    const draw = () => {
        if (!state.sourceElement || !state.overlayCanvas || !state.ctx) return;

        const rect = state.overlayCanvas.getBoundingClientRect();
        state.ctx.clearRect(0, 0, rect.width, rect.height);

        state.rects.forEach((rect, index) => {
            const canvasRect = normalizedToCanvasRect(rect);
            const selected = index === state.selectedIndex;
            state.ctx.save();
            state.ctx.lineWidth = selected ? 3 : 2;
            state.ctx.strokeStyle = selected ? '#ffb000' : '#00bcd4';
            state.ctx.fillStyle = selected ? 'rgba(255, 176, 0, 0.18)' : 'rgba(0, 188, 212, 0.16)';
            state.ctx.fillRect(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);
            state.ctx.strokeRect(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);
            state.ctx.restore();
        });

        if (state.draftRect) {
            const canvasRect = normalizedToCanvasRect(state.draftRect);
            state.ctx.save();
            state.ctx.lineWidth = 2;
            state.ctx.strokeStyle = '#ffffff';
            state.ctx.setLineDash([6, 4]);
            state.ctx.strokeRect(canvasRect.x, canvasRect.y, canvasRect.width, canvasRect.height);
            state.ctx.restore();
        }

        state.overlayCanvas.style.pointerEvents = state.mode === 'rect' ? 'auto' : 'none';
        state.overlayCanvas.classList.toggle('rect-mode', state.mode === 'rect');
        syncToTextbox();
    };

    const refresh = () => {
        const changed = updateSource();
        if (!state.sourceElement) return;
        if (changed && state.rects.length > 0) {
            parseTextboxValue();
        }
        draw();
    };

    function handleMouseDown(event) {
        if (state.mode !== 'rect') return;
        if (!state.sourceElement) return;
        if (state.overlayCanvas && typeof state.overlayCanvas.focus === 'function') {
            state.overlayCanvas.focus({ preventScroll: true });
        }
        const point = canvasPointToNormalized(event);
        const hitIndex = hitTest(point);
        state.selectedIndex = hitIndex;

        if (hitIndex >= 0) {
            state.dragMode = 'move';
            state.dragRectIndex = hitIndex;
            state.dragStart = {
                point,
                rect: { ...state.rects[hitIndex] },
            };
        } else {
            state.dragMode = 'draw';
            state.dragRectIndex = -1;
            state.dragStart = { point };
            state.draftRect = {
                x: point.x,
                y: point.y,
                width: 0,
                height: 0,
            };
        }

        draw();
        event.preventDefault();
    }

    window.addEventListener('mousemove', (event) => {
        if (state.mode !== 'rect') return;
        if (!state.dragMode || !state.dragStart) return;
        const point = canvasPointToNormalized(event);

        if (state.dragMode === 'move' && state.dragRectIndex >= 0) {
            const rect = state.dragStart.rect;
            const dx = point.x - state.dragStart.point.x;
            const dy = point.y - state.dragStart.point.y;
            state.rects[state.dragRectIndex] = {
                ...rect,
                x: clamp(rect.x + dx, 0, 1 - rect.width),
                y: clamp(rect.y + dy, 0, 1 - rect.height),
            };
        }

        if (state.dragMode === 'draw') {
            const x1 = clamp(Math.min(state.dragStart.point.x, point.x), 0, 1);
            const y1 = clamp(Math.min(state.dragStart.point.y, point.y), 0, 1);
            const x2 = clamp(Math.max(state.dragStart.point.x, point.x), 0, 1);
            const y2 = clamp(Math.max(state.dragStart.point.y, point.y), 0, 1);
            state.draftRect = {
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
            };
        }

        draw();
    });

    window.addEventListener('mouseup', () => {
        if (state.mode !== 'rect' && !state.dragMode) return;
        if (!state.dragMode) return;

        if (state.dragMode === 'draw' && state.draftRect) {
            if (state.draftRect.width > 0.01 && state.draftRect.height > 0.01) {
                state.rects.push(state.draftRect);
                state.selectedIndex = state.rects.length - 1;
            }
            state.draftRect = null;
        }

        state.dragMode = null;
        state.dragRectIndex = -1;
        state.dragStart = null;
        draw();
    });

    const brushModeButton = gradioApp().querySelector('#noobai_region_mode_brush');
    const rectModeButton = gradioApp().querySelector('#noobai_region_mode_rect');
    const deleteButton = gradioApp().querySelector('#noobai_region_delete');
    const clearButton = gradioApp().querySelector('#noobai_region_clear');

    const updateModeButtons = () => {
        if (brushModeButton) {
            brushModeButton.classList.toggle('selected', state.mode === 'brush');
            brushModeButton.setAttribute('aria-pressed', state.mode === 'brush' ? 'true' : 'false');
        }
        if (rectModeButton) {
            rectModeButton.classList.toggle('selected', state.mode === 'rect');
            rectModeButton.setAttribute('aria-pressed', state.mode === 'rect' ? 'true' : 'false');
        }
    };

    if (brushModeButton) {
        brushModeButton.addEventListener('click', () => {
            state.mode = 'brush';
            state.dragMode = null;
            state.draftRect = null;
            updateModeButtons();
            draw();
        });
    }

    if (rectModeButton) {
        rectModeButton.addEventListener('click', () => {
            state.mode = 'rect';
            updateModeButtons();
            draw();
        });
    }

    if (deleteButton) {
        deleteButton.addEventListener('click', () => {
            deleteSelectedRect();
        });
    }

    if (clearButton) {
        clearButton.addEventListener('click', () => {
            state.rects = [];
            state.selectedIndex = -1;
            draw();
        });
    }

    const deleteSelectedRect = () => {
        if (state.selectedIndex < 0) return;
        state.rects.splice(state.selectedIndex, 1);
        state.selectedIndex = state.rects.length === 0 ? -1 : Math.min(state.selectedIndex, state.rects.length - 1);
        draw();
    };

    const isDeleteShortcut = (event) => {
        if (event.key === 'Delete' || event.key === 'Backspace') return true;
        if (event.code === 'Backslash' || event.code === 'IntlYen' || event.code === 'IntlRo') return true;
        if (event.key === '\\' || event.key === '\u00A5') return true;
        return false;
    };

    document.addEventListener('keydown', (event) => {
        if (state.selectedIndex < 0) return;

        const activeElement = document.activeElement;
        const isTypingTarget = activeElement && (
            activeElement.tagName === 'INPUT' ||
            activeElement.tagName === 'TEXTAREA' ||
            activeElement.isContentEditable
        );
        if (isTypingTarget) return;

        if (isDeleteShortcut(event)) {
            deleteSelectedRect();
            event.preventDefault();
        }
    });

    parseTextboxValue();
    updateModeButtons();
    refresh();
    window.setInterval(refresh, 1200);

    editorRoot._noobaiRegionEditor = { refresh };
    return editorRoot._noobaiRegionEditor;
}

function initNoobaiRegionEditor() {
    createNoobaiRegionEditor();
}

onUiLoaded(initNoobaiRegionEditor);
onAfterUiUpdate(initNoobaiRegionEditor);
