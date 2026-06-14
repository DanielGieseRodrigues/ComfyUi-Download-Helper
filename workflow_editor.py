"""Friendly editor for ComfyUI workflows.

Given a workflow.json, this module locates the handful of values a user
actually wants to tweak (prompts, video duration, size, steps, cfg, seed,
LoRA strength, output name, reference image) and exposes them as flat,
labelled fields. Edited values are written straight back into the original
node widgets so the workflow can be re-exported unchanged except for the edits.

Design notes
------------
ComfyUI's ``widgets_values`` is a *positional* array (sometimes a dict) whose
meaning depends entirely on the node ``type``. So we keep a registry mapping
known node types -> which slot is which role. Unknown nodes degrade
gracefully (they simply produce no friendly fields).

Three node ecosystems show up in the wild:
  - native ComfyUI core (CLIPTextEncode, KSampler, WanImageToVideo, ...)
    -> reliably mapped here
  - kijai WanVideoWrapper (WanVideo*) -> partially mapped
  - fully custom packs (UmeAiRT_*, ...) -> not mapped (best effort)

Run as a CLI to inspect a file:
    python workflow_editor.py path/to/workflow.json
"""

import json
import os
import re as _re_alt
import sys

# --------------------------------------------------------------------------- #
# Node schema registry
#
# Each entry maps a node type to a list of widget specs:
#   (addr, role, label, vtype, group)
#     addr  : int index into widgets_values, or str key when it is a dict
#     role  : semantic role used for primary-node selection and duration math
#     label : friendly label shown in the UI
#     vtype : int | float | string | seed | combo | bool
#     group : "main" (always visible) or "advanced" (collapsed)
# --------------------------------------------------------------------------- #
SAMPLER_OPTIONS = [
    "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms",
    "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "dpmpp_sde", "ddim", "uni_pc",
    "res_multistep", "lcm",
]
SCHEDULER_OPTIONS = [
    "normal", "karras", "exponential", "simple", "sgm_uniform", "beta", "ddim_uniform",
]
SEED_CONTROL_OPTIONS = ["fixed", "increment", "decrement", "randomize"]
# UNETLoader's weight_dtype: loading an fp16/bf16 model as fp8 roughly halves
# its VRAM footprint at a small quality cost — the cheapest VRAM win available
# (no file swap, no custom node). "default" keeps the file's native precision.
WEIGHT_DTYPE_OPTIONS = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]

_LATENT_IMAGE = [
    (0, "width", "Width", "int", "main"),
    (1, "height", "Height", "int", "main"),
    (2, "batch_size", "Batch size", "int", "advanced"),
]

NODE_SCHEMAS = {
    # --- prompts -------------------------------------------------------------
    "CLIPTextEncode": [
        (0, "prompt", "Prompt", "string", "main"),
    ],
    # --- model loader (precision = VRAM lever) -------------------------------
    "UNETLoader": [
        (0, "unet_name", "Diffusion model", "string", "advanced"),
        (1, "weight_dtype", "Precision (VRAM)", "combo", "main"),
    ],
    # --- samplers ------------------------------------------------------------
    "KSampler": [
        (0, "seed", "Seed", "seed", "main"),
        (1, "seed_control", "Seed control", "combo", "advanced"),
        (2, "steps", "Steps", "int", "main"),
        (3, "cfg", "CFG", "float", "main"),
        (4, "sampler_name", "Sampler", "combo", "advanced"),
        (5, "scheduler", "Scheduler", "combo", "advanced"),
        (6, "denoise", "Denoise", "float", "advanced"),
    ],
    "KSamplerAdvanced": [
        (0, "add_noise", "Add noise", "combo", "advanced"),
        (1, "seed", "Seed", "seed", "main"),
        (2, "seed_control", "Seed control", "combo", "advanced"),
        (3, "steps", "Steps", "int", "main"),
        (4, "cfg", "CFG", "float", "main"),
        (5, "sampler_name", "Sampler", "combo", "advanced"),
        (6, "scheduler", "Scheduler", "combo", "advanced"),
        (7, "start_at_step", "Start step", "int", "advanced"),
        (8, "end_at_step", "End step", "int", "advanced"),
    ],
    # --- latent / video size -------------------------------------------------
    "EmptyLatentImage": list(_LATENT_IMAGE),
    "EmptySD3LatentImage": list(_LATENT_IMAGE),
    "WanImageToVideo": [
        (0, "width", "Width", "int", "main"),
        (1, "height", "Height", "int", "main"),
        (2, "video_length", "Frames", "int", "main"),
        (3, "batch_size", "Batch size", "int", "advanced"),
    ],
    "WanImageToVideoSVIPro": [
        (0, "video_length", "Frames", "int", "main"),
        (1, "batch_size", "Batch size", "int", "advanced"),
    ],
    "EmptyHunyuanLatentVideo": [
        (0, "width", "Width", "int", "main"),
        (1, "height", "Height", "int", "main"),
        (2, "video_length", "Frames", "int", "main"),
        (3, "batch_size", "Batch size", "int", "advanced"),
    ],
    # --- output (fps + filename) --------------------------------------------
    "VHS_VideoCombine": [
        ("frame_rate", "fps", "FPS", "int", "main"),
        ("filename_prefix", "output_name", "Output name", "string", "advanced"),
    ],
    "CreateVideo": [
        (0, "fps", "FPS", "int", "main"),
    ],
    "SaveVideo": [
        (0, "output_name", "Output name", "string", "advanced"),
    ],
    "SaveImage": [
        (0, "output_name", "Output name", "string", "advanced"),
    ],
    # --- reference image -----------------------------------------------------
    "LoadImage": [
        (0, "reference_image", "Reference image", "string", "main"),
    ],
    # --- loras ---------------------------------------------------------------
    "LoraLoaderModelOnly": [
        (0, "lora_name", "LoRA", "string", "advanced"),
        (1, "lora_strength", "LoRA strength", "float", "main"),
    ],
    "LoraLoader": [
        (0, "lora_name", "LoRA", "string", "advanced"),
        (1, "lora_strength", "LoRA strength (model)", "float", "main"),
        (2, "lora_strength_clip", "LoRA strength (CLIP)", "float", "advanced"),
    ],
}

COMBO_OPTIONS = {
    "sampler_name": SAMPLER_OPTIONS,
    "scheduler": SCHEDULER_OPTIONS,
    "seed_control": SEED_CONTROL_OPTIONS,
    "add_noise": ["enable", "disable"],
    "weight_dtype": WEIGHT_DTYPE_OPTIONS,
}

# Roles for which we only ever expose ONE field (the "primary" node).
SINGLE_ROLES = {
    "steps", "cfg", "seed", "seed_control", "denoise", "sampler_name",
    "scheduler", "width", "height", "batch_size", "video_length", "fps",
    "output_name", "reference_image", "add_noise", "start_at_step", "end_at_step",
}

# Node types that signal frame interpolation (output fps != model fps).
INTERPOLATION_TYPES = ("RIFE VFI", "FILM VFI", "GIMM VFI")


# --------------------------------------------------------------------------- #
# Traversal
# --------------------------------------------------------------------------- #
def _collect_nodes(obj, out):
    """Collect references to every node-like dict (has type + widgets_values).

    Deterministic order so field addresses stay stable between analyze/export.
    """
    if isinstance(obj, dict):
        if "type" in obj and "widgets_values" in obj:
            out.append(obj)
        for v in obj.values():
            _collect_nodes(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_nodes(v, out)


def _get_widget(node, addr):
    wv = node.get("widgets_values")
    try:
        if isinstance(wv, dict) and isinstance(addr, str):
            return wv.get(addr)
        if isinstance(wv, list) and isinstance(addr, int) and addr < len(wv):
            return wv[addr]
    except (TypeError, IndexError):
        pass
    return None


def _set_widget(node, addr, value):
    wv = node.get("widgets_values")
    if isinstance(wv, dict) and isinstance(addr, str):
        wv[addr] = value
        return True
    if isinstance(wv, list) and isinstance(addr, int) and addr < len(wv):
        wv[addr] = value
        return True
    return False


def _coerce(value, vtype):
    """Best-effort coercion of an incoming (string) value to the widget type."""
    if vtype in ("int", "seed"):
        return int(float(value))
    if vtype == "float":
        return float(value)
    if vtype == "bool":
        return bool(value) if not isinstance(value, str) else value.lower() == "true"
    return value


# --------------------------------------------------------------------------- #
# Prompt polarity
# --------------------------------------------------------------------------- #
def _prompt_polarity(node):
    """Return 'negative', 'positive' or None based on the node title."""
    title = (node.get("title") or "").lower()
    if "negative" in title:
        return "negative"
    if "positive" in title:
        return "positive"
    return None


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def _field_id(node_index, addr):
    return f"{node_index}:{addr}"


def _build_raw_fields(nodes):
    """Produce one field descriptor per recognised widget in every node."""
    fields = []
    for ni, node in enumerate(nodes):
        schema = NODE_SCHEMAS.get(node.get("type"))
        if not schema:
            continue
        for addr, role, label, vtype, group in schema:
            value = _get_widget(node, addr)
            if value is None and role not in ("reference_image", "output_name"):
                # Widget absent (e.g. shorter array variant) -> skip.
                if _get_widget(node, addr) is None and not _has_addr(node, addr):
                    continue
            fields.append({
                "id": _field_id(ni, addr),
                "node_index": ni,
                "addr": addr,
                "role": role,
                "label": label,
                "type": vtype,
                "value": value,
                "group": group,
                "node_type": node.get("type"),
                "node_title": node.get("title") or "",
            })
    return fields


def _has_addr(node, addr):
    wv = node.get("widgets_values")
    if isinstance(wv, dict):
        return isinstance(addr, str) and addr in wv
    if isinstance(wv, list):
        return isinstance(addr, int) and addr < len(wv)
    return False


SAMPLER_ROLES = {"steps", "cfg", "seed", "seed_control", "denoise",
                 "sampler_name", "scheduler", "add_noise",
                 "start_at_step", "end_at_step"}


def _is_active_sampler(node):
    """A two-stage workflow has a main sampler and a refiner. The main one
    adds noise / randomizes its seed; the refiner reuses a fixed seed. Prefer
    the main one."""
    wv = node.get("widgets_values")
    if not isinstance(wv, list):
        return False
    ntype = node.get("type")
    if ntype == "KSamplerAdvanced":
        return wv and wv[0] == "enable"            # add_noise
    if ntype == "KSampler":
        return len(wv) > 1 and wv[1] == "randomize"  # control_after_generate
    return False


def _pick_primary(cands, nodes):
    """Pick the primary field among same-role candidates.

    For sampler params, prefer the "active" sampler (adds noise / randomizes);
    otherwise keep the first in traversal order.
    """
    if len(cands) == 1:
        return cands[0]
    if cands[0]["role"] in SAMPLER_ROLES:
        for f in cands:
            if _is_active_sampler(nodes[f["node_index"]]):
                return f
    return cands[0]


def analyze(data):
    """Return the friendly field model for a workflow dict.

    Output:
      {
        "fields": [ ...primary fields, deduped by single-role... ],
        "duration": { ... } | None,
        "summary": {...},
        "warnings": [...],
      }
    """
    nodes = []
    _collect_nodes(data, nodes)
    raw = _build_raw_fields(nodes)
    warnings = []

    # --- prompts: classify polarity, keep one positive + one negative -------
    prompt_fields = [f for f in raw if f["role"] == "prompt"]
    positive = negative = None
    untitled = []
    for f in prompt_fields:
        pol = _prompt_polarity(nodes[f["node_index"]])
        if pol == "positive" and positive is None:
            positive = f
        elif pol == "negative" and negative is None:
            negative = f
        elif pol is None:
            untitled.append(f)
    # Fallbacks for untitled prompts: assume order positive-then-negative.
    if positive is None and untitled:
        positive = untitled.pop(0)
    if negative is None and untitled:
        negative = untitled.pop(0)
    if len(prompt_fields) > 1 and (positive is None or negative is None):
        warnings.append("Could not confidently tell positive/negative prompts apart.")

    final = []
    if positive:
        positive = dict(positive, role="prompt_positive", label="Positive prompt")
        final.append(positive)
    if negative:
        negative = dict(negative, role="prompt_negative", label="Negative prompt")
        final.append(negative)

    # --- single-value roles: keep only the primary -------------------------
    by_role = {}
    for f in raw:
        if f["role"] == "prompt":
            continue
        by_role.setdefault(f["role"], []).append(f)

    # All fps candidates, kept for duration math before single-role collapse.
    fps_cands = [f for f in by_role.get("fps", [])
                 if isinstance(f["value"], (int, float)) and f["value"]]

    for role, cands in by_role.items():
        if role == "fps":
            # The editable FPS field is the final/highest output rate (what the
            # user perceives); the base rate is handled by the duration block.
            final.append(max(cands, key=lambda f: f["value"] or 0))
        elif role in SINGLE_ROLES:
            final.append(_pick_primary(cands, nodes))
        else:
            final.extend(cands)

    # Disambiguate repeated labels (e.g. several LoRA strengths).
    _disambiguate_labels(final, nodes)

    # attach combo options
    for f in final:
        if f["role"] in COMBO_OPTIONS:
            f["options"] = COMBO_OPTIONS[f["role"]]
            f["type"] = "combo"

    # --- duration (frames / base fps) ---------------------------------------
    duration = _build_duration(final, fps_cands, nodes, warnings)

    # --- summary ------------------------------------------------------------
    recognised = sum(1 for n in nodes if n.get("type") in NODE_SCHEMAS)
    category = _guess_category(final)
    summary = {
        "node_count": len(nodes),
        "recognized_nodes": recognised,
        "category": category,
    }
    # stable order: prompts first, then by group(main first), then label
    final.sort(key=lambda f: (
        0 if f["role"].startswith("prompt") else 1,
        0 if f["group"] == "main" else 1,
        f["label"],
    ))
    return {
        "fields": final,
        "duration": duration,
        "summary": summary,
        "warnings": warnings,
    }


def _disambiguate_labels(fields, nodes):
    """When several fields share a label (e.g. multiple LoRA strengths),
    append a short hint so the UI stays unambiguous."""
    seen = {}
    for f in fields:
        seen.setdefault(f["label"], []).append(f)
    for label, group in seen.items():
        if len(group) < 2:
            continue
        for i, f in enumerate(group, 1):
            hint = f["node_title"]
            # For repeated loaders/loras, prefer the sibling model name (widget 0)
            # over a bare index — e.g. WAN 2.2's high-noise vs low-noise UNETs.
            if not hint and f["role"] in ("lora_strength", "weight_dtype"):
                name = _get_widget(nodes[f["node_index"]], 0)  # sibling name widget
                if isinstance(name, str):
                    hint = name.rsplit("/", 1)[-1].rsplit(".", 1)[0][:24]
            f["label"] = f"{label} ({hint or i})"


def _build_duration(fields, fps_cands, nodes, warnings):
    length_field = next((f for f in fields if f["role"] == "video_length"), None)
    if not length_field or not isinstance(length_field["value"], (int, float)):
        return None
    frames = int(length_field["value"])
    interpolated = any(n.get("type") in INTERPOLATION_TYPES for n in nodes)
    if not fps_cands:
        warnings.append("No FPS node found; duration shown in frames only.")
        return {
            "length_field_id": length_field["id"],
            "frames": frames, "fps": None, "seconds": None,
            "interpolated": interpolated,
        }
    # Base FPS = lowest detected rate. Interpolation (RIFE/FILM) only raises the
    # final playback rate, so the model's true duration uses the lowest.
    base_fps = min(f["value"] for f in fps_cands)
    if interpolated or len({f["value"] for f in fps_cands}) > 1:
        warnings.append(
            "This workflow interpolates frames; duration is computed from the "
            "model's base FPS (%g), the saved file may play at a higher FPS." % base_fps)
    return {
        "length_field_id": length_field["id"],
        "frames": frames,
        "fps": base_fps,
        "seconds": round(frames / base_fps, 2),
        "interpolated": interpolated,
    }


def _guess_category(fields):
    roles = {f["role"] for f in fields}
    has_video = "video_length" in roles or "fps" in roles
    has_ref = "reference_image" in roles
    if has_video and has_ref:
        return "img2vid"
    if has_video:
        return "txt2vid"
    if has_ref:
        return "img2img"
    return "txt2img"


# --------------------------------------------------------------------------- #
# VRAM estimate (deliberately rough, honest)
#
# Predicting ComfyUI's real VRAM peak exactly is impossible from the workflow
# alone — it depends on the attention impl, offload mode, block swap, VAE
# tiling, and more. So this is a coarse order-of-magnitude estimate in two parts:
#   1. model weight  — fairly reliable: inferred from the parameter count in the
#      file name and the chosen precision (fp8 ≈ half of fp16/bf16).
#   2. runtime overhead — latents + activations, which for video scale with
#      width × height × frames. Calibrated to land in the right ballpark only.
# The deliverable is the verdict (fits / tight / won't fit), not a precise GB.
# --------------------------------------------------------------------------- #
import re as _re  # local alias; keeps this block self-contained

# Bytes per parameter for a given precision.
_BYTES_PER_PARAM = {
    "bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "fp8": 1.0,
    # GGUF quant levels ≈ bits/8 plus a little metadata overhead.
    "q8": 1.06, "q6": 0.82, "q5": 0.68, "q4": 0.56, "q3": 0.43, "q2": 0.34,
}
# Calibrated against known-good runs: 832×480×81f WAN ≈ a few GB of overhead,
# 1280×720×81f markedly more, a single 1024² image well under 1 GB.
_OVERHEAD_GB_PER_UNIT = 5.0e-7
_OS_HEADROOM_GB = 1.5   # VRAM the OS/driver/compositor keeps for itself
_MISC_GB = 1.0          # VAE + framework working set, roughly


def _infer_param_count_b(name):
    """Guess parameter count in billions from a file name ('..._14B_...' -> 14).

    Note: '_' is a regex word char, so '\\b' fails on '14B_bf16'; we use explicit
    non-alphanumeric lookarounds instead so underscores act as separators."""
    m = _re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", (name or "").lower())
    return float(m.group(1)) if m else None


def _infer_file_bytes_per_param(name):
    """Bytes/param implied by precision tags in the file name (Qn, fp8, bf16...)."""
    s = (name or "").lower()
    m = _re.search(r"(?<![a-z0-9])q(\d)(?:_[0-9k]+)?(?![a-z0-9])", s)
    if m:
        return _BYTES_PER_PARAM.get("q" + m.group(1))
    if "fp8" in s or "e4m3" in s or "e5m2" in s:
        return _BYTES_PER_PARAM["fp8"]
    if "fp16" in s or "bf16" in s:
        return _BYTES_PER_PARAM["bf16"]
    if "fp32" in s:
        return _BYTES_PER_PARAM["fp32"]
    return None


def estimate_vram(category, width, height, frames,
                  unet_name=None, weight_dtype=None, vram_total_gb=None):
    """Rough VRAM estimate for a generation. See module section header.

    Returns a dict the UI can render. All GB numbers are estimates; ``verdict``
    is the useful part and is only set when ``vram_total_gb`` is known.
    """
    params_b = _infer_param_count_b(unet_name)
    file_bpp = _infer_file_bytes_per_param(unet_name)

    # The loader's weight_dtype overrides the file's native precision when fp8.
    if weight_dtype and "fp8" in str(weight_dtype).lower():
        bpp, precision = _BYTES_PER_PARAM["fp8"], "fp8"
    elif file_bpp is not None:
        bpp, precision = file_bpp, "file"
    else:
        bpp, precision = _BYTES_PER_PARAM["bf16"], "bf16 (assumed)"

    model_gb = round(params_b * bpp, 1) if params_b else None

    def _int(x, default=0):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return default

    w = _int(width)
    h = _int(height)
    f = _int(frames, 1) or 1
    is_video = category in ("img2vid", "txt2vid") and f > 1
    latent_frames = ((f - 1) // 4 + 1) if is_video else 1   # WAN temporal ≈ 4x
    overhead_gb = (round(w * h * latent_frames * _OVERHEAD_GB_PER_UNIT, 1)
                   if w and h else None)

    peak_gb = (round(model_gb + (overhead_gb or 0) + _MISC_GB, 1)
               if model_gb is not None else None)

    verdict, advice = "unknown", []
    if vram_total_gb and peak_gb is not None:
        usable = vram_total_gb - _OS_HEADROOM_GB
        if peak_gb <= usable * 0.85:
            verdict = "fits"
        elif peak_gb <= usable * 1.05:
            verdict = "tight"
        else:
            verdict = "over"

    usable = (vram_total_gb - _OS_HEADROOM_GB) if vram_total_gb else None
    # When the weights alone exceed usable VRAM, the model is the blocker —
    # shrinking resolution/frames can't help, so we don't suggest it.
    model_exceeds = (model_gb is not None and usable is not None
                     and model_gb > usable)

    if verdict in ("tight", "over"):
        if params_b and bpp > _BYTES_PER_PARAM["fp8"]:
            # Native fp16/bf16 weights — fp8 is the cheap, in-editor win.
            advice.append(
                "Set Precision to fp8 — roughly halves the model's VRAM "
                "(~%.0f GB saved)." % (model_gb / 2))
        elif params_b and model_exceeds:
            # Already fp8/GGUF and the weights alone don't fit.
            advice.append(
                "The model alone (~%.0f GB) is bigger than your usable VRAM, so "
                "lowering resolution or frames won't make it fit — switch to a "
                "GGUF quant (Q4) or enable block swap." % model_gb)
        elif params_b:
            advice.append(
                "Use a GGUF quant (Q4) or enable block swap to fit a model "
                "this large.")
        else:
            advice.append(
                "Couldn't read the model size from its file name, so this is a "
                "partial estimate.")
        if is_video and overhead_gb and overhead_gb > 2 and not model_exceeds:
            advice.append(
                "Lower the resolution or frame count — overhead scales with "
                "width × height × frames.")

    return {
        "is_estimate": True,
        "params_b": params_b,
        "precision": precision,
        "model_gb": model_gb,
        "overhead_gb": overhead_gb,
        "peak_gb": peak_gb,
        "vram_total_gb": vram_total_gb,
        "verdict": verdict,       # fits | tight | over | unknown
        "advice": advice,
        "is_video": is_video,
    }


# --------------------------------------------------------------------------- #
# Apply edits + export
# --------------------------------------------------------------------------- #
def apply_edits(data, edits, duration_seconds=None):
    """Mutate ``data`` in place with edited values and return it.

    ``edits``: {field_id: new_value}
    ``duration_seconds``: optional; converted to frames using the base fps.
    """
    nodes = []
    _collect_nodes(data, nodes)
    model = analyze(data)
    by_id = {f["id"]: f for f in model["fields"]}

    # Duration -> frames first. Uses the model's base FPS (the same rate the UI
    # showed), so "20 seconds" lands on the model's real frame count.
    if duration_seconds is not None and model["duration"] and model["duration"]["fps"]:
        fps = model["duration"]["fps"]
        frames = max(1, round(float(duration_seconds) * fps))
        edits = dict(edits)
        edits[model["duration"]["length_field_id"]] = frames

    applied = []
    for fid, value in edits.items():
        f = by_id.get(fid)
        if not f:
            continue
        node = nodes[f["node_index"]]
        try:
            coerced = _coerce(value, f["type"])
        except (TypeError, ValueError):
            coerced = value
        if _set_widget(node, f["addr"], coerced):
            applied.append(fid)
    return data, applied


# --------------------------------------------------------------------------- #
# Speed-up injection ("Generate Faster")
#
# Every meaningful speed lever for diffusion is a node that takes a MODEL and
# returns a patched MODEL (TeaCache, torch.compile, WaveSpeed first-block cache,
# a Sage-attention patch...). So injection is uniform: find the MODEL wire that
# feeds the sampler and splice the patch node(s) onto it. Plus one non-node
# lever — fp8 precision on UNETLoader — which is just a widget value.
#
# We inject TeaCache (the most reliable, compiler-free win — it caches similar
# diffusion steps) plus fp8. The injected TeaCache node needs the ComfyUI-TeaCache
# custom node installed; that, plus heavier install-level levers (SageAttention,
# the C++/Triton toolchain for torch.compile) are reported as "needs_setup".
#
# Two on-disk graph shapes exist and we handle both:
#   - flat litegraph  : top-level "links" are arrays [id,oid,os,tid,ts,type]
#   - subgraphs       : definitions.subgraphs[].links are objects {id,...}
# --------------------------------------------------------------------------- #
# A node type whose input named "model" we treat as "the sampler" — the splice
# point. Substring match catches WanVideoSampler etc.; the MODEL-type guard
# below keeps us from touching non-native (WANVIDEOMODEL) wires.
def _is_sampler(node):
    t = node.get("type") or ""
    return "Sampler" in t or "KSampler" in t


# Loaders whose precision (weight_dtype) widget we can flip to fp8.
_FP8_LOADERS = {"UNETLoader"}

_LINK_ARRAY_IDX = {"id": 0, "origin_id": 1, "origin_slot": 2,
                   "target_id": 3, "target_slot": 4, "type": 5}


def _iter_graphs(data):
    """Yield (graph, link_format) for the top-level graph and every subgraph."""
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        yield data, _graph_link_format(data, "array")
    subs = (((data.get("definitions") or {}).get("subgraphs")) or []
            ) if isinstance(data, dict) else []
    for s in subs:
        if isinstance(s, dict) and isinstance(s.get("nodes"), list):
            yield s, _graph_link_format(s, "object")


def _graph_link_format(graph, default):
    links = graph.get("links")
    if isinstance(links, list) and links:
        return "object" if isinstance(links[0], dict) else "array"
    return default


def _lget(link, key, fmt):
    return link[_LINK_ARRAY_IDX[key]] if fmt == "array" else link.get(key)


def _lset(link, key, val, fmt):
    if fmt == "array":
        link[_LINK_ARRAY_IDX[key]] = val
    else:
        link[key] = val


def _make_link(fmt, lid, oid, oslot, tid, tslot, ltype):
    if fmt == "array":
        return [lid, oid, oslot, tid, tslot, ltype]
    return {"id": lid, "origin_id": oid, "origin_slot": oslot,
            "target_id": tid, "target_slot": tslot, "type": ltype}


def _find_link(graph, link_id, fmt):
    for l in graph.get("links") or []:
        if _lget(l, "id", fmt) == link_id:
            return l
    return None


def _node_by_id(graph, node_id):
    for n in graph.get("nodes") or []:
        if n.get("id") == node_id:
            return n
    return None


def _next_ids(graph, fmt):
    """Highest node/link id in use, honoring both the actual contents and the
    bookkeeping fields (top-level last_*; subgraph state.last*)."""
    node_max = max([n.get("id") for n in graph.get("nodes") or []
                    if isinstance(n.get("id"), int)] + [0])
    link_max = max([_lget(l, "id", fmt) for l in graph.get("links") or []
                    if isinstance(_lget(l, "id", fmt), int)] + [0])
    st = graph.get("state") or {}
    node_max = max(node_max, graph.get("last_node_id") or 0, st.get("lastNodeId") or 0)
    link_max = max(link_max, graph.get("last_link_id") or 0, st.get("lastLinkId") or 0)
    return node_max, link_max


def _write_ids(graph, node_id, link_id):
    if "last_node_id" in graph:
        graph["last_node_id"] = node_id
    if "last_link_id" in graph:
        graph["last_link_id"] = link_id
    st = graph.get("state")
    if isinstance(st, dict):
        if "lastNodeId" in st:
            st["lastNodeId"] = node_id
        if "lastLinkId" in st:
            st["lastLinkId"] = link_id


def _model_input_slot(node):
    """Index of the node's native MODEL input (the splice point)."""
    for i, inp in enumerate(node.get("inputs") or []):
        if inp.get("type") == "MODEL" and "model" in (inp.get("name") or "").lower():
            return i
    return None


# ComfyUI-TeaCache "TeaCache" node. model_type MUST match the model family, or
# the caching uses the wrong polynomial coefficients (hurts quality), so we only
# inject when we can identify the family — otherwise we skip and say so.
TEACACHE_MODEL_TYPES = [
    "flux", "flux-kontext", "ltxv", "lumina_2", "hunyuan_video",
    "hidream_i1_full", "hidream_i1_dev", "hidream_i1_fast",
    "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B",
]
# Balanced cache aggressiveness: raise toward 0.4 for more speed, lower toward
# 0.15 for more fidelity. The node's own default is 0.4 (speed-leaning).
TEACACHE_THRESH = 0.25


def _teacache_node(model_type, thresh=TEACACHE_THRESH):
    """ComfyUI-TeaCache 'TeaCache' node (MODEL -> MODEL). Caches similar steps.
    widgets: [model_type, rel_l1_thresh, start_percent, end_percent, cache_device]."""
    return {
        "type": "TeaCache",
        "title": "⚡ TeaCache (Generate Faster)",
        "flags": {}, "mode": 0, "order": 0,
        # cnr_id + aux_id identify the node's pack to ComfyUI Manager. Without
        # them the frontend can't bind the node to the installed ComfyUI-TeaCache
        # pack, so it keeps reporting it as a missing node even after install.
        "properties": {
            "cnr_id": "teacache",
            "aux_id": "welltop-cn/ComfyUI-TeaCache",
            "Node name for S&R": "TeaCache",
        },
        "widgets_values": [model_type, thresh, 0.0, 1.0, "cuda"],
        "size": [320, 130],
        "inputs": [{"name": "model", "type": "MODEL"}],
        # output name is lowercase "model" to match the pack's real definition.
        "outputs": [{"name": "model", "type": "MODEL", "slot_index": 0, "links": []}],
    }


def _text_blobs(data):
    """Lowercased node types, titles and string widget values — the haystack we
    sniff the model family out of."""
    out, nodes = [], []
    _collect_nodes(data, nodes)
    for n in nodes:
        for key in ("type", "title"):
            v = n.get(key)
            if isinstance(v, str):
                out.append(v.lower())
        wv = n.get("widgets_values")
        vals = wv.values() if isinstance(wv, dict) else (wv or [])
        out += [v.lower() for v in vals if isinstance(v, str)]
    return " ".join(out)


def _detect_teacache_model_type(data):
    """Best-effort TeaCache model_type from the workflow's model files/nodes.
    Returns the type string, or None when we can't tell confidently."""
    b = _text_blobs(data)
    if "wan" in b:
        if "i2v" in b or "imagetovideo" in b or "image_to_video" in b:
            return "wan2.1_i2v_720p_14B" if "720" in b else "wan2.1_i2v_480p_14B"
        return "wan2.1_t2v_1.3B" if "1.3b" in b else "wan2.1_t2v_14B"
    if "flux" in b:
        return "flux-kontext" if "kontext" in b else "flux"
    if "ltx" in b:
        return "ltxv"
    if "hunyuan" in b:
        return "hunyuan_video"
    if "lumina" in b:
        return "lumina_2"
    if "hidream" in b:
        if "fast" in b:
            return "hidream_i1_fast"
        if "dev" in b:
            return "hidream_i1_dev"
        return "hidream_i1_full"
    return None


def _splice_patches(graph, fmt, sampler, factories, node_id, link_id):
    """Insert ``factories`` (model-patch node builders) between the sampler's
    current model source and the sampler. Returns (created, node_id, link_id)
    or (None, ...) when there's no native MODEL wire to splice."""
    slot = _model_input_slot(sampler)
    if slot is None:
        return None, node_id, link_id
    in_link_id = sampler["inputs"][slot].get("link")
    orig = _find_link(graph, in_link_id, fmt) if in_link_id is not None else None
    if orig is None:
        return None, node_id, link_id

    # Idempotency: if the wire already comes from one of our patch types, skip.
    src = _node_by_id(graph, _lget(orig, "origin_id", fmt))
    want_types = {f().get("type") for f in factories}
    if src and src.get("type") in want_types:
        return None, node_id, link_id

    base = sampler.get("pos") if isinstance(sampler.get("pos"), list) else [0, 0]
    x, y = base[0] - 340, base[1] - 40
    created = []
    # Hop 0: redirect the original link so the first patch consumes it.
    prev = orig
    for fac in factories:
        node_id += 1
        n = fac()
        n["id"] = node_id
        n["pos"] = [x, y]
        x -= 340
        n["inputs"][0]["link"] = _lget(prev, "id", fmt)
        if prev is orig:
            _lset(orig, "target_id", node_id, fmt)
            _lset(orig, "target_slot", 0, fmt)
        created.append(n)
        if prev is not orig:
            # link feeding this node was created last loop; nothing more to do.
            pass
        prev_node_id = node_id
        # Create the outgoing link; its target is fixed up on the next hop / end.
        link_id += 1
        new_link = _make_link(fmt, link_id, prev_node_id, 0, sampler["id"], slot, "MODEL")
        graph.setdefault("links", []).append(new_link)
        n["outputs"][0]["links"] = [link_id]
        # If there's a next patch, it will retarget this link to itself.
        prev = new_link
    # The last created link already targets the sampler; wire the sampler input.
    sampler["inputs"][slot]["link"] = _lget(prev, "id", fmt)
    # Fix intermediate links: each non-final new link must target the NEXT patch,
    # not the sampler. Walk created nodes and repoint.
    for i in range(len(created) - 1):
        feeding = _find_link(graph, created[i]["outputs"][0]["links"][0], fmt)
        _lset(feeding, "target_id", created[i + 1]["id"], fmt)
        _lset(feeding, "target_slot", 0, fmt)
        created[i + 1]["inputs"][0]["link"] = _lget(feeding, "id", fmt)
    graph["nodes"].extend(created)
    return created, node_id, link_id


# --------------------------------------------------------------------------- #
# Lighter-model suggestions ("swap the heavy model for a leaner one")
#
# A second, file-level VRAM lever: point the workflow at a smaller-but-compatible
# version of the same model (a pre-quantized fp8 file, or a GGUF quant). Unlike
# the weight_dtype fp8 cast — which needs no download but keeps the big file on
# disk — these are real, smaller files. The catalog (metadata/model-alternatives
# .json) maps a model family to verified, loader-appropriate alternatives.
#
# Safe to apply automatically: an fp8 *diffusion-model* file that loads in the
# SAME loader node (just a different filename). Everything else (GGUF needs the
# ComfyUI-GGUF node + a different loader; distilled/turbo changes the sampling
# recipe) is surfaced as a suggestion only.
# --------------------------------------------------------------------------- #
_ALTERNATIVES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "metadata", "model-alternatives.json")
_alt_cache = {"loaded": False, "families": []}

# Loader node type -> index of the model-filename widget in widgets_values.
_DIFFUSION_LOADER_FILENAME_IDX = {
    "UNETLoader": 0,
    "CheckpointLoaderSimple": 0,
    "CheckpointLoader": 0,
    "UnetLoaderGGUF": 0,
    "UnetLoaderGGUFAdvanced": 0,
}


_PREQUANT_RE = _re_alt.compile(
    r"(fp8|e4m3|e5m2|nf4|gguf|[-_]q\d|scaled)", _re_alt.IGNORECASE)


def _is_prequantized_filename(name):
    """True when the file name already signals a low-precision/quantized model
    (fp8, fp8_scaled, GGUF, Qn, nf4) — i.e. no fp8 re-cast is needed/wanted."""
    return bool(_PREQUANT_RE.search(os.path.basename(name or "")))


def _load_alternatives():
    if not _alt_cache["loaded"]:
        _alt_cache["loaded"] = True
        try:
            with open(_ALTERNATIVES_PATH, "r", encoding="utf-8") as f:
                _alt_cache["families"] = json.load(f).get("families", [])
        except (OSError, json.JSONDecodeError):
            _alt_cache["families"] = []
    return _alt_cache["families"]


def _diffusion_loader_refs(data):
    """Yield (node, filename_widget_idx, current_filename) for every diffusion-
    model loader whose filename widget we can read."""
    for graph, _fmt in _iter_graphs(data):
        for n in graph.get("nodes") or []:
            idx = _DIFFUSION_LOADER_FILENAME_IDX.get(n.get("type"))
            if idx is None:
                continue
            wv = n.get("widgets_values")
            if not (isinstance(wv, list) and len(wv) > idx):
                continue
            fn = wv[idx]
            if isinstance(fn, str) and fn.strip():
                yield n, idx, fn


def _alternatives_for(filename, loader_type):
    """Return catalog alternatives whose family matches ``filename`` and that
    apply to ``loader_type``. Empty when the file is already a light variant."""
    stem = os.path.basename(filename or "").lower()
    out = []
    for fam in _load_alternatives():
        try:
            if not _re_alt.search(fam.get("match", ""), stem):
                continue
            if fam.get("skip_if") and _re_alt.search(fam["skip_if"], stem):
                continue
        except _re_alt.error:
            continue
        for alt in fam.get("alternatives", []):
            fl = alt.get("for_loaders")
            if fl and loader_type not in fl:
                continue
            out.append({**alt, "family": fam.get("label", "")})
    return out


def suggest_lighter_models(data, apply_safe=True):
    """Find heavy diffusion models in ``data`` and suggest leaner equivalents.

    When ``apply_safe`` is true, a same-loader fp8 alternative is applied in
    place (the loader's filename widget is repointed). Other alternatives are
    only suggested. Returns ``{suggestions, applied}`` where each suggestion is
    a dict the UI can render (and download from)."""
    suggestions, applied = [], []
    seen = set()
    for node, idx, current in _diffusion_loader_refs(data):
        loader_type = node.get("type")
        for alt in _alternatives_for(current, loader_type):
            key = (current.lower(), alt["filename"].lower())
            if key in seen:
                continue
            seen.add(key)
            short = os.path.basename(current)
            auto = bool(apply_safe and alt.get("same_loader")
                        and alt.get("kind") == "fp8" and alt.get("url"))
            if auto:
                node["widgets_values"][idx] = alt["filename"]
                msg = (f"{short} → {alt['filename']} (fp8, "
                       f"{alt.get('size_text', 'smaller')}) — repointed in the "
                       "workflow; download it below to use it.")
                # The file is already fp8, so the loader must NOT re-cast it:
                # reset weight_dtype to "default" (required for fp8_scaled files,
                # harmless for plain fp8). UNETLoader keeps weight_dtype at idx 1.
                wv = node["widgets_values"]
                if (loader_type == "UNETLoader" and isinstance(wv, list)
                        and len(wv) > 1 and str(wv[1]).lower() != "default"):
                    wv[1] = "default"
                    msg += " Precision set back to 'default' (the file is already fp8)."
                applied.append(msg)
            suggestions.append({
                "current": short,
                "loader": loader_type,
                "filename": alt["filename"],
                "url": alt.get("url", ""),
                "directory": alt.get("directory", "diffusion_models"),
                "size_text": alt.get("size_text", ""),
                "kind": alt.get("kind", ""),
                "same_loader": bool(alt.get("same_loader")),
                "loader_node": alt.get("loader_node", ""),
                "family": alt.get("family", ""),
                "note": alt.get("note", ""),
                "auto_applied": auto,
            })
    return {"suggestions": suggestions, "applied": applied}


def inject_speedups(data, options=None):
    """Inject speed-up nodes into ``data`` (mutated in place).

    options:
      fp8_value      : weight_dtype to set on UNETLoaders ("fp8_e4m3fn" default,
                       None to skip). "fp8_e4m3fn_fast" is faster on RTX 40xx+.
      teacache       : splice a TeaCache node before each sampler (default True).
      teacache_thresh: rel_l1_thresh for TeaCache (default TEACACHE_THRESH).
      lighter        : suggest lighter compatible models (default True).
      lighter_apply  : auto-repoint safe fp8 swaps in the workflow (default True).

    Returns a report dict: {applied, skipped, needs_setup, suggestions}.
    """
    options = options or {}
    fp8_value = options.get("fp8_value", "fp8_e4m3fn")
    do_teacache = options.get("teacache", True)
    thresh = options.get("teacache_thresh", TEACACHE_THRESH)
    do_lighter = options.get("lighter", True)
    lighter_apply = options.get("lighter_apply", True)

    applied, skipped, needs = [], [], []

    # 0) Lighter-model suggestions (and safe fp8 file swaps) FIRST, so the fp8
    #    precision pass below sees the post-swap filenames and won't re-cast a
    #    file that is already fp8.
    suggestions = []
    if do_lighter:
        light = suggest_lighter_models(data, apply_safe=lighter_apply)
        suggestions = light["suggestions"]
        applied.extend(light["applied"])
        if any(s["kind"] == "gguf" for s in suggestions):
            needs.append("To use a GGUF alternative, install the ComfyUI-GGUF "
                         "custom node and swap the loader to 'Unet Loader (GGUF)'.")

    # TeaCache needs the model family up front so we can set model_type correctly.
    model_type = _detect_teacache_model_type(data) if do_teacache else None
    factories = []
    if do_teacache:
        if model_type:
            factories = [lambda mt=model_type, th=thresh: _teacache_node(mt, th)]
        else:
            skipped.append(
                "Couldn't tell the model family from the workflow, so TeaCache "
                "wasn't added — open it in ComfyUI, drop a TeaCache node before "
                "the sampler and pick the matching model_type.")

    teacache_any = False
    for graph, fmt in _iter_graphs(data):
        node_id, link_id = _next_ids(graph, fmt)

        # 1) fp8 precision on diffusion-model loaders.
        if fp8_value:
            for n in graph.get("nodes") or []:
                if n.get("type") not in _FP8_LOADERS:
                    continue
                wv = n.get("widgets_values")
                if not (isinstance(wv, list) and len(wv) >= 2):
                    continue
                cur = str(wv[1]).lower()
                name = (wv[0] if wv else "") or "model"
                short = str(name).rsplit("/", 1)[-1]
                if _is_prequantized_filename(name):
                    # Already a quantized file (incl. a file we just swapped in) —
                    # leave weight_dtype alone; re-casting would hurt fp8_scaled.
                    skipped.append(f"{short}: already a quantized file — left precision as is.")
                elif "fp8" in cur:
                    skipped.append(f"{short}: already fp8 ({wv[1]}).")
                else:
                    wv[1] = fp8_value
                    applied.append(f"{short}: precision → {fp8_value} (≈ half the VRAM).")

        # 2) TeaCache before each sampler.
        if factories:
            for n in list(graph.get("nodes") or []):
                if not _is_sampler(n):
                    continue
                created, node_id, link_id = _splice_patches(
                    graph, fmt, n, factories, node_id, link_id)
                if created:
                    teacache_any = True

        _write_ids(graph, node_id, link_id)

    if teacache_any:
        applied.append(
            f"Added TeaCache (model_type={model_type}, rel_l1_thresh={thresh}) "
            "before the sampler — caches similar steps for ~2× fewer compute steps.")
        needs.append("Install the ComfyUI-TeaCache custom node (or this node "
                     "will load as missing).")
    elif do_teacache and model_type:
        skipped.append("No MODEL→sampler wire found to add TeaCache.")

    if fp8_value == "fp8_e4m3fn":
        needs.append("On an RTX 40xx/50xx you can switch fp8 to the faster "
                     "'fp8_e4m3fn_fast' variant for extra speed.")

    return {"applied": applied, "skipped": skipped, "needs_setup": needs,
            "suggestions": suggestions}


# --------------------------------------------------------------------------- #
# CLI test harness
# --------------------------------------------------------------------------- #
def _cli(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    model = analyze(data)
    print(f"\n# {path}")
    print("summary:", model["summary"])
    if model["duration"]:
        d = model["duration"]
        print(f"duration: {d['seconds']}s  ({d['frames']} frames @ {d['fps']} fps"
              f"{', interpolated' if d['interpolated'] else ''})")
    for w in model["warnings"]:
        print("  ! warning:", w)
    print("fields:")
    for fld in model["fields"]:
        val = fld["value"]
        if isinstance(val, str) and len(val) > 40:
            val = val[:40] + "..."
        # never echo full prompt text
        if fld["role"].startswith("prompt"):
            val = f"<text, {len(fld['value'] or '')} chars>"
        print(f"   [{fld['group']}] {fld['label']:<22} = {val!r}"
              f"   ({fld['node_type']} #{fld['id']})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        _cli(p)
