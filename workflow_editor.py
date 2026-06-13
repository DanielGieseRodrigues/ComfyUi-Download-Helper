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
            if not hint and f["role"] == "lora_strength":
                name = _get_widget(nodes[f["node_index"]], 0)  # sibling lora_name
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
