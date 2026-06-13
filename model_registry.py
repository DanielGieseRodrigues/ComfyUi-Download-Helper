"""Curated model registry: turns a model name into a precise download URL.

This is the layer that makes "search by name" actually useful. Most models used
in ComfyUI workflows are well-known files whose exact download URL is already
cataloged by the ComfyUI-Manager community list. Instead of throwing the raw
query at the HuggingFace/Civitai APIs (which return a pile of loosely-related
repos), we first look the name up in this curated catalog and return the exact
file. The API search is only a fallback for things the catalog doesn't know.

Data sources (bundled snapshots under metadata/, refreshable from upstream):
  - model-list.json    : the ComfyUI-Manager model database (filename -> url)
  - model-aliases.json : maps name variants / quantization suffixes to canonicals
"""

import json
import logging
import os
import re
import threading
from difflib import SequenceMatcher

import requests

log = logging.getLogger("comfyui-helper")

BASE = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(BASE, "metadata")
BUNDLED_LIST = os.path.join(META_DIR, "model-list.json")
CACHED_LIST = os.path.join(META_DIR, "model-list.cache.json")  # refreshed copy wins
ALIASES_PATH = os.path.join(META_DIR, "model-aliases.json")

# Canonical ComfyUI-Manager model database.
UPSTREAM_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"

# For entries whose save_path is "default", derive the folder from the type.
TYPE_DIR = {
    "checkpoint": "checkpoints", "unclip": "checkpoints", "zero123": "checkpoints",
    "CustomNet": "checkpoints", "Janus-Pro": "checkpoints",
    "diffusion_model": "diffusion_models", "FramePackI2V": "diffusion_models",
    "controlnet": "controlnet", "T2I-Adapter": "controlnet",
    "T2I-Style": "style_models",
    "lora": "loras", "motion lora": "animatediff_motion_lora",
    "animatediff": "animatediff_models", "animatediff-pia": "animatediff_models",
    "clip": "text_encoders", "clip_vision": "clip_vision",
    "VAE": "vae", "vae": "vae",
    "upscale": "upscale_models", "RGT": "upscale_models",
    "embedding": "embeddings", "TAESD": "vae_approx",
    "IP-Adapter": "ipadapter", "gligen": "gligen",
    "Ultralytics": "ultralytics", "insightface": "insightface",
    "sam": "sams", "sam2": "sams", "sam2.1": "sams",
    "GFPGAN": "facerestore_models", "CodeFormer": "facerestore_models",
    "face_restore": "facerestore_models", "facexlib": "facedetection",
    "LLM": "LLM", "photomaker": "photomaker", "instantid": "instantid",
    "PuLID": "pulid", "depthanything": "depthanything",
}

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin",
              ".gguf", ".onnx", ".sft", ".vae")

# Quantization / variant suffixes to strip so "model_fp8.safetensors" still
# matches "model.safetensors" both in the catalog and in API queries.
_QUANT_RE = re.compile(
    r"[-_.]?(fp16|fp8|fp32|bf16|e4m3fn|e5m2|q\d+(_[0-9k]+)?|"
    r"int8|int4|nf4|gguf|scaled|pruned|emaonly|ema|fixed)\b",
    re.IGNORECASE,
)

_lock = threading.Lock()
_state = {"models": None, "aliases": None}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _entry_directory(m):
    sp = (m.get("save_path") or "").strip()
    if sp and sp.lower() != "default":
        return sp.replace("\\", "/")
    return TYPE_DIR.get(m.get("type", ""), "checkpoints")


def _load_raw_models():
    path = CACHED_LIST if os.path.exists(CACHED_LIST) else BUNDLED_LIST
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load model registry (%s): %s", path, exc)
        return []
    out = []
    for m in data.get("models", []):
        fn = (m.get("filename") or "").strip()
        url = (m.get("url") or "").strip()
        if not fn or not url:
            continue
        out.append({
            "filename": fn,
            "name": m.get("name") or fn,
            "type": m.get("type") or "",
            "base": m.get("base") or "",
            "directory": _entry_directory(m),
            "url": url,
            "size": m.get("size") or "",
        })
    return out


def _load_aliases():
    try:
        with open(ALIASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"aliases": {}, "patterns": []}
    return {"aliases": data.get("aliases", {}), "patterns": data.get("patterns", [])}


def _models():
    with _lock:
        if _state["models"] is None:
            _state["models"] = _load_raw_models()
            log.info("Model registry loaded: %d catalog entries.", len(_state["models"]))
        return _state["models"]


def _aliases():
    with _lock:
        if _state["aliases"] is None:
            _state["aliases"] = _load_aliases()
        return _state["aliases"]


def refresh_from_upstream(verify=True):
    """Download the latest ComfyUI-Manager model list and cache it locally.
    Returns (ok, message). Falls back silently to the bundled snapshot on error."""
    try:
        r = requests.get(UPSTREAM_URL, timeout=40, verify=verify,
                         headers={"User-Agent": "comfyui-helper"})
        r.raise_for_status()
        data = r.json()
        count = len(data.get("models", []))
        if count == 0:
            return False, "Upstream list was empty; kept the current catalog."
        os.makedirs(META_DIR, exist_ok=True)
        with open(CACHED_LIST, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with _lock:
            _state["models"] = None  # force reload on next use
        log.info("Model registry refreshed from upstream: %d entries.", count)
        return True, f"Catalog updated: {count} models."
    except Exception as exc:  # noqa: BLE001
        log.warning("Registry refresh failed: %s", exc)
        return False, f"Could not refresh (kept current catalog): {exc}"


# --------------------------------------------------------------------------- #
# Name normalization & matching
# --------------------------------------------------------------------------- #
def _stem(name):
    return os.path.splitext(os.path.basename(name or ""))[0].lower()


def strip_quant(name):
    """Remove a trailing quantization/variant tag from a filename or query."""
    stem, ext = os.path.splitext(name or "")
    cleaned = _QUANT_RE.sub("", stem).strip(" -_.")
    return (cleaned + ext) if ext else cleaned


def _tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) >= 2]


def resolve_alias(filename):
    """Return the canonical filename for a known alias / quantization variant,
    or the input unchanged."""
    al = _aliases()
    low = filename.lower()
    for canonical, variants in al.get("aliases", {}).items():
        if low in [v.lower() for v in variants] or low == canonical.lower():
            return canonical
    for pat in al.get("patterns", []):
        try:
            m = re.match(pat["pattern"], filename, re.IGNORECASE)
        except re.error:
            continue
        if m:
            base = pat["base"]
            for i, g in enumerate(m.groups(), start=1):
                base = base.replace(f"${i}", g or "")
            return base
    return filename


def score(query, entry):
    """Relevance score (0..1000) of a catalog entry for a query.

    Works for both filename-like queries ("flux1-dev.safetensors") and free
    text ("flux dev"). Higher is better; anything under ~250 is noise."""
    fn = entry["filename"].lower()
    fn_stem = _stem(fn)
    q = query.lower().strip()
    q_stem = _stem(q) if ("." in q and q.rsplit(".", 1)[-1] in
                          {e.lstrip(".") for e in MODEL_EXTS}) else q

    if q == fn or q_stem == fn_stem:
        return 1000

    canon = resolve_alias(query).lower()
    if canon == fn or _stem(canon) == fn_stem:
        return 970

    # quant-stripped exact match (model_fp8 -> model)
    if _stem(strip_quant(q)) == _stem(strip_quant(fn)):
        return 900

    best = 0
    # substring containment on the stem
    if q_stem and q_stem in fn_stem:
        best = max(best, 760)
    if fn_stem and fn_stem in q_stem:
        best = max(best, 700)

    # token coverage over filename + display name + base model
    hay = f"{fn} {entry['name']} {entry['base']}".lower()
    qtok = _tokens(q)
    cov = (sum(1 for t in qtok if t in hay) / len(qtok)) if qtok else 1.0
    best = max(best, int(cov * 650))

    # fuzzy similarity as a final tie-breaker / typo tolerance
    ratio = SequenceMatcher(None, q_stem, fn_stem).ratio()
    best = max(best, int(ratio * 520))

    # Hard ceiling: if the query has several words and barely any matched, it's
    # unrelated — don't let fuzzy character-overlap rescue it into the results.
    if len(qtok) >= 2 and cov < 0.5:
        best = min(best, 150)
    return best


def search(query, limit=12, min_score=250):
    """Search the curated catalog. Returns result dicts shaped like the API
    search results, sorted best-first, each tagged with a relevance score."""
    query = (query or "").strip()
    if not query:
        return []
    scored = []
    for e in _models():
        s = score(query, e)
        if s >= min_score:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, e in scored[:limit]:
        out.append({
            "filename": e["filename"],
            "model_name": e["name"],
            "version": e["base"],
            "model_type": e["type"],
            "directory": e["directory"],
            "source": "registry",            # UI badge; real source derived from url
            "url": e["url"],
            "size_kb": None,
            "size_text": e["size"],
            "score": s,
            "registry": True,
        })
    return out
