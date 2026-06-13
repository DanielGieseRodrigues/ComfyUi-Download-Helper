"""Extract the list of models/dependencies from a ComfyUI workflow.json.

Two sources are used:
  1. The `properties.models[]` lists inside each node (reliable source: provides
     directory + name + url). Nodes may be at the root or inside
     `definitions.subgraphs[].nodes`, which is why the scan is recursive.
  2. Model links written in `MarkdownNote` nodes (e.g. an optional LoRA that is
     not wired to any node). Flagged as optional.
"""

import re

MODEL_EXTS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin",
    ".gguf", ".onnx", ".sft", ".vae",
)

# Maps the loader node type -> subfolder inside models/. Used to guess where a
# model "referenced without a link" should go.
NODE_DIR = {
    "CheckpointLoaderSimple": "checkpoints",
    "CheckpointLoader": "checkpoints",
    "ImageOnlyCheckpointLoader": "checkpoints",
    "unCLIPCheckpointLoader": "checkpoints",
    "UNETLoader": "diffusion_models",
    "UnetLoaderGGUF": "diffusion_models",
    "VAELoader": "vae",
    "CLIPLoader": "text_encoders",
    "DualCLIPLoader": "text_encoders",
    "TripleCLIPLoader": "text_encoders",
    "QuadrupleCLIPLoader": "text_encoders",
    "CLIPVisionLoader": "clip_vision",
    "LoraLoader": "loras",
    "LoraLoaderModelOnly": "loras",
    "ControlNetLoader": "controlnet",
    "DiffControlNetLoader": "controlnet",
    "UpscaleModelLoader": "upscale_models",
    "StyleModelLoader": "style_models",
    "GLIGENLoader": "gligen",
}


def _walk_models(obj, found):
    """Recursively scan for `models` lists in any node."""
    if isinstance(obj, dict):
        models = obj.get("models")
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict) and m.get("url") and m.get("name"):
                    found.append({
                        "name": m.get("name"),
                        "directory": m.get("directory") or "checkpoints",
                        "url": m.get("url"),
                        "optional": False,
                    })
        for v in obj.values():
            _walk_models(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_models(v, found)


def _collect_markdown_texts(obj, texts):
    if isinstance(obj, dict):
        if obj.get("type") == "MarkdownNote":
            for t in (obj.get("widgets_values") or []):
                if isinstance(t, str):
                    texts.append(t)
        for v in obj.values():
            _collect_markdown_texts(v, texts)
    elif isinstance(obj, list):
        for v in obj:
            _collect_markdown_texts(v, texts)


_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_HEADER_RE = re.compile(r"\*\*([A-Za-z0-9_]+)\*\*")


def _parse_markdown_notes(obj, found):
    texts = []
    _collect_markdown_texts(obj, texts)
    for text in texts:
        current_dir = None
        for line in text.splitlines():
            header = _HEADER_RE.search(line)
            if header:
                current_dir = header.group(1)
            for name, url in _LINK_RE.findall(line):
                if url.lower().endswith(MODEL_EXTS):
                    found.append({
                        "name": name,
                        "directory": current_dir or "checkpoints",
                        "url": url,
                        "optional": True,
                    })


def _collect_nodes(obj, out):
    """Gather every dict that looks like a node (has widgets_values)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("widgets_values"), list):
            out.append(obj)
        for v in obj.values():
            _collect_nodes(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_nodes(v, out)


def extract_referenced(data):
    """Model filenames referenced in node widgets (with or without a link).

    Used to detect dependencies the workflow uses but provides no URL for.
    The subfolder is guessed from the loader node type.
    """
    nodes = []
    _collect_nodes(data, nodes)
    refs = []
    seen = set()
    for node in nodes:
        ntype = node.get("type", "")
        for w in node.get("widgets_values", []):
            if isinstance(w, str) and w.lower().endswith(MODEL_EXTS):
                key = w.lower()
                if key in seen:
                    continue
                seen.add(key)
                refs.append({
                    "name": w,
                    "directory": NODE_DIR.get(ntype, "checkpoints"),
                    "node_type": ntype,
                })
    return refs


def parse_workflow(data):
    """Return a list of dicts: {name, directory, url, optional}, deduplicated."""
    structured = []
    _walk_models(data, structured)

    markdown = []
    _parse_markdown_notes(data, markdown)

    # Dedupe by URL. If it shows up in both sources, treat it as required.
    by_url = {}
    for m in structured + markdown:
        key = m["url"]
        if key not in by_url:
            by_url[key] = m
        elif not m["optional"]:
            by_url[key]["optional"] = False

    return list(by_url.values())
