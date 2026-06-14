"""ComfyUI Helper - local server.

Serves a simple web page to:
  - map the ComfyUI folder
  - store tokens (HuggingFace / Civitai)
  - upload a workflow.json, list its dependencies and download everything
    automatically into the right folders (models/<directory>/<name>).
"""

import json
import logging
import os
import struct
import subprocess
import threading
import uuid

import requests
from flask import Flask, jsonify, request, send_from_directory

import custom_nodes
import model_registry
import workflow_editor
from workflow_parser import MODEL_EXTS, extract_referenced, parse_workflow

# Generic filenames that exist in almost every HF repo and tell us nothing about
# the model the user is after — skip them so they don't drown the results.
HF_SKIP_FILENAMES = {
    "pytorch_model.bin", "adapter_model.bin", "diffusion_pytorch_model.bin",
    "model.safetensors", "model.fp16.safetensors", "model.bin", "model.ckpt",
    "model.pt", "config.json", "tokenizer.json", "optimizer.pt",
    "training_args.bin", "diffusion_pytorch_model.safetensors",
}

# Minimum relevance (model_registry.score, 0..1000) for a result to be shown.
SEARCH_MIN_SCORE = 200

# Civitai model type -> ComfyUI subfolder.
CIVITAI_TYPE_DIR = {
    "Checkpoint": "checkpoints",
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "TextualInversion": "embeddings",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "Upscaler": "upscale_models",
    "MotionModule": "diffusion_models",
}

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
LOG_PATH = os.path.join(BASE, "helper.log")

# Config defaults. Includes verify_ssl: by default we validate certificates,
# but the user can turn it off if the environment (antivirus/proxy) blocks it.
DEFAULT_CONFIG = {
    "comfyui_path": "",
    # ComfyUI program folder (the one with custom_nodes/ and main.py). Only
    # needed when it differs from comfyui_path — e.g. a shared-models setup
    # where comfyui_path points at a models-only folder. Empty = same as
    # comfyui_path.
    "comfyui_install_path": "",
    "hf_token": "",
    "civitai_token": "",
    "civitai_red_token": "",
    "verify_ssl": True,
}

# ---------------------------------------------------------------------------
# Logging to file + console (helps diagnose download failures)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("comfyui-helper")

# Uses the Windows certificate store (fixes most SSL errors caused by
# antivirus/proxy software that injects its own root certificate).
try:
    import truststore
    truststore.inject_into_ssl()
    log.info("truststore active: using the operating system certificates.")
except Exception as exc:  # noqa: BLE001
    log.warning("truststore unavailable (%s); falling back to certifi certificates.", exc)

app = Flask(__name__, static_folder="static", static_url_path="")

# In-progress download jobs: job_id -> {name, downloaded, total, status, error}
jobs = {}
jobs_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = {}
    else:
        cfg = {}
    for k, default in DEFAULT_CONFIG.items():
        cfg.setdefault(k, default)
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def detect_source(url):
    u = url.lower()
    if "huggingface.co" in u or "hf.co" in u:
        return "huggingface"
    if "civitai.red" in u:
        return "civitai_red"
    if "civitai.com" in u:
        return "civitai"
    return "direct"


def target_path(cfg, item):
    return os.path.join(cfg["comfyui_path"], "models", item["directory"], item["name"])


def _looks_like_install(path):
    """A folder is a ComfyUI program install if it has custom_nodes/ or main.py."""
    return bool(path) and (
        os.path.isdir(os.path.join(path, "custom_nodes"))
        or os.path.isfile(os.path.join(path, "main.py")))


def resolve_install_path(cfg):
    """Folder where custom_nodes/ lives. Uses the explicit install path when set,
    otherwise the models path when it looks like a real install. Returns None when
    we can't tell (then custom-node install is disabled with a helpful message)."""
    explicit = (cfg.get("comfyui_install_path") or "").strip()
    if explicit:
        return explicit
    mp = (cfg.get("comfyui_path") or "").strip()
    if _looks_like_install(mp):
        return mp
    return None


def check_status(cfg, item):
    path = cfg.get("comfyui_path")
    if not path:
        return {"exists": False, "reason": "no_path"}
    fp = target_path(cfg, item)
    if os.path.isfile(fp) and os.path.getsize(fp) > 0:
        return {"exists": True, "size": os.path.getsize(fp), "path": fp}
    return {"exists": False, "path": fp}


# --------------------------------------------------------------------------- #
# Static routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        cfg = load_config()
        data = request.get_json(force=True) or {}
        for k in DEFAULT_CONFIG:
            if k in data:
                if k == "verify_ssl":
                    cfg[k] = bool(data[k])
                else:
                    cfg[k] = (data[k] or "").strip()
        save_config(cfg)
        return jsonify(ok=True)
    return jsonify(load_config())


@app.route("/api/validate-path", methods=["POST"])
def validate_path():
    data = request.get_json(force=True) or {}
    path = (data.get("path") or "").strip()
    if not path or not os.path.isdir(path):
        return jsonify(valid=False, msg="Folder not found.")
    has_models = os.path.isdir(os.path.join(path, "models"))
    msg = "OK - valid folder." if has_models else \
        "The folder exists but has no 'models' subfolder. The folders will be created on download."
    return jsonify(valid=True, has_models=has_models, msg=msg)


@app.route("/api/parse", methods=["POST"])
def parse():
    try:
        if "file" in request.files:
            data = json.load(request.files["file"])
        else:
            data = request.get_json(force=True)
    except (json.JSONDecodeError, ValueError):
        return jsonify(error="Invalid JSON."), 400

    cfg = load_config()
    models = parse_workflow(data)
    for m in models:
        m["source"] = detect_source(m["url"])
        m["status"] = check_status(cfg, m)

    # Models referenced in the workflow that do NOT have a download link.
    have = {m["name"].lower() for m in models}
    missing = [r for r in extract_referenced(data) if r["name"].lower() not in have]
    for r in missing:
        r["status"] = check_status(cfg, r)

    # Custom node packs the workflow depends on (from each node's cnr_id/aux_id).
    install_path = resolve_install_path(cfg)
    nodes = custom_nodes.extract_node_packs(data)
    for p in nodes:
        p["status"] = custom_nodes.pack_install_status(p, install_path or "")

    return jsonify(models=models, missing=missing, custom_nodes=nodes,
                   comfyui_path=cfg.get("comfyui_path", ""),
                   install_path=install_path or "")


# --------------------------------------------------------------------------- #
# Search by name (when there is no link or the download failed)
# --------------------------------------------------------------------------- #
def _rel_score(query, filename, model_name="", base=""):
    """Relevance of an API result for the query (reuses the registry scorer)."""
    return model_registry.score(
        query, {"filename": filename or "", "name": model_name or "", "base": base or ""})


def _search_civitai(base, query, token, source, verify):
    headers = {"User-Agent": "comfyui-helper"}
    if token:
        headers["Authorization"] = "Bearer " + token
    clean = model_registry.strip_quant(query) or query
    out = []
    try:
        r = requests.get(f"{base}/api/v1/models",
                         params={"query": clean, "limit": 10, "sort": "Most Downloaded"},
                         headers=headers, timeout=30, verify=verify)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Search on %s failed: %s", source, exc)
        return out
    for item in data.get("items", []):
        mtype = item.get("type", "")
        directory = CIVITAI_TYPE_DIR.get(mtype, "checkpoints")
        model_name = item.get("name", "")
        added = 0
        for ver in item.get("modelVersions", []):
            files = ver.get("files", [])
            # Only the actual model weight (primary / type=="Model"), not the
            # config/VAE/training-data extras Civitai bundles into a version.
            primary = next((f for f in files if f.get("primary")), None)
            chosen = primary or next((f for f in files if f.get("type") == "Model"), None)
            if not chosen:
                continue
            url = chosen.get("downloadUrl")
            if not url:
                continue
            fn = chosen.get("name") or ""
            out.append({
                "filename": fn,
                "model_name": model_name,
                "version": ver.get("name"),
                "model_type": mtype,
                "directory": directory,
                "source": source,
                "url": url,
                "size_kb": chosen.get("sizeKB"),
                "score": _rel_score(query, fn, model_name, model_name),
            })
            added += 1
            if added >= 3:  # cap versions per model so one model can't flood results
                break
    return out


def _search_huggingface(query, token, verify):
    headers = {"User-Agent": "comfyui-helper"}
    if token:
        headers["Authorization"] = "Bearer " + token
    clean = model_registry.strip_quant(query) or query
    out = []
    try:
        # Sort by downloads: the community-validated repos float to the top,
        # which is most of what makes a Google search land on the right link.
        r = requests.get("https://huggingface.co/api/models",
                         params={"search": clean, "limit": 10,
                                 "sort": "downloads", "direction": -1},
                         headers=headers, timeout=30, verify=verify)
        r.raise_for_status()
        repos = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Search on HuggingFace failed: %s", exc)
        return out
    for repo in repos[:6]:
        repo_id = repo.get("id") or repo.get("modelId")
        if not repo_id:
            continue
        try:
            d = requests.get(f"https://huggingface.co/api/models/{repo_id}",
                             headers=headers, timeout=30, verify=verify).json()
            siblings = d.get("siblings", [])
        except Exception:  # noqa: BLE001
            siblings = []
        # Score every weight file in the repo, then keep only the few best ones
        # so a repo with dozens of shards doesn't dump junk into the results.
        scored = []
        for s in siblings:
            path = s.get("rfilename", "")
            fn = path.split("/")[-1]
            if not fn.lower().endswith(MODEL_EXTS) or fn.lower() in HF_SKIP_FILENAMES:
                continue
            sc = _rel_score(query, fn, repo_id)
            scored.append((sc, fn, path))
        scored.sort(key=lambda x: x[0], reverse=True)
        for sc, fn, path in scored[:4]:
            if sc < SEARCH_MIN_SCORE:
                continue
            out.append({
                "filename": fn,
                "model_name": repo_id,
                "version": "",
                "model_type": "",
                "directory": "checkpoints",
                "source": "huggingface",
                "url": f"https://huggingface.co/{repo_id}/resolve/main/{path}",
                "size_kb": None,
                "score": sc,
            })
        if len(out) >= 40:
            break
    return out


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    sources = data.get("sources") or ["civitai", "civitai_red"]
    if not query:
        return jsonify(results=[])
    cfg = load_config()
    verify = cfg.get("verify_ssl", True)
    log.info("Search for '%s' on %s", query, sources)

    results = []

    # 1) Curated catalog first — exact name -> exact URL for well-known models.
    #    Always consulted; it's the highest-quality layer and needs no network.
    registry_hits = model_registry.search(query)
    for r in registry_hits:
        r["source"] = detect_source(r["url"])  # real provider for the badge/token
    results += registry_hits

    # 2) Live APIs as a supplement for anything the catalog doesn't know.
    if "civitai" in sources:
        results += _search_civitai("https://civitai.com", query,
                                    cfg.get("civitai_token"), "civitai", verify)
    if "civitai_red" in sources:
        results += _search_civitai("https://civitai.red", query,
                                   cfg.get("civitai_red_token"), "civitai_red", verify)
    if "huggingface" in sources:
        results += _search_huggingface(query, cfg.get("hf_token"), verify)

    # Make sure every result has a score, dedupe by URL (keeping the best),
    # drop low-relevance noise, and sort best-first.
    for r in results:
        r.setdefault("score", _rel_score(query, r.get("filename"),
                                         r.get("model_name"), r.get("version")))
    best = {}
    for r in results:
        key = (r.get("url") or "").split("?")[0].lower()
        if not key:
            continue
        if key not in best or r["score"] > best[key]["score"]:
            # Preserve the curated flag if either copy of the URL had it.
            if best.get(key, {}).get("registry"):
                r["registry"] = True
            best[key] = r
    merged = [r for r in best.values() if r["score"] >= SEARCH_MIN_SCORE]
    merged.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(results=merged[:40])


@app.route("/api/registry/refresh", methods=["POST"])
def registry_refresh():
    cfg = load_config()
    ok, msg = model_registry.refresh_from_upstream(verify=cfg.get("verify_ssl", True))
    return jsonify(ok=ok, msg=msg)


@app.route("/api/check", methods=["POST"])
def check():
    item = request.get_json(force=True) or {}
    return jsonify(check_status(load_config(), item))


def do_download(job_id, cfg, item):
    url = item["url"]
    source = detect_source(url)
    headers = {"User-Agent": "comfyui-helper"}

    if source == "huggingface" and cfg.get("hf_token"):
        headers["Authorization"] = "Bearer " + cfg["hf_token"]
    if source == "civitai" and cfg.get("civitai_token"):
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={cfg['civitai_token']}"
    if source == "civitai_red" and cfg.get("civitai_red_token"):
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={cfg['civitai_red_token']}"

    dest = target_path(cfg, item)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    verify = cfg.get("verify_ssl", True)

    log.info("Downloading %s (%s) from %s [verify_ssl=%s]",
             item["name"], source, url.split("?")[0], verify)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60,
                          allow_redirects=True, verify=verify) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with jobs_lock:
                jobs[job_id]["total"] = total
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    with jobs_lock:
                        jobs[job_id]["downloaded"] = downloaded
        os.replace(tmp, dest)
        with jobs_lock:
            jobs[job_id]["status"] = "done"
        log.info("OK: %s (%d bytes) -> %s", item["name"], downloaded, dest)
    except requests.exceptions.SSLError as exc:
        msg = ("SSL/certificate error. Try: keeping truststore active, or "
               "ticking 'Skip SSL verification' in the settings. Detail: "
               + str(exc))
        _fail_job(job_id, tmp, msg)
        log.error("SSL failed on %s: %s", item["name"], exc)
    except Exception as exc:  # noqa: BLE001 - report any network/IO failure
        _fail_job(job_id, tmp, str(exc))
        log.error("Failure on %s: %s", item["name"], exc)


def _fail_job(job_id, tmp, message):
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    with jobs_lock:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = message


@app.route("/api/download", methods=["POST"])
def download():
    cfg = load_config()
    if not cfg.get("comfyui_path"):
        return jsonify(error="Configure the ComfyUI folder first."), 400

    item = request.get_json(force=True) or {}
    status = check_status(cfg, item)
    if status["exists"]:
        return jsonify(skipped=True, reason="exists")

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "name": item["name"],
            "downloaded": 0,
            "total": 0,
            "status": "downloading",
            "error": None,
        }
    threading.Thread(target=do_download, args=(job_id, cfg, item), daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/progress/<job_id>")
def progress(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify(error="job not found"), 404
        return jsonify(dict(job))


# --------------------------------------------------------------------------- #
# Custom node packs (git clone into custom_nodes/ + optional pip install)
# --------------------------------------------------------------------------- #
def do_install_node(job_id, cfg, pack, install_path):
    def log_step(step, message):
        with jobs_lock:
            jobs[job_id]["step"] = step
            jobs[job_id]["message"] = message
        log.info("Node install [%s]: %s", step, message)

    try:
        ok, msg = custom_nodes.install_pack(
            pack, install_path, verify=cfg.get("verify_ssl", True),
            run_pip=True, log=log_step)
    except Exception as exc:  # noqa: BLE001
        ok, msg = False, str(exc)
        log.error("Node install crashed: %s", exc)
    with jobs_lock:
        jobs[job_id]["status"] = "done" if ok else "error"
        jobs[job_id]["message"] = msg
        if not ok:
            jobs[job_id]["error"] = msg


@app.route("/api/node/install", methods=["POST"])
def node_install():
    cfg = load_config()
    install_path = resolve_install_path(cfg)
    if not install_path:
        return jsonify(error="Set the ComfyUI program folder (the one with "
                             "custom_nodes/) in step 1 first."), 400
    pack = request.get_json(force=True) or {}
    if not (pack.get("cnr_id") or pack.get("aux_id") or pack.get("git_url")):
        return jsonify(error="No pack identifier."), 400

    job_id = uuid.uuid4().hex
    label = pack.get("repo") or pack.get("cnr_id") or pack.get("aux_id")
    with jobs_lock:
        jobs[job_id] = {
            "name": label,
            "kind": "node",
            "status": "installing",
            "step": "queued",
            "message": "",
            "error": None,
        }
    threading.Thread(target=do_install_node, args=(job_id, cfg, pack, install_path),
                     daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/node/check", methods=["POST"])
def node_check():
    pack = request.get_json(force=True) or {}
    cfg = load_config()
    return jsonify(custom_nodes.pack_install_status(
        pack, resolve_install_path(cfg) or ""))


# --------------------------------------------------------------------------- #
# Workflow editor (second tab): friendly field editing
# --------------------------------------------------------------------------- #
def _load_workflow_from_request():
    """Read a workflow JSON from either a multipart file or the JSON body."""
    if "file" in request.files:
        return json.load(request.files["file"])
    return request.get_json(force=True)


@app.route("/api/workflow/analyze", methods=["POST"])
def workflow_analyze():
    try:
        data = _load_workflow_from_request()
    except (json.JSONDecodeError, ValueError):
        return jsonify(error="Invalid JSON."), 400
    if not isinstance(data, dict):
        return jsonify(error="Not a ComfyUI workflow."), 400
    return jsonify(workflow_editor.analyze(data))


@app.route("/api/workflow/export", methods=["POST"])
def workflow_export():
    body = request.get_json(force=True) or {}
    data = body.get("workflow")
    if not isinstance(data, dict):
        return jsonify(error="Missing workflow."), 400
    edits = body.get("edits") or {}
    duration_seconds = body.get("duration_seconds")
    modified, applied = workflow_editor.apply_edits(data, edits, duration_seconds)
    log.info("Workflow export: applied %d edit(s)", len(applied))
    return jsonify(workflow=modified, applied=applied)


def _fp8_variant_for_gpu(gpu):
    """fp8_e4m3fn_fast uses fp8 matmul, which only pays off on Ada/Blackwell
    (RTX 40xx/50xx). Everywhere else the plain fp8_e4m3fn is the safe pick —
    same VRAM saving, no risk of an unsupported-op error."""
    name = (gpu or {}).get("name", "").upper()
    if "RTX 50" in name or "RTX 40" in name or "RTX 6000 ADA" in name or "H100" in name:
        return "fp8_e4m3fn_fast"
    return "fp8_e4m3fn"


@app.route("/api/workflow/optimize", methods=["POST"])
def workflow_optimize():
    """"Generate Faster" (DEMO): apply the user's edits, then inject the native
    speed-up nodes (fp8 precision + torch.compile) into the workflow. Returns the
    optimized workflow plus a report of what was applied and what still needs a
    one-time change to the ComfyUI install (the second phase)."""
    body = request.get_json(force=True) or {}
    data = body.get("workflow")
    if not isinstance(data, dict):
        return jsonify(error="Missing workflow."), 400
    edits = body.get("edits") or {}
    duration_seconds = body.get("duration_seconds")
    swap_models = bool(body.get("swap_models"))
    modified, applied = workflow_editor.apply_edits(data, edits, duration_seconds)

    gpu = detect_gpu()
    report = workflow_editor.inject_speedups(
        modified, options={"fp8_value": _fp8_variant_for_gpu(gpu),
                           "lighter": swap_models, "lighter_apply": swap_models})
    log.info("Workflow optimize: %d edit(s), %d speed-up(s)",
             len(applied), len(report["applied"]))
    return jsonify(workflow=modified, report=report, gpu=gpu)


_gpu_cache = {"done": False, "gpu": None}


def detect_gpu():
    """Detect the primary NVIDIA GPU's name and total VRAM (GB) via nvidia-smi.

    Cached after the first call. Returns ``{"name", "total_gb"}`` or ``None``
    when no NVIDIA GPU / nvidia-smi is available (the estimate then degrades to
    showing numbers without a fits/over verdict)."""
    if _gpu_cache["done"]:
        return _gpu_cache["gpu"]
    _gpu_cache["done"] = True
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        lines = (out.stdout or "").strip().splitlines()
        if out.returncode == 0 and lines:
            name, mem = (x.strip() for x in lines[0].split(",", 1))
            _gpu_cache["gpu"] = {"name": name, "total_gb": round(float(mem) / 1024, 1)}
            log.info("Detected GPU: %s (%s GB)", name, _gpu_cache["gpu"]["total_gb"])
    except Exception as exc:  # noqa: BLE001  (nvidia-smi missing, timeout, etc.)
        log.info("GPU detection unavailable: %s", exc)
    return _gpu_cache["gpu"]


@app.route("/api/workflow/estimate", methods=["POST"])
def workflow_estimate():
    """Rough VRAM estimate for the current editor values, judged against the
    detected GPU. Numbers are estimates; the verdict is the useful part."""
    body = request.get_json(force=True) or {}
    gpu = detect_gpu()
    est = workflow_editor.estimate_vram(
        category=body.get("category"),
        width=body.get("width"),
        height=body.get("height"),
        frames=body.get("frames"),
        unet_name=body.get("unet_name"),
        weight_dtype=body.get("weight_dtype"),
        vram_total_gb=gpu["total_gb"] if gpu else None,
    )
    return jsonify(estimate=est, gpu=gpu)


def _image_size(path):
    """Return (width, height) for common image formats using only the stdlib.

    Supports PNG, GIF, BMP, WebP and JPEG by reading just the header bytes —
    no Pillow dependency. Returns ``None`` when the format is unknown or the
    file can't be parsed.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(26)
            if len(head) < 24:
                return None
            # PNG
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            # GIF
            if head[:6] in (b"GIF87a", b"GIF89a"):
                w, h = struct.unpack("<HH", head[6:10])
                return int(w), int(h)
            # BMP
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return int(w), abs(int(h))
            # WebP
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                chunk = head[12:16]
                if chunk == b"VP8 ":
                    fh.seek(26)
                    data = fh.read(4)
                    w = struct.unpack("<H", data[0:2])[0] & 0x3FFF
                    h = struct.unpack("<H", data[2:4])[0] & 0x3FFF
                    return int(w), int(h)
                if chunk == b"VP8L":
                    fh.seek(21)
                    bits = struct.unpack("<I", fh.read(4))[0]
                    w = (bits & 0x3FFF) + 1
                    h = ((bits >> 14) & 0x3FFF) + 1
                    return int(w), int(h)
                if chunk == b"VP8X":
                    fh.seek(24)
                    data = fh.read(6)
                    w = (data[0] | data[1] << 8 | data[2] << 16) + 1
                    h = (data[3] | data[4] << 8 | data[5] << 16) + 1
                    return int(w), int(h)
            # JPEG: walk the segment markers until a Start-Of-Frame.
            if head[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    byte = fh.read(1)
                    while byte and byte != b"\xff":
                        byte = fh.read(1)
                    marker = fh.read(1)
                    while marker == b"\xff":  # skip fill bytes
                        marker = fh.read(1)
                    if not marker:
                        return None
                    m = marker[0]
                    if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                        fh.read(3)  # segment length (2) + sample precision (1)
                        h = struct.unpack(">H", fh.read(2))[0]
                        w = struct.unpack(">H", fh.read(2))[0]
                        return int(w), int(h)
                    seg = fh.read(2)
                    if len(seg) < 2:
                        return None
                    fh.seek(struct.unpack(">H", seg)[0] - 2, 1)
    except (OSError, struct.error):
        return None
    return None


@app.route("/api/workflow/image-size", methods=["GET"])
def workflow_image_size():
    """Return the dimensions of a reference image already living in
    ComfyUI's input/ folder, so the editor can offer to match the
    workflow's Width/Height to it."""
    name = os.path.basename(request.args.get("name") or "")
    if not name:
        return jsonify(error="No image name."), 400
    cfg = load_config()
    path = cfg.get("comfyui_path")
    if not path:
        return jsonify(error="ComfyUI folder not set."), 400
    img = os.path.join(path, "input", name)
    if not os.path.isfile(img):
        return jsonify(error="Image not found in input/."), 404
    size = _image_size(img)
    if not size:
        return jsonify(error="Could not read image size."), 422
    return jsonify(width=size[0], height=size[1])


@app.route("/api/workflow/upload-image", methods=["POST"])
def workflow_upload_image():
    """Copy a chosen reference image into ComfyUI's input/ folder so the
    LoadImage node can find it, and return the filename to store in the JSON."""
    cfg = load_config()
    path = cfg.get("comfyui_path")
    if not path or not os.path.isdir(path):
        return jsonify(error="Set the ComfyUI folder in the Downloader tab first."), 400
    if "image" not in request.files:
        return jsonify(error="No image uploaded."), 400
    f = request.files["image"]
    name = os.path.basename(f.filename or "")
    if not name:
        return jsonify(error="Invalid file name."), 400
    input_dir = os.path.join(path, "input")
    os.makedirs(input_dir, exist_ok=True)
    dest = os.path.join(input_dir, name)
    f.save(dest)
    log.info("Saved reference image -> %s", dest)
    return jsonify(filename=name)


if __name__ == "__main__":
    print("ComfyUI Helper running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
