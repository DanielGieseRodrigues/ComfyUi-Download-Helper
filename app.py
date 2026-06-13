"""ComfyUI Helper - servidor local.

Sobe uma pagina web simples para:
  - mapear a pasta do ComfyUI
  - guardar tokens (HuggingFace / Civitai)
  - subir um workflow.json, listar as dependencias e baixar tudo
    automaticamente nas pastas certas (models/<directory>/<name>).
"""

import json
import logging
import os
import threading
import uuid

import requests
from flask import Flask, jsonify, request, send_from_directory

from workflow_parser import MODEL_EXTS, extract_referenced, parse_workflow

# Tipo de modelo no Civitai -> subpasta do ComfyUI.
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

# Defaults da config. Inclui verify_ssl: por padrao validamos certificados,
# mas o usuario pode desligar caso o ambiente (antivirus/proxy) impeca.
DEFAULT_CONFIG = {
    "comfyui_path": "",
    "hf_token": "",
    "civitai_token": "",
    "civitai_red_token": "",
    "verify_ssl": True,
}

# ---------------------------------------------------------------------------
# Logging para arquivo + console (ajuda a diagnosticar falhas de download)
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

# Usa a loja de certificados do Windows (resolve a maioria dos erros de SSL
# causados por antivirus/proxy que injetam um certificado raiz proprio).
try:
    import truststore
    truststore.inject_into_ssl()
    log.info("truststore ativo: usando os certificados do sistema operacional.")
except Exception as exc:  # noqa: BLE001
    log.warning("truststore indisponivel (%s); usando certificados do certifi.", exc)

app = Flask(__name__, static_folder="static", static_url_path="")

# Jobs de download em andamento: job_id -> {name, downloaded, total, status, error}
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


def check_status(cfg, item):
    path = cfg.get("comfyui_path")
    if not path:
        return {"exists": False, "reason": "no_path"}
    fp = target_path(cfg, item)
    if os.path.isfile(fp) and os.path.getsize(fp) > 0:
        return {"exists": True, "size": os.path.getsize(fp), "path": fp}
    return {"exists": False, "path": fp}


# --------------------------------------------------------------------------- #
# Rotas estaticas
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
        return jsonify(valid=False, msg="Pasta nao encontrada.")
    has_models = os.path.isdir(os.path.join(path, "models"))
    msg = "OK - pasta valida." if has_models else \
        "A pasta existe, mas nao tem subpasta 'models'. As pastas serao criadas no download."
    return jsonify(valid=True, has_models=has_models, msg=msg)


@app.route("/api/parse", methods=["POST"])
def parse():
    try:
        if "file" in request.files:
            data = json.load(request.files["file"])
        else:
            data = request.get_json(force=True)
    except (json.JSONDecodeError, ValueError):
        return jsonify(error="JSON invalido."), 400

    cfg = load_config()
    models = parse_workflow(data)
    for m in models:
        m["source"] = detect_source(m["url"])
        m["status"] = check_status(cfg, m)

    # Modelos citados no workflow que NAO tem link de download.
    have = {m["name"].lower() for m in models}
    missing = [r for r in extract_referenced(data) if r["name"].lower() not in have]
    for r in missing:
        r["status"] = check_status(cfg, r)

    return jsonify(models=models, missing=missing,
                   comfyui_path=cfg.get("comfyui_path", ""))


# --------------------------------------------------------------------------- #
# Busca por nome (quando nao ha link ou o download falhou)
# --------------------------------------------------------------------------- #
def _search_civitai(base, query, token, source, verify):
    headers = {"User-Agent": "comfyui-helper"}
    if token:
        headers["Authorization"] = "Bearer " + token
    out = []
    try:
        r = requests.get(f"{base}/api/v1/models",
                         params={"query": query, "limit": 8},
                         headers=headers, timeout=30, verify=verify)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Busca em %s falhou: %s", source, exc)
        return out
    for item in data.get("items", []):
        mtype = item.get("type", "")
        directory = CIVITAI_TYPE_DIR.get(mtype, "checkpoints")
        for ver in item.get("modelVersions", []):
            for f in ver.get("files", []):
                url = f.get("downloadUrl")
                if not url:
                    continue
                out.append({
                    "filename": f.get("name"),
                    "model_name": item.get("name"),
                    "version": ver.get("name"),
                    "model_type": mtype,
                    "directory": directory,
                    "source": source,
                    "url": url,
                    "size_kb": f.get("sizeKB"),
                })
    return out


def _search_huggingface(query, token, verify):
    headers = {"User-Agent": "comfyui-helper"}
    if token:
        headers["Authorization"] = "Bearer " + token
    out = []
    try:
        r = requests.get("https://huggingface.co/api/models",
                         params={"search": query, "limit": 5},
                         headers=headers, timeout=30, verify=verify)
        r.raise_for_status()
        repos = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("Busca em HuggingFace falhou: %s", exc)
        return out
    for repo in repos[:5]:
        repo_id = repo.get("id") or repo.get("modelId")
        if not repo_id:
            continue
        try:
            d = requests.get(f"https://huggingface.co/api/models/{repo_id}",
                             headers=headers, timeout=30, verify=verify).json()
            siblings = d.get("siblings", [])
        except Exception:  # noqa: BLE001
            siblings = []
        for s in siblings:
            fn = s.get("rfilename", "")
            if fn.lower().endswith(MODEL_EXTS):
                out.append({
                    "filename": fn.split("/")[-1],
                    "model_name": repo_id,
                    "version": "",
                    "model_type": "",
                    "directory": "checkpoints",
                    "source": "huggingface",
                    "url": f"https://huggingface.co/{repo_id}/resolve/main/{fn}",
                    "size_kb": None,
                })
                if len(out) >= 40:
                    return out
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
    log.info("Busca por '%s' em %s", query, sources)
    results = []
    if "civitai" in sources:
        results += _search_civitai("https://civitai.com", query,
                                    cfg.get("civitai_token"), "civitai", verify)
    if "civitai_red" in sources:
        results += _search_civitai("https://civitai.red", query,
                                   cfg.get("civitai_red_token"), "civitai_red", verify)
    if "huggingface" in sources:
        results += _search_huggingface(query, cfg.get("hf_token"), verify)
    return jsonify(results=results)


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

    log.info("Baixando %s (%s) de %s [verify_ssl=%s]",
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
        msg = ("Erro de SSL/certificado. Tente: deixar o truststore ativo, ou "
               "marcar 'Ignorar verificacao SSL' nas configuracoes. Detalhe: "
               + str(exc))
        _fail_job(job_id, tmp, msg)
        log.error("SSL falhou em %s: %s", item["name"], exc)
    except Exception as exc:  # noqa: BLE001 - reportar qualquer falha de rede/IO
        _fail_job(job_id, tmp, str(exc))
        log.error("Falha em %s: %s", item["name"], exc)


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
        return jsonify(error="Configure a pasta do ComfyUI primeiro."), 400

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
            return jsonify(error="job nao encontrado"), 404
        return jsonify(dict(job))


if __name__ == "__main__":
    print("ComfyUI Helper rodando em http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
