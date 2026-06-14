"""Detect and install the custom node packs a ComfyUI workflow depends on.

Modern ComfyUI workflows record, on every node, which pack it came from inside
``node["properties"]``:

  - ``cnr_id``  -> id in the Comfy Registry. ``comfy-core`` means it's a native
                   node (nothing to install); any other value is an installable
                   pack.
  - ``aux_id``  -> the GitHub repo as ``owner/repo`` (set when the pack isn't in
                   the registry, e.g. installed straight from git).
  - ``ver``     -> the version that was used.

This is the same metadata ComfyUI-Manager uses to know what to install. We scan
it (recursively, including subgraphs), resolve each pack to a git URL and
``git clone`` it into ``custom_nodes/`` — optionally running its
``requirements.txt`` in the ComfyUI Python.
"""

import os
import re
import subprocess

import requests

# Comfy Registry endpoint: resolves a cnr_id to its GitHub repository.
REGISTRY_NODE_API = "https://api.comfy.org/nodes/{node_id}"

# cnr_id values that are native ComfyUI and need no install.
CORE_IDS = {"comfy-core", "comfy_core", "comfyui"}


def _git_url_from_aux(aux_id):
    """``owner/repo`` -> a clonable GitHub URL. Returns None if it doesn't look
    like a GitHub slug (some packs store a full URL or junk in aux_id)."""
    aux_id = (aux_id or "").strip()
    if not aux_id:
        return None
    if aux_id.startswith("http://") or aux_id.startswith("https://"):
        return aux_id
    if re.fullmatch(r"[\w.-]+/[\w.-]+", aux_id):
        return f"https://github.com/{aux_id}"
    return None


def _repo_name(git_url):
    """Folder name git clone would create from a repo URL (strips .git)."""
    name = (git_url or "").rstrip("/").split("/")[-1]
    return re.sub(r"\.git$", "", name)


def _walk_nodes(obj, out):
    """Collect every dict that looks like a node (has a ``type``)."""
    if isinstance(obj, dict):
        if isinstance(obj.get("type"), str) and "properties" in obj:
            out.append(obj)
        for v in obj.values():
            _walk_nodes(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_nodes(v, out)


def extract_node_packs(data):
    """Return the installable custom-node packs referenced by a workflow.

    Each entry: ``{cnr_id, aux_id, ver, git_url, repo, node_types[], count}``.
    Native (comfy-core) nodes and nodes without pack metadata are ignored.
    Packs are grouped so a workflow using ten nodes from one pack lists it once.
    """
    nodes = []
    _walk_nodes(data, nodes)

    packs = {}
    for node in nodes:
        props = node.get("properties") or {}
        cnr_id = (props.get("cnr_id") or "").strip()
        aux_id = (props.get("aux_id") or "").strip()
        if not cnr_id and not aux_id:
            continue  # legacy/unknown node — can't tell which pack it's from
        if cnr_id.lower() in CORE_IDS:
            continue  # native node, nothing to install
        git_url = _git_url_from_aux(aux_id)
        # Group by whatever stable id we have (prefer cnr_id, else the repo).
        key = cnr_id or aux_id
        pack = packs.get(key)
        if not pack:
            pack = packs[key] = {
                "cnr_id": cnr_id,
                "aux_id": aux_id,
                "ver": (props.get("ver") or "").strip(),
                "git_url": git_url,
                "repo": _repo_name(git_url) if git_url else "",
                "node_types": [],
                "count": 0,
            }
        # Backfill a git url / aux_id if a later node of the same pack has it.
        if not pack["git_url"] and git_url:
            pack["git_url"] = git_url
            pack["repo"] = _repo_name(git_url)
        if not pack["aux_id"] and aux_id:
            pack["aux_id"] = aux_id
        ntype = node.get("type")
        if ntype and ntype not in pack["node_types"]:
            pack["node_types"].append(ntype)
        pack["count"] += 1

    return list(packs.values())


def resolve_git_url(pack, verify=True, timeout=20):
    """Make sure a pack has a git URL, querying the Comfy Registry by cnr_id when
    only that is known. Returns the URL (also stored on ``pack``) or None."""
    if pack.get("git_url"):
        return pack["git_url"]
    cnr_id = pack.get("cnr_id")
    if not cnr_id:
        return None
    try:
        r = requests.get(REGISTRY_NODE_API.format(node_id=cnr_id),
                         headers={"User-Agent": "comfyui-helper"},
                         timeout=timeout, verify=verify)
        r.raise_for_status()
        repo = (r.json() or {}).get("repository")
    except Exception:  # noqa: BLE001 - offline / not in registry
        return None
    if repo:
        pack["git_url"] = repo
        pack["repo"] = _repo_name(repo)
    return pack.get("git_url")


def installed_dirs(comfyui_path):
    """Set of lowercase folder names already in custom_nodes/."""
    cn = os.path.join(comfyui_path or "", "custom_nodes")
    if not os.path.isdir(cn):
        return set()
    return {d.lower() for d in os.listdir(cn)
            if os.path.isdir(os.path.join(cn, d))}


def pack_install_status(pack, comfyui_path):
    """Is this pack already present in custom_nodes/?

    Matched by the folder git clone would create (the repo name) or, as a
    fallback, the cnr_id — covers packs installed via ComfyUI-Manager."""
    dirs = installed_dirs(comfyui_path)
    candidates = {c.lower() for c in (pack.get("repo"), pack.get("cnr_id")) if c}
    return {"installed": bool(candidates & dirs)}


def find_comfy_python(comfyui_path):
    """Locate the Python that runs ComfyUI, so requirements land in the right
    environment. Checks the portable build's python_embeded and common venvs.
    Returns the path or None (then we skip pip and say so)."""
    p = comfyui_path or ""
    parent = os.path.dirname(p.rstrip("\\/"))
    candidates = [
        os.path.join(parent, "python_embeded", "python.exe"),    # windows portable
        os.path.join(p, "python_embeded", "python.exe"),
        os.path.join(parent, "standalone-env", "python.exe"),    # desktop standalone
        os.path.join(p, "standalone-env", "python.exe"),
        os.path.join(p, "venv", "Scripts", "python.exe"),        # windows venv
        os.path.join(p, ".venv", "Scripts", "python.exe"),
        os.path.join(p, "venv", "bin", "python"),                # linux/mac venv
        os.path.join(p, ".venv", "bin", "python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def install_pack(pack, comfyui_path, verify=True, run_pip=True, log=None):
    """git clone a pack into custom_nodes/ and (optionally) install its
    requirements in the ComfyUI Python.

    ``log`` is an optional callback ``log(step, message)`` for progress.
    Returns ``(ok, message)``.
    """
    def emit(step, message):
        if log:
            log(step, message)

    url = resolve_git_url(pack, verify=verify)
    if not url:
        return False, ("Couldn't find a git repository for this pack "
                       "(no aux_id and not in the Comfy Registry).")

    cn_dir = os.path.join(comfyui_path, "custom_nodes")
    os.makedirs(cn_dir, exist_ok=True)
    repo = pack.get("repo") or _repo_name(url)
    dest = os.path.join(cn_dir, repo)

    if os.path.isdir(dest):
        emit("done", f"Already present: {repo}")
        return True, f"{repo} already installed."

    emit("cloning", f"Cloning {url} …")
    try:
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", "--recursive", url, dest],
            capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        return False, "git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return False, "git clone timed out."
    if proc.returncode != 0:
        return False, "git clone failed: " + (proc.stderr or proc.stdout or "").strip()

    if not run_pip:
        emit("done", f"Cloned {repo} (pip skipped).")
        return True, f"Cloned {repo}."

    req = os.path.join(dest, "requirements.txt")
    if not os.path.isfile(req):
        emit("done", f"Cloned {repo} (no requirements.txt).")
        return True, f"Cloned {repo} (no requirements)."

    python = find_comfy_python(comfyui_path)
    if not python:
        emit("done", f"Cloned {repo}. Couldn't find the ComfyUI Python — "
                     "run its requirements.txt manually.")
        return True, (f"Cloned {repo}, but the ComfyUI Python wasn't found; "
                      "install its requirements.txt manually.")

    emit("pip", f"Installing requirements for {repo} …")
    try:
        proc = subprocess.run(
            [python, "-m", "pip", "install", "-r", req],
            capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return False, f"Cloned {repo}, but pip install timed out."
    if proc.returncode != 0:
        return False, (f"Cloned {repo}, but pip failed: "
                       + (proc.stderr or proc.stdout or "").strip()[-400:])

    emit("done", f"Installed {repo} + requirements.")
    return True, f"Installed {repo} and its requirements."
