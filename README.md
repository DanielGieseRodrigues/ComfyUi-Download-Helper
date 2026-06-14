# ComfyUI Download Helper

A local web page with two tools that make working with ComfyUI workflows less painful:
a **model downloader** and a **workflow editor**. Everything runs on your machine.

## 📥 Model downloader

Reads a ComfyUI `workflow.json` and downloads all of its dependencies (models, VAEs,
text encoders, LoRAs) straight into the right folders.

- Reads the model links from the workflow and downloads from **HuggingFace, Civitai, Civitai.red** or a direct URL.
- Places each file in the correct folder (`models/diffusion_models/`, `models/loras/`, etc.).
- Skips files that already exist.
- Searches by name when a model has no link.
- **Installs missing custom node packs automatically.** Reads each node's pack info (`cnr_id` / `aux_id`, the same metadata ComfyUI-Manager uses) from the workflow, `git clone`s the missing packs into `custom_nodes/` and runs each one's `requirements.txt` in your ComfyUI Python. Packs only known by registry id are resolved through the Comfy Registry. Requires `git` on PATH.

## 🎛️ Workflow editor

Upload a `workflow.json` and get a friendly form instead of a wall of nodes. Tweak the
values that matter, then export a modified `workflow.json` to load back into ComfyUI.

- Edits **positive / negative prompts**, steps, CFG, seed (+ randomize), size, batch, LoRA strength, sampler and output name.
- **Video duration in seconds** — type "6 seconds" and it converts to the right frame count using the workflow's FPS (and accounts for frame interpolation).
- **⚡ Generate Faster (DEMO)** — injects the native speed-ups straight into the workflow: sets the diffusion model to **fp8** precision (≈ half the VRAM) and splices a **torch.compile** node before the sampler. It then reports the levers that need a one-time change to the ComfyUI install itself (Triton, SageAttention, TeaCache). Works on both flat and subgraph workflows; re-running won't duplicate the nodes.
- Recognizes native ComfyUI nodes (and common WAN video nodes); anything it doesn't understand is left untouched.

## How to use

1. Run **`run.bat`** (it sets up the environment, installs the dependencies and opens it in your browser).
2. **Model downloader:** point it to your ComfyUI folder, optionally paste your HuggingFace / Civitai / Civitai.red tokens, upload the `workflow.json` and click **Download everything that's missing**.
3. **Workflow editor:** switch to the editor tab, upload a `workflow.json`, tweak the fields and click **Export workflow.json**.
