# ComfyUI Download Helper

A local web page that reads a ComfyUI `workflow.json` and downloads all of its
dependencies (models, VAEs, text encoders, LoRAs) straight into the right folders.

## What it does

- Reads the model links from the workflow and downloads from **HuggingFace, Civitai, Civitai.red** or a direct URL.
- Places each file in the correct folder (`models/diffusion_models/`, `models/loras/`, etc.).
- Skips files that already exist.
- Searches by name when a model has no link.

## How to use

1. Run **`run.bat`** (it sets up the environment, installs the dependencies and opens it in your browser).
2. Point it to your ComfyUI folder.
3. (Optional) Paste your HuggingFace / Civitai / Civitai.red tokens.
4. Upload the `workflow.json` and click **Download everything that's missing**.
