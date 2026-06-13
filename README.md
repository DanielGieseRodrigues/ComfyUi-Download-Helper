# ComfyUI Download Helper

Página local que lê um `workflow.json` do ComfyUI e baixa todas as dependências
(modelos, VAE, text encoders, LoRAs) direto nas pastas certas.

## O que faz

- Lê os links de modelos do workflow e baixa de **HuggingFace, Civitai, Civitai.red** ou URL direta.
- Coloca cada arquivo na pasta certa (`models/diffusion_models/`, `models/loras/`, etc.).
- Pula o que já existe.
- Busca por nome quando um modelo não tem link.

## Como usar

1. Rode o **`run.bat`** (cria o ambiente, instala as dependências e abre no navegador).
2. Informe a pasta do seu ComfyUI.
3. (Opcional) Cole os tokens de HuggingFace / Civitai / Civitai.red.
4. Suba o `workflow.json` e clique em **Baixar tudo o que falta**.
