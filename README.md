# HSLU DLM03 - DevOps & LLMs

This repository contains the lecture materials for the HSLU DLM03 course "DevOps & LLMs".

# Setup

```bash
mamba create -y -f environment.yaml
```

# API Key Setup

In order to access the different LLM Backend, we need to provide API Keys.
All the secrets will be stored in a `.env` file in the root of this repo (WARNING: do not commit this file to git!).
You can add as many API Keys as you want as long as they are from different providers/names.

```bash
touch .env

# For Gemini
echo GOOGLE_API_KEY=<your_api_key> >> .env

# For OpenAI
echo OPENAI_API_KEY=<your_api_key> >> .env
```
# Jupyter Lab Server

You can then launch the jupyter lab server with:

```bash
mamba run -n hslu-dlm03 jupyter-lab
```