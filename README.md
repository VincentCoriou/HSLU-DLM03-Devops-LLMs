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

# LLama.cpp

In order to run local LLM models using llama.cpp, you need to install it using:

```bash
brew install llama.cpp
```

or download the binaries for your system from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases). Don't forget to add the binary to your PATH environment variable for the script to run.

## Models

You can download open-weights LLMs from [Huggingface](https://huggingface.co/models).

For example, you can download the `EmbeddingGemma-300M` model from [here](https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/tree/main) and `Gemma-1b` from [here](https://huggingface.co/brittlewis12/gemma-3-1b-it-GGUF/tree/main) (select the right quantization for your machine, note that in general `Q5_K_S` offer the best balance between performance and memory requirements).

Please add a line to your environment variable file (`.env`) to point to the model directory:

```bash
export MODEL_ROOT_DIR=/path/to/your/model_directory
```

You can then modify the relative path to your wanted model in the `llm-server.sh` and `embeddings-server.sh` scripts.

and run the servers with:

```bash
sh llm-server.sh
sh embeddings-server.sh
```
