#!/bin/bash

set -a
source .env
set +a

MODEL=google/embeddinggemma-300m-GGUF/embeddinggemma-300M-F32.gguf
llama-server \
  -m "${MODEL_ROOT_DIR}/${MODEL}" \
  --host 0.0.0.0 \
  --port 8081 \
  -ngl 1000 \
  --jinja \
  --embeddings \
  --ubatch-size 8096 \
  --batch-size 8096