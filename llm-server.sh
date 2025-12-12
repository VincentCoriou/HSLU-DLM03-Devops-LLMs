llama-server \#!/bin/bash

set -a
source .env
set +a

MODEL=google/gemma-3-12b-it-GGUF/google_gemma-3-12b-it-Q5_K_S.gguf
llama-server \
  -m "${MODEL_ROOT_DIR}/${MODEL}" \
  -c 128000 \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 1000 \
  --jinja