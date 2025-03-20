#!/usr/bin/bash

curl -sSf https://astral.sh/uv/install.sh | bash
uv pip install vllm
uv sync
source .venv/bin/activate  #virtual environment
pkill vllm # vLLM processes have become unresponsive or are consuming too many resources
vllm serve OpenGVLab/InternVL2_5-8B-MPO-AWQ --trust-remote-code  --quantization awq --dtype half # to connect to vllm server
uv run main.py -m OpenGVLab/InternVL2_5-8B-MPO-AWQ --pdf-file SlamonetalSCIENCE1987.pdf --use-llm-eval #to run the main program
