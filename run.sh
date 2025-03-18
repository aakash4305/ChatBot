#!/usr/bin/bash

uv sync
vllm serve OpenGVLab/InternVL2_5-1B-MPO --dtype=half &
sleep 20
uv run main.py -m OpenGVLab/InternVL2_5-1B-MPO
