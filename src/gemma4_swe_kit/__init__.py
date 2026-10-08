"""Scorer-faithful local evaluation tools for Gemma 4 software-engineering agents.

Modules:
    chat       Gemma 4 chat-template rendering with vLLM 0.19's message preprocessing.
    toolcalls  vLLM-compatible reasoning split and gemma4 tool-call parsing, malformed-call detection.
    proxy      OpenAI-compatible server that behaves like the scorer's vLLM front end.
    timing     Scorer wall-time estimate from a local run's per-call token counts.
    smoke      Scripted fake model for end-to-end harness smoke tests without a GPU.
    harness    Local runner for the official harness with the scorer's compaction settings.
    distill    OpenHands trajectory conversion, trajectory curation and training-window rendering.
    lora       MLX LoRA trainer, mlx_lm to PEFT conversion and PEFT key checks.
    canary     Logit canary: do the adapters a vLLM server serves change the model?
    replay     Replay logged requests under prompt or sampling changes, with pluggable metrics.
    logstats   Per-session loop, edit-failure and malformed-call statistics from proxy logs.
"""

__version__ = "0.1.0"
