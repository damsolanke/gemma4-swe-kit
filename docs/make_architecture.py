"""Writes docs/images/architecture.svg (and a 2x PNG through rsvg-convert when it is installed).

Usage: python docs/make_architecture.py
"""
import shutil
import subprocess
from pathlib import Path

OUT = Path(__file__).parent / "images"
FONT = "Geist, 'Helvetica Neue', 'Inter', Arial, sans-serif"
INK, MUTED, FAINT, ACCENT, GREEN = "#0F172A", "#64748B", "#94A3B8", "#4F46E5", "#047857"
W, H = 1700, 840
parts = []


def text(x, y, s, size=15, fill=INK, weight=500, anchor="start"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
                 f'text-anchor="{anchor}">{s}</text>')


def card(x, y, w, h, title, lines, fill="#FFFFFF", stroke="#CBD5E1", title_fill=INK, line_fill=MUTED, tag=None):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" ry="14" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="1.5" filter="url(#soft)"/>')
    if tag:
        parts.append(f'<rect x="{x + 18}" y="{y + 16}" width="{9 * len(tag) + 18}" height="24" rx="12" fill="#E0E7FF"/>')
        text(x + 27, y + 33, tag, size=12.5, fill=ACCENT, weight=700)
        ty = y + 66
    else:
        ty = y + 38
    text(x + 18, ty, title, size=19, fill=title_fill, weight=800)
    for k, ln in enumerate(lines):
        text(x + 18, ty + 28 + 23 * k, ln, size=14, fill=line_fill)


def arrow(x1, y1, x2, y2, color=ACCENT, marker="arrow", dash=False):
    d = ' stroke-dasharray="7 6"' if dash else ""
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.4"{d} '
                 f'marker-end="url(#{marker})"/>')


parts.append(f'<rect width="{W}" height="{H}" fill="#FBFCFE"/>')
text(60, 64, "gemma4-swe-kit", size=34, weight=800)
text(62, 92, "Local evaluation for Gemma 4 SWE agents that behaves like the Kaggle scorer's vLLM front end",
     size=16, fill=MUTED)

card(60, 170, 300, 230, "Agent bundle", ["ADK YAML agents and sub-agents", "prompts and skills", "PEFT LoRA adapters",
                                         "eval_config.yaml budget"], tag="your submission")
card(420, 170, 320, 230, "Official harness", ["swegemma + ADK runner", "compaction at 14,336 tokens", "tool results as JSON",
                                              "nudges on unparsed calls"], tag="g4kit-harness run")
parts.append('<rect x="800" y="150" width="440" height="270" rx="16" ry="16" fill="url(#fuse)" stroke="#4338CA" '
             'stroke-width="1.5" filter="url(#soft)"/>')
text(824, 190, "g4kit-proxy", size=24, fill="#FFFFFF", weight=800)
text(824, 214, "OpenAI-compatible /v1/chat/completions", size=14, fill="#C7D2FE")
for k, (head, sub) in enumerate([("Chat template", "rendered like vLLM 0.19 (openai content format)"),
                                 ("gemma4 call parser", "vLLM 0.19.1 rules: malformed calls stay text"),
                                 ("Limits", "32,768-token context, thinking-token budget")]):
    yy = 238 + 58 * k
    parts.append(f'<rect x="824" y="{yy}" width="392" height="48" rx="10" fill="#FFFFFF" fill-opacity="0.12"/>')
    text(840, yy + 21, head, size=15, fill="#FFFFFF", weight=700)
    text(840, yy + 40, sub, size=13, fill="#E0E7FF")
card(1300, 150, 340, 120, "Ollama (raw mode)", ["any OS, template bypassed"])
card(1300, 300, 340, 120, "MLX (Apple silicon)", ["prefix KV cache, LoRA adapters"])

arrow(360, 285, 412, 285)
arrow(740, 285, 792, 285)
arrow(1240, 210, 1292, 210)
arrow(1240, 360, 1292, 360)

parts.append('<rect x="800" y="470" width="440" height="56" rx="12" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1.5"/>')
text(820, 504, "Proxy logs: every request, completion and token count", size=15, fill=INK, weight=600)
arrow(1020, 420, 1020, 462, color=FAINT, marker="arrowD", dash=True)

tools = [("Distillation data", ["convert-openhands, render-distill:", "windows in the harness format"]),
         ("g4kit-fake-llm", ["scripted model walks the bundle", "through the real harness, no GPU"]),
         ("g4kit-scorer-time", ["projects scorer hours for", "~120 tasks from local runs"]),
         ("g4kit-replay / log-stats", ["re-sends logged decision points;", "loops, failed edits, overflows"])]
for k, (t, ls) in enumerate(tools):
    card(60 + 405 * k, 640, 375, 122, t, ls, stroke="#A7F3D0")
parts.append(f'<path d="M247 632 C247 520, 210 500, 210 408" fill="none" stroke="{GREEN}" stroke-width="2.2" '
             f'stroke-dasharray="7 6" marker-end="url(#arrowG)"/>')
text(262, 560, "trains LoRA adapters", size=13, fill=GREEN, weight=600)
parts.append(f'<path d="M652 632 C652 520, 580 500, 580 408" fill="none" stroke="{GREEN}" stroke-width="2.2" '
             f'stroke-dasharray="7 6" marker-end="url(#arrowG)"/>')
text(667, 560, "stands in for the model", size=13, fill=GREEN, weight=600)
for x in (870 + 187, 1275 + 187):
    parts.append(f'<path d="M1020 526 C1020 580, {x} 580, {x} 632" fill="none" stroke="{GREEN}" stroke-width="2.2" '
                 f'stroke-dasharray="7 6" marker-end="url(#arrowG)"/>')
text(60, 808, "Ships no competition data, weights or chat template: the kit reads the user's own copies "
     "(g4kit-assets fetches and fingerprints them).", size=14, fill=MUTED)

defs = ('<defs><filter id="soft" x="-25%" y="-25%" width="150%" height="150%"><feDropShadow dx="0" dy="3" '
        'stdDeviation="7" flood-color="#1E293B" flood-opacity="0.13"/></filter>'
        '<linearGradient id="fuse" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6366F1"/>'
        '<stop offset="1" stop-color="#4338CA"/></linearGradient>'
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 1 L9 5 L0 9 z" fill="#6366F1"/></marker>'
        '<marker id="arrowG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 1 L9 5 L0 9 z" fill="#10B981"/></marker>'
        '<marker id="arrowD" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0 1 L9 5 L0 9 z" fill="#94A3B8"/></marker></defs>')
OUT.mkdir(parents=True, exist_ok=True)
svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">{defs}{"".join(parts)}</svg>\n'
(OUT / "architecture.svg").write_text(svg)
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-z", "2", "-o", str(OUT / "architecture.png"), str(OUT / "architecture.svg")], check=True)
print("wrote", OUT / "architecture.svg")
