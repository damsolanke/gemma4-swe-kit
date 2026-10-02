"""Render OpenAI chat requests into Gemma 4 prompts the way the scorer's vLLM 0.19 server does.

What vLLM does before the template sees a request, reproduced here:

1. Content format. vLLM inspects the template's syntax tree. If the template loops over ``message['content']``
   (Gemma 4's templates do), every message's content is passed as a list of ``{"type": "text"}`` parts
   ("openai" format), otherwise as a plain string. For Gemma 4 this matters: the system turn renders each
   text part followed by a space, so the system prompt reaches the model with a trailing space that a
   plain-string renderer drops.
2. Message rebuilding. Only role, content, tool_calls, tool_call_id, name and ``reasoning`` survive;
   ``reasoning`` is copied to ``reasoning_content``. A message that carries only ``reasoning_content`` loses its
   thought (vLLM 0.19 behaviour; the competition harness patches the client to send both keys).
3. Tool-call arguments given as JSON strings are parsed into dicts; empty arguments become ``{}``; an empty
   ``tool_calls`` list is dropped.
4. Tools are passed as full ``{"type", "function": {"name", "description", "parameters"}}`` dicts.
5. The template is rendered in a sandboxed Jinja environment configured like transformers'
   ``apply_chat_template`` (trim_blocks, lstrip_blocks, loopcontrols, ``tojson``, ``raise_exception``).
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Iterable

import jinja2
import jinja2.ext
import jinja2.nodes
from jinja2.sandbox import ImmutableSandboxedEnvironment

DEFAULT_BOS = "<bos>"
SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}
RESERVED_KWARGS = {"messages", "tools", "add_generation_prompt", "documents"}


class _GenerationTag(jinja2.ext.Extension):
    """Accept transformers' ``{% generation %}...{% endgeneration %}`` blocks and render their body."""

    tags = {"generation"}

    def parse(self, parser: Any) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

    def _render(self, caller: Any) -> str:
        return caller()


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x: Any, ensure_ascii: bool = False, indent: Any = None, separators: Any = None, sort_keys: bool = False) -> str:
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_environment() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_GenerationTag, jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


# ---------------------------------------------------------------------------------------------------------
# content-format detection (same rule as vLLM's chat-template content format detection)
# ---------------------------------------------------------------------------------------------------------

def _unwrap(node: jinja2.nodes.Node) -> jinja2.nodes.Node:
    """Strip filters, tests and slices: ``messages[1:] | selectattr(...)`` refers to ``messages``."""
    while True:
        if isinstance(node, jinja2.nodes.Filter) and node.node is not None:
            node = node.node
        elif isinstance(node, jinja2.nodes.Test):
            node = node.node
        elif isinstance(node, jinja2.nodes.Getitem) and isinstance(node.arg, jinja2.nodes.Slice):
            node = node.node
        else:
            return node


def _is_name(node: jinja2.nodes.Node, names: set[str]) -> bool:
    node = _unwrap(node)
    return isinstance(node, jinja2.nodes.Name) and node.ctx == "load" and node.name in names


def _is_content_of(node: jinja2.nodes.Node, names: set[str]) -> bool:
    node = _unwrap(node)
    if isinstance(node, jinja2.nodes.Getitem):
        return (isinstance(node.arg, jinja2.nodes.Const) and node.arg.value == "content"
                and isinstance(node.node, jinja2.nodes.Name) and node.node.name in names)
    if isinstance(node, jinja2.nodes.Getattr):
        return node.attr == "content" and isinstance(node.node, jinja2.nodes.Name) and node.node.name in names
    return False


def detect_content_format(template_text: str) -> str:
    """'openai' if the template iterates over a message's content parts, else 'string'."""
    try:
        ast = make_environment().parse(template_text)
    except jinja2.TemplateSyntaxError:
        return "string"
    lists = {"messages"}
    grew = True
    while grew:
        grew = False
        for node in ast.find_all(jinja2.nodes.Assign):
            if (isinstance(node.target, jinja2.nodes.Name) and node.target.name not in lists
                    and _is_name(node.node, lists)):
                lists.add(node.target.name)
                grew = True
    message_vars = {loop.target.name for loop in ast.find_all(jinja2.nodes.For)
                    if isinstance(loop.target, jinja2.nodes.Name) and _is_name(loop.iter, lists)}
    for loop in ast.find_all(jinja2.nodes.For):
        if _is_content_of(loop.iter, message_vars):
            return "openai"
    return "string"


# ---------------------------------------------------------------------------------------------------------
# request preprocessing
# ---------------------------------------------------------------------------------------------------------

_TEXT_PART_TYPES = {"text", "input_text", "output_text", "refusal", "thinking"}


def _text_parts(content: Any) -> list[str]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") in _TEXT_PART_TYPES:
            value = part.get("text", part.get(part.get("type", "text")))
            if value is not None:
                texts.append(str(value))
    return texts


def to_conversation(messages: Iterable[dict], content_format: str = "openai") -> list[dict]:
    """Rebuild OpenAI messages the way vLLM 0.19's parse_chat_messages does before templating."""
    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        texts = _text_parts(msg.get("content"))
        if content_format == "openai":
            conv: dict[str, Any] = {"role": role, "content": [{"type": "text", "text": t} for t in texts]}
        else:
            conv = {"role": role, "content": "\n".join(texts)}
        if role == "assistant":
            if msg.get("tool_calls") is not None:
                conv["tool_calls"] = copy.deepcopy(list(msg["tool_calls"]))
            if msg.get("reasoning") is not None:
                conv["reasoning"] = msg["reasoning"]
                conv["reasoning_content"] = msg["reasoning"]
        elif role == "tool" and "tool_call_id" in msg:
            conv["tool_call_id"] = msg["tool_call_id"]
        if isinstance(msg.get("name"), str):
            conv["name"] = msg["name"]
        if role == "developer":
            conv["tools"] = msg.get("tools")
        out.append(conv)
    for conv in out:
        if conv["role"] != "assistant" or "tool_calls" not in conv:
            continue
        if not conv["tool_calls"]:
            conv.pop("tool_calls")
            continue
        for call in conv["tool_calls"]:
            fn = call.setdefault("function", {})
            args = fn.get("arguments")
            if args:
                if not isinstance(args, (dict, list)):
                    fn["arguments"] = json.loads(args)   # invalid JSON is a 400 on vLLM as well
            else:
                fn["arguments"] = {}
    return out


def normalize_tools(tools: list[dict] | None) -> list[dict] | None:
    """Tool dicts as vLLM's pydantic model_dump() produces them (missing optional fields become None)."""
    if tools is None:
        return None
    out = []
    for tool in tools:
        tool = copy.deepcopy(tool)
        fn = tool.setdefault("function", {})
        fn.setdefault("description", None)
        fn.setdefault("parameters", None)
        tool.setdefault("type", "function")
        out.append(tool)
    return out


class ChatRenderer:
    """Gemma 4 prompt renderer with the scorer's server defaults.

    ``default_kwargs`` plays the role of vLLM's ``--default-chat-template-kwargs``; the scorer starts vLLM with
    ``{"enable_thinking": true}``, so thinking is on unless a request turns it off.
    """

    def __init__(self, template_text: str, *, content_format: str = "auto", bos_token: str = DEFAULT_BOS,
                 default_kwargs: dict | None = None):
        self.template_text = template_text
        self.content_format = detect_content_format(template_text) if content_format == "auto" else content_format
        if self.content_format not in ("openai", "string"):
            raise ValueError(f"content_format must be auto, openai or string, not {content_format!r}")
        self.special_tokens = dict(SPECIAL_TOKENS, bos_token=bos_token)
        self.default_kwargs = {"enable_thinking": True} if default_kwargs is None else dict(default_kwargs)
        self.template = make_environment().from_string(template_text)

    @classmethod
    def from_file(cls, path: str, **kw: Any) -> "ChatRenderer":
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), **kw)

    def template_kwargs(self, chat_template_kwargs: dict | None) -> dict:
        kw = dict(self.default_kwargs)
        kw.update(chat_template_kwargs or {})
        return {k: v for k, v in kw.items() if k not in RESERVED_KWARGS}

    def render_conversation(self, conversation: list[dict], tools: list[dict] | None = None, *,
                            add_generation_prompt: bool = True, **template_kwargs: Any) -> str:
        """Render messages that are already in the template's input format (no preprocessing)."""
        kw = dict(self.special_tokens)
        kw.update(template_kwargs)
        return self.template.render(messages=conversation, tools=tools, documents=None,
                                    add_generation_prompt=add_generation_prompt, **kw)

    def render(self, messages: list[dict], tools: list[dict] | None = None, *, add_generation_prompt: bool = True,
               chat_template_kwargs: dict | None = None) -> str:
        """Preprocess OpenAI messages like vLLM, then render."""
        conversation = to_conversation(messages, self.content_format)
        return self.render_conversation(conversation, normalize_tools(tools),
                                        add_generation_prompt=add_generation_prompt,
                                        **self.template_kwargs(chat_template_kwargs))

    def render_request(self, body: dict) -> tuple[str, bool]:
        """Prompt for a /v1/chat/completions body, plus whether thinking is enabled for it.

        Tools stay in the prompt even with tool_choice "none", as on a vLLM server started without
        --exclude-tools-when-tool-choice-none.
        """
        kwargs = self.template_kwargs(body.get("chat_template_kwargs"))
        prompt = self.render(body.get("messages") or [], body.get("tools"),
                             add_generation_prompt=body.get("add_generation_prompt", True),
                             chat_template_kwargs=body.get("chat_template_kwargs"))
        return prompt, bool(kwargs.get("enable_thinking", False))


def role_key(messages: list[dict], width: int = 40) -> str:
    """Short label of a request's agent role: the start of its system prompt."""
    if not messages or messages[0].get("role") not in ("system", "developer"):
        return ""
    return " ".join(_text_parts(messages[0].get("content")))[:width]
