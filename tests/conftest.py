import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gemma4_swe_kit.chat import ChatRenderer

FIXTURES = Path(__file__).parent / "fixtures"
OFFICIAL_TEMPLATE = os.environ.get("G4KIT_TEST_OFFICIAL_TEMPLATE")   # set locally to run parity tests
OFFICIAL_PARSER = os.environ.get("G4KIT_TEST_OFFICIAL_PARSER")
FUZZ_PARSER = os.environ.get("G4KIT_TEST_FUZZ_PARSER")                # the same file; runs the slow 60,000-draw fuzz


@pytest.fixture
def mini_renderer() -> ChatRenderer:
    return ChatRenderer.from_file(str(FIXTURES / "mini_chat_template.jinja"))


@pytest.fixture
def tools() -> list[dict]:
    return json.loads((FIXTURES / "tools.json").read_text())


def whitespace_counter(text: str) -> int:
    """Mock tokenizer: one token per whitespace-separated word."""
    return len(text.split())


class ServerThread:
    """Run an http.server instance on a free port for the duration of a test."""

    def __init__(self, server):
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def post_json(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)
