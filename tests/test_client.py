"""The Decider client against a real local HTTP server (no Decider model needed)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import decider_patterns.client as client_module
from decider_patterns.client import DECIDER_ERRORS, Decider, DownFlag, ServerUnavailable, ask_serialized


class Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Handler.requests.append({"path": self.path, "body": body})
        payload = json.dumps(
            {
                "answers": {k: {"type": "noul", "noul": 0.5} for k in body["questions"]},
                "usage": {"input_tokens": 11},
                "latency_ms": 42.0,
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        payload = json.dumps(
            {"checkpoint": "ckpt", "base_model": "base", "device": "cpu", "max_length": 512}
        ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def server():
    Handler.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


class TestDecider:
    def test_ask_posts_state_and_questions_and_returns_answers(self, server):
        decider = Decider(server)
        questions = {"q": Decider.noul("is it?", {"true": "yes", "false": "no"})}
        answers = decider.ask("some state", questions)
        assert answers["q"]["noul"] == 0.5
        sent = Handler.requests[0]
        assert sent["path"] == "/v1/systemone"
        assert sent["body"]["state"] == "some state" and sent["body"]["questions"] == questions

    def test_usage_and_latency_are_tracked(self, server):
        decider = Decider(server)
        decider.ask("s", {"q": Decider.noul("x")})
        decider.ask("s", {"q": Decider.noul("x")})
        assert (decider.calls, decider.input_tokens, decider.last_latency_ms) == (2, 22, 42.0)

    def test_question_builders(self):
        assert Decider.choice("pick", {"a": "A"}) == {"type": "choice", "instructions": "pick", "criteria": {"a": "A"}}
        assert Decider.noul("q") == {"type": "noul", "instructions": "q"}
        assert Decider.score("s", ["low", "high"])["criteria"] == ["low", "high"]

    def test_banner_reads_health(self, server):
        assert "ckpt" in Decider(server).banner()

    def test_nothing_listening_raises_server_unavailable_with_the_serve_command(self):
        with pytest.raises(ServerUnavailable, match="strands-decider serve"):
            Decider("http://127.0.0.1:1", timeout=1).ask("s", {})

    def test_decider_errors_cover_the_raw_connection_failures(self):
        assert ServerUnavailable in DECIDER_ERRORS and OSError in DECIDER_ERRORS


class FakeDecider:
    def __init__(self):
        self.calls = []

    def ask(self, state, questions, timeout=None):
        self.calls.append((state, questions, timeout))
        return {"q": {"type": "noul", "noul": 0.5}}


class TestAskSerialized:
    def test_without_fcntl_it_still_asks(self, monkeypatch):
        monkeypatch.setattr(client_module, "fcntl", None)
        fake = FakeDecider()
        assert ask_serialized(fake, "s", {"q": {}}) == {"q": {"type": "noul", "noul": 0.5}}
        assert len(fake.calls) == 1

    def test_timeout_is_forwarded_only_when_given(self):
        fake = FakeDecider()
        ask_serialized(fake, "s", {})
        ask_serialized(fake, "s", {}, timeout=1.5)
        assert [c[2] for c in fake.calls] == [None, 1.5]


class GarbageHandler(BaseHTTPRequestHandler):
    body = b"not json"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(type(self).body)


def serve(body: bytes):
    handler = type("H", (GarbageHandler,), {"body": body})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class TestMalformedReplies:
    @pytest.mark.parametrize("body", [b"not json", b"{}", b"[]", b"null", b'{"answers": 3, "usage": 5}'])
    def test_malformed_200_becomes_server_unavailable(self, body):
        httpd = serve(body)
        try:
            with pytest.raises(ServerUnavailable):
                Decider(f"http://127.0.0.1:{httpd.server_port}").ask("s", {"q": Decider.noul("x")})
        finally:
            httpd.shutdown()

    def test_per_request_timeout_overrides_the_default(self):
        class Silent(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                import time

                time.sleep(1.0)

        httpd = HTTPServer(("127.0.0.1", 0), Silent)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            with pytest.raises(DECIDER_ERRORS):  # a read timeout is a TimeoutError (OSError)
                Decider(f"http://127.0.0.1:{httpd.server_port}", timeout=60).ask("s", {}, timeout=0.1)
        finally:
            httpd.shutdown()


class TestDownFlag:
    def test_sticky_by_default(self):
        now = [0.0]
        flag = DownFlag(clock=lambda: now[0])
        flag.mark()
        now[0] = 1e9
        assert flag.down

    def test_cooldown_expires_and_reset_clears(self):
        now = [0.0]
        flag = DownFlag(retry_after_s=10, clock=lambda: now[0])
        flag.mark()
        now[0] = 9.9
        assert flag.down
        now[0] = 10.0
        assert not flag.down
        flag.mark()
        flag.reset()
        assert not flag.down
