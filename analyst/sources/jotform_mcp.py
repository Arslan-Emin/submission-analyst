"""Jotform MCP data source: the same data as the REST source, through Jotform's
official remote MCP server (https://mcp.jotform.com, Streamable HTTP, OAuth 2.0).

Read from the server's public OAuth metadata (/.well-known/oauth-protected-resource
and /.well-known/oauth-authorization-server): authorization server
https://oauth2.jotform.com, scopes "readOnly" and "full", dynamic client
registration (register-public-client), PKCE S256, refresh tokens. The MCP Python
SDK's OAuthClientProvider runs that whole flow.

Least privilege, twice:
- the OAuth scope is "readOnly", so the server refuses writes,
- this client only calls the read tools in READ_TOOLS; asking for any other tool
  (create_form, edit_form, assign_form, ...) raises before a request is sent.

Tokens are stored in .mcp_tokens/ (git-ignored) and never printed.

    python -m analyst.sources.jotform_mcp login    # one-time browser consent
    python -m analyst.sources.jotform_mcp tools    # list the server's tools
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import webbrowser
from collections.abc import Awaitable, Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from analyst.sources.base import SourceData

MCP_SERVER_URL = "https://mcp.jotform.com"
SCOPE = "readOnly"
CALLBACK_PORT = 8765
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/callback"
TOKEN_FILE = Path(__file__).resolve().parents[2] / ".mcp_tokens" / "jotform.json"
# Tools this client may call. Everything else on the server writes or sends something.
READ_TOOLS = {"get_submissions", "fetch", "search"}
PAGE_SIZE = 100


class MCPError(RuntimeError):
    """A failed or refused MCP call. Never contains tokens."""


class FileTokenStorage(TokenStorage):
    """Keeps the OAuth client registration and tokens in one local JSON file."""

    def __init__(self, path: Path = TOKEN_FILE) -> None:
        self.path = path

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, key: str, value: dict[str, Any]) -> None:
        data = self._read()
        data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._read().get("tokens")
        return OAuthToken.model_validate(raw) if raw else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write("tokens", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._read().get("client_info")
        return OAuthClientInformationFull.model_validate(raw) if raw else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write("client_info", client_info.model_dump(mode="json", exclude_none=True))


# --------------------------------------------------------------------------- #
# Browser consent: open the authorize URL, catch the redirect on localhost
# --------------------------------------------------------------------------- #


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict[str, str] = {}

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        query = parse_qs(urlparse(self.path).query)
        type(self).result = {key: values[0] for key, values in query.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Jotform authorization received. You can close this tab.")

    def log_message(self, *args: Any) -> None:  # keep the code out of the console
        pass


async def _open_browser(url: str) -> None:
    print("Opening the Jotform consent page in your browser.")
    webbrowser.open(url)


async def _wait_for_callback() -> AuthorizationCodeResult:
    _CallbackHandler.result = {}
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    await asyncio.to_thread(thread.join, 300)
    server.server_close()
    result = _CallbackHandler.result
    if "code" not in result:
        raise MCPError(f"Authorization failed: {result.get('error', 'no code received')}")
    return AuthorizationCodeResult(code=result["code"], state=result.get("state"))


async def _no_browser(url: str) -> None:
    raise MCPError("Not logged in to Jotform MCP. Run: python -m analyst.sources.jotform_mcp login")


async def _no_callback() -> AuthorizationCodeResult:
    raise MCPError("Not logged in to Jotform MCP.")


def make_auth(storage: TokenStorage, *, interactive: bool) -> OAuthClientProvider:
    metadata = OAuthClientMetadata(
        client_name="Submission Analyst (local)",
        redirect_uris=[REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=SCOPE,
        token_endpoint_auth_method="none",
    )
    return OAuthClientProvider(
        MCP_SERVER_URL,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=_open_browser if interactive else _no_browser,
        callback_handler=_wait_for_callback if interactive else _no_callback,
    )


# --------------------------------------------------------------------------- #
# Read-only tool calls
# --------------------------------------------------------------------------- #


class ToolSession(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


async def call_read_tool(session: ToolSession, name: str, arguments: dict[str, Any]) -> Any:
    """Call an allowed read tool and return its result as JSON data."""
    if name not in READ_TOOLS:
        raise MCPError(
            f"Tool {name!r} is not allowed: this client only reads ({sorted(READ_TOOLS)})."
        )
    result = await session.call_tool(name, arguments)
    if getattr(result, "isError", False):
        raise MCPError(f"{name} failed: {_text(result)[:300]}")
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    text = _text(result)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise MCPError(f"{name} returned non-JSON content.") from error


def _text(result: Any) -> str:
    return "".join(getattr(part, "text", "") for part in getattr(result, "content", []) or [])


def questions_from_answers(submissions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rebuild {qid: question} from submissions: every answer entry carries the
    question's name, order, text and type (unanswered questions included)."""
    questions: dict[str, dict[str, Any]] = {}
    for submission in submissions:
        for qid, entry in (submission.get("answers") or {}).items():
            if qid not in questions:
                questions[qid] = {"qid": qid, **{k: v for k, v in entry.items() if k != "answer"}}
    return questions


async def load_from_session(session: ToolSession, form_id: str) -> SourceData:
    form = await call_read_tool(session, "fetch", {"id": form_id})
    submissions: list[dict[str, Any]] = []
    while True:
        page = await call_read_tool(
            session,
            "get_submissions",
            {"form_id": form_id, "filter": {"limit": PAGE_SIZE, "offset": len(submissions)}},
        )
        batch = page.get("submissions") or []
        submissions.extend(batch)
        if not batch or len(submissions) >= int(page.get("total_count") or 0):
            break
    title = form.get("title") if isinstance(form, dict) else None
    return SourceData(
        form_id=form_id,
        form_title=str(title or f"Form {form_id}"),
        questions=questions_from_answers(submissions),
        submissions=submissions,
    )


SessionRunner = Callable[[Callable[[ClientSession], Awaitable[Any]]], Awaitable[Any]]


async def run_with_session(
    action: Callable[[ClientSession], Awaitable[Any]],
    *,
    interactive: bool = False,
    storage: TokenStorage | None = None,
) -> Any:
    auth = make_auth(storage or FileTokenStorage(), interactive=interactive)
    async with create_mcp_http_client(auth=auth) as http:
        async with streamable_http_client(MCP_SERVER_URL, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await action(session)


class JotformMCPSource:
    """Live data through the MCP server. Needs a prior `login` (tokens on disk)."""

    def __init__(self, form_id: str, runner: SessionRunner | None = None) -> None:
        self.form_id = form_id
        self._runner = runner or run_with_session

    def load(self) -> SourceData:
        return asyncio.run(self._runner(lambda session: load_from_session(session, self.form_id)))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args == ["login"]:

        async def check(session: ClientSession) -> int:
            return len((await session.list_tools()).tools)

        count = asyncio.run(run_with_session(check, interactive=True))
        print(f"Logged in (scope {SCOPE}). The server offers {count} tools.")
        return 0
    if args == ["tools"]:

        async def describe(session: ClientSession) -> None:
            for tool in (await session.list_tools()).tools:
                marker = "read" if tool.name in READ_TOOLS else "not used"
                print(f"- {tool.name} [{marker}]: {(tool.description or '').splitlines()[0][:100]}")

        asyncio.run(run_with_session(describe))
        return 0
    print("Usage: python -m analyst.sources.jotform_mcp login|tools", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
