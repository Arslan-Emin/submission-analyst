# Adapted from conversational-form-agent (commit 6e67866), app/jotform.py:
# only `JotformError` and `JotformClient` were kept; `get_submissions` is new.
"""Jotform REST API client.

Verified against Jotform's official Python client (github.com/jotform/jotform-api-python):
- endpoints live under `/v1`, the API key goes in the `APIKEY` header,
- responses wrap the payload in a `content` key,
- GET /form/{id}/submissions takes `offset`, `limit`, `filter` (JSON) and `orderby`.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from analyst.sources.base import SourceData

API_VERSION_PREFIX = "/v1"


class JotformError(RuntimeError):
    """Raised for any failed Jotform API call. Never contains the API key."""


class JotformClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.jotform.com",
        *,
        http: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # `http` lets tests inject an httpx.Client with a MockTransport.
        self._http = http or httpx.Client(timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self._base_url}{API_VERSION_PREFIX}{path}"
        try:
            response = self._http.request(
                method, url, headers={"APIKEY": self._api_key}, data=data, params=params
            )
        except httpx.HTTPError as error:
            raise JotformError(f"Could not reach Jotform: {error.__class__.__name__}") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise JotformError(
                f"Jotform returned non-JSON (HTTP {response.status_code})."
            ) from error

        response_code = payload.get("responseCode", response.status_code)
        if response.status_code != 200 or response_code != 200:
            message = payload.get("message") or "unknown error"
            raise JotformError(f"Jotform API error (HTTP {response.status_code}): {message}")
        return payload.get("content")

    def get_form(self, form_id: str) -> dict[str, Any]:
        content = self._request("GET", f"/form/{form_id}")
        if not isinstance(content, dict):
            raise JotformError("Unexpected form format from Jotform.")
        return content

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        content = self._request("GET", f"/submission/{submission_id}")
        if not isinstance(content, dict):
            raise JotformError("Unexpected submission format from Jotform.")
        return content

    def get_questions(self, form_id: str) -> dict[str, dict[str, Any]]:
        content = self._request("GET", f"/form/{form_id}/questions")
        if not isinstance(content, dict):
            raise JotformError("Unexpected questions format from Jotform.")
        return content

    def get_submissions(
        self,
        form_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """One page of a form's submissions (GET /form/{id}/submissions).

        Query names follow Jotform's official Python client: `offset`, `limit`,
        `filter` (a JSON-encoded object) and `orderby`. Like that client, empty or
        zero values are not sent.
        """
        params: dict[str, str] = {"limit": str(limit)}
        if offset:
            params["offset"] = str(offset)
        if filters:
            params["filter"] = json.dumps(filters)
        if order_by:
            params["orderby"] = order_by
        content = self._request("GET", f"/form/{form_id}/submissions", params=params)
        if not isinstance(content, list):
            raise JotformError("Unexpected submissions format from Jotform.")
        return content

    def get_all_submissions(self, form_id: str, *, page_size: int = 1000) -> list[dict[str, Any]]:
        """Every submission of the form, fetched page by page.

        Stops at the first empty page, not at the first short one: the API's maximum
        page size is not documented in the sources we checked, and a server-side cap
        below `page_size` must not silently truncate the data.
        """
        submissions: list[dict[str, Any]] = []
        while page := self.get_submissions(form_id, limit=page_size, offset=len(submissions)):
            submissions.extend(page)
        return submissions

    def create_submission(self, form_id: str, data: dict[str, str]) -> dict[str, Any]:
        content = self._request("POST", f"/form/{form_id}/submissions", data=data)
        return content if isinstance(content, dict) else {}


class JotformRESTSource:
    """Live data from a Jotform form through the REST API (read-only calls only).

    There is no snapshot date: "today" for relative questions is the real date.
    """

    def __init__(self, client: JotformClient, form_id: str) -> None:
        self.client = client
        self.form_id = form_id

    def load(self) -> SourceData:
        form = self.client.get_form(self.form_id)
        return SourceData(
            form_id=self.form_id,
            form_title=str(form.get("title") or f"Form {self.form_id}"),
            questions=self.client.get_questions(self.form_id),
            submissions=self.client.get_all_submissions(self.form_id),
        )
