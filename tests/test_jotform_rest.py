# Adapted from conversational-form-agent (commit 6e67866), tests/test_jotform.py:
# kept the client tests; the submission-listing tests are new.
"""Jotform REST client tests. All HTTP is mocked with httpx.MockTransport."""

from __future__ import annotations

from urllib.parse import parse_qsl

import httpx
import pytest

from analyst.sources.jotform_rest import JotformClient, JotformError, JotformRESTSource

QUESTIONS = {"3": {"qid": "3", "type": "control_textbox", "name": "comment", "text": "Comment"}}


def make_client(handler) -> JotformClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return JotformClient("secret-key", "https://api.jotform.com/", http=http)


def ok(content) -> httpx.Response:
    return httpx.Response(200, json={"responseCode": 200, "message": "success", "content": content})


# ------------------------------------------------------------------ client


def test_get_questions_uses_v1_path_and_apikey_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok(QUESTIONS)

    questions = make_client(handler).get_questions("123")

    assert questions == QUESTIONS
    request = seen[0]
    assert request.method == "GET"
    assert str(request.url) == "https://api.jotform.com/v1/form/123/questions"
    assert request.headers["APIKEY"] == "secret-key"


def test_create_submission_posts_form_urlencoded_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok({"submissionID": "999", "URL": "https://api.jotform.com/submission/999"})

    content = make_client(handler).create_submission("123", {"submission[3][first]": "Ali"})

    assert content["submissionID"] == "999"
    request = seen[0]
    assert request.method == "POST"
    assert request.headers["content-type"].startswith("application/x-www-form-urlencoded")
    assert dict(parse_qsl(request.content.decode())) == {"submission[3][first]": "Ali"}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"responseCode": 401, "message": "You're not authorized"}),
        httpx.Response(200, json={"responseCode": 400, "message": "Bad request"}),
        httpx.Response(502, text="<html>Bad gateway</html>"),
    ],
)
def test_api_errors_raise_without_leaking_key(response: httpx.Response) -> None:
    client = make_client(lambda request: response)

    with pytest.raises(JotformError) as error:
        client.get_questions("123")

    assert "secret-key" not in str(error.value)


def test_network_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(JotformError, match="Could not reach Jotform"):
        make_client(handler).get_questions("123")


# ------------------------------------------------------------------ submissions


def test_get_submissions_sends_paging_filter_and_order_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok([{"id": "1"}])

    rows = make_client(handler).get_submissions(
        "123", limit=50, offset=100, filters={"status": "ACTIVE"}, order_by="created_at"
    )

    assert rows == [{"id": "1"}]
    request = seen[0]
    assert request.url.path == "/v1/form/123/submissions"
    assert dict(request.url.params) == {
        "limit": "50",
        "offset": "100",
        "filter": '{"status": "ACTIVE"}',
        "orderby": "created_at",
    }


def test_get_submissions_omits_zero_offset_and_empty_options() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok([])

    make_client(handler).get_submissions("123")

    assert dict(seen[0].url.params) == {"limit": "1000"}


def test_get_submissions_rejects_unexpected_content() -> None:
    with pytest.raises(JotformError, match="Unexpected submissions format"):
        make_client(lambda request: ok({"not": "a list"})).get_submissions("123")


def test_get_all_submissions_pages_until_an_empty_page() -> None:
    # The server caps pages at 2 rows although 3 were asked for: a short page
    # must not end the loop early.
    data = [{"id": str(i)} for i in range(5)]
    offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        offsets.append(offset)
        return ok(data[offset : offset + 2])

    rows = make_client(handler).get_all_submissions("123", page_size=3)

    assert rows == data
    assert offsets == [0, 2, 4, 5]


def test_rest_source_loads_form_questions_and_all_submissions() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/v1/form/123":
            return ok({"id": "123", "title": "Survey"})
        if request.url.path == "/v1/form/123/questions":
            return ok(QUESTIONS)
        offset = int(request.url.params.get("offset", "0"))
        return ok([{"id": "1", "answers": {}}] if offset == 0 else [])

    data = JotformRESTSource(make_client(handler), "123").load()

    assert (data.form_id, data.form_title) == ("123", "Survey")
    assert data.questions == QUESTIONS
    assert [s["id"] for s in data.submissions] == ["1"]
    assert data.as_of is None  # live data: today is the real date
    assert paths.count("/v1/form/123/submissions") == 2
