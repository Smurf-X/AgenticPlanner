# -*- coding: utf-8 -*-
import httpx
import pytest

from data_juicer.planner.generate.http_llm import OpenAICompatibleJsonClient


def test_openai_compatible_json_client_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"operator_names": ["text_length_filter"]}',
                        },
                    },
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    llm = OpenAICompatibleJsonClient(
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        http_client=client,
    )
    out = llm.complete_json("sys", "user")
    assert out == {"operator_names": ["text_length_filter"]}


def test_openai_compatible_requires_api_key():
    llm = OpenAICompatibleJsonClient(model="x", api_key="", base_url="https://api.example.com/v1")
    with pytest.raises(ValueError, match="api_key"):
        llm.complete_json("a", "b")
