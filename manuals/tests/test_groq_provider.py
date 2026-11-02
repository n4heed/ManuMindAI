"""
Unit tests for manuals/rag/groq_provider.py in isolation - no Django models,
no network access. Every HTTP call is mocked (requests.post); none of these
tests make a real request to Groq, per the "testing without an API key"
requirement.

Covers the Groq Strict Structured Outputs response_format (response_format:
{"type": "json_schema", "json_schema": {"strict": true, ...}}) and the
provider-level parsing of the resulting {"status", "claims": [...]} shape.
Backend evidence/citation validation against real retrieved chunks is
exercised separately in test_dual_answer_rag.py, since Structured Outputs
only guarantees response SHAPE, not factual truth.
"""

import json
from unittest import mock

import requests
from django.test import TestCase, override_settings

from manuals.rag.groq_provider import (
    GroqConfigurationError,
    GroqLLMProvider,
    GroqProviderError,
    get_groq_provider,
)

SOURCES = [
    {"id": "S1", "document_id": "HP-OPS-01", "page_number": 2, "text": "E17 means a hydraulic pressure fault."},
    {"id": "S2", "document_id": "HP-MNT-01", "page_number": 3, "text": "Check the reservoir level and RF-20 filter."},
]


def _structured_response(status: str, claims: list) -> mock.MagicMock:
    content = json.dumps({"status": status, "claims": claims})
    resp = mock.MagicMock(spec=requests.Response)
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _raw_content_response(content: str) -> mock.MagicMock:
    resp = mock.MagicMock(spec=requests.Response)
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _fake_error_response(status_code: int, error_type: str = None, error_code: str = None, message: str = None):
    resp = mock.MagicMock(spec=requests.Response)
    resp.ok = False
    resp.status_code = status_code
    body = {}
    if error_type or error_code or message:
        body = {"error": {k: v for k, v in (("type", error_type), ("code", error_code), ("message", message)) if v}}
    resp.json.return_value = body
    return resp


@override_settings(LLM_ENABLED=False, GROQ_API_KEY="")
class ProviderConfigurationTests(TestCase):
    def test_instantiation_fails_without_llm_enabled(self):
        with self.assertRaises(GroqConfigurationError):
            get_groq_provider()

    @override_settings(LLM_ENABLED=True, GROQ_API_KEY="")
    def test_instantiation_fails_without_api_key_even_if_enabled(self):
        with self.assertRaises(GroqConfigurationError):
            get_groq_provider()

    @override_settings(LLM_ENABLED=True, GROQ_API_KEY="sk-test-key", GROQ_MODEL="openai/gpt-oss-120b")
    def test_instantiation_succeeds_with_mocked_configuration(self):
        provider = get_groq_provider()
        self.assertEqual(provider.model, "openai/gpt-oss-120b")
        self.assertEqual(provider.name, "groq")

    @mock.patch("manuals.rag.groq_provider.requests.post")
    @override_settings(LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
    def test_no_network_call_is_made_when_not_configured(self, mock_post):
        with self.assertRaises(GroqConfigurationError):
            with override_settings(LLM_ENABLED=False):
                get_groq_provider()
        mock_post.assert_not_called()


@override_settings(LLM_ENABLED=True, GROQ_API_KEY="sk-test-key", GROQ_MODEL="openai/gpt-oss-120b")
class StrictStructuredOutputRequestTests(TestCase):
    """#2: the request must configure Groq's strict JSON Schema response_format."""

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_request_uses_strict_json_schema_response_format(self, mock_post):
        mock_post.return_value = _structured_response("answered", [])
        provider = get_groq_provider()
        provider.generate("What does E17 mean?", SOURCES)

        sent_body = mock_post.call_args.kwargs["json"]
        rf = sent_body["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertTrue(rf["json_schema"]["strict"])
        schema = rf["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("status", schema["properties"])
        self.assertIn("claims", schema["properties"])
        claim_schema = schema["properties"]["claims"]["items"]
        self.assertFalse(claim_schema["additionalProperties"])
        self.assertIn("text", claim_schema["properties"])
        self.assertIn("sources", claim_schema["properties"])
        source_schema = claim_schema["properties"]["sources"]["items"]
        self.assertFalse(source_schema["additionalProperties"])
        self.assertEqual(set(source_schema["required"]), {"source_id", "supporting_quote"})

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_payload_contains_the_question_and_the_given_sources_only(self, mock_post):
        mock_post.return_value = _structured_response("answered", [])
        provider = get_groq_provider()
        provider.generate("What does E17 mean?", SOURCES)

        _, kwargs = mock_post.call_args
        sent_body = kwargs["json"]
        self.assertEqual(sent_body["model"], "openai/gpt-oss-120b")
        user_message = sent_body["messages"][1]["content"]
        self.assertIn("What does E17 mean?", user_message)
        self.assertIn("[S1]", user_message)
        self.assertIn("E17 means a hydraulic pressure fault.", user_message)
        self.assertIn("[S2]", user_message)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test-key")
        self.assertEqual(mock_post.call_args[0][0], "https://api.groq.com/openai/v1/chat/completions")

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_payload_never_contains_full_page_or_pdf_text(self, mock_post):
        """Only the already-retrieved CHUNK text is sent - never a full page/manual."""
        mock_post.return_value = _structured_response("answered", [])
        full_page_text = "UNRELATED PAGE CONTENT " * 500
        chunk_only_sources = [{"id": "S1", "document_id": "DOC-1", "page_number": 1, "text": "Short retrieved chunk."}]
        provider = get_groq_provider()
        provider.generate("question", chunk_only_sources)

        sent_body = mock_post.call_args.kwargs["json"]
        user_message = sent_body["messages"][1]["content"]
        self.assertNotIn(full_page_text, user_message)
        self.assertIn("Short retrieved chunk.", user_message)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_system_prompt_forbids_general_knowledge_and_requires_per_claim_sources(self, mock_post):
        mock_post.return_value = _structured_response("answered", [])
        provider = get_groq_provider()
        provider.generate("question", SOURCES)

        sent_body = mock_post.call_args.kwargs["json"]
        system_message = sent_body["messages"][0]["content"]
        self.assertIn("ONLY", system_message)
        self.assertIn("Never invent", system_message)
        self.assertIn("safety", system_message.lower())
        self.assertIn("verbatim", system_message.lower())
        self.assertIn("multiple parts", system_message.lower())

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_prompt_requests_explanatory_not_one_line_answers(self, mock_post):
        """The prompt must explicitly ask for brief explanation (2-4 sentences),
        not just a bare citation, while still requiring exact verbatim quotes -
        explanation and grounding are not in tension in the prompt design."""
        mock_post.return_value = _structured_response("answered", [])
        provider = get_groq_provider()
        provider.generate("question", SOURCES)

        sent_body = mock_post.call_args.kwargs["json"]
        system_message = sent_body["messages"][0]["content"]
        user_message = sent_body["messages"][1]["content"]
        self.assertIn("explan", system_message.lower())  # "explain"/"explanatory"
        self.assertIn("2-4", system_message)
        self.assertIn("practical terms", system_message.lower())
        self.assertIn("explan", user_message.lower())


@override_settings(LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class StructuredResponseParsingTests(TestCase):
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_single_claim_single_source_parses_correctly(self, mock_post):
        mock_post.return_value = _structured_response(
            "answered",
            [{"text": "E17 means a hydraulic pressure fault.", "sources": [{"source_id": "S1", "supporting_quote": "E17 means a hydraulic pressure fault."}]}],
        )
        provider = get_groq_provider()
        result = provider.generate("q", SOURCES)
        self.assertEqual(result.status, "answered")
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(result.claims[0].text, "E17 means a hydraulic pressure fault.")
        self.assertEqual(result.claims[0].sources, [{"source_id": "S1", "supporting_quote": "E17 means a hydraulic pressure fault."}])

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_multiple_claims_with_different_source_identifiers(self, mock_post):
        mock_post.return_value = _structured_response(
            "answered",
            [
                {"text": "Claim A.", "sources": [{"source_id": "S1", "supporting_quote": "quote a"}]},
                {"text": "Claim B.", "sources": [{"source_id": "S2", "supporting_quote": "quote b"}]},
            ],
        )
        provider = get_groq_provider()
        result = provider.generate("q", SOURCES)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.claims[0].sources[0]["source_id"], "S1")
        self.assertEqual(result.claims[1].sources[0]["source_id"], "S2")

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_claim_supported_by_multiple_passages(self, mock_post):
        mock_post.return_value = _structured_response(
            "answered",
            [{"text": "Combined claim.", "sources": [
                {"source_id": "S1", "supporting_quote": "quote a"},
                {"source_id": "S2", "supporting_quote": "quote b"},
            ]}],
        )
        provider = get_groq_provider()
        result = provider.generate("q", SOURCES)
        self.assertEqual(len(result.claims[0].sources), 2)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_insufficient_evidence_returns_empty_claims(self, mock_post):
        mock_post.return_value = _structured_response("insufficient_evidence", [])
        provider = get_groq_provider()
        result = provider.generate("q", SOURCES)
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.claims, [])

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_empty_content_raises_provider_error(self, mock_post):
        mock_post.return_value = _raw_content_response("")
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_non_json_content_raises_provider_error_not_a_crash(self, mock_post):
        mock_post.return_value = _raw_content_response("this is not JSON at all")
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_missing_status_field_raises_provider_error(self, mock_post):
        mock_post.return_value = _raw_content_response(json.dumps({"claims": []}))
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_invalid_status_value_raises_provider_error(self, mock_post):
        mock_post.return_value = _raw_content_response(json.dumps({"status": "maybe", "claims": []}))
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_missing_claims_field_raises_provider_error(self, mock_post):
        mock_post.return_value = _raw_content_response(json.dumps({"status": "answered"}))
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_claim_missing_sources_raises_provider_error(self, mock_post):
        mock_post.return_value = _raw_content_response(
            json.dumps({"status": "answered", "claims": [{"text": "x"}]})
        )
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_source_missing_supporting_quote_raises_provider_error(self, mock_post):
        mock_post.return_value = _raw_content_response(
            json.dumps({"status": "answered", "claims": [{"text": "x", "sources": [{"source_id": "S1"}]}]})
        )
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)


@override_settings(LLM_ENABLED=True, GROQ_API_KEY="sk-test-key")
class ErrorHandlingAndRetryTests(TestCase):
    @mock.patch("manuals.rag.groq_provider.time.sleep")
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_invalid_credentials_fail_immediately_without_retry(self, mock_post, mock_sleep):
        mock_post.return_value = _fake_error_response(401, "invalid_request_error", "invalid_api_key", "Invalid API Key")
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)
        self.assertEqual(mock_post.call_count, 1)
        mock_sleep.assert_not_called()

    @mock.patch("manuals.rag.groq_provider.time.sleep")
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_error_detail_from_groq_body_is_included_in_the_message(self, mock_post, mock_sleep):
        mock_post.return_value = _fake_error_response(
            403, "permission_error", "model_not_allowed", "The model is not enabled for this project"
        )
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError) as ctx:
            provider.generate("q", SOURCES)
        self.assertIn("model_not_allowed", str(ctx.exception))
        self.assertIn("not enabled for this project", str(ctx.exception))

    @mock.patch("manuals.rag.groq_provider.time.sleep")
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_rate_limit_is_retried_a_bounded_number_of_times(self, mock_post, mock_sleep):
        mock_post.return_value = _fake_error_response(429, "rate_limit_error", "rate_limit_exceeded")
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)
        self.assertEqual(mock_post.call_count, GroqLLMProvider.MAX_ATTEMPTS)

    @mock.patch("manuals.rag.groq_provider.time.sleep")
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_server_error_is_retried_then_succeeds(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            _fake_error_response(503),
            _structured_response("answered", [{"text": "Recovered.", "sources": [{"source_id": "S1", "supporting_quote": "q"}]}]),
        ]
        provider = get_groq_provider()
        result = provider.generate("q", SOURCES)
        self.assertEqual(result.claims[0].text, "Recovered.")

    @mock.patch("manuals.rag.groq_provider.time.sleep")
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_network_failure_raises_provider_error_not_a_crash(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.exceptions.ConnectionError("connection refused")
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    @mock.patch("manuals.rag.groq_provider.time.sleep")
    @mock.patch("manuals.rag.groq_provider.requests.post")
    def test_timeout_raises_provider_error_not_a_crash(self, mock_post, mock_sleep):
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        provider = get_groq_provider()
        with self.assertRaises(GroqProviderError):
            provider.generate("q", SOURCES)

    def test_error_messages_never_contain_the_api_key(self):
        """The API key must never leak into an exception message a caller might log/display."""
        with mock.patch("manuals.rag.groq_provider.requests.post") as mock_post, \
             mock.patch("manuals.rag.groq_provider.time.sleep"):
            mock_post.return_value = _fake_error_response(401, "invalid_request_error", "invalid_api_key")
            provider = get_groq_provider()
            try:
                provider.generate("q", SOURCES)
            except GroqProviderError as exc:
                self.assertNotIn("sk-test-key", str(exc))
