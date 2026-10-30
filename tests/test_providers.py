import base64
from copy import deepcopy
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from acuexact.providers import ModelClient, ProviderConfig, ProviderError, _NoRedirect, provider_options


SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["强相关", "弱相关"]}}, "required": ["label"], "additionalProperties": False}
RESULT = {"label": "强相关"}
KEY = "test-secret-never-log"
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode()


def completion(content=None, **overrides):
    result = {"id": "req_test", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(RESULT, ensure_ascii=False) if content is None else content}}], "usage": {"prompt_tokens": 17, "completion_tokens": 9, "private_field": KEY}}
    result.update(overrides)
    return result


def responses(content=None, **overrides):
    result = {"id": "resp_test", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(RESULT, ensure_ascii=False) if content is None else content}]}], "usage": {"input_tokens": 12, "output_tokens": 5}}
    result.update(overrides)
    return result


def messages(content=None, **overrides):
    result = {"id": "msg_test", "stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(RESULT, ensure_ascii=False) if content is None else content}]}
    result.update(overrides)
    return result


class FakeResponse:
    def __init__(self, body, headers=None, status=200):
        self.body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.headers = headers or {}
        self.status = status
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def getcode(self): return self.status
    def read(self, limit=-1): return self.body[:limit] if limit >= 0 else self.body


class ProviderTests(unittest.TestCase):
    def client(self, provider="openai", **kwargs):
        return ModelClient(ProviderConfig(provider, "test-model", api_key=KEY, **kwargs))

    def run_result(self, client, result, **kwargs):
        with patch.object(client._opener, "open", return_value=FakeResponse(result)) as mocked:
            self.assertEqual(client.generate_json("Extract", {"source": "private-document"}, SCHEMA, **kwargs), RESULT)
        request = mocked.call_args.args[0]
        return request, json.loads(request.data)

    def test_constructor_offline_and_key_not_in_repr(self):
        with patch("acuexact.providers.build_opener") as opener:
            client = self.client()
            opener.return_value.open.assert_not_called()
        self.assertEqual(client.calls, [])
        self.assertNotIn(KEY, repr(client.config))

    def test_options_are_copies_and_not_claims_of_live_inventory(self):
        options = provider_options()
        self.assertEqual({o["id"] for o in options}, {"openai", "anthropic", "minimax", "openai_compatible"})
        minimax = next(o for o in options if o["id"] == "minimax")
        self.assertFalse(minimax["vision"])
        self.assertEqual(minimax["structured_output"], "prompt_and_local_validation")
        options[0]["model"] = "changed"
        self.assertNotEqual(provider_options()[0]["model"], "changed")

    def test_openai_responses_strict_and_image(self):
        client = self.client(vision=True)
        request, body = self.run_result(client, responses(), images=[PNG])
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + KEY)
        self.assertFalse(body["store"])
        self.assertEqual(body["text"]["format"]["schema"], SCHEMA)
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["max_output_tokens"], 8000)
        self.assertEqual(body["input"][0]["content"][1], {"type": "input_image", "image_url": PNG, "detail": "high"})
        self.assertEqual(client.calls[-1]["usage"], {"input_tokens": 12, "output_tokens": 5})
        self.assertEqual(client.calls[-1]["status"], "completed")

    def test_claude_native_schema_and_image(self):
        request, body = self.run_result(self.client("anthropic", vision=True), messages(), images=[PNG])
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(request.get_header("X-api-key"), KEY)
        self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(body["output_config"]["format"], {"type": "json_schema", "schema": SCHEMA})
        self.assertNotIn("output_format", body)
        self.assertNotIn("anthropic-beta", request.headers)
        source = body["messages"][0]["content"][0]["source"]
        self.assertEqual(source, {"type": "base64", "media_type": "image/png", "data": PNG.split(",")[1]})

    def test_claude_unsupported_constraint_kept_local(self):
        schema = deepcopy(SCHEMA)
        schema["properties"]["label"]["minLength"] = 5
        client = self.client("anthropic")
        with patch.object(client._opener, "open", return_value=FakeResponse(messages())) as mocked:
            with self.assertRaisesRegex(ProviderError, "Schema"):
                client.generate_json("x", {}, schema)
        sent = json.loads(mocked.call_args.args[0].data)["output_config"]["format"]["schema"]
        self.assertNotIn("minLength", sent["properties"]["label"])
        self.assertEqual(schema["properties"]["label"]["minLength"], 5)

    def test_minimax_prompt_schema_thinking_and_metadata(self):
        client = self.client("minimax")
        request, body = self.run_result(client, completion("<think>private reasoning</think>\n```json\n" + json.dumps(RESULT) + "\n```"))
        self.assertEqual(request.full_url, "https://api.minimax.io/v1/chat/completions")
        self.assertNotIn("response_format", body)
        self.assertIn("JSON Schema", body["messages"][0]["content"])
        self.assertEqual(body["max_tokens"], 8000)
        serialized = json.dumps(client.calls)
        self.assertNotIn(KEY, serialized)
        self.assertNotIn("private-document", serialized)
        self.assertNotIn("private reasoning", serialized)
        self.assertNotIn("private_field", serialized)

    def test_minimax_china_endpoint(self):
        request, _ = self.run_result(self.client("minimax", base_url="https://api.minimaxi.com/v1/"), completion())
        self.assertEqual(request.full_url, "https://api.minimaxi.com/v1/chat/completions")

    def test_custom_payload_and_explicit_vision(self):
        client = self.client("openai_compatible", base_url="https://gateway.example/api/v1", vision=True)
        request, body = self.run_result(client, completion(), images=[PNG])
        self.assertEqual(request.full_url, "https://gateway.example/api/v1/chat/completions")
        self.assertEqual(body["messages"][1]["content"][1]["image_url"]["url"], PNG)

    def test_environment_keys_and_custom_does_not_inherit(self):
        for provider, env in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"), ("minimax", "MINIMAX_API_KEY")):
            with self.subTest(provider=provider), patch.dict(os.environ, {env: KEY}, clear=True):
                client = ModelClient(ProviderConfig(provider, "model"))
                self.assertEqual(client._key(), KEY)
        with patch.dict(os.environ, {"OPENAI_API_KEY": KEY}, clear=True):
            custom = ModelClient(ProviderConfig("openai_compatible", "model", base_url="https://gateway.example/v1"))
            with self.assertRaisesRegex(ProviderError, "API Key"):
                custom.generate_json("x", {}, SCHEMA)

    def test_missing_key_is_preflight(self):
        with patch.dict(os.environ, {}, clear=True):
            client = ModelClient(ProviderConfig("openai", "model"))
            with patch.object(client._opener, "open") as mocked, self.assertRaisesRegex(ProviderError, "OPENAI_API_KEY"):
                client.generate_json("x", {}, SCHEMA)
            mocked.assert_not_called()

    def test_unsafe_and_misdirected_endpoints_rejected(self):
        for provider, url in (("openai", "https://attacker.example/v1"), ("minimax", "https://api.minimax.io.attacker.example/v1"), ("anthropic", "https://api.anthropic.com:8443/v1"), ("openai_compatible", "http://gateway.example/v1"), ("openai_compatible", "https://user:pass@gateway.example/v1"), ("openai_compatible", "https://gateway.example/v1?key=secret"), ("openai_compatible", "https://gateway.example/#fragment")):
            with self.subTest(url=url), self.assertRaises(ProviderError):
                self.client(provider, base_url=url)

    def test_redirect_handler_never_forwards_key(self):
        request = Request("https://api.openai.com/v1/responses", headers={"Authorization": "Bearer " + KEY})
        self.assertIsNone(_NoRedirect().redirect_request(request, None, 307, "", {}, "https://attacker.example/"))
        client = self.client()
        with patch.object(client._opener, "open", side_effect=HTTPError(request.full_url, 302, KEY, {"location": "https://attacker.example/"}, BytesIO(KEY.encode()))) as mocked:
            with self.assertRaisesRegex(ProviderError, "重定向") as caught:
                client.generate_json("x", {}, SCHEMA)
        self.assertEqual(mocked.call_count, 1)
        self.assertNotIn(KEY, str(caught.exception))

    def test_429_and_5xx_retry_at_most_two_times(self):
        client = self.client()
        errors = [HTTPError("https://api.openai.com/v1/responses", code, KEY, {}, BytesIO(KEY.encode())) for code in (429, 503)]
        with patch.object(client._opener, "open", side_effect=errors + [FakeResponse(responses())]) as mocked, patch("acuexact.providers.time.sleep"):
            self.assertEqual(client.generate_json("x", {}, SCHEMA), RESULT)
        self.assertEqual(mocked.call_count, 3)
        self.assertEqual([c["attempt"] for c in client.calls], [1, 2, 3])
        self.assertNotIn(KEY, json.dumps(client.calls))

    def test_http_failure_and_network_error_redacted(self):
        for error in (HTTPError("https://api.openai.com/v1/responses", 401, KEY, {}, BytesIO(KEY.encode())), URLError(KEY), TimeoutError(KEY)):
            client = self.client()
            with patch.object(client._opener, "open", side_effect=error) as mocked:
                with self.assertRaises(ProviderError) as caught:
                    client.generate_json("private input", {}, SCHEMA)
            self.assertEqual(mocked.call_count, 1)
            self.assertNotIn(KEY, str(caught.exception))
            self.assertNotIn("private input", str(caught.exception))

    def test_retry_exhausted(self):
        client = self.client()
        with patch.object(client._opener, "open", side_effect=[HTTPError("https://x", 503, KEY, {}, BytesIO()) for _ in range(3)]) as mocked, patch("acuexact.providers.time.sleep"):
            with self.assertRaisesRegex(ProviderError, "503"):
                client.generate_json("x", {}, SCHEMA)
        self.assertEqual(mocked.call_count, 3)

    def test_request_budget_blocks_network_and_metadata_before_first_attempt(self):
        client = self.client()
        failure = RuntimeError("budget exhausted")
        def before_request():
            raise failure
        client.before_request = before_request
        with patch.object(client._opener, "open") as mocked:
            with self.assertRaises(RuntimeError) as caught:
                client.generate_json("x", {}, SCHEMA)
        self.assertIs(caught.exception, failure)
        mocked.assert_not_called()
        self.assertEqual(client.calls, [])

    def test_request_budget_applies_to_transport_retry(self):
        client = self.client()
        attempted = []
        def before_request():
            if attempted:
                raise RuntimeError("budget exhausted")
            attempted.append(True)
        client.before_request = before_request
        with patch.object(client._opener, "open", side_effect=HTTPError("https://x", 503, "retry", {}, BytesIO())) as mocked, patch("acuexact.providers.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                client.generate_json("x", {}, SCHEMA)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(len(client.calls), 1)

    def test_request_budget_applies_to_json_repair(self):
        client = self.client("minimax")
        attempted = []
        def before_request():
            if attempted:
                raise RuntimeError("budget exhausted")
            attempted.append(True)
        client.before_request = before_request
        with patch.object(client._opener, "open", return_value=FakeResponse(completion("{}"))) as mocked:
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                client.generate_json("x", {}, SCHEMA)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["status"], "schema_invalid")

    def test_prompted_schema_one_repair(self):
        client = self.client("minimax")
        with patch.object(client._opener, "open", side_effect=[FakeResponse(completion('{"label":"unknown"}')), FakeResponse(completion())]) as mocked:
            self.assertEqual(client.generate_json("x", {}, SCHEMA), RESULT)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual([c["status"] for c in client.calls], ["schema_invalid", "completed"])
        self.assertTrue(client.calls[-1]["repair"])
        self.assertIn("previous output failed", json.loads(mocked.call_args.args[0].data)["messages"][0]["content"])

    def test_invalid_json_schema_is_never_accepted(self):
        for invalid in ('{"label":"unknown"}', '{"label":"强相关","extra":1}', "not json", '{"label":"强相关","label":"弱相关"}', '{"label":NaN}'):
            client = self.client("minimax")
            with self.subTest(invalid=invalid), patch.object(client._opener, "open", return_value=FakeResponse(completion(invalid))) as mocked:
                with self.assertRaisesRegex(ProviderError, "Schema"):
                    client.generate_json("x", {}, SCHEMA)
            self.assertEqual(mocked.call_count, 2)

    def test_native_schema_failure_does_not_replay(self):
        client = self.client()
        with patch.object(client._opener, "open", return_value=FakeResponse(responses("{}"))) as mocked:
            with self.assertRaisesRegex(ProviderError, "Schema"):
                client.generate_json("x", {}, SCHEMA)
        self.assertEqual(mocked.call_count, 1)

    def test_overflowed_json_number_is_not_accepted_as_finite_data(self):
        client = self.client()
        schema = {"type": "object", "properties": {"confidence": {"type": "number"}}, "required": ["confidence"], "additionalProperties": False}
        with patch.object(client._opener, "open", return_value=FakeResponse(responses('{"confidence":1e999}'))):
            with self.assertRaisesRegex(ProviderError, "Schema"):
                client.generate_json("x", {}, schema)

    def test_unresolvable_local_schema_reference_is_sanitized(self):
        client = self.client()
        schema = {"type": "object", "properties": {"label": {"$ref": "#/$defs/missing"}}}
        with patch.object(client._opener, "open", return_value=FakeResponse(responses())):
            with self.assertRaisesRegex(ProviderError, "Schema"):
                client.generate_json("x", {}, schema)

    def test_refusal_and_truncation_rejected_before_json_validation(self):
        cases = [("openai", responses(status="incomplete")), ("openai", responses(output=[{"type": "message", "content": [{"type": "refusal", "refusal": KEY}]}])), ("anthropic", messages(stop_reason="max_tokens")), ("anthropic", messages(stop_reason="refusal")), ("minimax", completion(choices=[{"finish_reason": "length", "message": {"content": json.dumps(RESULT)}}])), ("minimax", completion(choices=[{"finish_reason": "stop", "message": {"refusal": KEY}}]))]
        for provider, result in cases:
            client = self.client(provider)
            with self.subTest(provider=provider, result=result), patch.object(client._opener, "open", return_value=FakeResponse(result)) as mocked:
                with self.assertRaises(ProviderError) as caught:
                    client.generate_json("x", {}, SCHEMA)
            self.assertEqual(mocked.call_count, 1)
            self.assertNotIn(KEY, str(caught.exception))

    def test_malformed_wire_json_and_protocol_are_safe(self):
        for result in (b"not json " + KEY.encode(), {"status": "completed", "output": [None]}, responses(output=[])):
            client = self.client()
            with patch.object(client._opener, "open", return_value=FakeResponse(result)):
                with self.assertRaises(ProviderError) as caught:
                    client.generate_json("x", {}, SCHEMA)
            self.assertNotIn(KEY, str(caught.exception))

    def test_images_rejected_preflight(self):
        for config, images in ((ProviderConfig("minimax", "MiniMax-M2.5", KEY, vision=True), [PNG]), (ProviderConfig("openai", "gpt-4.1", KEY), [PNG]), (ProviderConfig("openai", "gpt-4.1", KEY, vision=True), ["https://example.com/image.png"]), (ProviderConfig("openai", "gpt-4.1", KEY, vision=True), ["data:image/png;base64,AAAA"]), (ProviderConfig("openai", "gpt-4.1", KEY, vision=True), ["data:image/png;base64,invalid="])):
            client = ModelClient(config)
            with patch.object(client._opener, "open") as mocked, self.assertRaises(ProviderError):
                client.generate_json("x", {}, SCHEMA, images=images)
            mocked.assert_not_called()

    def test_schema_rejected_preflight_without_remote_resolution(self):
        for schema in ({"type": "array"}, {"type": "object", "properties": {"x": {"type": "bogus"}}}, {"type": "object", "$ref": "https://attacker.example/schema"}):
            client = self.client()
            with patch.object(client._opener, "open") as mocked, self.assertRaises(ProviderError):
                client.generate_json("x", {}, schema)
            mocked.assert_not_called()

    def test_metadata_does_not_echo_key_from_remote(self):
        client = self.client()
        with patch.object(client._opener, "open", return_value=FakeResponse(responses(id=KEY), headers={"x-request-id": KEY})):
            client.generate_json("x", {}, SCHEMA)
        self.assertIsNone(client.calls[-1]["request_id"])
        self.assertNotIn(KEY, json.dumps(client.calls))


if __name__ == "__main__":
    unittest.main()
