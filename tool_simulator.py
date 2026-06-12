"""
stable_tool_router.py  (Apache-2.0)
─────────────────────
LLM-based API simulator with optional response caching for deterministic replays.

ENV vars:
  TOOL_CACHE_DIR   path to cache directory (default: tool_cache)
  TOOL_SIM_MODEL   OpenAI model for simulation (default: gpt-4o)
  OPENAI_API_KEY   API key for simulation requests
"""

from __future__ import annotations
import os, json, pathlib, hashlib, typing as T
import requests

Json = T.Dict[str, T.Any]
_VOLATILE_KEYS = {"timestamp", "ts", "time", "nonce", "auth", "token", "access_token", "session_id", "request_id"}


class StableToolRouter:
    def __init__(self, cache_dir: str = "tool_cache", sim_model: str = "gpt-4o"):
        self.cache = pathlib.Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sim_model = sim_model

    @classmethod
    def from_env(cls) -> "StableToolRouter":
        return cls(
            cache_dir=os.getenv("TOOL_CACHE_DIR", "tool_cache"),
            sim_model=os.getenv("TOOL_SIM_MODEL", "gpt-4o"),
        )

    def invoke(
        self,
        *,
        base_url: str,
        path: str,
        method: str = "GET",
        headers: Json | None = None,
        query: Json | None = None,
        json_body: Json | None = None,
        op_id: str | None = None,
        spec_version: str | None = None,
        response_schema: Json | None = None,
        simulate_hint: str | None = None,
        timeout: int | None = None,
    ) -> Json:
        headers = headers or {}
        query = query or {}

        key = self._cache_key(base_url, path, method, headers, query, json_body, op_id, spec_version)
        fp = self.cache / f"{key}.json"

        if fp.exists():
            try:
                data = json.loads(fp.read_text())
                data["source"] = "cache"
                return data
            except Exception:
                pass

        sim = self._simulate_via_llm(
            base_url=base_url, path=path, method=method, headers=headers, query=query, json_body=json_body,
            op_id=op_id, spec_version=spec_version, response_schema=response_schema, simulate_hint=simulate_hint
        )
        sim["source"] = "sim"
        self._write_cache(fp, sim)
        return sim

    def _normalize(self, obj: T.Any) -> T.Any:
        if isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in sorted(obj.items()) if k.lower() not in _VOLATILE_KEYS}
        if isinstance(obj, list):
            return [self._normalize(v) for v in obj]
        return obj

    def _cache_key(self, base_url, path, method, headers, query, json_body, op_id, spec_version) -> str:
        norm = {
            "u": base_url.rstrip("/"), "p": path, "m": method.upper(),
            "h": self._normalize(headers or {}), "q": self._normalize(query or {}),
            "b": self._normalize(json_body or {}), "op": op_id or "", "v": spec_version or ""
        }
        return hashlib.sha256(json.dumps(norm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _write_cache(self, fp: pathlib.Path, payload: Json) -> None:
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(fp)
        except Exception as e:
            print(f"[ERROR] Failed to write cache {fp}: {e}")

    def _simulate_via_llm(
        self, *, base_url: str, path: str, method: str, headers: Json, query: Json, json_body: Json,
        op_id: str | None, spec_version: str | None, response_schema: Json | None, simulate_hint: str | None
    ) -> Json:
        sys_prompt = (
            "You are an API simulator for offline tool evaluation.\n"
            "Return VALID JSON only, no prose. Conform to provided schema if present.\n"
            "Avoid timestamps/random IDs. Use stable fields. Be concise but plausible.\n"
        )
        user = {
            "endpoint": {"base_url": base_url, "path": path, "method": method, "operation_id": op_id, "spec_version": spec_version},
            "request": {"headers": headers, "query": query, "json": json_body},
            "schema_hint": response_schema or {},
            "description": simulate_hint or ""
        }

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"status": 200, "headers": {}, "body": {"ok": True, "simulated": True, "path": path, "query": query, "json": json_body}, "fetched_at": None}

        url = "https://api.openai.com/v1/chat/completions"
        req_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.sim_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }

        resp = requests.post(url, headers=req_headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            body = json.loads(content)
        except Exception:
            body = {"ok": True, "simulated": True, "raw": content}
        return {"status": 200, "headers": {}, "body": body, "fetched_at": None}


class OperationMeta(T.TypedDict, total=False):
    base_url: str
    path: str
    method: str
    headers: Json
    op_id: str
    spec_version: str
    response_schema: Json
    simulate_hint: str
    timeout: int
    rapidapi_bool_query: bool


def make_stable_text_call_fn(router: StableToolRouter, meta: OperationMeta):
    """Return a callable(**kwargs)->str that simulates API calls with optional caching."""
    base_url = meta.get("base_url", "").rstrip("/")
    path_tmpl = meta.get("path", "")
    method = (meta.get("method", "GET") or "GET").upper()
    default_headers = meta.get("headers", {})
    op_id = meta.get("op_id")
    spec_version = meta.get("spec_version")
    response_schema = meta.get("response_schema")
    simulate_hint = meta.get("simulate_hint")
    timeout = meta.get("timeout")
    rapidapi_bool_query = meta.get("rapidapi_bool_query", False)

    def _call(**kwargs) -> str:
        path = path_tmpl
        for k, v in list(kwargs.items()):
            token = "{" + k + "}"
            if token in path:
                path = path.replace(token, str(v))
                kwargs.pop(k, None)

        if method in {"GET", "DELETE"}:
            query = dict(kwargs)
            if rapidapi_bool_query:
                for k, v in list(query.items()):
                    if isinstance(v, bool):
                        query[k] = "true" if v else "false"
        else:
            query = {}
        json_body = None if method in {"GET", "DELETE"} else (kwargs if kwargs else None)

        res = router.invoke(
            base_url=base_url, path=path, method=method,
            headers=default_headers, query=query, json_body=json_body,
            op_id=op_id, spec_version=spec_version,
            response_schema=response_schema, simulate_hint=simulate_hint,
            timeout=timeout
        )
        body = res.get("body")
        if isinstance(body, (dict, list)):
            text = json.dumps(body, ensure_ascii=False)
        else:
            text = str(body)
        return text[:4000]

    _call.__name__ = (op_id or f"{method} {path_tmpl}").replace(" ", "_").replace("/", "_")
    _call.__doc__ = f"Simulated call for {method} {path_tmpl} (op_id={op_id}) – returns str[:4000]"
    return _call
