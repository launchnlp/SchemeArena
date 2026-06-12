"""
tools/selective_loader.py
─────────────────────────
Wrap RapidAPI & OpenAPI specs into AutoGen FunctionTool objects.

• _wrap_rapid  – builds a tool per endpoint, generating an explicit Pydantic
  signature so required parameters (e.g. `query`) are validated.

• _wrap_openapi – minimal fallback for raw OpenAPI specs.

• load_external_tools(folder) – loads *.json specs; keeps only filenames
  that match chosen keywords.
"""

from __future__ import annotations
import os, re
from inspect import Signature, Parameter
from typing  import Any, Dict, List

from langchain_core.tools import Tool

from tool_simulator import StableToolRouter, make_stable_text_call_fn
ROUTER = StableToolRouter.from_env()

# ────────────────────────────────────────────────────────────────────
RAPID_KEY = os.getenv("RAPIDAPI_KEY", "SIGN-UP-FOR-KEY")
PH        = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
slug      = lambda s: re.sub(r"[^A-Za-z0-9]", "_", s).strip("_")[:60] or "api"

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HEADERISH = {"content-type", "accept", "authorization", "x-rapidapi-key", "x-rapidapi-host"}

def to_py_name(name: str) -> str:
    """Make a Python-safe parameter name, preserving simple identifiers."""
    return name if IDENT.match(name) else re.sub(r"[^A-Za-z0-9_]", "_", name)

def is_headerish(name: str) -> bool:
    """Treat header-like params as headers, not function args."""
    return "-" in name or name.lower() in HEADERISH

# Utility to map API‑type strings → Python types
TYPE_MAP: Dict[str, Any] = {
    "STRING": str,
    "NUMBER": int,
    "BOOLEAN": bool,
    "FLOAT": float,
}
def map_type(t: str) -> Any:
    return TYPE_MAP.get(t.upper(), str)

def _wrap_rapid(host: str, spec_slug: str, ep: dict) -> Tool:
    """Wrap a single RapidAPI endpoint dict into a LangChain Tool."""
    method  = ep.get("method", "GET").upper()
    rel_url = ep["url"].replace(f"https://{host}", "")
    phs: List[str] = PH.findall(rel_url)

    name = f"{spec_slug}_{slug(ep['name'])}"[:60]

    required_params: List[dict] = ep.get("required_parameters", []) # get required parameters listed in the tool json
    optional_params: List[dict] = ep.get("optional_parameters", []) # get optional parameters listed in the tool json

    # Sanitize names, drop header-like params, build py_name → original name map
    req_clean: List[dict] = []
    opt_clean: List[dict] = []
    name_map: Dict[str, str] = {}

    for p in required_params:
        orig = p["name"]
        if is_headerish(orig):
            continue  # don't expose header params as function args
        py = to_py_name(orig)
        name_map[py] = orig
        q = dict(p); q["py_name"] = py
        req_clean.append(q)

    for p in optional_params:
        orig = p["name"]
        if is_headerish(orig):
            continue
        py = to_py_name(orig)
        name_map[py] = orig
        q = dict(p); q["py_name"] = py
        opt_clean.append(q)

    # Precompute py-version of placeholders (for exclusion below)
    phs_py = {to_py_name(p) for p in phs}

    # -------- inner callable --------
    # build a stable transport-backed callable once
    stable_fn = make_stable_text_call_fn(ROUTER, {
        "base_url": f"https://{host}",
        "path": rel_url,  # may contain {placeholders}
        "method": method,
        "headers": {
            "X-RapidAPI-Key": RAPID_KEY,
            "X-RapidAPI-Host": host,
            # optional default for POST/PUT/PATCH:
            # "Content-Type": "application/json",
        },
        "op_id": name,
        "spec_version": "",
        "response_schema": None,
        "simulate_hint": (ep.get("description") or "")[:400],
        "rapidapi_bool_query": True,
        "timeout": 20,
    })


    def call_func(*args, **kwargs) -> str:
        error_prefix = f"Tool Call Error [{name}]:"

        try:
            bound = call_func.__signature__.bind(*args, **kwargs)
            bound.apply_defaults()
            passed = bound.arguments
        except TypeError as e:
            passed = kwargs
            # Capture errors like: "missing a required argument" or "got an unexpected keyword"
            error_msg = f"{error_prefix} Invalid arguments provided. {str(e)}"
            print(f"[LOG_DOCUMENTATION] {error_msg}")

        # 2. Verify all required parameters (from req_clean) are actually present
        # This is a secondary safety check for your API requirements
        required_names = {p['py_name'] for p in req_clean}
        missing = [rn for rn in required_names if rn not in passed or passed[rn] is None]
        
        if missing:
            error_msg = f"{error_prefix} Missing required parameters: {', '.join(missing)}"
            print(f"[LOG_DOCUMENTATION] {error_msg}")
        
        
        # Translate bound arguments to stable_fn kwargs
        translated = {}
        for k, v in passed.items():
            # Skip the *args/**kwargs catch-alls added for empty signatures
            if k in ("args", "kwargs"):
                continue

            # Path placeholder — pass through as-is
            if k in phs_py:
                translated[k] = v
                continue

            # Skip optional None
            if v is None:
                continue

            # Restore the original API parameter name
            out_key = name_map.get(k, k)

            # RapidAPI expects string "true"/"false" for booleans
            if isinstance(v, bool):
                translated[out_key] = "true" if v else "false"
            else:
                translated[out_key] = v

        # 4. Final execution call
        try:
            return stable_fn(**translated)
        except Exception as e:
            error_msg = f"{error_prefix} Execution failed with error: {str(e)}"
            print(f"[LOG_DOCUMENTATION] {error_msg}")

        
    # -------- build dynamic signature with py-safe names --------
    params: List[Parameter] = []
    for p in req_clean:
        p_type = map_type(p.get("type", "STRING"))
        params.append(Parameter(p["py_name"], kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=p_type))

    for p in opt_clean:
        p_type      = map_type(p.get("type", "STRING"))
        default_raw = p.get("default")
        default_py  = None
        if default_raw not in (None, ""):
            if p["type"].upper() == "NUMBER":
                try:
                    default_py = int(default_raw)
                except ValueError:
                    default_py = float(default_raw)
            elif p["type"].upper() == "BOOLEAN":
                default_py = str(default_raw).lower() == "true"
                p_type = bool
            else:
                default_py = default_raw

        annotation = p_type if default_py is not None else Optional[p_type]
        params.append(Parameter(p["py_name"], kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation, default=default_py))

    # Add path placeholder parameters to the signature
    for p_py in phs_py:
        params.append(
            Parameter(p_py, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
        )

    # Fallback: if the API has no parameters at all, accept *args/**kwargs
    if not params:
        params.append(Parameter("args", kind=Parameter.VAR_POSITIONAL))
        params.append(Parameter("kwargs", kind=Parameter.VAR_KEYWORD))



    call_func.__signature__ = Signature(params)
    call_func.__annotations__ = {p.name: p.annotation for p in params}
    usage = ", ".join([f"{p['py_name']}=…" for p in req_clean + opt_clean])
    description = (ep.get("description") or "")[:300] + f"\nUsage: {name}({usage})"

    return Tool(
        func=call_func,
        name=name,
        description=description,
    )
