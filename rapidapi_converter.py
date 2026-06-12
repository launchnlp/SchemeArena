"""
rapidapi_converter.py
────────────────────
Simple RapidAPI to OpenAPI converter for StableToolBench.
"""

import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _convert_rapidapi_to_openapi_spec(spec: dict) -> dict:
    """Convert RapidAPI format to OpenAPI 3.0 format (internal helper)."""
    host = spec.get("host", "")
    title = spec.get("title", spec.get("name", "API"))
    description = spec.get("tool_description", "")
    
    # Create OpenAPI structure
    openapi_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": title,
            "description": description,
            "version": "1.0.0"
        },
        "servers": [{"url": f"https://{host}"}],
        "components": {
            "securitySchemes": {
                "RapidAPI": {
                    "type": "apiKey",
                    "in": "header", 
                    "name": "X-RapidAPI-Key"
                }
            }
        },
        "security": [{"RapidAPI": []}],
        "paths": {}
    }
    
    # Convert each endpoint
    for endpoint in spec.get("api_list", []):
        name = endpoint.get("name", "")
        url = endpoint.get("url", "")
        method = endpoint.get("method", "GET").lower()
        desc = endpoint.get("description", "")
        
        # Extract path from full URL
        path = url.replace(f"https://{host}", "") or "/"
        
        if path not in openapi_spec["paths"]:
            openapi_spec["paths"][path] = {}
        
        # Simple parameter handling
        parameters = []
        for param in endpoint.get("required_parameters", []) + endpoint.get("optional_parameters", []):
            if param.get("type") != "CREDENTIALS":  # Skip API keys
                parameters.append({
                    "name": param.get("name", ""),
                    "in": "query",
                    "required": param in endpoint.get("required_parameters", []),
                    "description": param.get("description", ""),
                    "schema": {"type": "string"}
                })
        
        openapi_spec["paths"][path][method] = {
            "summary": name,
            "description": desc,
            "parameters": parameters,
            "responses": {
                "200": {"description": "Success"}
            }
        }
    
    return openapi_spec

def wrap_rapidapi_as_openapi_tool(rapidapi_file: str):
    """Convert RapidAPI file to OpenAPI and return as FunctionTool.
    
    Args:
        rapidapi_file: Path to RapidAPI JSON file
        
    Returns:
        FunctionTool that uses OpenAPI-style calls with RapidAPI auth
    """
    import os
    import requests
    from autogen_core.tools import FunctionTool
    from typing import Optional, Dict, Any
    
    # Load original RapidAPI spec for host info
    with open(rapidapi_file, 'r') as f:
        rapidapi_spec = json.load(f)
    
    # Convert to OpenAPI format
    openapi_spec = _convert_rapidapi_to_openapi_spec(rapidapi_spec)
    
    # Create custom tool with RapidAPI authentication
    title = openapi_spec.get("info", {}).get("title", "API")[:60]
    desc = openapi_spec.get("info", {}).get("description", "")[:200]
    host = rapidapi_spec.get("host", "")
    base_url = f"https://{host}"
    
    # Add endpoint information to description
    paths = openapi_spec.get("paths", {})
    endpoint_info = "\\nAvailable endpoints:\\n"
    for path, methods in paths.items():
        for method in methods.keys():
            endpoint_info += f"- {method.upper()} {path}\\n"
    
    desc = desc + endpoint_info
    
    def call(method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 20) -> str:
        """Call the API with proper RapidAPI authentication."""
        url = f"{base_url}{endpoint}"
        
        headers = {
            "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY", "SIGN-UP-FOR-KEY"),
            "X-RapidAPI-Host": host,
        }
        
        try:
            r = requests.request(
                method.upper(),
                url,
                headers=headers,
                params=payload if method.lower() == "get" else None,
                json=payload if method.lower() in ["post", "put", "patch"] else None,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.text[:4000]
        except requests.exceptions.ConnectionError as e:
            return f"Connection error: {str(e)}"
        except requests.exceptions.RequestException as e:
            return f"Request error: {str(e)}"
    
    return FunctionTool(call, name=title.lower().replace(" ", "_"), description=desc)