"""
utils.py
────────
Shared utilities used across the SchemeArena codebase.
"""

import os


def find_tool_json_file(tool_name: str, dev_tools_v3_path: str = "./dev_tools_v3/") -> str | None:
    """
    Search recursively through dev_tools_v3 to find the JSON file for a tool.

    Args:
        tool_name: The tool name to search for.
        dev_tools_v3_path: Base path to the dev_tools_v3 directory.

    Returns:
        Absolute path to the JSON file if found and readable, else None.
    """
    if not tool_name or not isinstance(tool_name, str):
        print(f"Invalid tool name provided: {tool_name}")
        return None

    target_filename = tool_name.lower().replace(" ", "_") + ".json"

    if not os.path.exists(dev_tools_v3_path):
        print(f"Base directory does not exist: {dev_tools_v3_path}")
        return None

    if not os.path.isdir(dev_tools_v3_path):
        print(f"Path is not a directory: {dev_tools_v3_path}")
        return None

    try:
        for root, dirs, files in os.walk(dev_tools_v3_path):
            if target_filename in files:
                full_path = os.path.join(root, target_filename)
                if os.access(full_path, os.R_OK):
                    return full_path
                else:
                    print(f"Found file but cannot read: {full_path}")
    except Exception as e:
        print(f"Error searching for tool file '{target_filename}': {e}")

    return None
