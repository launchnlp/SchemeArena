"""
json_extract.py
───────────────
Robust JSON extraction and repair utilities used across the SchemeArena codebase.
"""

import json
import re


def robust_json_fix(text):
    """
    Handle JSON with nested quotes and multi-line content by parsing more carefully.
    """
    # Extract from markdown if present
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        content = text.strip()
        if content.startswith("'") and content.endswith("'"):
            content = content[1:-1]

    # Fix escaped single quotes first
    content = content.replace("\\'", "'")

    # Try parsing as-is first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    def fix_json_string(json_str):
        """Fix common JSON string issues: unescaped newlines and quotes in string values."""
        result = []
        i = 0
        in_string = False
        escape_next = False

        while i < len(json_str):
            char = json_str[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                if not in_string:
                    # Starting a string
                    in_string = True
                    result.append(char)
                else:
                    # Could be ending a string or a quote within
                    # Look ahead to see if this looks like the end of a string
                    next_non_space = i + 1
                    while next_non_space < len(json_str) and json_str[next_non_space].isspace():
                        next_non_space += 1
                    
                    if (next_non_space >= len(json_str) or 
                        json_str[next_non_space] in ',}]\n'):
                        # This is the end of the string
                        in_string = False
                        result.append(char)
                    else:
                        # This is a quote within the string, escape it
                        result.append('\\"')
                i += 1
                continue

            if in_string:
                # We're inside a string, need to escape special characters
                if char == '\n':
                    result.append('\\n')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                else:
                    result.append(char)
            else:
                result.append(char)

            i += 1

        return ''.join(result)

    fixed_content = fix_json_string(content)

    try:
        return json.loads(fixed_content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed after fixes: {e}")
        print(f"Around position {e.pos}:")
        start = max(0, e.pos - 100)
        end = min(len(fixed_content), e.pos + 100)
        print(f"Context: {repr(fixed_content[start:end])}")
        
        # Last resort: try a more aggressive approach
        return aggressive_json_fix(content)


def aggressive_json_fix(content):
    """
    More aggressive JSON fixing for badly malformed JSON.
    """
    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # If line contains ": " followed by a quote, it's likely a JSON string value
        if '": "' in line and not line.strip().endswith('",') and not line.strip().endswith('"'):
            # This line starts a multi-line string value
            # Find the start of the content after ": "
            match = re.search(r'": "(.*)$', line)
            if match:
                prefix = line[:match.start(1)]
                content_part = match.group(1)
                # Escape the content and close the line properly
                escaped_content = content_part.replace('\\', '\\\\').replace('"', '\\"')
                fixed_lines.append(f'{prefix}{escaped_content}')
            else:
                fixed_lines.append(line)
        elif line.strip().startswith('"') and ': "' not in line:
            # This might be a continuation of a multi-line string
            # Escape the content
            stripped = line.strip()
            if stripped.endswith('"'):
                # This is the end of the multi-line string
                content_part = stripped[:-1]  # Remove the closing quote
                escaped_content = content_part.replace('\\', '\\\\').replace('"', '\\"')
                fixed_lines[-1] += f'\\n{escaped_content}"'
            else:
                # Middle of multi-line string
                escaped_content = stripped.replace('\\', '\\\\').replace('"', '\\"')
                fixed_lines[-1] += f'\\n{escaped_content}'
        else:
            fixed_lines.append(line)

    fixed_content = '\n'.join(fixed_lines)

    try:
        return json.loads(fixed_content)
    except json.JSONDecodeError as e:
        print(f"Aggressive fix also failed: {e}")
        print("Final attempt with regex replacement...")
        
        # Final attempt: use regex to fix the most common pattern
        # Multi-line string values in JSON
        def fix_multiline_strings(text):
            # Pattern to match: "key": "value with
            # actual newlines
            # more content"
            
            # First, let's handle the case where we have actual newlines in JSON strings
            # Replace actual newlines within quoted strings with \n
            result = []
            in_string = False
            i = 0

            while i < len(text):
                char = text[i]

                if char == '"' and (i == 0 or text[i-1] != '\\'):
                    in_string = not in_string
                    result.append(char)
                elif in_string and char == '\n':
                    result.append('\\n')
                elif in_string and char == '\t':
                    result.append('\\t')
                elif in_string and char == '\r':
                    result.append('\\r')
                else:
                    result.append(char)

                i += 1

            return ''.join(result)

        final_content = fix_multiline_strings(content)

        try:
            return json.loads(final_content)
        except json.JSONDecodeError:
            print("All JSON fixing attempts failed")
            return None
