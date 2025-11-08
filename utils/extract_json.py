# Helper to robustly parse any JSON substring containing "chart_json"
from typing import Optional, Dict, Any

def _extract_chart_json_from_text(txt: str) -> Optional[Dict[str, Any]]:
            import json as _json
            import re as _re
            s = txt.strip()
            # Strip code fences if present
            if s.startswith("```"):
                # remove first fence line
                nl = s.find("\n")
                if nl != -1:
                    s = s[nl+1:]
                # trim trailing fence if present
                if s.endswith("```"):
                    s = s[:-3]
                s = s.strip()
            # Remove console/markup artifacts like <co> ... </co:...>
            s = _re.sub(r"<[^>]+>", "", s)
            # Remove ANSI escape sequences if any
            s = _re.sub(r"\x1b\[[0-9;]*m", "", s)
            # Fix mangled chart_json key like chart_json</co: 2:[0]>
            s = _re.sub(r'"?chart_json[^"\s]*"?:', '"chart_json":', s)
            # Fast path: direct parse
            try:
                obj = _json.loads(s)
                if isinstance(obj, dict) and "chart_json" in obj:
                    return obj["chart_json"]
            except Exception:
                pass
            # Heuristic: find the substring that encloses chart_json
            key_idx = s.find('"chart_json"')
            if key_idx == -1:
                key_idx = s.find("'chart_json'")
            if key_idx != -1:
                # expand to nearest surrounding braces
                start = s.rfind('{', 0, key_idx)
                end = s.find('}', key_idx)
                # grow end to the matching closing brace by counting
                if start != -1:
                    depth = 0
                    for i in range(start, len(s)):
                        if s[i] == '{':
                            depth += 1
                        elif s[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if end and end > start:
                        try:
                            obj = _json.loads(s[start:end+1])
                            if isinstance(obj, dict) and "chart_json" in obj:
                                return obj["chart_json"]
                        except Exception:
                            pass
            # Fallback: global outermost braces
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    obj = _json.loads(s[start:end+1])
                    if isinstance(obj, dict) and "chart_json" in obj:
                        return obj["chart_json"]
                except Exception:
                    pass
            return None