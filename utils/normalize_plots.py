from typing import Any, Optional, Dict
import json

def normalize_plotly_chart(chart_data: Any) -> Optional[dict]:
    """Ensure chart_data always follows {data: [...], layout: {...}} structure."""
    if chart_data is None:
        return None
    
    # If it's a JSON string, parse it
    if isinstance(chart_data, str):
        try:
            chart_data = json.loads(chart_data)
        except Exception:
            return None

    # If the agent gave a raw data array, wrap it
    if isinstance(chart_data, list):
        chart_data = {"data": chart_data}

    # Ensure 'data' exists and is a list
    if not isinstance(chart_data, dict) or "data" not in chart_data:
        return None

    # Default layout if missing
    if "layout" not in chart_data:
        chart_data["layout"] = {
            "title": {"text": "Auto Generated Chart"},
            "xaxis": {"title": {"text": "X-axis"}},
            "yaxis": {"title": {"text": "Y-axis"}},
            "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
        }

    # Convert y values to numeric when possible
    for trace in chart_data["data"]:
        if "y" in trace:
            try:
                trace["y"] = [float(y) if isinstance(y, (int, float, str)) and str(y).replace('.', '', 1).isdigit() else y for y in trace["y"]]
            except Exception:
                pass

    return chart_data