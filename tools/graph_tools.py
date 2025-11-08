# chart_agent.py
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# If you're using LangChain @tool decorator:
try:
    from langchain.tools import tool
except Exception:
    # Fallback no-op decorator
    def tool(func):
        return func

# Load CSV from project root (works regardless of where script is run from)
project_root = Path(__file__).parent.parent
csv_file = project_root / "trade_data.csv"

df = pd.read_csv(csv_file)


# ------------------------------
# Configuration / Helpers
# ------------------------------

SAFE_GLOBALS = {"pd": pd}
# Allowed pandas expressions: df[...] or df.groupby(...).agg(...) etc.
SQL_KEYWORDS = {"SELECT", "FROM", "WHERE", "JOIN", "ORDER BY", "LIMIT"}

PNG_OUTPUT_DIR = "/tmp"  # change to your storage path (or /mnt/data)

# ---- Utility functions ----
@tool
def detect_chart_type(nl_request: str, df: Optional[pd.DataFrame] = None) -> str:
    """Detect the type of chart to be generated based on the natural language request."""
    r = nl_request.lower()
    if "trend" in r or "over time" in r or "month" in r or "year" in r:
        return "line"
    if "compare" in r or "vs" in r or "versus" in r:
        # if multiple series present, grouped bar or line
        return "line"
    if "top" in r or "rank" in r or "largest" in r:
        return "bar"
    if "share" in r or "distribution" in r or "percentage" in r:
        return "pie"
    if "scatter" in r or "relation" in r or "correl" in r:
        return "scatter"
    # fallback: if DF has time index -> line else bar
    if df is not None:
        # simple heuristic
        if any(c.lower() in ["month", "date", "time", "year"] for c in df.columns):
            return "line"
    return "bar"


def sanitize_and_eval(query: str, local_df: pd.DataFrame) -> Any:
    """Safely evaluate a pandas expression that starts with 'df' against local_df."""
    # Clean up the query: remove trailing punctuation that LLMs might add
    query = query.strip()
    # Remove trailing periods, commas, semicolons that might be added by LLM
    while query.endswith(('.', ',', ';')):
        query = query[:-1].strip()
    
    if not query.startswith("df"):
        raise ValueError("Query must start with 'df' (the DataFrame variable).")

    # Check for common Python 'and' usage (should be '&' for pandas)
    # Look for patterns like: df[...] and df[...] or df[...] and condition
    # Only flag if 'and' appears after a closing bracket or before df/condition
    import re
    if re.search(r'\]\s+and\s+', query) or re.search(r'\]\s+and\(', query) or re.search(r'and\s+df\[', query):
        raise ValueError(
            "❌ ERROR: Found Python 'and' operator! Use '&' (bitwise AND) for pandas, not 'and'.\n\n"
            "WRONG: df[df['A'] == 'x'] and df[df['B'] == 'y']\n"
            "CORRECT: df[(df['A'] == 'x') & (df['B'] == 'y')]\n\n"
            "Remember: Each condition must be in parentheses when using & operator."
        )
    
    # Check for inefficient multiple != conditions (anti-pattern)
    # Count occurrences of != in the query
    if query.count(' != ') >= 3:
        raise ValueError(
            "❌ ERROR: Using multiple != conditions is inefficient! Use == or .isin() instead.\n\n"
            "WRONG: df[(df['Month'] != '2025-01') & (df['Month'] != '2025-02') & ...]\n"
            "CORRECT (single value): df[df['Month'] == '2025-06']\n"
            "CORRECT (multiple values): df[df['Month'].isin(['2025-06', '2025-07'])]\n\n"
            "Always filter for what you WANT, not what you want to exclude!"
        )

    # quick SQL keyword detection
    q_upper = query.upper()
    for kw in SQL_KEYWORDS:
        if kw in q_upper:
            raise ValueError(f"SQL keyword detected ('{kw}'). Use pandas syntax.")

    # Allowed locals for eval: df -> local_df, pd
    allowed = {"df": local_df, "pd": pd}
    # Use eval but limit builtins:
    try:
        result = eval(query, {"__builtins__": {}}, allowed)
    except SyntaxError as e:
        raise ValueError(f"Pandas syntax error: {e}. Check for trailing punctuation or incomplete expressions.")
    except Exception as e:
        raise ValueError(f"Pandas evaluation error: {e}")
    return result

@tool
def df_to_plotly_json(
    chart_type: str,
    x: List[Any],
    y: Union[List[Any], List[List[Any]]],
    series: Optional[List[str]] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> Dict[str, Any]:
    """Return a Plotly figure as a serializable dict (json), minimized for size."""
    # Handle empty data
    if not y or len(y) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available to plot",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white")
        return fig.to_dict()
    
    fig = go.Figure()
    if chart_type == "line":
        # y can be 1D or 2D (multiple series)
        if len(y) > 0 and (isinstance(y[0], list) or isinstance(y[0], tuple)):
            for i, ser in enumerate(y):
                name = series[i] if series and i < len(series) else f"series_{i}"
                fig.add_trace(go.Scatter(x=x, y=ser, mode="lines+markers", name=name))
        else:
            name = series[0] if series and len(series) > 0 else y_label or "value"
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name))
    elif chart_type == "bar":
        if series and len(y) > 0 and (isinstance(y[0], list) or isinstance(y[0], tuple)):
            # grouped bar
            for i, ser in enumerate(y):
                name = series[i] if i < len(series) else f"series_{i}"
                fig.add_trace(go.Bar(x=x, y=ser, name=name))
        else:
            fig.add_trace(go.Bar(x=x, y=y, name=y_label or "value"))
    elif chart_type == "pie":
        fig = go.Figure(go.Pie(labels=x, values=y))
    elif chart_type == "scatter":
        # assume y is 1D and x is 1D
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers"))
    else:
        # fallback to bar
        fig.add_trace(go.Bar(x=x, y=y))

    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    # Build minimal dict to avoid massive embedded templates
    fig_dict = fig.to_dict()
    # Keep only essential layout fields
    minimal_layout = {
        "title": {"text": title} if isinstance(fig_dict.get("layout", {}).get("title"), dict) else title,
        "xaxis": {"title": {"text": x_label}},
        "yaxis": {"title": {"text": y_label}},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    }
    # Minimize data traces to essential fields
    minimal_data = []
    for tr in fig_dict.get("data", []):
        m = {k: tr[k] for k in ("type", "x", "y") if k in tr}
        if "mode" in tr:
            m["mode"] = tr["mode"]
        if "name" in tr:
            m["name"] = tr["name"]
        minimal_data.append(m)
    return {"data": minimal_data, "layout": minimal_layout}


def save_fig_png(fig_dict: Dict[str, Any], filename_prefix: str = "chart") -> str:
    """Save Plotly figure dict as PNG and return file path."""
    fig = go.Figure(fig_dict)
    fd, path = tempfile.mkstemp(prefix=filename_prefix, suffix=".png", dir=PNG_OUTPUT_DIR)
    os.close(fd)
    # write image using kaleido
    fig.write_image(path, scale=2)
    return path

# ---- Tools ----
@tool
def get_schema() -> str:
    """Get CSV schema including columns, types, and sample data. ALWAYS CALL THIS FIRST!"""
    info_str = f"Dataset: {len(df)} rows × {len(df.columns)} columns\n\n"
    info_str += "Columns and Types:\n"
    for col in df.columns:
        info_str += f"  • {col} ({df[col].dtype})\n"
    info_str += "\nSample Rows:\n"
    info_str += df.head(3).to_string()
    return info_str

def _call_tool_function(maybe_tool, *args, **kwargs):
    """Call underlying function even if it's decorated as a LangChain tool."""
    func = getattr(maybe_tool, "func", None)
    if callable(func):
        return func(*args, **kwargs)
    return maybe_tool(*args, **kwargs)

# ------------------------------
# The Chart Agent tool
# ------------------------------

@tool
def chart_agent(
    user_request: str,
    title: str,
    pandas_query: Optional[str] = None,
    explicit_chart_type: Optional[str] = None,
    return_image: bool = False,
    max_rows: int = 500,
) -> str:
    """Create a chart from pandas query. Keep queries simple!
    
    Args:
        user_request: What the user wants to see
        pandas_query: Simple pandas expression like df[filter][columns]
        explicit_chart_type: 'line'|'bar'|'pie'|'scatter' (optional)
        return_image: Keep False (frontend renders JSON)
        max_rows: Max rows to plot
    

    Returns JSON with chart_json, title, x_label, y_label, raw_data_preview.
    
    Example query: df[df['commodity'].str.contains('milk', case=False, na=False)][['month','rupees_2025']]
    """

    # --- Acquire DataFrame ---
    # If you have a CSV Agent & want to call it, implement run_data_query(...) to call that agent.
    # For now we evaluate pandas_query directly against the global `df`.
    try:
        # Ensure the global `df` exists in the environment
        if 'df' not in globals():
            return json.dumps({"error": "Global DataFrame 'df' not found. Make sure df is loaded."})

        local_df: pd.DataFrame = globals()["df"]

        # If user provided a pandas_query, execute it safely:
        if pandas_query:
            result = sanitize_and_eval(pandas_query, local_df)
            # If result is a DataFrame or Series, convert to DataFrame for plotting
            if isinstance(result, pd.Series):
                result_df = result.reset_index()
                result_df.columns = [str(c) for c in result_df.columns]
            elif isinstance(result, pd.DataFrame):
                result_df = result
            else:
                # scalar result -> return as text
                return json.dumps({"error": "Query returned scalar, not suitable for charting", "value": str(result)})
        else:
            # Attempt to infer a query from user_request (basic heuristics)
            # This is a simple heuristic parser: expected format "plot <column> over <time_col>"
            # For robust operation you should call your CSV agent to generate pandas_query.
            tokens = user_request.lower().split()
            # naive fallback: try to plot first numeric column vs month/year if present
            if "month" in [c.lower() for c in local_df.columns]:
                time_col = next(c for c in local_df.columns if c.lower() == "month")
            elif "date" in [c.lower() for c in local_df.columns]:
                time_col = next(c for c in local_df.columns if c.lower() == "date")
            else:
                # pick first non-object numeric
                numeric = local_df.select_dtypes(include='number').columns.tolist()
                if len(numeric) >= 2:
                    # choose first two numeric columns
                    result_df = local_df[[numeric[0], numeric[1]]].head(max_rows)
                elif len(numeric) == 1:
                    result_df = local_df[[numeric[0]]].head(max_rows)
                else:
                    return json.dumps({"error": "Couldn't infer a meaningful query. Please provide 'pandas_query'."})
            # If we have time_col, try to find the metric column from tokens
            if 'time_col' in locals():
                metric_candidates = [c for c in local_df.columns if c.lower() in user_request.lower()]
                # fallback metric pick: first numeric
                if metric_candidates:
                    metric_col = metric_candidates[0]
                else:
                    numeric = local_df.select_dtypes(include='number').columns.tolist()
                    if not numeric:
                        return json.dumps({"error": "No numeric columns available for plotting."})
                    metric_col = numeric[0]
                result_df = local_df[[time_col, metric_col]].head(max_rows)

        # Simplify names
        result_df = result_df.copy()
        # Reset index and ensure columns are strings
        result_df.columns = [str(c) for c in result_df.columns]
        result_df = result_df.reset_index(drop=True)

        # Check if DataFrame is empty
        if result_df.empty or len(result_df) == 0:
            return json.dumps({
                "error": "No data found matching the query",
                "message": f"The query returned an empty DataFrame with {len(result_df)} rows. Please check your filters or try a different query.",
                "query": pandas_query if pandas_query else "auto-inferred query",
                "raw_data_preview": []
            })

        # Try to pick x and y columns
        # If two columns: x = col0, y = col1
        if result_df.shape[1] == 1:
            col0 = result_df.columns[0]
            x = list(range(len(result_df)))
            y = result_df[col0].astype(object).tolist()
            series = [col0]
            x_label = ""
            y_label = col0
            title = title
        elif result_df.shape[1] >= 2:
            col0 = result_df.columns[0]
            col1 = result_df.columns[1]
            # Cast x to string to ensure compatibility with Plotly serialization
            x = result_df[col0].astype(str).tolist()
            y = result_df[col1].astype(object).tolist()
            series = [col1]
            x_label = col0
            y_label = col1
            title = title
            # If more than 2 columns, treat as multiple series
            if result_df.shape[1] > 2:
                # every other column is a series
                multi_y = []
                series = []
                for c in result_df.columns[1:]:
                    multi_y.append(result_df[c].tolist())
                    series.append(c)
                y = multi_y

        # Decide chart type
        chart_type = explicit_chart_type or _call_tool_function(detect_chart_type, user_request, result_df)

        # Build plotly JSON (already minimized by df_to_plotly_json)
        chart_json = _call_tool_function(
            df_to_plotly_json,
            chart_type=chart_type,
            x=x,
            y=y,
            series=series,
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        response = {
            "chart_json": chart_json,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
            "raw_data_preview": result_df.head(10).to_dict(orient="records"),
            "chart_type": chart_type,
        }

        # Optionally save a PNG and return path
        if return_image:
            try:
                img_path = save_fig_png(chart_json, filename_prefix="chart_")
                response["image_path"] = img_path
            except Exception as e:
                response["image_error"] = f"Failed to save PNG: {e}"

        return json.dumps(response, default=str)

    except Exception as e:
        return json.dumps({"error": str(e)})

   