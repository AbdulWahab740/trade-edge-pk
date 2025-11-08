
from dotenv import load_dotenv
import os

load_dotenv()  # loads only if .env exists locally

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
import pandas as pd
# from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional, List, Dict, Any
from pathlib import Path
from typing import Any

# ---- LOAD DATA ----
# Load CSV from project root (works regardless of where script is run from)
project_root = Path(__file__).parent.parent
csv_file = project_root / "trade_data.csv"

df = pd.read_csv(csv_file)
# Clean whitespace
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

# ---- Configuration ----
# Quick SQL keyword detection to prevent accidental SQL-like queries
SQL_KEYWORDS = {"SELECT", "FROM", "WHERE", "JOIN", "ORDER BY", "LIMIT", "GROUP BY"}

GEMINI_API_KEY = "AIzaSyA6Qt6_n4LBwX9G03h3kzi0Eoh8RYH4KYE"
def load_llm():
    """Loads and caches the LLM for use in the application."""
    if not GEMINI_API_KEY:
        return "API key not avaliable"
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            max_tokens=2000,
            api_key=GEMINI_API_KEY
        )
        return llm
    except Exception as e:
        pass

# ---- Tools ----
@tool
def get_schema_for_csv() -> str:
    """Get CSV schema including columns, types, and sample data. ALWAYS CALL THIS FIRST!"""
    info_str = f"Dataset: {len(df)} rows × {len(df.columns)} columns\n\n"
    info_str += "Columns and Types:\n"
    for col in df.columns:
        info_str += f"  • {col} ({df[col].dtype})\n"
    info_str += "\nSample Rows:\n"
    info_str += df.head(3).to_string()
    return info_str


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

def _call_tool_function(maybe_tool, *args, **kwargs):
    """Call underlying function even if it's decorated as a LangChain tool."""
    func = getattr(maybe_tool, "func", None)
    if callable(func):
        return func(*args, **kwargs)
    return maybe_tool(*args, **kwargs)

@tool
def csv_query(
    
    pandas_query: Optional[str] = None,
    max_rows: int = 100,
) -> str:
    """Create a chart from pandas query. Keep queries simple!
    
    Args:
        user_request: What the user wants to see
        pandas_query: Simple pandas expression like df[filter][columns]
       
        max_rows: Max rows to plot
    ⚠️ CRITICAL: Use PANDAS syntax, NOT SQL!
    
    CORRECT Pandas Examples:
    ✓ df[df['month'] == '2022-06']
    ✓ df.nlargest(1, 'Quantity_2025')
    ✓ df[df['group'] == 'Food Group']['quantity_2025'].max()
    ✓ df.groupby('group')['rupees_2025'].sum()
    ✓ df.sort_values('dollar_2025', ascending=False).head(1)
    
    WRONG SQL Examples (will fail):
    ✗ SELECT * FROM df WHERE month='2022-06'
    ✗ SELECT MAX(quantity_2025) FROM df
    ✗ ORDER BY, LIMIT, FROM - these are SQL keywords!
    
    The DataFrame is called 'df'. Always start queries with 'df'.
    Returns the response of the pandas query with general information don't over explain and head that to economics agent.
    Example query: df[df['commodity'].str.contains('milk', case=False, na=False)][['month','rupees_2025']]
    """

    try:
        # Ensure the global `df` exists in the environment
        if 'df' not in globals():
            return "Global DataFrame 'df' not found. Make sure df is loaded."

        local_df: pd.DataFrame = globals()["df"]

        # If user provided a pandas_query, execute it safely:
        if pandas_query:
            result = sanitize_and_eval(pandas_query, local_df)
            # Normalize output to string for tool safety
            if isinstance(result, pd.DataFrame):
                if max_rows and max_rows > 0:
                    return result.head(max_rows).to_string(index=False)
                else:
                    return result.to_string(index=False)
            elif isinstance(result, pd.Series):
                return result.to_string()
            else:
                return str(result)
        return "No pandas_query provided. Please supply a valid pandas expression starting with 'df'."
    except Exception as e:
        return str(e)

class SafeAgentExecutor(AgentExecutor):
    """A safer AgentExecutor that auto-fixes duplicated tool calls."""
    def _call(self, inputs: dict, run_manager=None):
        # Patch: Fix double tool names like 'get_schema_for_csvget_schema_for_csv'
        if "input" in inputs:
            fixed_input = inputs["input"].replace("get_schema_for_csvget_schema_for_csv", "get_schema_for_csv")
            fixed_input = fixed_input.replace("csv_querycsv_query", "csv_query")
            inputs["input"] = fixed_input
        return super()._call(inputs, run_manager=run_manager)

def csv_analyzer(query: str, conversation_history: Optional[List] = None) -> str:
    """
    Analyze CSV data using pandas syntax.
    
    Args:
        query: Pandas query to execute
        conversation_history: Previous messages for context
    
    Returns:
        summary analyses from the data using pandas syntax.
    """
    try:
        tools = [get_schema_for_csv, csv_query]

        llm = ChatCohere(model="command-r7b-12-2024", temperature=0.2)
        # llm = load_llm()
        # CRITICAL: Very explicit prompt about pandas vs SQL
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a **Data Analytics AI** for Pakistan's trade data.

Your job: Answer questions with SIMPLE pandas queries and provide clear summaries.

---

### 📊 Dataset Info

- Dataframe name: `df`
- Key columns: `month`, `group`, `commodity`, `unit`, `tradetype` (Import/Export)
- Value columns: `quantity_2025`, `rupees_2025`, `dollar_2025`
- ALWAYS call `get_schema_for_csv()` first!

---

### ✅ Query Rules

1. **ONE query at a time** - Never combine multiple statements
2. **Simple syntax** - Use basic pandas operations
3. **No comments** - Don't add # comments in queries
4. **No variables** - Don't try to create variables like `total_rupees = ...`

---

### 💡 Good Query Patterns

**Filtering:**
```python
df[df['tradetype'] == 'Export']
df[df['month'] == '2025-06']
df[(df['tradetype'] == 'Import') & (df['month'] == '2025-06')]
```

**Aggregations (ONE calculation per query):**
```python
df[df['group'] == 'Food Group']['rupees_2025'].sum()
df.groupby('group')['rupees_2025'].sum()
df['rupees_2025'].mean()
```

**Top/Bottom:**
```python
df.nlargest(5, 'rupees_2025')[['commodity', 'rupees_2025']]
df.groupby('group')['rupees_2025'].sum().nlargest(5)
```

---

### 🎯 Analysis Approach

For each question, run **3-5 SEPARATE simple queries**:

**Example: "Analyze Food Group exports"**

Query 1: Get the data
```python
df[(df['tradetype'] == 'Export') & (df['group'] == 'Food Group')][['commodity', 'rupees_2025', 'dollar_2025']]
```

Query 2: Calculate total
```python
df[(df['tradetype'] == 'Export') & (df['group'] == 'Food Group')]['rupees_2025'].sum() # if needed
if asked a single value to return like valued import in June 2025?
df[(df['tradetype'] == 'Import') & (df['month'] == '2025-06')][['commodity', 'rupees_2025']].sort_values(by='rupees_2025', ascending=False).head(1)
```

Query 3: Count entries
```python
df[(df['tradetype'] == 'Export') & (df['group'] == 'Food Group')].shape[0]
```

Query 4: Get top items
```python
df[(df['tradetype'] == 'Export') & (df['group'] == 'Food Group')].nlargest(3, 'rupees_2025')[['commodity', 'rupees_2025']]
```
**Keep It Simple:**
   - Use `df[filters][['month','rupees_2025']]` or `[['month','dollar_2025']]` depending on what’s relevant.
   - Only add month filters if the user explicitly mentions months (e.g., "in June 2025").
   - If the user asks for a **trend**, include `'month'` and a **value column** (`'rupees_2025'` or `'dollar_2025'`).
   - Default to `'rupees_2025'` unless the user mentions dollar or USD.
   
---

### 📋 Response Format

After running queries, format like this:

FOOD GROUP EXPORTS ANALYSIS

Data Overview:
[Show top 5 rows from Query 1]

Key Metrics you must need to find if applicable:
- Total Value: PKR X 
- Total Items: Y entries 
- Average Value: PKR Z (calculate from total/count)
- Best Performing Commodity: XYZ (highest value)


Summary for Economics Expert:
Food Group exports total PKR X across Y items. Top 3 commodities account for A% of total value. [Add 1-2 more insights]

Return the output in markdown format and write the response clear, attractive and professional so it should be best for analysis

🚫 What NOT to Do

❌ **Multi-line queries:**
❌ **Comments in queries:**
❌ **Variable assignments:**
```python
# DON'T DO THIS
total = df['rupees_2025'].sum()
count = df.shape[0]
average = total / count
```

✅ **Do this instead - ONE query at a time:**
```python
df[df['month'] == '2025-06']['rupees_2025'].sum()
```
double check the pandas rules!
---

### 🔍 Workflow

1. Call `get_schema_for_csv()` to check columns
2. after checking columns, perform the analysis based on the user query
3. if the user asks for a trend, include `'month'` and a **value column** (`'rupees_2025'` or `'dollar_2025'`)
4. if the user asks for a **total value**, include `'rupees_2025'` or `'dollar_2025'`
5. if the user asks for a **top items**, include `'commodity'` and `'rupees_2025'` or `'dollar_2025'`
6. Format all results into clear summary

---

Keep it simple. One query = one calculation. Multiple queries = complete analysis.
"""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        csv_agent_executor = SafeAgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True
)

        # Convert conversation history to the format expected by the agent
        history = []
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, dict):
                    # If it's already a dictionary, use it as is
                    history.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
                else:
                    # If it's a message object, extract role and content
                    history.append({
                        "role": getattr(msg, 'role', 'user'),
                        "content": getattr(msg, 'content', str(msg))
                    })
        
        # Invoke the agent with the query and history
        result = csv_agent_executor.invoke({
            "input": query,
            "conversation_history": history
        })
        
        return result["output"]
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    query = "what are the export trends of Food group in 2025?"
    print(csv_analyzer(query))