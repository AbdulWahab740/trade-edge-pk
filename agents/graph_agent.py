import sys
import os
from pathlib import Path

# Add parent directory to path so we can import from tools
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_cohere import ChatCohere
from tools.graph_tools import get_schema, chart_agent

tools = [get_schema, chart_agent]
# Note: df is loaded in graph_tools.py and used by all tools
os.environ["COHERE_API_KEY"] = "oM4BkpsY3NEdlFxktdmYEtOHPJ7TBHJD5WcvvC2e"

def graph_agent(user_request: str) -> str:
    llm = ChatCohere(
        model="command-r7b-12-2024",
        temperature=0.3
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a trade analytics AI that analyzes import and export data from Pakistan.  
Your task is to generate **simple, working pandas queries** that extract relevant data for visualization.

The dataset `df` contains **both import and export** data combined together with a key column `tradetype`  
that specifies whether a row belongs to "Import" or "Export".
Also extract the title of the chart from the user request and give it the title of the chart in the format "Chart Title".
---

### 🧠 Context and Rules

- Always begin by calling `get_schema()` to see the columns.
- The dataframe includes these main columns:
  `month`, `group`, `commodity`, `unit`, `quantity_2025`, `rupees_2025`, `dollar_2025`, `quantity_2024`, `rupees_2024`, `dollar_2024`, `tradetype`
- `tradetype` is either `"Import"` or `"Export"`.

---

### 💡 How to Think

1. **Identify Trade Type:**
   - If the user mentions *import(s)*, *import trend*, or *import data*, filter by:
     `df['tradetype'] == 'Import'`
   - If the user mentions *export(s)*, *export trend*, or *export data*, filter by:
     `df['tradetype'] == 'Export'`
   - If the user doesn’t mention it, **do not filter** by `tradetype`.

2. **Commodity Search:**
   - Always use case-insensitive search:
     `df['commodity'].str.contains('keyword', case=False, na=False)`

3. **Combine Filters:**
   - Use `&` between filters, not `and`.
   - Example:  
     ```python
     df[(df['tradetype'] == 'Export') & (df['commodity'].str.contains('MILK', case=False, na=False))][['month','rupees_2025']]
     ```
if multiple queries:
❌ Wrong
df['month'] == '2025-06' & df['tradetype'] == 'Import'
✅ Correct
(df['month'] == '2025-06') & (df['tradetype'] == 'Import')
4. **Keep It Simple:**
   - Use `df[filters][['month','rupees_2025']]` or `[['month','dollar_2025']]` depending on what’s relevant.
   - Only add month filters if the user explicitly mentions months (e.g., "in June 2025").
   - If the user asks for a **trend**, include `'month'` and a **value column** (`'rupees_2025'` or `'dollar_2025'`).
   - Default to `'rupees_2025'` unless the user mentions dollar or USD.
   
    """
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    result = agent_executor.invoke({"input": user_request})
    # Prefer the direct tool output from chart_agent to avoid any LLM-added noise
    try:
        steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
        for step in reversed(steps):
            # step is expected to be a tuple: (AgentAction, observation)
            try:
                action, observation = step
            except Exception:
                action, observation = None, None
            tool_name = getattr(action, "tool", None)
            if tool_name == "chart_agent" and isinstance(observation, str):
                # chart_agent returns a JSON string via json.dumps(response)
                return observation
        # Fallback to the agent's output
        if isinstance(result, dict):
            return result.get("output", "")
        return result
    except Exception:
        # Final fallback
        if isinstance(result, dict):
            return result.get("output", "")
        return result


if __name__ == "__main__":
    print(graph_agent("Milk imports trend"))