# pricing_bot
## Folder structure
datagen contains code for sample data generation <br>
agent contains python scripts used to build AI agents to process the data and respond to conversations
## Gemini API integration 
Refer this link for Gemini Quickstart integration: https://ai.google.dev/gemini-api/docs/quickstart <br>
Refer this link to check usage details for your google login & google project: https://aistudio.google.com/

---

## MCP CSV file server 🔧

A small Model Context Protocol (MCP) server that loads CSV files from `src/data` (or the folder set by the `CSV_FOLDER` env var) into an in-memory SQLite DB and exposes tools for listing datasets, getting summaries/descriptions, and running read-only SQL queries. The server is implemented using `mcp.server.fastmcp` when available and exposes the following tools (tool names shown):

- Run:

```bash
pip install -r requirements.txt
# Install MCP runtime if needed, e.g. `pip install mcp` (package name may vary)
python -m src.mcp.file_server
```

- Tools:
  - `list_datasets` — list available datasets
  - `describe` (params: `{ "name": "dataset_name" }`) — detailed description of dataset
  - `summary` (params: `{ "name": "dataset_name" }`) — compact numeric/categorical summary
  - `query` (params: `{ "sql": "SELECT ...", "limit": 100 }`) — run a read-only SQL query (read-only `SELECT`/`WITH` enforced)
  - `reload` — reload CSV files from disk

Notes: only `SELECT`/`WITH` queries are accepted; results are limited to 1000 rows by default. If `mcp.server.fastmcp` isn't installed, the module will raise an informative error when run as an MCP server.

---

## Sample Crew AI client (demo)

A sample agent client is provided in `src/agent/mcp_client.py` that demonstrates how an agent can translate KPI-style requests into SQL, run them using the MCP file server (in-process), and return results. The included fallback agent does not require external Crew AI packages and can be used for development and testing.

- Run the scripted demo:

```bash
python -m src.agent.mcp_client
```

- Run an interactive demo:

```bash
python -m src.agent.mcp_client
```

- If you have a Crew AI runtime installed, you can extend `create_crewai_agent_if_available` in `src/agent/mcp_client.py` to integrate directly with your agent framework and register MCP tools as tool calls.