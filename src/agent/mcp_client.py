"""Sample Crew AI agent client for the MCP CSV file server.

This module implements a small agent that demonstrates how to use the MCP
server (see `src.mcp.file_server`) to answer KPI-style requests by:

- parsing a user's KPI request into an SQL query (simple heuristics/templates)
- running that SQL against the MCP FileServer (in-process)
- returning nicely formatted results

The agent prefers to use the official Crew AI / crewai package when available
(and will attempt to register the MCP tools if a runtime is present). If the
package isn't installed, a local interactive fallback agent is provided which
matches the required behavior for demonstration and testing.

Usage (interactive):
    python -m src.agent.mcp_client

Requirements: none additional for the fallback agent. For full Crew AI
integration, install the crew runtime (package name may vary).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import os

# Try to import crew AI runtime (best-effort). If not available, fall back to a
# local simple agent implementation that mimics the behaviour.
try:
    import crewai  # type: ignore
    HAVE_CREW = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_CREW = False

# We will import the FileServer from the MCP module to query CSVs in-process.
from src.mcp.file_server import FileServer


class MCPClient:
    """Simple in-process client for the MCP FileServer.

    For demo purposes this client directly instantiates and calls
    FileServer methods. In a real deployment the client would call the
    MCP runtime (e.g., via mcp client libraries or networked RPC).
    """

    def __init__(self, csv_folder: Optional[str] = None):
        self.server = FileServer(csv_folder)

    def list_datasets(self) -> List[Dict[str, Any]]:
        return self.server.list_datasets()

    def describe(self, name: str) -> Dict[str, Any]:
        return self.server.describe(name)

    def summary(self, name: str) -> Dict[str, Any]:
        return self.server.summary(name)

    def query(self, sql: str, limit: Optional[int] = None) -> Dict[str, Any]:
        print(sql)
        qr = self.server.run_query(sql, limit=limit)
        print(qr)
        return {"columns": qr.columns, "rows": qr.rows}

    def reload(self) -> Dict[str, Any]:
        self.server.reload()
        return {"datasets": self.list_datasets()}


# A very small intent parser that maps user KPI requests to SQL templates.
@dataclass
class SQLCandidate:
    sql: str
    confidence: float
    reason: str


class KPIIntentParser:
    """Convert a natural-language KPI request into an SQL query using
    simple heuristics and dataset introspection."""

    def __init__(self, client: MCPClient):
        self.client = client

    def guess_table(self, tokens: List[str]) -> Optional[str]:
        # pick a table that matches any token in its name or columns
        datasets = self.client.list_datasets()
        token_set = set(t.lower() for t in tokens)
        for ds in datasets:
            name = ds["name"].lower()
            if any(t in name for t in token_set):
                return ds["name"]
        # fallback: first dataset
        return datasets[0]["name"] if datasets else None

    def available_columns(self, table: str) -> List[str]:
        try:
            desc = self.client.describe(table)
            return desc.get("columns", [])
        except Exception:
            return []

    def parse(self, text: str) -> SQLCandidate:
        t = text.lower()
        tokens = re.findall(r"[a-zA-Z_]+", t)
        table = self.guess_table(tokens)

        cols = self.available_columns(table) if table else []

        # common KPI patterns
        if "revenue" in tokens or "sales" in tokens or "sales_amount" in ":".join(cols):
            # try to pick date col and amount col
            date_col = next((c for c in cols if "date" in c), None)
            amt_col = next((c for c in cols if c in ("sales_amount", "sales_amt", "sales", "amount", "salesamount")), None)
            if amt_col is None:
                # try numeric column names
                amt_col = next((c for c in cols if "amount" in c or "sales" in c or "price" in c), None)

            if "by month" in t or "per month" in t or "monthly" in t:
                if date_col and amt_col:
                    sql = f"SELECT strftime('%Y-%m', {date_col}) AS month, SUM({amt_col}) AS total_revenue FROM {table} GROUP BY month ORDER BY month"
                    return SQLCandidate(sql=sql, confidence=0.9, reason="revenue by month (date & amount detected)")
            # fallback: total revenue
            if amt_col:
                sql = f"SELECT SUM({amt_col}) AS total_revenue FROM {table}"
                return SQLCandidate(sql=sql, confidence=0.7, reason="total revenue")

        if "count" in tokens or "transactions" in tokens or "opportunity" in tokens:
            # count rows possibly by month
            date_col = next((c for c in cols if "date" in c), None)
            if "by month" in t and date_col:
                sql = f"SELECT strftime('%Y-%m', {date_col}) AS month, COUNT(*) AS cnt FROM {table} GROUP BY month ORDER BY month"
                return SQLCandidate(sql=sql, confidence=0.85, reason="count by month")
            sql = f"SELECT COUNT(*) AS total FROM {table}"
            return SQLCandidate(sql=sql, confidence=0.6, reason="count rows")

        if "average" in tokens or "avg" in tokens or "mean" in tokens:
            # find numeric col
            num_col = next((c for c in cols if c in ("sales_amount", "list_price", "price", "amount", "quantity")), None)
            if num_col:
                sql = f"SELECT AVG({num_col}) AS avg_{num_col} FROM {table}"
                return SQLCandidate(sql=sql, confidence=0.7, reason="average detected")

        # default: attempt a head query
        if table:
            sql = f"SELECT * FROM {table} LIMIT 10"
            return SQLCandidate(sql=sql, confidence=0.4, reason="sample rows")

        # ultimate fallback
        return SQLCandidate(sql="", confidence=0.0, reason="no suitable mapping found")


class SQLAgent:
    """Class defined to build SQL agent using Crew AI runtime."""

    def __init__(self):
        self.server_params = {
            "url": "http://localhost:8000/filemcp",  # Replace with your actual Streamable HTTP server URL
            "transport": "streamable-http"
        }

        # Create a StdioServerParameters object
        self.stdio_params=StdioServerParameters(
            command="python", 
            args=["D:\\SNK\\codes\\repos\\pricing_bot\\src\\mcp\\file_server.py"],
            # env={"UV_PYTHON": "3.12", **os.environ},
        )

        self.mcp_server_adapter = None # MCPServerAdapter(server_params=self.stdio_params)

        # my-mcp-server-495272d2
        # Ensure instance attrs always exist
        self.mcp_server_adapter = None
        self.tools: List[Any] = []
        self.llm = None
        self.sql_agent = None

        try:
            # store adapter on self so __del__ and other methods can access it
            #self.mcp_server_adapter = MCPServerAdapter(self.server_params)
            self.mcp_server_adapter = MCPServerAdapter(self.stdio_params)
            self.mcp_server_adapter.start()
            self.tools = getattr(self.mcp_server_adapter, "tools", [])
            print(f"Available tools (manual Streamable HTTP): {[getattr(t, 'name', str(t)) for t in self.tools]}")

            # Fix typo: 'temperature' not 'temperatue'
            self.llm = LLM(model="ollama/deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0)
            print(self.llm)

            # Create agent only if tools/llm are available; guard to avoid raising if any not present
            self.sql_agent = Agent(
                role="SQL code generation and execution",
                goal="Create SQL Queries for sqlite database",
                verbose=True,
                tools=self.tools,
                backstory=(
                    """
                    You are a highly skilled sqlite3 expert with 10 years of experience that writes SQL code for sqlite database.
                    You are fully capable to write explorative queries to find tables available in the given database.
                    You can create SQL queries to fetch the schema of a table.
                    You can also create SQL queries to filter data in a table and run 'group by' summarization queries.
                    Make sure you provide error free SQL queries using the correct columns from the tables.
                    """
                ),
                llm=self.llm,
            )
        except Exception as e:
            # Keep instance in a clean fallback state and log the error
            print(f"Error setting up MCP Server Adapter / Agent / LLM: {e}")
            # leave self.mcp_server_adapter / self.tools / self.llm / self.sql_agent as-is for safe fallback

    def handle(self, input_query: str):
        # if self.sql_agent is None:
        #     return "SQL agent is not initialized (MCP adapter or LLM failed to start). Please check logs; falling back is not configured for this agent."

        desc = f"""
                The Model Context Protocol server running at http://localhost/filemcp offers many interface to query datasets.
                The datasets can be considered as tables in a sqlite database.
                It offers tools like list datasets, describe dataset, summary of dataset and also run SQL queries on the datasets.
                Create relevant SQL queries to leverage the data present in the datasets, to answer the questions or execute 
                the instructions given by the user. Also run the queries you create using the sql_runner tool to get results 
                from the datasets, and return the output data in your final answer.
                The instruction from user is provided in input: .
        """

        expo = f"""
            I need SQL queries which can be run in a mysql database. You must create the SQL queries based on the schema of the tables
            available in datasets available in MCP server. You can as well create SQL queries to read schema from the databse, to understand the 
            columns present in certain table.

            I need the SQL queries to answer the questions of the user. The questions or instructions from the user are available in the 
            input: .
            The questions can request informations like below:
            1. Data descriptions like columns and their data types in a particular table.
            2. Select some rows of data from a table
            3. Select some columns of data from a table
            4. Apply filters on rows based on some columns
            5. Apply 'group by' based summarization on some columns like calculating 'sum', 'mean', 'percentage of total' etc.
            6. Or a combination of above

            You can run the queries using the SQL runner tool. The tool takes an SQL query as argument to run the SQL queries and 
            obtain the results as a pandas dataframe.

            Some important points to note while creating SQL queries are:
            1. Always use the full qualified name of tables to avoid errors in the queries.
            2. In case of error in the queries, recheck in the scrapper tool to find resolution of the error and modify the sql query
            to fix the error.
            3. Use the specified database only.
            4. Do not update any rows in the existing data.

            You should first find out the correct table to be used in your query.
            You can use nested queries in order to accomplish complex tasks. You can create new table by running appropriate SQL queries,
            and also insert data into tables from another table.

            Review the final answer thoroughly before presenting. Prefer to produce the results using minimum number of queries.
            Optimize the queries well before running them and presenting the final answer.

            The final answer you give must contain below items: 
            1. The output from the queries you ran. The output must be presented as tabular data with exact column 
            names as provided by the tool. 
            2. You must also provide all the SQL queries you run using the tool in the same chronological 
            order as you run them.
        """



        self.sql_task = Task(
            description = f"""
                The Model Context Protocol server running at http://localhost/filemcp offers many interface to query datasets.
                The datasets can be considered as tables in a sqlite database.
                It offers tools like list datasets, describe dataset, summary of dataset and also run SQL queries on the datasets.
                Create relevant SQL queries to leverage the data present in the datasets, to answer the questions or execute 
                the instructions given by the user. Also run the queries you create using the sql_runner tool to get results 
                from the datasets, and return the output data in your final answer.
                The instruction from user is provided in input: .
            """,
            expected_output = f"""
                I need SQL queries which can be run in a mysql database. You must create the SQL queries based on the schema of the tables
                available in datasets available in MCP server. You can as well create SQL queries to read schema from the databse, to understand the 
                columns present in certain table.

                I need the SQL queries to answer the questions of the user. The questions or instructions from the user are available in the 
                input: .
                The questions can request informations like below:
                1. Data descriptions like columns and their data types in a particular table.
                2. Select some rows of data from a table
                3. Select some columns of data from a table
                4. Apply filters on rows based on some columns
                5. Apply 'group by' based summarization on some columns like calculating 'sum', 'mean', 'percentage of total' etc.
                6. Or a combination of above

                You can run the queries using the SQL runner tool. The tool takes an SQL query as argument to run the SQL queries and 
                obtain the results as a pandas dataframe.

                Some important points to note while creating SQL queries are:
                1. Always use the full qualified name of tables to avoid errors in the queries.
                2. In case of error in the queries, recheck in the scrapper tool to find resolution of the error and modify the sql query
                to fix the error.
                3. Use the specified database only.
                4. Do not update any rows in the existing data.

                You should first find out the correct table to be used in your query.
                You can use nested queries in order to accomplish complex tasks. You can create new table by running appropriate SQL queries,
                and also insert data into tables from another table.

                Review the final answer thoroughly before presenting. Prefer to produce the results using minimum number of queries.
                Optimize the queries well before running them and presenting the final answer.

                The final answer you give must contain below items: 
                1. The output from the queries you ran. The output must be presented as tabular data with exact column 
                names as provided by the tool. 
                2. You must also provide all the SQL queries you run using the tool in the same chronological 
                order as you run them.
            """,
            #tools = [sql_runner], #scrap_tool,
            tools = self.tools, 
            agent = self.sql_agent
        )


        self.sql_crew = Crew(
            agents=[self.sql_agent],
            tasks=[self.sql_task],
            process=Process.sequential,
            verbose=True,
        )

        results = self.sql_crew.kickoff(inputs={"input_query": input_query})
        return results


    def __del__(self):
        if self.mcp_server_adapter is not None: #and self.mcp_server_adapter.is_connected:
            print("Stopping Streamable HTTP MCP server connection (manual)...")
            self.mcp_server_adapter.stop()  # **Crucial: Ensure stop is called**
        elif self.mcp_server_adapter:
            print("Streamable HTTP MCP server adapter was not connected. No stop needed or start failed.")




class CrewAgentFallback:
    """A simple interactive agent used when no Crew AI runtime is available.

    The agent accepts a user request, parses an SQL candidate, runs it,
    and returns a short text response including the results.
    """

    def __init__(self, client: MCPClient):
        self.client = client
        self.parser = KPIIntentParser(client)

    def handle(self, user_text: str) -> str:
        candidate = self.parser.parse(user_text)
        print(candidate)
        if not candidate.sql:
            return "Sorry, I couldn't map that request to an SQL query. Try asking for basic KPIs like 'total revenue by month' or 'count transactions'"

        # run the query and format results
        try:    
            res = self.client.query(candidate.sql, limit=1000)
        except Exception as e:
            return f"Error running query: {e}"

        out = [f"SQL (confidence={candidate.confidence:.2f}): {candidate.reason}\n{candidate.sql}\n"]
        cols = res.get("columns", [])
        rows = res.get("rows", [])
        if not rows:
            out.append("No rows returned.")
        else:
            # pretty-print first few rows as JSON for readability
            out.append("Results (first 10 rows):")
            for r in rows[:10]:
                out.append(json.dumps(dict(zip(cols, r)), default=str))
        return "\n".join(out)





# If crew AI runtime exists, attempt to register an agent that can use the MCP tools.
# This is best-effort because different crew versions may have different APIs.
def create_crewai_agent_if_available(client: MCPClient):
    if not HAVE_CREW:
        return None

    # This is a placeholder to show where integration would happen. The real
    # integration depends on the Crew AI API (agent, tools, tool invocation APIs).
    # We provide a helpful exception to guide the user to wire the agent up.
    raise RuntimeError("Crew AI runtime detected but automatic integration is not implemented in this demo. Please open an issue if you need a specific integration.")


def demo_interactive(csv_folder: Optional[str] = None):
    client = MCPClient(csv_folder)
    agent = CrewAgentFallback(client)
    agent = SQLAgent()

    print("CrewAI MCP demo agent (fallback). Type 'exit' or Ctrl-C to quit.")
    print("Try requests like: 'total revenue by month', 'count transactions', 'average list_price'")

    while True:
        try:
            text = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye")
            break
        if not text or text.strip().lower() in ("exit", "quit"):
            print("Goodbye")
            break
        resp = agent.handle(text)
        print("Agent:\n", resp)


def demo_scripted(csv_folder: Optional[str] = None):
    client = MCPClient(csv_folder)
    #dl = client.query("SELECT product_family, AVG(list_price) AS average_list_price FROM saas_pricing_data3 GROUP BY product_family")
    dl = client.query("SELECT product_family, AVG(list_price) AS average_list_price FROM saas_pricing_data3 GROUP BY product_sub_group")
    
    print("Datasets:", dl)
    # for d in dl:
    #     print(f"  {d['name']} ({d['rows']} rows, {len(d['columns'])} columns)")

    #agent = CrewAgentFallback(client)
    # agent = SQLAgent()

    # samples = [
    #     "total revenue by month",
    #     "count transactions",
    #     "average list_price",
    #     "show me 10 example rows",
    # ]

    # for s in samples:
    #     print(f">>> User: {s}")
    #     print(agent.handle(s))
    #     print("---")


if __name__ == "__main__":
    # Run a small scripted demo -- change csv_folder to point at your CSVs if needed
    demo_scripted()
