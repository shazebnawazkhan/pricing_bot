"""MCP-like file server for CSV datasets.

Provides endpoints to list datasets, get descriptions and summaries,
and run read-only SQL queries against the loaded CSV files.

Usage:
    python -m src.mcp.file_server

Default CSV folder: ../datagen (relative to this file)
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


CSV_FOLDER_ENV = "CSV_FOLDER"
DEFAULT_CSV_FOLDER = Path(__file__).resolve().parents[1] / "data"
VALID_SQL_RE = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
MAX_QUERY_ROWS = 1000


class QueryRequest(BaseModel):
    sql: str
    limit: Optional[int] = None


class QueryResult(BaseModel):
    columns: List[str]
    rows: List[List[Any]]


class FileServer:
    def __init__(self, csv_folder: Optional[Path] = None):
        self.csv_folder = Path(csv_folder or os.getenv(CSV_FOLDER_ENV) or DEFAULT_CSV_FOLDER)
        #print(self.csv_folder)
        self._conn: Optional[sqlite3.Connection] = None
        self._dfs: Dict[str, pd.DataFrame] = {}
        self.reload()

    def reload(self) -> None:
        """Load all CSV files from the csv_folder into an in-memory sqlite DB and keep DataFrames."""
        if not self.csv_folder.exists():
            raise FileNotFoundError(f"CSV folder not found: {self.csv_folder}")

        # create in-memory sqlite DB
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA case_sensitive_like = OFF")

        csv_files = sorted(self.csv_folder.glob("*.csv"))
        self._dfs = {}

        for csv_path in csv_files:
            table_name = csv_path.stem
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                # skip unreadable CSVs
                continue

            # normalize column names to avoid SQL issues
            df.columns = [str(c) for c in df.columns]
            # store DataFrame
            self._dfs[table_name] = df
            # write to sqlite
            try:
                df.to_sql(table_name, self._conn, if_exists="replace", index=False)
            except Exception:
                # fallback: coerce object columns to text
                for c in df.select_dtypes(["object"]).columns:
                    df[c] = df[c].astype(str)
                df.to_sql(table_name, self._conn, if_exists="replace", index=False)

    def list_datasets(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "rows": int(df.shape[0]),
                "columns": list(df.columns),
            }
            for name, df in self._dfs.items()
        ]

    def describe(self, name: str) -> Dict[str, Any]:
        if name not in self._dfs:
            raise KeyError(name)
        df = self._dfs[name]
        desc = df.describe(include="all", datetime_is_numeric=True).to_dict()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        sample = df.head(5).to_dict(orient="records")
        return {"name": name, "rows": int(df.shape[0]), "columns": list(df.columns), "dtypes": dtypes, "missing": missing, "describe": desc, "sample": sample}

    def summary(self, name: str) -> Dict[str, Any]:
        # lighter-weight summary
        if name not in self._dfs:
            raise KeyError(name)
        df = self._dfs[name]
        numeric = df.select_dtypes(include=["number"]).describe().to_dict()
        categorical = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            top = df[col].value_counts(dropna=False).head(5).to_dict()
            categorical[col] = top
        missing = {col: int(df[col].isna().sum()) for col in df.columns}
        return {"name": name, "rows": int(df.shape[0]), "columns": list(df.columns), "numeric": numeric, "categorical_top": categorical, "missing": missing}

    def run_query(self, sql: str, limit: Optional[int] = None) -> QueryResult:
        if not VALID_SQL_RE.match(sql):
            raise ValueError("Only read-only SELECT/WITH queries are allowed")

        # enforce a safe limit
        effective_limit = min(limit or MAX_QUERY_ROWS, MAX_QUERY_ROWS)

        # naive approach: wrap query to apply a limit if none exists
        # check if there's a top-level LIMIT clause
        if re.search(r"\blimit\b", sql, re.IGNORECASE) is None:
            sql = f"SELECT * FROM ({sql}) LIMIT {effective_limit}"

        try:
            df = pd.read_sql_query(sql, self._conn)
        except Exception as e:
            raise

        rows = df.fillna(value=0).values.tolist()
        return QueryResult(columns=list(df.columns), rows=rows)


# MCP server integration (using mcp.server.fastmcp)
# server instance (module-level)
_server: Optional[FileServer] = None


def get_server() -> FileServer:
    global _server
    if _server is None:
        _server = FileServer()
    return _server


# Try to create an MCP server using mcp.server.fastmcp. If not available, raise a helpful error
try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except Exception:  # pragma: no cover - import-time fallback
    FastMCP = None


def _register_tools_with_fastmcp(mcp: "FastMCP") -> None:
    """Register tools on a FastMCP instance.

    This function attempts different registration APIs to be compatible with
    multiple versions of `mcp.server.fastmcp`.
    """

    # tool: list_datasets
    def list_datasets_tool(params: Optional[dict] = None) -> dict:
        return {"datasets": get_server().list_datasets()}

    # tool: describe
    def describe_tool(params: dict) -> dict:
        name = params.get("name")
        if not name:
            return {"error": "missing parameter 'name'"}
        try:
            return get_server().describe(name)
        except KeyError:
            return {"error": f"Dataset not found: {name}"}

    # tool: summary
    def summary_tool(params: dict) -> dict:
        name = params.get("name")
        if not name:
            return {"error": "missing parameter 'name'"}
        try:
            return get_server().summary(name)
        except KeyError:
            return {"error": f"Dataset not found: {name}"}

    # tool: query
    def query_tool(params: dict) -> dict:
        sql = params.get("sql")
        limit = params.get("limit")
        if not sql:
            return {"error": "missing parameter 'sql'"}
        try:
            qr = get_server().run_query(sql, limit=limit)
            return {"columns": qr.columns, "rows": qr.rows}
        except ValueError as ve:
            return {"error": str(ve)}
        except Exception as e:
            return {"error": f"Query error: {e}"}

    # tool: reload
    def reload_tool(params: Optional[dict] = None) -> dict:
        try:
            get_server().reload()
            return {"status": "ok", "datasets": get_server().list_datasets()}
        except Exception as e:
            return {"error": str(e)}

    # Register using decorator-style API if available
    if hasattr(mcp, "tool"):
        # decorator API
        mcp.tool("list_datasets")(list_datasets_tool)
        mcp.tool("describe")(describe_tool)
        mcp.tool("summary")(summary_tool)
        mcp.tool("query")(query_tool)
        mcp.tool("reload")(reload_tool)
        return

    # Register using register_tool/add_tool style
    if hasattr(mcp, "register_tool"):
        mcp.register_tool("list_datasets", list_datasets_tool, description="List datasets")
        mcp.register_tool("describe", describe_tool, description="Describe a dataset")
        mcp.register_tool("summary", summary_tool, description="Summary for a dataset")
        mcp.register_tool("query", query_tool, description="Run a read-only SQL query")
        mcp.register_tool("reload", reload_tool, description="Reload CSV files from disk")
        return

    if hasattr(mcp, "add_tool"):
        mcp.add_tool("list_datasets", list_datasets_tool)
        mcp.add_tool("describe", describe_tool)
        mcp.add_tool("summary", summary_tool)
        mcp.add_tool("query", query_tool)
        mcp.add_tool("reload", reload_tool)
        return

    # Unknown registration API
    raise RuntimeError("Unable to register tools on FastMCP: incompatible API")


def create_mcp_server(name: str = "csv-file-server"):
    if FastMCP is None:
        raise ImportError(
            "mcp.server.fastmcp is not installed. Install it (e.g. `pip install mcp`) and ensure the package exposes `mcp.server.fastmcp.FastMCPServer`."
        )

    mcp = FastMCP(name=name)
    _register_tools_with_fastmcp(mcp)
    return mcp


# Provide a simple CLI to start the MCP server
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", os.getenv("MCP_PORT", 0)))
    if FastMCP is None:
        raise RuntimeError("mcp.server.fastmcp is required to run as MCP server. Install it and try again.")

    mcp = create_mcp_server()
    # prefer explicit run/serve methods if available
    if hasattr(mcp, "serve"):
        #mcp.serve(port=port or None)
        mcp.serve(transport="stdio")
        #mcp.serve(transport="streamable-http", mount_path="/filemcp")
    elif hasattr(mcp, "run"):
        #mcp.run(port=port or None)
        mcp.run(transport="stdio")
        #mcp.run(transport="streamable-http", mount_path="/filemcp")
    else:
        raise RuntimeError("FastMCP has no 'serve' or 'run' method to start the server")
