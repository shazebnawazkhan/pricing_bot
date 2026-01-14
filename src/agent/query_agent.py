from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool
from typing import List
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Model for a single row/item — allow arbitrary columns
class ResRow(BaseModel):
    # keep a flexible model that can accept any keys returned by SQL queries
    class Config:
        extra = "allow"

# Model for the entire table (a list of rows with arbitrary schema)
class ResTable(BaseModel):
    products: List[Dict[str, Any]] = Field(description="A list of rows where each item is a dict mapping column names to values.")


class QueryCrew:
    """Build a simple Crew that runs a single SQL agent task.
    
        2. You must also provide all the SQL queries you run using the tool in the same chronological 
                order as you run them.

                
    
    """

    def __init__(self):
        # LLM and tools

        #self.llm = LLM(model="ollama/deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.9)

        self.llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.7, verbose=True)

        self.stdio_params=StdioServerParameters(
            command="python", 
            args=["D:\\SNK\\codes\\repos\\pricing_bot\\src\\mcp\\file_server.py"]
        )

        self.mcp_server_adapter = None # MCPServerAdapter(server_params=self.stdio_params)

        self.mcp_server_adapter = MCPServerAdapter(self.stdio_params)
        #self.mcp_server_adapter.start()
        self.tools = getattr(self.mcp_server_adapter, "tools", [])
        print(f"Available tools (manual Streamable HTTP): {[getattr(t, 'name', str(t)) for t in self.tools]}")


        # Create Agent instance directly (no undefined config usage)
        self.sql_agent = Agent(
            role="Respond to user queries based on tools and datasets available in MCP server",
            goal="Leverage the MCP server to answer user queries using SQL queries on datasets",
            verbose=True,
            # tools=[self.scrap_tool],
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
            llm=self.llm
        )

        # Create a Task that references the input by placeholder (provided at kickoff time)
        self.sql_task = Task(
            description=(f"""
                The Model Context Protocol server running at http://localhost/filemcp offers many interface to query datasets.
                The datasets can be considered as tables in a sqlite database.
                It offers tools like list datasets, describe dataset, summary of dataset and also run SQL queries on the datasets.
                The task is to use the MCP server to get information about the datasets available using the tools provided by the MCP server. 
                One of the tools offered by the MCP server is query which allows to provide custom SQL as parameter named 'sql' which can be run on the dataset. 
                Create relevant SQL queries and pass into the tool, and obtain the results from the tool to share as response to the chat.
                ALso use other tools to leverage the data present in the datasets, to answer the questions or execute
                the instructions given by the user. Also run the queries you create using the tools to get results 
                from the datasets, and return the output data in your final answer.
                The instruction from user is provided in input: {input_query}.
            """),
            expected_output=(f"""
                I need responses to the queries posted by the user in the chat conversations. 
                You can create SQL queries which can be run using the available tools from the MCP server. 
                You must create the SQL queries based on the schema of the tables available in datasets available in MCP server. 
                You can leverage the tools available to fetch schema and column names of the datasets in order to understand the 
                columns present in certain table.

                I need the data response fetched from the tool to be furnished as answers to the questions of the user.
                The data reseponse must be presented in tabular format with exact column names as provided by the tool.
                The data must be accurate and precise as per the queries run on the datasets. 
                Make sure to use the correct Markdown formatting for the data tables.
                The questions or instructions from the user are available in the input: {input_query}.
                The questions can request informations like below:
                1. Data descriptions like columns and their data types in a particular table.
                2. Select some rows of data from a table
                3. Select some columns of data from a table
                4. Apply filters on rows based on some columns
                5. Apply 'group by' based summarization on some columns like calculating 'sum', 'mean', 'percentage of total' etc.
                6. Or a combination of above

                You can run the queries using the available 'query' tool from MCP server. The tool takes an SQL query as argument to run the SQL queries and 
                obtain the results as a pandas dataframe. Return the results obtained from the queries as part of your final answer to the user.

                Some important points to note while creating SQL queries are:
                1. Always use the full qualified name of tables to avoid errors in the queries.
                2. Do not update any rows in the existing data.

                You should first find out the correct table (dataset) to be used in your query.
                You can use nested queries in order to accomplish complex tasks. You can create new table by running appropriate SQL queries,
                and also insert data into tables from another table.

                Review the final answer thoroughly before presenting. Prefer to produce the results using minimum number of queries.
                Optimize the queries well before running them and presenting the final answer.

                The final answer you give must contain below items: 
                1. The output from the queries you ran. The output must be presented as tabular data with exact column 
                names as provided by the tool. Return the tabular result as per the pydantic Model: ResTable. Do not hallucinate any data.
                
            """),
            #tools=[self.scrap_tool],
            tools=self.tools,
            agent=self.sql_agent,
            output_pydantic=ResTable,
            markdown=True,
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[self.sql_agent],
            tasks=[self.sql_task],
            process=Process.sequential,
            verbose=True,
        )
    
    def handle(self, input_query: str) -> str:  
        crew = self.crew()
        results = crew.kickoff(inputs={"input_query": input_query})
        print("results.pydantic:", results.pydantic)
    
        markdown_table = ""

        if results.pydantic:
            # Generic formatting: determine columns from the first row and render a markdown table
            products = results.pydantic.products
            if not products:
                markdown_table = "No rows returned"
            else:
                # If rows are Pydantic models, convert them to dicts
                first = products[0]
                if isinstance(first, BaseModel):
                    first = first.dict()
                # Use the keys of the first row as the table columns (preserves order if available)
                columns = list(first.keys())

                # Build header
                markdown_table = "| " + " | ".join(columns) + " |\n"
                markdown_table += "|" + "|".join(["---"] * len(columns)) + "|\n"

                # Build rows
                for row in products:
                    if isinstance(row, BaseModel):
                        row = row.dict()
                    values = [str(row.get(c, "")) for c in columns]
                    markdown_table += "| " + " | ".join(values) + " |\n"

            print(markdown_table)
        else:
            # If output_pydantic wasn't used or failed validation, access raw output
            print("Failed validation, printing raw results:")
            print(results.raw)
            markdown_table = str(results.raw)
        return markdown_table
        


if __name__ == "__main__":
    # Example use: provide inputs at kickoff time via the inputs dict
    input_query = "show me average list_price by product_family and product_group from saas_pricing_data3"
    #input_query = "SELECT product_category, AVG(list_price) AS average_list_price FROM saas_pricing_data3 GROUP BY product_category"
    qc = QueryCrew()
    # crew = qc.crew()
    # results = crew.kickoff(inputs={"input_query": input_query})
    results = qc.handle(input_query)
    print(results)
    print("Exiting...")
