from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

# Connect to MySQL database
mysql_uri = 'mysql+mysqlconnector://root:password@localhost:3306/chinook'
db = SQLDatabase.from_uri(mysql_uri)

@tool
def get_schema():
    """Returns the database schema."""
    return db.get_table_info()

@tool
def run_query(query: str):
    """Runs an SQL query and returns the result."""
    return db.run(query)

# List of tools available to the agent
tools = [get_schema, run_query]

# Define the language model with specific parameters
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant designed to help users interact with a SQL database. "
     "You have access to the following tools:\n"
     "1. `get_schema` - Retrieves the database schema.\n"
     "2. `run_query` - Executes an SQL query and returns the results.\n\n"
     "When answering user queries:\n"
     "- First, call `get_schema` to understand the database structure.\n"
     "- Based on the schema, generate an appropriate SQL query.\n"
     "- Use `run_query` to execute the generated SQL query.\n"
     "- Provide a natural language response based on the results.\n\n"
     "Ensure your responses are clear and concise. If a query cannot be answered, explain why."
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent with tool-calling ability
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor to handle the execution of the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run an example query to demonstrate functionality
user_question = "How many albums are there in the database?"
response = agent_executor.invoke({"input": user_question})
print(response)