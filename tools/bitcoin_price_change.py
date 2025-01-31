from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize Tavily search tool
tavily_search = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True
)

@tool
def calculate_percentage_change(yesterday_price: float, today_price: float) -> str:
    """
    Calculates the percentage change between two values and determines if it is an increase or decrease.
    
    Args:
        yesterday_price (float): The price of Bitcoin yesterday.
        today_price (float): The price of Bitcoin today.
    
    Returns:
        str: A message indicating the percentage change and whether it is an increase or decrease.
    """
    if yesterday_price <= 0:
        return "Invalid yesterday price. It must be greater than zero."
    
    change = today_price - yesterday_price
    percent_change = (change / yesterday_price) * 100
    direction = "increased" if change > 0 else "decreased"
    return f"The price has {direction} by {abs(percent_change):.2f}%."

# List of tools to be used by the agent
tools = [tavily_search, calculate_percentage_change]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial assistant that retrieves Bitcoin prices and calculates percentage changes."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent with the specified tools and prompt
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Define the question to be asked to the agent
question = "Find yesterday's and today's Bitcoin prices in USD. Calculate whether the price has increased or decreased and by what percentage."

# Invoke the agent executor with the question
response = agent_executor.invoke({"input": question})

# Print the response
print(response)