from typing import TypedDict, Annotated, Any

import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import AnyMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode

class QueryState(TypedDict):
    """Represents the current state of the query workflow."""
    query: str
    messages: Annotated[list[AnyMessage], add_messages]

class IntelligentAgent:
    def __init__(self):
        """Initializes the agent with tools and workflow setup."""
        self.tools = [DuckDuckGoSearchRun(), self.get_stock_price]
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.workflow = self.build_workflow()
        
    def get_stock_price(self, ticker: str) -> float:
        """
        Retrieve the previous closing stock price for a given ticker symbol.

        Args:
            ticker (str): The ticker symbol of the stock.

        Returns:
            float: The previous closing price of the stock.
        """
        stock = yf.Ticker(ticker)
        return stock.info['previousClose']

    def build_workflow(self):
        """Constructs the LangGraph workflow with necessary nodes and edges."""
        workflow = StateGraph(QueryState)
        workflow.add_node("reasoning", self.reasoning)
        workflow.add_node("tools", ToolNode(self.tools))
        
        workflow.add_edge(START, "reasoning")
        workflow.add_edge("tools", "reasoning")
        workflow.add_conditional_edges("reasoning", tools_condition)
        
        return workflow.compile()
    
    def reasoning(self, state):
        """Processes the user query and determines if external tools are needed."""
        query = state["query"]
        messages = state["messages"]
        
        system_message = SystemMessage(
            content=(
                "You are an intelligent assistant capable of performing web searches and retrieving stock market data. "
                "Please provide accurate and helpful responses to user queries by utilizing the available tools when necessary."
            )
        )
        user_message = HumanMessage(content=query)
        
        messages.append(user_message)
        
        result = [self.llm_with_tools.invoke([system_message] + messages)]
        
        state['messages'] = result
        
        return state
    
    def run(self, query: str):
        """Runs the workflow for a given user query and prints responses."""
        response = self.workflow.invoke({"query": query, "messages": []})
        
        for message in response['messages']:
            message.pretty_print()

if __name__ == "__main__":
    agent = IntelligentAgent()
    agent.run("Who won the Best Actor award at the most recent Oscars?")
    # agent.run("How many years has it been since the fall of the Berlin Wall?")
    # agent.run("What is the current stock price of Microsoft, and how does it compare to its price one month ago?")
    # agent.run("What is the stock price of the company that Sundar Pichai is CEO of?")
    # agent.run("If the stock price of Amazon increases by 20%, what will be its new price?")