from typing import TypedDict, Annotated, Any

import yfinance as yf
from langchain_community.retrievers import WikipediaRetriever
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
        self.duckduckgo_search_tool = DuckDuckGoSearchRun()
        self.wikipedia_retriever = WikipediaRetriever(load_all_available_meta=False, top_k_results=1)
        self.tools = [
            self.duckduckgo_search,
            self.wikipedia_search,
            self.get_stock_price
        ]
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.workflow = self.build_workflow()
        
    def duckduckgo_search(self, query: str) -> Any:
        """
        Perform a search using DuckDuckGoSearchRun.

        Args:
            query (str): The search query.

        Returns:
            Any: The search results.
        """
        return self.duckduckgo_search_tool.invoke(query)
        
    def wikipedia_search(self, query: str) -> Any:
        """
        Perform a search using WikipediaRetriever.

        Args:
            query (str): The search query.

        Returns:
            Any: The search results.
        """
        return self.wikipedia_retriever.invoke(query)
        
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
                "You are an intelligent assistant capable of performing web searches, retrieving stock market data, and searching Wikipedia. "
                "When a user query is received, determine the most appropriate tool to use based on the nature of the query. "
                "If the query involves retrieving information from the web, use the DuckDuckGo search tool. "
                "If the query involves retrieving stock market data, use the stock price retrieval tool. "
                "If the query involves retrieving information from Wikipedia, use the Wikipedia search tool. "
                "If a tool fails to provide the necessary information, respond with an appropriate message indicating the limitation. "
                "Always provide accurate and helpful responses to user queries by utilizing the available tools when necessary."
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
    agent.run("Tell me some historical facts about Paris including latest news and current weather conditions.")