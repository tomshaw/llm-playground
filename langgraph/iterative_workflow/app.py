import os
import sys
import logging
from typing import TypedDict, List, Dict, Annotated, Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from tavily import TavilyClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TavilySearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    raw_content: Any

class TavilySearchResponse(BaseModel):
    query: str
    follow_up_questions: Any
    answer: Any
    images: List[Any]
    results: List[TavilySearchResult]
    response_time: float
    
    def get_data(self) -> List[dict]:
        """Returns a list of dictionaries with title, url, and content from results."""
        return [{"title": result.title, "url": result.url, "content": result.content} for result in self.results]
    
class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: List[dict]
    summaries: Annotated[List[dict], Field(description="The summaries created by the summariser")]
    approved: Annotated[bool, Field(description="Indicates if the summary has been marked as approved")]
    iteration: int
    
class SummariserOutput(BaseModel):
    message: str = Field(description="Status message about the summarization process.")
    summary: str = Field(description="A concise summary of the input text.")
    
class ReviewerOutput(BaseModel):
    message: str = Field(description="Review feedback message.")
    approved: bool = Field(description="Indicates if the summary is approved.")

class Agent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.summariser_chain = None
        self.reviewer_chain = None
        self.max_iterations = 25
        self.setup_chains()
        
    def tavily_keyword_search(self, search_term: str) -> TavilySearchResponse:
        """Search Tavily for articles based on a keyword search term."""
        client = TavilyClient(api_key=self.api_key)
        response = client.search(search_term)
        return TavilySearchResponse(**response)
    
    def setup_chains(self):
        """Setup the summarizer and reviewer chains."""
        summarizer_template = ChatPromptTemplate.from_messages([
            ("system", 
            "You are an expert summarizer. Summarize the provided articles clearly, accurately, and concisely. "
            "Include key points and direct links to the original sources. Ensure the summary is well-structured and readable.\n\n"
            "Articles to summarize: {articles}"
            ),
            ("placeholder", "{messages}"),
        ])

        reviewer_template = ChatPromptTemplate.from_messages([
            ("system", 
            "You are an expert reviewer. Assess the summary for accuracy, clarity, and completeness. "
            "Provide constructive feedback on any needed improvements. If the summary is satisfactory, approve it."
            ),
            ("placeholder", "{messages}"),
        ])

        self.summariser_chain = summarizer_template | self.llm.with_structured_output(SummariserOutput)
        self.reviewer_chain = reviewer_template | self.llm.with_structured_output(ReviewerOutput)

    def summarizer(self, state: State) -> Dict:
        """ Summarizes the search results. """
        summarizer_output = self.summariser_chain.invoke({
            "messages": state["messages"],
            "articles": state["search_results"]
        })
        
        new_messages = [
            AIMessage(content=summarizer_output.summary),
            AIMessage(content=summarizer_output.message)
        ]
        
        state["messages"].extend(new_messages)
        state["summaries"] = [summarizer_output.summary]
        state["iteration"] += 1
        
        return state
    
    def reviewer(self, state: State) -> Dict:
        """ Reviews the summary and provides feedback. """       
        reviewer_output = self.reviewer_chain.invoke({
            "messages": state["messages"]
        })
        
        new_messages = [
            AIMessage(content=reviewer_output.message)
        ]
        
        state["messages"].extend(new_messages)
        state["approved"] = reviewer_output.approved
        
        return state
        
    def conditional_edge(self, state: State):
        if state["approved"]:
            return "final_step"
        elif state["iteration"] < self.max_iterations:
            return "summarizer"
        else:
            return "final_step"
    
    def final_step(self, state: State) -> Dict:
        state["messages"].append(AIMessage(content="**Workflow Completed**"))
        return state
    
    def calculate_token_usage(self, prompt: str) -> int:
        return self.llm.get_num_tokens(prompt)
        
    def run(self):
        search_topic = input("Enter search criteria: ")
        search_topic = f"Give me the latest news on the following subject: {search_topic}"
        search_results = self.tavily_keyword_search(search_topic)
        
        config = RunnableConfig(recursion_limit=self.max_iterations)
        
        # Define LangGraph Workflow
        workflow = StateGraph(State)
        workflow.add_node("summarizer", self.summarizer)
        workflow.add_node("reviewer", self.reviewer)
        workflow.add_node("final_step", self.final_step)

        # Define Workflow Edges
        workflow.add_edge(START, "summarizer")
        workflow.add_edge("summarizer", "reviewer")

        # Conditional Routing to Avoid Infinite Loops
        workflow.add_conditional_edges('reviewer', self.conditional_edge)
        workflow.add_edge("final_step", END)

        graph = workflow.compile()

        initial_state = {
            "messages": [],
            "search_results": search_results.get_data(),
            "summaries": [],
            "approved": False,
            "iteration": 0
        }

        output = graph.invoke(initial_state, config)
        print(output["summaries"][-1])
        
        prompt = " ".join([msg.content for msg in output["messages"]])
        num_tokens = self.calculate_token_usage(prompt)
        print(f"Our prompt has {num_tokens} tokens")

if __name__ == "__main__":
    required_env_vars = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY"
    ]

    missing_environment_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_environment_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_environment_vars)}")
        sys.exit(1)

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    agent = Agent(api_key=tavily_api_key)
    agent.run()