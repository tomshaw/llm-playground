import time
from typing import List, Any

from pydantic import BaseModel
from langchain_community.retrievers import WikipediaRetriever

class WikipediaSearchResult(BaseModel):
    title: str
    content: str

class WikipediaSearchResponse(BaseModel):
    query: str
    results: List[WikipediaSearchResult]
    response_time: float

class WikipediaSearcher:
    """
    A class to retrieve Wikipedia summaries using WikipediaRetriever.
    """
    def __init__(self, top_k: int = 3) -> None:
        """
        Initializes the WikipediaSearcher with a specified number of results.
        
        :param top_k: The number of results to retrieve.
        """
        self.retriever = WikipediaRetriever(top_k_results=top_k)
    
    def get_summaries(self, query: str) -> WikipediaSearchResponse:
        """
        Retrieves a summary of Wikipedia articles based on the query.
        
        :param query: The search term for Wikipedia.
        :return: A WikipediaSearchResponse object containing Wikipedia summaries.
        """
        start_time = time.time()
        results = self.retriever.invoke(query)
        search_results = [WikipediaSearchResult(title=doc.metadata.get("title", "Unknown"), content=doc.page_content) for doc in results]
        response_time = time.time() - start_time
        return WikipediaSearchResponse(query=query, results=search_results, response_time=response_time)

if __name__ == "__main__":
    query = input("What do you want to search for on Wikipedia? ")
    
    searcher = WikipediaSearcher()
    response = searcher.get_summaries(query)
    
    print(f"Query: {response.query}")
    print(f"Response Time: {response.response_time:.2f} seconds")
    for i, result in enumerate(response.results, 1):
        print(f"Result {i}:\nTitle: {result.title}\nContent: {result.content}\n")