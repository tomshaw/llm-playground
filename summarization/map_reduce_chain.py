import os
from langchain import hub
from langchain.chains import MapReduceChain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tavily import TavilyClient

def main():
    """Main function to fetch web search results, process them, and summarize using an LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Retrieve API key from environment variables
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Error: TAVILY_API_KEY environment variable not set.")
        return

    # Prompt user for a search term
    search_term = input("Enter the search term: ")

    # Fetch search results using Tavily API
    tavily_client = TavilyClient(api_key=api_key)
    response = tavily_client.search(search_term)

    # Extract content from search results
    docs = [result['content'] for result in response['results']]
    print(docs)

    # Load LangChain map prompt
    map_prompt = hub.pull("rlm/map-prompt")

    # Initialize text splitter for document processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Convert documents to a single string for processing
    combined_text = "\n".join(docs)

    # Create and execute MapReduceChain for summarization
    map_reduce_chain = MapReduceChain.from_params(
        llm=llm,
        prompt=map_prompt,
        text_splitter=text_splitter
    )
    result = map_reduce_chain.invoke(combined_text)

    print(result["output_text"])

if __name__ == "__main__":
    main()