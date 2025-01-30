from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

def main():
    # Prompt the user for the URL to load documents from
    url = input("Enter the URL to load documents from: ")

    # Load documents from the specified URL
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Instantiate the chat model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\n\n{context}")]
    )

    # Create the chain to combine documents and generate the summary
    chain = create_stuff_documents_chain(llm, prompt)

    # Invoke the chain with the loaded documents
    result = chain.invoke({"context": docs})

    # Print the result
    print(result)

if __name__ == "__main__":
    main()