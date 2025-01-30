from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

def main():
    """
    Main function to load a PDF document, split it into pages, and summarize the content using
    the 'stuff' chain type from langchain.
    """
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Path to the PDF file
    file_path = "../data/practitioners_guide_to_mlops_whitepaper.pdf"
    
    # Load and split the PDF document into pages
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    
    # Select the first three pages for summarization
    three_pages = pages[:3]
    print(three_pages)
    
    # Define the prompt template for summarization
    prompt_template = """
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
     
    # Load the summarize chain with the 'stuff' chain type
    stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    
    # Invoke the chain to summarize the pages
    result = stuff_chain.invoke(pages)
    print(result)
     
if __name__ == "__main__":
    main()