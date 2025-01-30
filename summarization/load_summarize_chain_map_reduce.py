from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

def main():
    """
    Main function to load a PDF document, split it into pages, and summarize the content using
    the 'map_reduce' chain type from langchain.
    """
    # Initialize the language model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Path to the PDF file
    file_path = "../data/practitioners_guide_to_mlops_whitepaper.pdf"

    # Load and split the PDF document into pages
    loader = PyPDFLoader(file_path)
    split_pages = loader.load_and_split()

    # Select the first three pages for summarization
    pages = split_pages[:3]
    print(pages)

    # Define the prompt template for the map step
    map_prompt_template = """
        Write a summary of this chunk of text that includes the main points and any important details.
        {text}
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # Define the prompt template for the combine step
    combine_prompt_template = """
        Write a summary of the entire document that includes the main points from all of the individual summaries.
        {text}
    """
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    # Load the summarize chain with the 'map_reduce' chain type
    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )

    # Invoke the chain to summarize the pages
    map_reduce_outputs = map_reduce_chain.invoke({"input_documents": pages})
    print(map_reduce_outputs)

if __name__ == "__main__":
    main()