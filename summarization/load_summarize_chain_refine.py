from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

def main():
    """
    Main function to load a PDF document, split it into pages, and summarize the content using
    the 'refine' chain type from langchain.
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

    # Define the prompt template for the initial summary
    question_prompt_template = """
        Please provide a summary of the following text.
        TEXT: {text}
        SUMMARY:
    """
    question_prompt = PromptTemplate(template=question_prompt_template, input_variables=["text"])

    # Define the prompt template for refining the summary
    refine_prompt_template = """
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        BULLET POINT SUMMARY:
    """
    refine_prompt = PromptTemplate(template=refine_prompt_template, input_variables=["text"])

    # Load the summarize chain with the 'refine' chain type
    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )

    # Invoke the chain to summarize the pages
    refine_outputs = refine_chain.invoke({"input_documents": pages})
    print(refine_outputs)

if __name__ == "__main__":
    main()