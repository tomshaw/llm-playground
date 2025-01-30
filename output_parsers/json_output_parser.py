from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from pprint import pprint

# Define a JSON-compatible prompt for a technical explanation
template = PromptTemplate.from_template(
    "Explain the concept of {topic} in the context of artificial intelligence. Format the response as a JSON object with 'definition', 'applications', and 'challenges' keys."
)

# Instantiate a chat model with the specified model name
llm = ChatOpenAI(model="gpt-4o-mini")

# JSON output parser to parse the response into a JSON object
json_parser = JsonOutputParser()

# Use OutputFixingParser to handle minor formatting issues in the JSON response
fixed_parser = OutputFixingParser.from_llm(parser=json_parser, llm=llm)

# Create a chain of the prompt template, the chat model, and the fixed JSON output parser
chain = template | llm | fixed_parser

# Invoke the chain with a specific topic and get the result
result = chain.invoke({"topic": "neural networks"})

# Pretty-print the result
pprint(result)