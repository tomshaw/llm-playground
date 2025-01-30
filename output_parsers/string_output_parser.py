from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define a technical prompt template
template = PromptTemplate.from_template(
    "Explain the concept of {topic} in one short paragraph."
)

# Instantiate a chat model with the specified model name
llm = ChatOpenAI(model="gpt-4o-mini")

# Create a chain of the prompt template, the chat model, and the string output parser
chain = template | llm | StrOutputParser()

# Invoke the chain with a specific topic and get the result
result = chain.invoke({"topic": "quantum entanglement"})

# Print the result
print(result)