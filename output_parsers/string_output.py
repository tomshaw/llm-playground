from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Define the messages to be sent to the chat model
messages = [
    SystemMessage(content="Translate the following text from English to Spanish."),
    HumanMessage(content="If you need my help, please let me know.")
]

# Instantiate a chat model with the specified model name
model = ChatOpenAI(model="gpt-4o-mini")

# Invoke the model with the defined messages and get the response
response = model.invoke(messages)

# Instantiate a string output parser to parse the model's response
output_parser = StrOutputParser()

# Parse the model's response using the output parser
parsed_response = output_parser.invoke(response)

# Print the parsed response
print(parsed_response)