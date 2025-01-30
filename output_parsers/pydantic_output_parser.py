from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pprint import pprint

# Define a Pydantic model for the joke structure
class Joke(BaseModel):
    setup: str = Field(..., description="The setup or question part of the joke")
    punchline: str = Field(..., description="The punchline or answer part of the joke")

# Create a parser based on the Pydantic model
pydantic_parser = PydanticOutputParser(pydantic_object=Joke)

# Define a prompt template instructing the model to follow the Pydantic schema
template = PromptTemplate.from_template(
    "Tell me a joke about {topic}. Respond with a JSON object that follows this schema:\n{format_instructions}"
)

# Format instructions for ensuring correct output format
format_instructions = pydantic_parser.get_format_instructions()

# Inject format instructions into the prompt
prompt = template.format(topic="programming", format_instructions=format_instructions)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Invoke the model with the prompt
response = llm.invoke(prompt).content

# Parse the output into a Pydantic model
joke = pydantic_parser.parse(response)

# Print the structured response
pprint(joke.model_dump())