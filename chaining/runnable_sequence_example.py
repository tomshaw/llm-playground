from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableSequence

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Step 1: Generate a project idea
idea_prompt = PromptTemplate.from_template("Generate a unique project idea related to {topic}.")
runnable_idea = idea_prompt | llm

# Step 2: Generate Python code based on the idea
code_prompt = PromptTemplate.from_template("Write Python code for this project: {idea}.")
runnable_code = code_prompt | llm

# Extract the text content from the first output and pass it as input to the second step
def extract_idea_and_generate_code(response):
    idea_text = response.content  # Extracting text content from LLM response
    return runnable_code.invoke({"idea": idea_text})  # Generating code based on the idea

# Create a sequence that first generates an idea, then uses it to generate code
sequence = RunnableSequence(runnable_idea, RunnableLambda(extract_idea_and_generate_code))

# Running the sequence
topic = "machine learning"
code_result = sequence.invoke({"topic": topic})

# Print results
print("Generated Code:\n", code_result.content)
