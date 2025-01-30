from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define prompts
idea_prompt = PromptTemplate.from_template("Generate a unique project idea related to {topic}.")
code_prompt = PromptTemplate.from_template("Write Python code for this project: {idea}.")

# Chain the prompts using the pipe (`|`) operator
idea_chain = idea_prompt | llm
code_chain = code_prompt | llm

# Running the first chain (Generate idea)
topic = "machine learning"
idea_result = idea_chain.invoke({"topic": topic})

# Running the second chain (Generate code using the idea)
code_result = code_chain.invoke({"idea": idea_result.content})

# Print results
print("Generated Idea:", idea_result.content)
print("\nGenerated Code:\n", code_result.content)
