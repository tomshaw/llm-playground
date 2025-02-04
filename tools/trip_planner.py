import os
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Initialize Tavily search tool
tavily_search = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True
)

@tool
def get_weather_forecast(city: str, number_days: int = 5) -> str:
    """Fetch the 5-day weather forecast for a given city."""
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": OPENWEATHERMAP_API_KEY, "units": "metric", "cnt": number_days}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        forecast_summary = [
            f"Date: {item['dt_txt']}, Weather: {item['weather'][0]['description']}, Temp: {item['main']['temp']}°C"
            for item in data['list']
        ]
        return "\n".join(forecast_summary)
    return "Failed to fetch weather data."

@tool
def get_exchange_rate(target_currency: str) -> str:
    """Retrieve the exchange rate from USD to the given currency."""
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
    data = response.json()
    return f"Exchange rate from USD to {target_currency}: {data['rates'].get(target_currency, 0.0)}"

@tool
def get_latest_news(city: str) -> str:
    """Fetch the latest news for a city using Tavily."""
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(f"latest news in {city}")
    if "results" not in response:
        return "Failed to fetch news."
    return "\n".join([f"{item['title']}: {item['url']}" for item in response["results"]])

@tool
def translate_common_phrases(language: str) -> str:
    """Translate common travel phrases into the given language."""
    common_phrases = ["Hello", "Thank you", "Where is the restroom?", "How much does this cost?", "Goodbye"]
    translations = []
    for phrase in common_phrases:
        response = requests.get(f"https://api.mymemory.translated.net/get?q={phrase}&langpair=en|{language}")
        translations.append(f"{phrase} -> {response.json()['responseData']['translatedText']}")
    return "\n".join(translations)

# Define tools
tools = [tavily_search, get_weather_forecast, get_exchange_rate, get_latest_news, translate_common_phrases]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Get user input for city
city = input("Where are you planning to travel to? ")

# Define agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     f"""
     ### **Travel Planner:**  
     I am traveling to **{city}** and need a comprehensive travel report. Please gather and summarize the following details:  

     #### **1. Local Information:**  
     - What is the official language spoken in **{city}**?  
     - What is the local currency used in **{city}**?  

     #### **2. 5-Day Weather Forecast:**  
     - Provide a **detailed 5-day forecast** for **{city}**, including **temperature, precipitation, humidity, and general weather conditions**.  
     - Summarize the expected weather trends for each day (e.g., "Expect a warm and sunny day with mild breezes").  

     #### **3. Exchange Rate:**  
     - Retrieve the current exchange rate from **USD to the local currency**.  

     #### **4. Latest News:**  
     - Summarize the **most recent and relevant news articles** related to **{city}**.  

     #### **5. Common Phrase Translations:**  
     - Provide **translations of essential travel phrases** into the **local language** of **{city}**.  

     #### **Tools for Data Retrieval:**  
     Use the following tools to gather accurate and up-to-date information:  
     - **get_weather_forecast** - Fetch the 5-day weather forecast for **{city}**.  
     - **get_exchange_rate** - Retrieve the exchange rate from **USD to the local currency**.  
     - **get_latest_news** - Fetch and summarize the latest news articles about **{city}**.  
     - **translate_common_phrases** - Translate common travel phrases into the **local language**.  

     #### **Output Formatting:**  
     - Present the information in a well-structured travel report.  
     - Ensure the weather section includes **daily summaries** for quick reference.  
     - Make the report **engaging, concise, and informative**.
     - Do not mention the tools used in the final report.  
     """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute agent
response = agent_executor.invoke({"input": prompt})
print(response['output'])