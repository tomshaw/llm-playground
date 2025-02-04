import os

import requests
from dotenv import load_dotenv
from tavily import TavilyClient

import ollama

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

# Common phrases to translate
COMMON_PHRASES = ["Hello", "Thank you", "Where is the restroom?", "How much does this cost?", "Goodbye"]
    
# Function to get 5-day weather forecast
def get_weather_forecast(city: str, number_days: int = 0) -> str:
    """
    This function uses the OpenWeatherMap API to get the 5-day weather forecast for a given location.
    """ 

    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise Exception("OPENWEATHERMAP_API_KEY environment variable not set.")

    base_url = "http://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
        "cnt": number_days
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        forecast_list = data['list']
        forecast_summary = []

        for forecast in forecast_list:
            date = forecast['dt_txt']
            description = forecast['weather'][0]['description']
            temp = forecast['main']['temp']
            humidity = forecast['main']['humidity']
            wind_speed = forecast['wind']['speed']
            forecast_summary.append(f"Date: {date}, Weather: {description}, Temp: {temp}°C, Humidity: {humidity}%, Wind Speed: {wind_speed} m/s")

        return "\n".join(forecast_summary)
    else:
        raise Exception(f"Error fetching weather data: {response.status_code}")

# Function to get exchange rate
def get_exchange_rate(target_currency: str) -> str:
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
    data = response.json()
    rate = data['rates'].get(target_currency, 0.0)
    return f"The exchange rate from USD to {target_currency} is {rate}."

# Function to get latest news from Tavily
def get_latest_news(city: str) -> str:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(f"latest news in {city}")
    
    if "results" not in response:
        return "Failed to fetch news from Tavily."
    
    news = "\n".join([f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}\n" for result in response["results"]])
    return f"Latest news in {city}:\n{news}"

# Function to translate common phrases
def translate_common_phrases(language: str) -> str:
    translations = []
    for phrase in COMMON_PHRASES:
        response = requests.get(f"https://api.mymemory.translated.net/get?q={phrase}&langpair=en|{language}")
        data = response.json()
        translations.append(f"{phrase} -> {data['responseData']['translatedText']}")
    return "\n".join(translations)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Fetch the current weather for a given location.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days_ahead": {"type": "integer"}}, "required": ["city"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Retrieve the exchange rate from USD to the given currency.",
            "parameters": {"type": "object", "properties": {"target_currency": {"type": "string"}}, "required": ["target_currency"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_news",
            "description": "Fetch the latest news articles for a given location using Tavily.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "translate_common_phrases",
            "description": "Translate common phrases into the local language.",
            "parameters": {"type": "object", "properties": {"language": {"type": "string"}}, "required": ["language"]}
        }
    }
]

city = "Berlin"
language = "de"

print("Testing get_weather_forecast:")
print(get_weather_forecast(city))

print("\nTesting get_exchange_rate:")
print(get_exchange_rate("EUR"))

print("\nTesting get_latest_news:")
print(get_latest_news(city))

print("\nTesting translate_common_phrases:")
print(translate_common_phrases(language))

# Get user input
city = input("Where are you planning to take a trip to? ")

# Detailed prompt for Ollama query
prompt = f"""
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
"""

# Ollama query
response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}],
    tools=tools
)

print(response['message']['content'])