import os
from datetime import datetime, timedelta

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
def get_weather_forecast(city: str, days_ahead: int = 0) -> str:
    """
    This function uses the OpenWeatherMap API to get the current weather for a given location.
    """ 

    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    if not api_key:
        raise Exception("OPENWEATHERMAP_API_KEY environment variable not set.")

    if days_ahead > 0:
        target_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    else:
        target_date = None

    base_url = "http://api.openweathermap.org/data/2.5/weather" 

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if target_date:
            return f"The weather in {data['name']}, {data['sys']['country']} on {target_date} is currently {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C. The humidity is {data['main']['humidity']}% and the wind speed is {data['wind']['speed']} m/s."
        else:
            return f"The weather in {data['name']}, {data['sys']['country']} is currently {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C. The humidity is {data['main']['humidity']}% and the wind speed is {data['wind']['speed']} m/s."
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
I am traveling to {city}. Please provide a detailed travel report including the following information:
1. The local currency and official language of {city}.
2. A 5-day weather forecast for {city}.
3. The exchange rate from USD to the local currency.
4. The latest news articles related to {city}.
5. Translations of common phrases into the local language.

Use the following tools to gather the information:
- get_weather_forecast: Fetch the current weather for a given location.
- get_exchange_rate: Retrieve the exchange rate from USD to the given currency.
- get_latest_news: Fetch the latest news articles for a given location.
- translate_common_phrases: Translate common phrases into the local language.

After gathering the information, format the output as a detailed travel report.
"""

# Ollama query
response = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": prompt}],
    tools=tools
)

print(response['message']['content'])