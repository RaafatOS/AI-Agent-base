"""This file contians the utility functions used in the program.
    These functions are used to interact with the Hugging Face API and the Mistral API."""
import os
import json
import requests

def get_weather_data(name, longitude, latitude, date = None):
    """This function gets weather data from the weatherapi API."""
    weather_api_key = os.getenv("WEATHER_API_KEY")
    url = "https://api.weatherapi.com/v1/"
    # Check for historical data request
    if date is None:
        url += f"current.json?key={weather_api_key}"
    else:
        url += f"history.json?key={weather_api_key}&dt={date}"

    # Check for name or coordinates
    if name:
        url += f"&q={name}"
    elif longitude and latitude:
        url += f"&q={longitude},{latitude}"
    else:
        return "Error: Missing location information"
    
    response = requests.get(url)
    data = response.json()
    return data['current']
