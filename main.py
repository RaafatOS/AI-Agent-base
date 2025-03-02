"""implementation of the software agent"""
import os
import sys
import time
import logging
import requests
import json
from structures import Chatbot, MistralAgent, bookStructure, weatherStructure
from utils import get_weather_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
def main():
    """Main function"""

    # model_name = "facebook/opt-125m"  # Smaller model for CPU efficiency
    # chat = Chatbot(model_name)

    # # print(chat.generate_response("Hello, how are you?"))
    # print(chat.generate_response_with_structure("what is the weather like in New York?"))

    chat = MistralAgent()
    chat_response = chat.generate_weather("what is the temperature in Paris", weatherStructure)
    print(chat_response)
    # print(get_weather_data("New York", None, None))

main()