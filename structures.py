"""This file contains the classes for the data structures used in the program.
    those classes define the struture to be returned by the LLM and passed to the
    LLM for the next iteration."""

import json
import os
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from mistralai import Mistral
import dotenv

class weatherStructure(BaseModel):
    """This class defines the format in which the response from the LLM is to be
        returned."""
    longitude: float
    latitude: float
    date: str
    name: str

class bookStructure(BaseModel):
    """This class defines the format in which the response from the LLM is to be
        returned."""
    title: str
    authors: List[str]
    publisher: str

class Chatbot:
    """This class defines the Chatbot class that is used to interact with the
        Hugging Face API. The model is chosen by the user."""
    
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt):
        """This function generates a response from the Hugging Face API model."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=100)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text

    def generate_response_with_structure(self, prompt):
        """This function generates a response from the Hugging Face API model
            and returns the response in the weatherStructure format."""
        # Add prompt to the model input
        formatted_prompt = """
            You are an AI assistant that provides weather data.
            Return the response in the following JSON format:
            {
            "longitude": float,
            "latitude": float,
            "date": "YYYY-MM-DD",
            "name": "City Name"
            }
            Question: """ + prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=300)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract JSON from model output
        try:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            json_response = json.loads(text[json_start:json_end])
            return json_response, text
        except:
            return "Error parsing JSON", text

class MistralAgent:
    """This class defines the MistralAgent class that is used to interact with the
        Mistral API."""
    def __init__(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        self.model_name = "mistral-small-latest"
        self.client = Mistral(api_key=api_key)

    def generate_response(self, prompt):
        """This function generates a response from the Mistral API."""
        chat_response = self.client.chat.complete(
            model= self.model_name,
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        return chat_response.choices[0].message.content
    
    def generate_response_with_structure(self, prompt, structure):
        """This function generates a response from the Mistral API and returns the
            response in the weatherStructure format."""
        chat_response = self.client.chat.parse(
            model=self.model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "Extract the books information."
                },
                {
                    "role": "user", 
                    "content": prompt
                },
            ],
            response_format=structure,
            max_tokens=256,
            temperature=0
        )
        return chat_response.choices[0].message.content
