"""This file contains the classes for the data structures used in the program.
    those classes define the struture to be returned by the LLM and passed to the
    LLM for the next iteration."""

from pydantic import BaseModel
from typing import List, Optional

class weatherStructure(BaseModel):
    """This class defines the format in which the response from the LLM is to be
        returned."""
    longitude: float
    latitude: float
    date: str
    name: str