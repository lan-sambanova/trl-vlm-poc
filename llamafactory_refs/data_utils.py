# Copied from: src/llamafactory/data/data_utils.py
from enum import Enum, unique
from typing import Dict, Sequence, Set, Union


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"
