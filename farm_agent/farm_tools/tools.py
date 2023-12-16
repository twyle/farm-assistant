from .find_aggrovets import get_agrovets
from langchain.agents.tools import Tool


tools: list[Tool] = [
    get_agrovets
]
