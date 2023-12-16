from .llms import open_ai
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from ..farm_tools import tools


memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools,
    open_ai,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)