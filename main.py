from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent."""
    url : str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for a response from the agent."""
    answer : str = Field(description="The answer from the agent to the query")
    sources : List[Source] = Field(default_factory=list, description="The sources used by the agent to arrive at the answer")

tavily_client = TavilyClient()

@tool
def search(query : str) -> str:
    """
    Tool that searches over the internet
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """
    print(f"Searching for: {query}")
    return tavily_client.search(query)

llm = ChatOllama(model="qwen3:8b", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages" : [HumanMessage(content="Search for 3 job posting for data scientist in India")]})
    print(result)


if __name__ == "__main__":
    main()
