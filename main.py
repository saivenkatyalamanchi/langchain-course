from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

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
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages" : [HumanMessage(content="Search for 3 job posting for data scientist in India")]})
    print(result)


if __name__ == "__main__":
    main()
