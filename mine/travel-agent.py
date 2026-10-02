from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model


import asyncio
from IPython.display import Markdown, display
from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.getenv("MODEL")
print(f"Model - {MODEL}")


def get_model() -> BaseChatModel:
    base_model = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
    )

    return base_model


async def get_tools():
    mcp_client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            },
            "wiki_server": {
                "transport": "streamable_http",
                "url": "https://mcp.deepwiki.com/mcp",
            },
            "ai_data_server": {
                "transport": "streamable_http",
                "url": "https://mcp.exa.ai/mcp",
            },
            "data_index_server": {
                "transport": "streamable_http",
                "url": "https://index.mintlify.com",
            },
            "finance_server":{
                "transport":"streamable_http",
                "url":"https://wealthville.net/mcp"
            },
            "currency_convert_server":{
                "transport":"streamable_http",
                "url":"https://currency-mcp.wesbos.com/mcp"
            },
            "time_server":{
                "transport":"streamable_http",
                "url":"https://worldclock.pro/mcp"
            }
        }
    )

    tools = await mcp_client.get_tools()
    print(f"No of tools - {len(tools)}")
    for tool in tools:
        print(f"Name - {tool.name}")
    return tools


def get_system_prompt():
    system_prompt = SystemMessage(content="""
You are a helpfull travel agent to support flight related quries.
NO FOLLOW UP QUESTIONS.""")
    return system_prompt


async def run_agent():
    question = HumanMessage(
        content="Get me a best 5 flights and price in INR from Dubai to Delhi on December 31st 2026."
        "include the airline name, flight number, departure and arrival time, and layover information if any."
    )
    config = {"configurable": {"thread_id": "1"}}

    tools = await get_tools()
    # MODEL = get_model()

    # travel_agent = create_agent(
    #     model=MODEL,
    #     tools=tools,
    #     system_prompt=get_system_prompt(),
    #     checkpointer=InMemorySaver(),
    # )

    # response = await travel_agent.ainvoke({"messages": [question]}, config)
    # print(response["messages"][-1].content)

    # display(Markdown(response['messages'][-1].content))


if __name__ == "__main__":
    asyncio.run(run_agent())
