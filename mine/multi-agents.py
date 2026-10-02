from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain.chat_models import BaseChatModel
from langchain.tools import tool


import asyncio
from dotenv import load_dotenv
import os
import json

load_dotenv()



def get_model() -> BaseChatModel:
    MODEL = os.getenv("MODEL","gpt-4o-mini")
    # print(f"Model - {MODEL}")
    base_model = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
    )

    return base_model


def get_mcp_config(server_name: str)->dict:
    with open("mcp_list.json", "r") as file:
        mcp_list = json.load(file)

    mcp_config = mcp_list.get(server_name, None)

    if not mcp_config:
        raise ValueError(f"MCP Server {server_name} configuration not availble.")
    mcp_config = {server_name: mcp_config}
    # print(mcp_config)
    return mcp_config


async def get_tools(server_config: dict)->list[tool]:

    mcp_client = MultiServerMCPClient(server_config)

    tools = await mcp_client.get_tools()
    # print(f"No of tools available - {len(tools)}")
    # for tool in tools:
    #     print(f"Name - {tool.name}")
    return tools

def create_subagents(
        model:BaseChatModel,
        tools:list[any],
        system_prompt:str=None,
        subagent_name:str=None,
        is_tool:bool=True):

    agent=create_agent(
        model=model,
        tools=tools,
        name=subagent_name,
        system_prompt=system_prompt,
    )
    if is_tool:
        return agent.as_tool()
    return agent

async def run_main_agent(agent):
    question = HumanMessage(
        content="Get me a best 5 flights and price in INR from Dubai to Delhi on December 31st 2026."
        "include the airline name, flight number, departure and arrival time, and layover information if any."
    )
    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke(
        {
            'messages':[question]
        },
        config
    )


async def main():
    mcp_server_name='travel_server'
    model=get_model()
    # print(model)
    server_config=get_mcp_config(mcp_server_name)
    mcp_tools=await get_tools(server_config)
    print(f"No of tools available - {len(mcp_tools)}")
    for tool in mcp_tools:
        print(f"Name - {tool.name}")
    agent=create_subagents(model,tools=mcp_tools,subagent_name='Travel Agent')
    print(type(agent))
    print(agent.name)
    # print(agent)
    main_agent=create_subagents(
        model=model,
        tools=agent,
        system_prompt="You are a help agent to call sub agents for user queries.",
        is_tool=False,
    )
    response=await run_main_agent(main_agent)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
