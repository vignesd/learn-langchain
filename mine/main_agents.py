from langchain.chat_models import BaseChatModel
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


import asyncio
from dotenv import load_dotenv
import os
import json

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")
BASE_MODEL = BaseChatModel(
    model=MODEL,
    temperature=0.2,
)

CONFIG_TRAVEL = {
    "travel_server": {
        "transport": "streamable_http",
        "url": "https://mcp.kiwi.com",
    }
}

CONFIG_TRAVEL = {
    "currency_convert_server": {
        "transport": "streamable_http",
        "url": "https://currency-mcp.wesbos.com/mcp",
    }
}

CONFIG_TRAVEL = {
    "time_server": {
        "transport": "streamable_http", 
        "url": "https://worldclock.pro/mcp"
        }
}


class TravelAgent():

    def __init__(self):
        pass

    