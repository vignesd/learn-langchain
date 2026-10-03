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

MODEL = os.getenv("MODEL", "gpt-4o-mini")
base_model = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
)


async def get_travel_agent():
    travel_mcp_server = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            }
        }
    )

    travel_mcp_tools=await travel_mcp_server.get_tools()

    # for tool in travel_mcp_tools:
    #     print(f"{tool.name=}\n{tool.description=}")
    #     print("\n")

    travel_agent = create_agent(
    model=base_model,
    tools=travel_mcp_tools,
    system_prompt="You are the helpfull AI Assistant for reponse only travel releated quries using tools. NO FOLLOW UP Question.",
)
    return travel_agent


async def get_currency_agent():
    currency_mcp_server = MultiServerMCPClient(
        {
    "currency_convert_server": {
        "transport": "streamable_http",
        "url": "https://currency-mcp.wesbos.com/mcp"
    }
        }
    )

    currency_mcp_tools=await currency_mcp_server.get_tools()

    # for tool in currency_mcp_tools:
    #     print(f"{tool.name=}\n{tool.description=}")
    #     print("\n")
    currency_agent = create_agent(
    model=base_model,
    tools=currency_mcp_tools,
    system_prompt="You are the helpfull AI Assistant for reponse only currency releated quries using tools. NO FOLLOW UP Question.",
)
    return currency_agent


# @tool
# async def ask_travel_agent(query:str):
#     """
#     Search for flights.
#     Searches Kiwi.com for available flights between two locations for the given dates and passengers. 
#     City or airport names are resolved automatically, 
#     so call this whenever the user wants to search for flights — whether they gave IATA codes or just place names
#     """

#     question=HumanMessage(content=query)

#     result=await get_travel_agent.ainvoke(
#         {
#             'messages':[question]
#         }
#     )
#     return result['messages'][-1].content

# @tool
# async def ask_currency_agent(query:str):
#     """
#     Handles all currency-related operations, including currency conversion, 
#     supported-currency lookup, latest exchange-rate retrieval, and historical exchange-rate queries.
#     """

#     question=HumanMessage(content=query)

#     result=await get_currency_agent.ainvoke(
#         {
#             'messages':[question]
#         }
#     )
#     return result['messages'][-1].content

async def create_agents():
    travel_agent = await get_travel_agent()
    currency_agent = await get_currency_agent()

    return travel_agent, currency_agent

async def run_main_agent():
    # # ---------------------------------------------------------
    # # 1. Create specialist agents once
    # # ---------------------------------------------------------
    travel_agent = await get_travel_agent()
    currency_agent = await get_currency_agent()

    # ---------------------------------------------------------
    # 2. Create wrapper tools for the main agent
    # ---------------------------------------------------------
    @tool
    async def ask_travel_agent(query: str) -> str:
        """
        Handle travel-related requests such as flights, hotels,
        transportation, destinations, and itineraries.
        """

        result = await travel_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=query)
                ]
            }
        )

        return result["messages"][-1].content

    @tool
    async def ask_currency_agent(query: str) -> str:
        """
        Handle currency conversion, supported currencies,
        latest exchange rates, and historical exchange rates.
        """

        result = await currency_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(content=query)
                ]
            }
        )

        return result["messages"][-1].content

    # ---------------------------------------------------------
    # 3. Main Agent system prompt
    # ---------------------------------------------------------
    system_message = """
You are the Main Agent responsible for handling user requests
and coordinating specialized agents.

Available specialized agents:

1. Travel Agent
   - Handles flights, hotels, transportation, destinations,
     itineraries, and other travel-related requests.

2. Currency Agent
   - Handles currency conversion, supported currencies,
     latest exchange rates, and historical exchange rates.

Rules:

1. Identify the user's intent and delegate to the appropriate
   specialized agent.

2. For travel-related requests, use the Travel Agent.

3. For currency-related requests, use the Currency Agent.

4. If a request requires both travel and currency information,
   use both agents and combine their results.

5. When multiple independent agents are required, invoke them
   in parallel when possible.

6. Do not expose internal agent names, tool calls, or
   implementation details to the user.

7. Never invent or guess information.

8. Use information returned by the specialized agents.

9. Do not fabricate flight prices, availability, exchange rates,
   airlines, dates, times, or other factual information.

10. Do not ask follow-up questions. Use the information already
    provided by the user and make the best possible request to
    the appropriate specialized agent.

11. If an agent cannot provide the requested information, clearly
    state that the information is unavailable.

12. Keep the final response concise and directly answer the user.

13. For multiple independent requests, call the required
    specialized agents rather than answering from your own
    knowledge.
"""


    # ---------------------------------------------------------
    # 4. Create the Main Agent
    # ---------------------------------------------------------
    main_agent = create_agent(
        model=base_model,
        tools=[
            ask_travel_agent,
            ask_currency_agent,
        ],
        system_prompt=system_message,
        checkpointer=InMemorySaver(),
    )

    # ---------------------------------------------------------
    # 5. Conversation configuration
    # ---------------------------------------------------------
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # ---------------------------------------------------------
    # 6. User request
    # ---------------------------------------------------------
    query = HumanMessage(
        content="""
Get me 2 available flight options from Dubai to Delhi
on December 5th 2026.

Include:
- Airline name
- Flight number
- Departure time
- Arrival time
- Layover information
- Price in INR
"""
    )

    query=HumanMessage(content="Convert 100 USD to INR using the latest available exchange rate. Provide the converted amount and the exchange rate used.")
    # query=HumanMessage(content="Find 2 available flights from Dubai to Delhi for December 5, 2026, and show their prices in USD. " \
    # "Then convert those prices to INR using the latest exchange rate.")

    # ---------------------------------------------------------
    # 7. Invoke Main Agent
    # ---------------------------------------------------------
    response = await main_agent.ainvoke(
        {
            "messages": [query]
        },
        config=config,
    )

    # ---------------------------------------------------------
    # 8. Print final response
    # ---------------------------------------------------------
    print("\n===== FINAL RESPONSE =====\n")
    print(response["messages"][-1].content)



if __name__ == "__main__":
    asyncio.run(run_main_agent())
