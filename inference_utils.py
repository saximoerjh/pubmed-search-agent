import os
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import ssl
import certifi
from dotenv import load_dotenv
import json
from pubmed_utils.pubmed_server import *
from functools import partial



def chat_without_model(message, llm):
    """
    用于与模型对话的函数

    Args:
        message (str): The message to send to the model
        llm(ChatTongyi): The language model instance
    """
    agent = create_agent(model=llm)
    message = {"messages": [HumanMessage(message)]}
    response = agent.invoke(
        message
    )
    return response["messages"][-1].content[0]['text']