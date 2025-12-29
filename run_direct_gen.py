import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.messages import SystemMessage, HumanMessage, AIMessage


def direct_main(message, agent):
    response = agent.invoke(
        message
    )
    return response["messages"][-1].content


if __name__ == '__main__':
    load_dotenv()
    llm = ChatTongyi(
        model="qwen-vl-plus",  # 注意所用模型
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    agent = create_agent(model=llm)
    message = {"messages": [HumanMessage("介绍一下三体这本书")]}
    response = direct_main(message, agent)
    print(response)
