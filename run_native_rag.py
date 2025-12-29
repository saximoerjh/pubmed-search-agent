import os
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import ssl
import certifi
from dotenv import load_dotenv
from pubmed_utils.pubmed_server import *
from functools import partial

def native_rag_main(query, llm):
    results = web_search(query, num_result=3)
    top_k = 3
    max_doc_len = 1000
    print(results)

    relevant_info = extract_relevant_info(results)[:top_k]

    # Collect all unique URLs to fetch
    unique_urls = set()
    url_snippets_map = {}

    for info in relevant_info:
        url = info['url']
        snippet = info.get('snippet', "")
        unique_urls.add(url)
        url_snippets_map[url] = query

    # Determine which URLs need to be fetched
    urls_to_fetch = [url for url in unique_urls]

    print(f"Fetching {len(urls_to_fetch)} unique URLs...")
    fetched_contents = fetch_page_content(
        urls_to_fetch,
        snippets=url_snippets_map
    )

    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        # snippet = doc_info.get('snippet', "")
        snippet = query
        raw_context = fetched_contents[url]
        success, context = extract_snippet_with_context(raw_context, snippet, context_chars=max_doc_len)
        if success:
            context = context
        else:
            context = raw_context[:max_doc_len]

        # Clean snippet from HTML tags if any
        clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

        formatted_documents += f"**Document {i + 1}:**\n"
        formatted_documents += f"**Title:** {doc_info.get('title', '')}\n"
        formatted_documents += f"**URL:** {url}\n"
        formatted_documents += f"**Snippet:** {clean_snippet}\n"
        formatted_documents += f"**Content:** {context}\n\n"

    # Construct the instruction with documents and question
    # instruction = get_naive_rag_instruction(question, formatted_documents)
    print(formatted_documents)

    instruction = (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{query}\n"
        "Documents:\n"
        f"{formatted_documents}\n"
    )

    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{query}\n\n'
    )

    full_prompt = instruction + "\n\n" + user_prompt

    agent = create_agent(model=llm, system_prompt=full_prompt)
    response = agent.invoke(
        {"messages": [HumanMessage(query)]}
    )
    return response["messages"][-1].content



def extract_boxed_answer(agent_result: Any) -> Optional[str]:
    """
    从 agent.invoke 的返回值中提取 \\boxed{...} 内的最终答案文本。
    兼容：
    - [{'text': '...\\boxed{...}...'}]
    - {'messages': [...]} / 直接是字符串
    """
    # 1) 取出承载文本的 content/text
    content = None

    if isinstance(agent_result, list) and agent_result:
        first = agent_result[0]
        if isinstance(first, dict):
            content = first.get("text") or first.get("content")
        elif isinstance(first, str):
            content = first

    if content is None and isinstance(agent_result, dict):
        content = agent_result.get("text") or agent_result.get("content")

    if content is None and isinstance(agent_result, str):
        content = agent_result

    if not content:
        return None

    # 2) 提取 \\boxed{...}
    # 支持内容中出现换行；非贪婪匹配 boxed 内部
    m = re.search(r"\\boxed\{([\s\S]*?)}", content)
    if m:
        return m.group(1).strip()

    return None


def show_final_answer(agent_result: Any) -> None:
    ans = extract_boxed_answer(agent_result)
    if ans is not None:
        print("Final Answer:")
        print(ans)
    else:
        # 退化：直接展示原始文本最后一段
        text = None
        if isinstance(agent_result, list) and agent_result and isinstance(agent_result[0], dict):
            text = agent_result[0].get("text")
        print("Final Answer (raw):")
        print((text or str(agent_result))[-800:])



if __name__ == '__main__':
    load_dotenv()
    ssl._create_default_https_context = partial(ssl.create_default_context, cafile=certifi.where())

    llm = ChatTongyi(
        model="qwen-vl-plus",  # 注意所用模型
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    query = "analyse the relationship between cancer and obesity"

    response = native_rag_main(query, llm)
    show_final_answer(response)
    # print(response)

