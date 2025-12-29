import os
import ssl
import certifi
from functools import partial
from dotenv import load_dotenv
from langchain_community.retrievers import PubMedRetriever

# 避免自签名证书验证失败
ssl._create_default_https_context = partial(ssl.create_default_context, cafile=certifi.where())


load_dotenv()
# 创建PubMed检索器实例
retriever = PubMedRetriever(api_key=os.getenv("PUBMED_API_KEY"), top_k_results=3)

# 检索与"chatgpt"相关的文献
documents = retriever.invoke("cancer")

# 输出检索结果
for document in documents:
    print(f"Title: {document.metadata['Title']}")
    print(f"Published: {document.metadata['Published']}")
    print(f"Copyright Information: {document.metadata['Copyright Information']}")
    print(f"Content: {document.page_content}\n")