import os
import ssl
import certifi
import urllib.error
import urllib.parse
import urllib.request
from functools import partial
from dotenv import load_dotenv
from langchain_community.retrievers import PubMedRetriever
from langchain_core.documents import Document
import re
import string
import time
import jieba
import requests
import concurrent.futures
from typing import Tuple, Optional
from tqdm import tqdm
from concurrent.futures.thread import ThreadPoolExecutor


class customRetriever(PubMedRetriever):
    def _dict2document(self, doc: dict) -> Document:
        """
        ATTENTION:
            旧写法：summary = doc.pop("Summary")，所以metadata中缺少Summary字段
        """
        summary = doc["Summary"]
        return Document(page_content=summary, metadata=doc)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"][
                "Article"
            ]
        except KeyError:
            ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = [
            f"{txt['@Label']}: {txt['#text']}"
            for txt in abstract_text
            if "#text" in txt and "@Label" in txt
        ]
        summary = (
            "\n".join(summaries)
            if summaries
            else (
                abstract_text
                if isinstance(abstract_text, str)
                else (
                    "\n".join(str(value) for value in abstract_text.values())
                    if isinstance(abstract_text, dict)
                    else "No abstract available"
                )
            )
        )
        a_d = ar.get("ArticleDate", {})
        pub_date = "-".join(
            [
                a_d.get("Year", ""),
                a_d.get("Month", ""),
                a_d.get("Day", ""),
            ]
        )

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "Copyright Information": ar.get("Abstract", {}).get(
                "CopyrightInformation", ""
            ),
            "Summary": summary,
        }

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        """
        获取文章的详细信息
         通过efetch接口获取文章的详细信息，包括标题、摘要、出版日期等
        :param uid:
        :param webenv:
        :return:
        """
        url = (
            self.base_url_efetch
            + "db=pubmed&retmode=xml&id="
            + uid
            + "&webenv="
            + webenv
        )
        if self.api_key != "":
            url += f"&api_key={self.api_key}"

        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(  # noqa: T201
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1
                else:
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = self.parse(xml_text)
        ret_dict =  self._parse_article(uid, text_dict)
        # 加入url字段
        ret_dict["url"] = url
        return ret_dict


class NCBIRetriever():
    def __init__(self):
        # 创建PubMed检索器实例
        # 如果取不到环境变量则报错
        if os.getenv("PUBMED_API_KEY") is None:
            raise ValueError("PUBMED_API_KEY environment variable not set")
        self.retriever = customRetriever(api_key=os.getenv("PUBMED_API_KEY"))

    def web_search(self, query: str, top_k_results: int = 3, MAX_QUERY_LENGTH: int = 2000) -> list[Document]:
        """
        发送查询请求，获取相关文献
        这里可以根据需要调整top_k_results参数
        例如：top_k_results=5表示获取前5条相关文献
        :param query: 查询关键词
        :param top_k_results: 获取的文献数量
        :return: 文献列表
        """
        self.retriever.top_k_results = top_k_results
        self.retriever.MAX_QUERY_LENGTH = MAX_QUERY_LENGTH
        results = self.retriever.invoke(query)
        return results

    def extract_text_from_url(self, url, snippet: Optional[str] = None):
        """
        Extract text from a URL. If a snippet is provided, extract the context related to it.

        Args:
            url (str): URL of a webpage or PDF.
            use_jina (bool): Whether to use Jina for extraction.
            snippet (Optional[str]): The snippet to search for.

        Returns:
            str: Extracted text or context.
        """
        try:
            retry = 0
            while True:
                try:
                    result = urllib.request.urlopen(url)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and retry < self.retriever.max_retry:
                        # Too Many Requests errors
                        # wait for an exponentially increasing amount of time
                        print(  # noqa: T201
                            f"Too Many Requests, "
                            f"waiting for {self.retriever.sleep_time:.2f} seconds..."
                        )
                        time.sleep(self.retriever.sleep_time)
                        self.retriever.sleep_time *= 2
                        retry += 1
                    else:
                        raise e

            xml_text = result.read().decode("utf-8")
            text_dict = self.retriever.parse(xml_text)
            # TODO: 这里需要根据实际的网页结构进行调整获取text
            text = text_dict.get("content", "")

            if snippet:
                # 通过摘要提取相关内容（因为文本根据摘要进行匹配）
                success, context = extract_snippet_with_context(text, snippet)
                if success:
                    return context
                else:
                    return text
            else:
                # If no snippet is provided, return directly
                return text[:8000]
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err}"
        except requests.exceptions.ConnectionError:
            return "Error: Connection error occurred"
        except requests.exceptions.Timeout:
            return "Error: Request timed out after 20 seconds"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def fetch_page_content(self, urls, max_workers=4, snippets: Optional[dict] = None):
        """
        Concurrently fetch content from multiple URLs.

        Args:
            urls (list): List of URLs to scrape.
            max_workers (int): Maximum number of concurrent threads.
            use_jina (bool): Whether to use Jina for extraction.
            snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.

        Returns:
            dict: A dictionary mapping URLs to the extracted content or context.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to display a progress bar
            futures = {
                executor.submit(self.extract_text_from_url, url, snippets.get(url) if snippets else None): url
                for url in urls
            }
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
                url = futures[future]
                try:
                    data = future.result()
                    results[url] = data
                except Exception as exc:
                    results[url] = f"Error fetching {url}: {exc}"
                time.sleep(0.2)  # Simple rate limiting
        return results

def extract_relevant_info(search_results: list[Document]) -> list[dict]:
    useful_info = []

    for id, result in enumerate(search_results):
        info = {
            'id': id + 1,  # Increment id for easier subsequent operations
            'title': result.metadata['Title'],
            'url': result.metadata['url'],
            'date': result.metadata['Published'],
            'snippet': result.metadata['Summary'],
            'context': ''  # Reserved field to be filled later
        }
        useful_info.append(info)
    return useful_info


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        # snippet_words = set(snippet.split())
        snippet_words = set(jieba.lcut_for_search(snippet))

        best_sentence = None
        best_f1 = 0.2

        sentences = re.split(r'(?<=[。？！]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        # sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            # sentence_words = set(key_sentence.split())
            sentence_words = set(jieba.lcut_for_search(key_sentence))
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


# 避免自签名证书验证失败
ssl._create_default_https_context = partial(ssl.create_default_context, cafile=certifi.where())
load_dotenv()
# 创建PubMed检索器实例
retrieverManager = NCBIRetriever()

# 检索与"chatgpt"相关的文献
documents = retrieverManager.web_search("cancer", top_k_results=3)
useful_info = extract_relevant_info(documents)
# 输出检索结果
for info in useful_info:
    print(f"Title: {info['title']}")
    print(f"Published: {info['date']}")
    print(f"URL: {info['url']}")
    print(f"Date: {info['date']}")
    print(f"Snippet: {info['snippet']}")
