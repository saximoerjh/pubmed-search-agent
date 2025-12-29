import re
import string
import time
import jieba
import requests
import concurrent.futures
from pubmed_utils.pubmed_search import PubMedSearch
from pubmed_utils.pubmed_fetch import PubMedFetch
from typing import Tuple, Optional, List, Dict, Any
import asyncio
from tqdm import tqdm
from concurrent.futures.thread import ThreadPoolExecutor


def web_search(query, num_result=10) -> List[Dict[str, Any]]:
    """
    Perform a web search using PubMedSearch.
    :param query:
    :param num_result: return number of results
    :return: JSON response from the PubMedSearch API.
    """
    results = asyncio.run(PubMedSearch().search_articles(query, num_result))
    return results


def extract_relevant_info(search_results: List[Dict[str, Any]]):
    """
    Extract relevant information from Bing search results.

    Args:
        search_results (dict): JSON response from the Bing Web Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []

    for id, result in enumerate(search_results):
        info = {
            'id': id + 1,  # Increment id for easier subsequent operations
            'title': result['title'],
            'pmid': result['pmid'],
            'url': result['pmid'],
            'date': result['publication_date'],
            'snippet': result['abstract'],  # Remove HTML tags
            # Add context content to the information
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


def extract_text_from_pmid(pmid, snippet: Optional[str] = None):
    """
    Extract text from a pmid. If a snippet is provided, extract the context related to it.

    Args:
        pmid (str): pmid of a article.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        text = asyncio.run(PubMedFetch().get_full_text(pmid=pmid))
        # 返回text包括摘要和正文，去掉摘要部分 ABSTRACT 为摘要标志  MAIN TEXT为正文标志
        # 去掉ABSTRACT开头到MAIN TEXT开头的部分
        main_text_index = text.find("MAIN TEXT")
        if main_text_index != -1:
            text = text[main_text_index + len("MAIN TEXT"):]

        # print(text)

        if snippet:
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

def fetch_page_content(urls, max_workers=4, snippets: Optional[dict] = None):
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
            executor.submit(extract_text_from_pmid, url, snippets.get(url) if snippets else None): url
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