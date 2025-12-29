import ssl
import certifi
from dotenv import load_dotenv
from pubmed_utils.pubmed_server import *
from functools import partial
from tqdm import tqdm

if __name__ == '__main__':
    load_dotenv()
    # 避免自签名证书验证失败
    ssl._create_default_https_context = partial(ssl.create_default_context, cafile=certifi.where())
    query = "cancer"
    # response = exa_web_search(query, url)
    # print(response)
    results = web_search(query, num_result=3)
    print(results)
    extracted_info = extract_relevant_info(results)
    full_text = extract_text_from_pmid(extracted_info[0]["pmid"])
    # print(full_text)

    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_pmid(info['url'], snippet=query)  # Get full webpage text
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"


    print(extracted_info)