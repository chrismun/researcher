import requests
from xml.etree import ElementTree

def search_arxiv(keywords):
    query = '+AND+'.join(keywords.split())
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1'
    response = requests.get(url)
    root = ElementTree.fromstring(response.content)

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        print(f"Title: {title}\nSummary:\n{summary}\n")

    return f"Title: {title}\nSummary:\n{summary}\n"

# paper = search_arxiv("deep learning")