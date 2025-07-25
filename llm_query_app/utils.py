import requests

def download_file_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file.")
    content = response.content
    if ".pdf" in url:
        extension = "pdf"
    elif ".docx" in url:
        extension = "docx"
    else:
        raise Exception("Unsupported file type in URL.")
    return content, extension
