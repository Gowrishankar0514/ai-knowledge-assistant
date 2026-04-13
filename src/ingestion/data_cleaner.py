import re
from bs4 import BeautifulSoup

def clean_html(html_content: str) -> str:
    """
    Remove scripts, styles, and other noise from HTML content.
    Returns cleaned text.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
        script_or_style.decompose()
        
    text = soup.get_text(separator=' ')
    
    # Collapse multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(raw_text: str) -> str:
    """
    General text cleaning (removes excess whitespaces, unusual characters).
    """
    text = re.sub(r'\s+', ' ', raw_text)
    return text.strip()
