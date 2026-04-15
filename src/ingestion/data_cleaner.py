import re
from bs4 import BeautifulSoup

def clean_html(html_content: str) -> str:
    """
    Extract meaningful readable text from HTML content.
    Uses a multi-strategy approach:
      1. Try to find main content areas (<article>, <main>, role="main")
      2. Fall back to <p>, <h1>-<h6>, <li>, <td> tags
      3. Last resort: get all text after removing scripts/styles
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove script, style, and non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Strategy 1: Look for semantic content containers
    main_content = None
    for selector in ['article', 'main', '[role="main"]', '.content', '.post-content', '.article-body', '#content', '#main']:
        try:
            found = soup.select(selector)
            if found:
                main_content = ' '.join([el.get_text(separator=' ') for el in found])
                break
        except Exception:
            continue
    
    # Strategy 2: If no semantic container, extract from readable tags
    if not main_content or len(main_content.strip()) < 100:
        readable_tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'blockquote', 'pre', 'span', 'div'])
        texts = []
        for tag in readable_tags:
            tag_text = tag.get_text(separator=' ').strip()
            # Only keep text fragments that are meaningful (more than just a word)
            if len(tag_text) > 20:
                texts.append(tag_text)
        if texts:
            main_content = ' '.join(texts)
    
    # Strategy 3: Last resort — get ALL text from body
    if not main_content or len(main_content.strip()) < 100:
        body = soup.find('body')
        if body:
            # Remove nav, header, footer for cleaner output
            for noise_tag in body.find_all(['nav', 'header', 'footer']):
                noise_tag.decompose()
            main_content = body.get_text(separator=' ')
        else:
            main_content = soup.get_text(separator=' ')
    
    if not main_content:
        return ""
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', main_content).strip()
    
    # Remove very short results (likely just boilerplate)
    if len(text) < 50:
        return ""
    
    return text

def clean_text(raw_text: str) -> str:
    """
    General text cleaning (removes excess whitespaces, unusual characters).
    """
    text = re.sub(r'\s+', ' ', raw_text)
    return text.strip()
