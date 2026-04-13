import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from src.utils.logger import get_logger
from src.ingestion.data_cleaner import clean_html

logger = get_logger(__name__)

class WebScraper:
    def __init__(self, base_url: str, max_depth: int = 2, max_pages: int = 15):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited = set()
        self.documents = []

    def fetch_page(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            # Check content type
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logger.warning(f"URL {url} is not HTML. Skipping.")
                return ""
            return response.text
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return ""

    def scrape(self, url: str = None, current_depth: int = 0):
        if url is None:
            url = self.base_url

        # Check stopping conditions
        if (
            current_depth > self.max_depth 
            or len(self.visited) >= self.max_pages 
            or url in self.visited 
            or urlparse(url).netloc != self.domain
        ):
            return

        logger.info(f"Scraping: {url} (Depth: {current_depth})")
        self.visited.add(url)
        
        html_content = self.fetch_page(url)
        if not html_content:
            return
            
        cleaned_text = clean_html(html_content)
        
        if cleaned_text:
            self.documents.append({
                "source": url,
                "content": cleaned_text,
                "type": "web"
            })

        # Extract internal links and recurse
        soup = BeautifulSoup(html_content, "html.parser")
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if not href or href.startswith(('mailto:', 'javascript:', 'tel:')):
                continue
                
            # Resolve relative URLs
            full_url = urljoin(url, href)
            # Remove fragments
            full_url = full_url.split('#')[0]
            
            if self.domain in full_url and full_url not in self.visited:
                self.scrape(full_url, current_depth + 1)
                
    def get_documents(self):
        return self.documents
