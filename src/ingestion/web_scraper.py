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
        # Use a session with proper headers so websites don't block us
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
        })

    def fetch_page(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
            # Check content type — accept html and xhtml
            content_type = response.headers.get('Content-Type', '').lower()
            if content_type and 'html' not in content_type and 'xml' not in content_type and 'text/plain' not in content_type:
                logger.warning(f"URL {url} is not HTML (Content-Type: {content_type}). Skipping.")
                return ""
            # Handle encoding properly
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching URL {url}: {e}")
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching URL {url}: {e}")
            return ""
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching URL {url}")
            return ""
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
            logger.warning(f"No HTML content received from {url}")
            return
            
        cleaned_text = clean_html(html_content)
        
        if cleaned_text and len(cleaned_text.strip()) > 50:
            self.documents.append({
                "source": url,
                "content": cleaned_text,
                "type": "web"
            })
            logger.info(f"Extracted {len(cleaned_text)} chars from {url}")
        else:
            logger.warning(f"Insufficient text content from {url} (got {len(cleaned_text) if cleaned_text else 0} chars)")

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
        logger.info(f"Total documents scraped: {len(self.documents)}")
        return self.documents
