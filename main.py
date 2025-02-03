import re
import asyncio
import aiohttp
import urllib.parse
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
from bs4 import BeautifulSoup
import re
from urllib import robotparser
from playwright.async_api import async_playwright
from xml.etree import ElementTree as ET
import gzip
import io
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to INFO if you want less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global constant for maximum pagination pages.
MAX_PAGE = 10

class ProductFinderCrawler:
    def __init__(self, domains, concurrency=5, request_delay=1):
        self.domains = domains
        self.concurrency = concurrency
        self.request_delay = request_delay  # rate-limiting
        self.product_patterns = [
            re.compile(r'/product/'),
            re.compile(r'/item/'),
            re.compile(r'/p/'),
            re.compile(r'/prod/'),
            re.compile(r'/collections/[^/]+/products/'),  # Shopify pattern
            re.compile(r'/products/'),
            re.compile(r'/shapewear/'),
            re.compile(r'/dp/'),
            re.compile(r'/\d{8,}')  # Shopify often uses numeric product IDs
        ]
        self.visited_urls = set()
        self.robots_parsers = {}
        self.sitemap_urls = {}
        self.max_page = MAX_PAGE  # Limit pagination to MAX_PAGE pages
        self.results = {}

    async def get_robot_text(self, domain):
        logger.debug(f"Entering get_robot_text for domain: {domain}")
        url = f"https://{domain}/robots.txt"
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url) as response:
                    logger.debug(f"Fetched robots.txt for {domain} with status {response.status}")
                    if response.status == 200:
                        content = await response.text()
                        rp = robotparser.RobotFileParser()
                        rp.parse(content.splitlines())
                        # Use crawl-delay from robots.txt if available.
                        crawl_delay = rp.crawl_delay("*") or 1.0
                        self.request_delay = max(self.request_delay, crawl_delay)
                        self.robots_parsers[domain] = rp
                        logger.debug(f"Parsed robots.txt for {domain}. Using crawl delay: {self.request_delay}")
        except Exception as e:
            logger.error(f"Error fetching robots.txt for {domain}: {e}")

    async def fetch_sitemap(self, domain):
        logger.debug(f"Entering fetch_sitemap for domain: {domain}")
        sitemap_urls = [
            f"https://{domain}/sitemap.xml",
            # f"http://{domain}/sitemap_index.xml",
            # f"http://{domain}/sitemap.xml.gz"
        ]
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
            "Accept": "application/xml, text/xml, */*"
        }
        
        for sitemap_url in sitemap_urls:
            try:
                logger.debug(f"Requesting sitemap: {sitemap_url}")
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(sitemap_url) as response:
                        logger.debug(f"Response status for sitemap {sitemap_url}: {response.status}")
                        logger.debug(f"Response Headers: {response.headers}")
                        if response.status == 200:
                            if sitemap_url.endswith('.gz'):
                                raw_content = await response.read()
                                with gzip.GzipFile(fileobj=io.BytesIO(raw_content)) as f:
                                    content = f.read().decode('utf-8')
                            else:
                                content = await response.text()
                            root = ET.fromstring(content)
                            urls = []
                            # Check for namespace
                            namespace = ""
                            if root.tag.startswith("{") and "}" in root.tag:
                                namespace = root.tag[1:root.tag.find("}")]
                            
                            if root.tag.endswith("sitemapindex"):
                                # Recursively process child sitemaps
                                sitemap_tag = f"{{{namespace}}}sitemap" if namespace else "sitemap"
                                loc_tag = f"{{{namespace}}}loc" if namespace else "loc"
                                for sitemap in root.iter(sitemap_tag):
                                    child_url = sitemap.find(loc_tag).text.strip()
                                    logger.debug(f"Found child sitemap URL: {child_url}")
                                    async with session.get(child_url) as child_response:
                                        if child_response.status == 200:
                                            child_content = await child_response.text()
                                            child_root = ET.fromstring(child_content)
                                            loc_child_tag = f"{{{namespace}}}loc" if namespace else "loc"
                                            for url_elem in child_root.iter(loc_child_tag):
                                                if url_elem.text:
                                                    urls.append(url_elem.text.strip())
                            else:
                                # Not a sitemap index—just a normal sitemap
                                loc_tag = f"{{{namespace}}}loc" if namespace else "loc"
                                for elem in root.iter(loc_tag):
                                    if elem.text:
                                        urls.append(elem.text.strip())
                            
                            self.sitemap_urls[domain] = urls
                            logger.debug(f"Extracted {len(urls)} URLs from sitemap for {domain}")
                        else:
                            logger.warning(f"Got status {response.status} when fetching sitemap for {domain}")
            except Exception as e:
                logger.error(f"Error fetching sitemap for {domain}: {e}")

    def is_allowed(self, domain, url):
        if domain not in self.robots_parsers:
            return True
        allowed = self.robots_parsers[domain].can_fetch("*", url)
        logger.debug(f"URL {url} allowed by robots.txt for {domain}: {allowed}")
        return allowed

    async def get_links(self, html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            abs_url = urljoin(base_url, href)
            parsed = urlparse(abs_url)
            # Remove fragment and query parameters for normalization
            normalized = parsed._replace(fragment='', query='').geturl()
            links.append(normalized)
        logger.debug(f"Extracted {len(links)} links from {base_url}")
        return links

    async def is_product_url(self, url):
        for pattern in self.product_patterns:
            if pattern.search(url):
                logger.debug(f"URL {url} matches product pattern: {pattern.pattern}")
                return True
        return False

    def requires_js_rendering(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # Shopify-specific checks and other patterns
        if soup.find('div', class_=re.compile('product-grid|shopify-section')):
            logger.debug("Page appears to require JS rendering (product-grid/shopify-section found)")
            return True
        if soup.find('div', {'data-product-id': True}):
            logger.debug("Page appears to require JS rendering (data-product-id found)")
            return True
        if soup.find('script', {'data-src': True}):
            logger.debug("Page appears to require JS rendering (script with data-src found)")
            return True
        if soup.find('div', {'id': '__next'}) or soup.find('div', {'id': 'app'}):
            logger.debug("Page appears to require JS rendering (React/Vue mounting point found)")
            return True
        return False

    def get_next_page_url(self, url):
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if 'page' in qs:
            try:
                current_page = int(qs['page'][0])
            except ValueError:
                current_page = 1
            if current_page >= self.max_page:
                return None
            qs['page'] = [str(current_page + 1)]
            new_query = urlencode(qs, doseq=True)
            next_url = parsed._replace(query=new_query).geturl()
            logger.debug(f"Next page URL based on query param: {next_url}")
            return next_url
        elif '/page/' in parsed.path:
            path_parts = parsed.path.rstrip('/').split('/')
            try:
                page_index = path_parts.index('page') + 1
                current_page = int(path_parts[page_index])
                if current_page >= self.max_page:
                    return None
                path_parts[page_index] = str(current_page + 1)
                new_path = '/'.join(path_parts)
                next_url = parsed._replace(path=new_path).geturl()
                logger.debug(f"Next page URL based on path: {next_url}")
                return next_url
            except (ValueError, IndexError):
                return None
        else:
            # If there’s no explicit page information, append '/page/2' to the path.
            new_path = parsed.path.rstrip('/') + '/page/2'
            next_url = parsed._replace(path=new_path).geturl()
            logger.debug(f"Default next page URL: {next_url}")
            return next_url

    async def crawl_page(self, url, session, domain):
        logger.debug(f"Starting crawl_page for URL: {url}")
        if url in self.visited_urls or not self.is_allowed(domain, url):
            logger.debug(f"Skipping URL (visited or disallowed): {url}")
            return [], []
        self.visited_urls.add(url)
        await asyncio.sleep(self.request_delay)  # rate limiting
        try:
            async with session.get(url) as response:
                logger.debug(f"Fetched {url} with status {response.status}")
                if response.status != 200:
                    logger.warning(f"Non-200 response for {url}: {response.status}")
                    return [], []
                html = await response.text()
                product_links = []
                links = await self.get_links(html, url)
                
                # Check for product links on the page
                for link in links:
                    if await self.is_product_url(link):
                        product_links.append(link)
                
                # If no product links found and the page appears to require JS rendering, try rendering.
                if domain == "www.bewakoof.com" or (not product_links and self.requires_js_rendering(html)):
                    logger.debug(f"Page {url} seems to require JS rendering. Launching Playwright.")
                    async with async_playwright() as p:
                        browser = await p.chromium.launch(headless=True)
                        page = await browser.new_page()
                        logger.debug(f"Attempting to navigate to {url} with Playwright.")
                        try:
                            await page.goto(url, timeout=60000, wait_until='domcontentloaded')
                        except Exception as e:
                            logger.error(f"Page.goto failed for {url}: {e}")
                            await browser.close()
                            return [], []
                        rendered_html = await page.content()
                        rendered_links = await self.get_links(rendered_html, url)
                        for link in rendered_links:
                            if await self.is_product_url(link):
                                product_links.append(link)
                        await browser.close()
                
                # Collect new links that are on the same domain and haven't been visited.
                new_links = [link for link in links if urlparse(link).netloc == domain and link not in self.visited_urls]

                # --- Pagination handling ---
                next_page_url = self.get_next_page_url(url)
                if next_page_url and next_page_url not in self.visited_urls:
                    new_links.append(next_page_url)
                    logger.debug(f"Pagination: added next page URL {next_page_url}")

                logger.debug(f"crawl_page for {url} found {len(product_links)} product links and {len(new_links)} new links")
                return product_links, new_links
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return [], []

    async def crawl_domain(self, domain):
        logger.info(f"Starting crawl for domain: {domain}")
        await self.get_robot_text(domain)
        await self.fetch_sitemap(domain)
        product_urls = set()
        queue = asyncio.Queue()

        # Using HTTPS for the starting URL.
        await queue.put(f"https://{domain}")
        # Initialize the queue with sitemap URLs if available; otherwise, start with the homepage.
        if domain in self.sitemap_urls:
            for url in self.sitemap_urls[domain]:
                if any(p.search(url) for p in self.product_patterns):
                    product_urls.add(url)  # Directly add sitemap product URLs
                    logger.debug(f"Added product URL from sitemap: {url}")
                else:
                    await queue.put(f"https://{domain}")
                    logger.debug(f"Added non-product URL from sitemap to queue: {url}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(self.concurrency):
                task = asyncio.create_task(self.worker(queue, session, domain, product_urls))
                tasks.append(task)
            await queue.join()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Finished crawling domain: {domain}. Found {len(product_urls)} product URLs.")
        return list(product_urls)

    async def worker(self, queue, session, domain, product_urls):
        while True:
            url = await queue.get()
            logger.debug(f"Worker processing URL: {url}")
            parsed_url = urlparse(url)
            if parsed_url.netloc != domain:
                logger.debug(f"Skipping URL (different domain): {url}")
                queue.task_done()
                continue
            products, new_links = await self.crawl_page(url, session, domain)
            for product in products:
                product_urls.add(product)
                self.results.setdefault(domain, set()).add(product)
            for link in new_links:
                await queue.put(link)
            queue.task_done()

    async def crawl_infinite_scrolling_page(self, page, base_url, max_scroll_attempts=5, scroll_pause=2):
        logger.debug(f"Starting infinite scrolling crawl for: {base_url}")
        previous_height = await page.evaluate("document.body.scrollHeight")
        for _ in range(max_scroll_attempts):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(scroll_pause)
            current_height = await page.evaluate("document.body.scrollHeight")
            if current_height == previous_height:
                logger.debug("No further scrolling detected.")
                break
            previous_height = current_height
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')
        product_links = []
        for a in soup.select("a.product-link"):
            href = a.get("href")
            if href:
                product_links.append(urljoin(base_url, href))
        logger.debug(f"Infinite scrolling found {len(product_links)} product links")
        return product_links

    async def run(self):
        logger.info("Starting crawler run.")
        results = {}
        for domain in self.domains:
            try:
                # can increase this timeout, timeout set for testing
                # Apply a timeout for each domain's crawl
                product_urls = await asyncio.wait_for(self.crawl_domain(domain), timeout=60)
                results[domain] = product_urls
                logger.info(f"Finished crawling {domain} with {len(product_urls)} URLs.")
            except asyncio.TimeoutError:
                logger.info(f"Timeout reached for domain {domain}. Using partial results.")
                # Retrieve any partial results you might have stored in self.results
                results[domain] = self.results.get(domain, [])
        logger.info("Crawler run complete.")
        return results


if __name__ == "__main__":
    import os

    output_path = Path("output")
    output_path.mkdir(parents=True, exist_ok=True)

    domains = [ "www.bewakoof.com"]
    # domains = ["www.botnia.in", "www.bewakoof.com"]
    crawler = ProductFinderCrawler(domains, concurrency=5, request_delay=10.0)
    try:
        results = asyncio.run(crawler.run())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Writing partial results to file.")
        results = crawler.results

    for domain, urls in results.items():
        output_path = os.path.abspath(f'output/{domain}.txt')
        print(f"Writing file to: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(urls))
        logger.info(f"Wrote {len(urls)} URLs for {domain} to {domain}.txt")
