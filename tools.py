import json
import os
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web with Tavily and return relevant titles, URLs, and summaries."""
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in the .env file.")

    response = TavilyClient(api_key=api_key).search(
        query=query,
        max_results=5,
        include_images=True,
    )
    results = response.get("results", [])
    images = response.get("images", [])
    if not results and not images:
        return "No search results found."

    output = "\n\n".join(
        f"{result.get('title', 'Untitled')}\n"
        f"{result.get('url', '')}\n"
        f"{result.get('content', '')}"
        for result in results
    )
    image_entries = []
    for image in images:
        if isinstance(image, str):
            image_entry = {"url": image, "description": query}
        elif isinstance(image, dict):
            image_entry = {
                "url": image.get("url", ""),
                "description": image.get("description") or query,
            }
        else:
            continue

        if image_entry["url"].startswith(("https://", "http://")):
            image_entries.append(
                "__TAVILY_IMAGE__:" + json.dumps(image_entry, ensure_ascii=True)
            )

    if image_entries:
        output += "\n\n" + "\n".join(image_entries)
    return output


@tool
def scrape_page(url: str) -> str:
    """Fetch a public webpage and return its readable text content."""
    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty.")

    for redirect_count in range(6):
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ValueError("URL must be an absolute HTTP or HTTPS URL.")

        try:
            addresses = socket.getaddrinfo(parsed_url.hostname, None)
        except socket.gaierror as error:
            raise ValueError(f"Could not resolve URL host: {parsed_url.hostname}") from error

        if any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
            raise ValueError("URL must point to a public host.")

        try:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; TextScraper/1.0)"},
                timeout=15,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            return f"Could not scrape {url}: {error}"

        if response.is_redirect or response.is_permanent_redirect:
            if redirect_count == 5:
                raise ValueError("The webpage redirected too many times.")
            location = response.headers.get("Location")
            if not location:
                return f"Could not scrape {url}: redirect response had no Location header."
            url = urljoin(url, location)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            status = response.status_code
            if status in {401, 403, 429}:
                return (
                    f"Could not scrape {url}: the site returned HTTP {status} "
                    f"({response.reason}), possibly because it blocks automated "
                    "requests. Use the Tavily search result snippets or another source."
                )
            return f"Could not scrape {url}: {error}"

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)
        if not text:
            return "No readable text content found on the page."
        return text[:3000]

    raise ValueError("Could not fetch the webpage.")
