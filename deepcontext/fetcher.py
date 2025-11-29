"""Content fetcher for various sources."""

import base64
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from gitingest import ingest
from markdownify import markdownify as md

from deepcontext.models import Document, SourceType


class ContentFetcher:
    """Fetches content from markdown files, GitHub repos, webpages, and Confluence."""

    def __init__(
        self,
        timeout: float = 30.0,
        confluence_email: str | None = None,
        confluence_token: str | None = None,
    ):
        self.timeout = timeout
        self._client: httpx.Client | None = None
        
        # Confluence auth (can also be set via env vars)
        self.confluence_email = confluence_email or os.environ.get("CONFLUENCE_EMAIL")
        self.confluence_token = confluence_token or os.environ.get("CONFLUENCE_TOKEN")

    @property
    def client(self) -> httpx.Client:
        """Lazy-load HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "DeepContext/0.1 (Semantic Chunking Service)"
                },
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "ContentFetcher":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _generate_id(self, source: str) -> str:
        """Generate a unique ID for a document."""
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def fetch(self, source: str) -> Document:
        """Fetch content from any supported source."""
        source_type = self._detect_source_type(source)

        if source_type == SourceType.MARKDOWN:
            return self.fetch_markdown(source)
        elif source_type == SourceType.GITHUB:
            return self.fetch_github(source)
        elif source_type == SourceType.CONFLUENCE:
            return self.fetch_confluence(source)
        else:
            return self.fetch_webpage(source)

    def _detect_source_type(self, source: str) -> SourceType:
        """Detect the type of source."""
        # Local file
        if Path(source).exists():
            return SourceType.MARKDOWN

        # GitHub URL patterns
        github_patterns = [
            r"^https?://github\.com/",
            r"^https?://raw\.githubusercontent\.com/",
        ]
        for pattern in github_patterns:
            if re.match(pattern, source):
                return SourceType.GITHUB

        # Confluence URL patterns
        confluence_patterns = [
            r"^https?://[^/]+\.atlassian\.net/wiki/",
            r"^https?://[^/]+/wiki/spaces/",
            r"^https?://[^/]+/confluence/",
        ]
        for pattern in confluence_patterns:
            if re.match(pattern, source):
                return SourceType.CONFLUENCE

        # Default to webpage
        return SourceType.WEBPAGE

    def fetch_markdown(self, path: str) -> Document:
        """Fetch content from a local markdown file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = file_path.read_text(encoding="utf-8")
        title = self._extract_markdown_title(content) or file_path.stem

        return Document(
            id=self._generate_id(path),
            source=str(file_path.absolute()),
            source_type=SourceType.MARKDOWN,
            title=title,
            content=content,
            metadata={"filename": file_path.name},
        )

    def fetch_github(self, url: str) -> Document:
        """Fetch content from a GitHub URL."""
        raw_url = self._convert_to_raw_github_url(url)
        response = self.client.get(raw_url)
        response.raise_for_status()

        content = response.text
        title = self._extract_markdown_title(content) or self._title_from_url(url)

        return Document(
            id=self._generate_id(url),
            source=url,
            source_type=SourceType.GITHUB,
            title=title,
            content=content,
            metadata={
                "raw_url": raw_url,
                "repo": self._extract_repo_info(url),
            },
        )

    def fetch_webpage(self, url: str) -> Document:
        """Fetch and convert a webpage to markdown."""
        response = self.client.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style first
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Find the main content area - strategy:
        # 1. Look for main/article tags
        # 2. Find the container with the h1 (usually the content)
        # 3. Fall back to body
        
        main_content = soup.find("article")
        
        if not main_content:
            main_content = soup.find("main")
            
        if not main_content:
            # Find h1 and get its content container
            h1 = soup.find("h1")
            if h1:
                # Walk up to find a good container (not too big, not too small)
                parent = h1.parent
                while parent and parent.name != "body":
                    text_len = len(parent.get_text())
                    if 1000 < text_len < 50000:
                        # Check if this looks like content (has h1 and paragraphs/code)
                        if parent.find("p") or parent.find("pre") or parent.find("code"):
                            main_content = parent
                            break
                    parent = parent.parent
        
        if not main_content:
            main_content = soup.find("body") or soup

        # Now remove navigation from within the content
        for nav in main_content.find_all(["nav", "aside"]):
            nav.decompose()
        
        for element in main_content.find_all(attrs={"role": "navigation"}):
            element.decompose()

        # Remove elements that look like navigation (lists of many links)
        for ul in main_content.find_all("ul"):
            links = ul.find_all("a", recursive=False)
            items = ul.find_all("li", recursive=False)
            # If >80% of list items are links and there are many, it's nav
            if len(items) > 8 and len(links) >= len(items) * 0.6:
                ul.decompose()

        # Convert to markdown
        content = md(str(main_content), heading_style="ATX", code_language="")
        
        # Clean up the markdown
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'^Copy\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\[(Previous|Next)\].*$', '', content, flags=re.MULTILINE)
        # Remove lines that are just markdown links with no context
        content = re.sub(r'^\s*\*\s*\[[^\]]+\]\([^)]+\)\s*$', '', content, flags=re.MULTILINE)
        # Clean up multiple blank lines again
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else self._title_from_url(url)

        return Document(
            id=self._generate_id(url),
            source=url,
            source_type=SourceType.WEBPAGE,
            title=title,
            content=content.strip(),
            metadata={"original_url": url},
        )

    def fetch_sitemap(self, sitemap_url: str) -> list[str]:
        """
        Fetch and parse a sitemap XML file.

        Returns a list of URLs found in the sitemap.
        Handles both regular sitemaps and sitemap index files.
        """
        urls: list[str] = []

        response = self.client.get(sitemap_url)
        response.raise_for_status()

        # Parse XML
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            # Try to extract URLs from text if XML parsing fails
            url_pattern = r'<loc>([^<]+)</loc>'
            matches = re.findall(url_pattern, response.text)
            return matches

        # Handle namespace
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        # Check if this is a sitemap index
        sitemap_tags = root.findall('.//sm:sitemap/sm:loc', ns)
        if sitemap_tags:
            # This is a sitemap index, recursively fetch each sitemap
            for sitemap_tag in sitemap_tags:
                if sitemap_tag.text:
                    child_urls = self.fetch_sitemap(sitemap_tag.text)
                    urls.extend(child_urls)
        else:
            # Regular sitemap, extract URLs
            url_tags = root.findall('.//sm:url/sm:loc', ns)
            for url_tag in url_tags:
                if url_tag.text:
                    urls.append(url_tag.text)

            # Try without namespace if no URLs found
            if not urls:
                url_tags = root.findall('.//url/loc')
                for url_tag in url_tags:
                    if url_tag.text:
                        urls.append(url_tag.text)

        return urls

    def fetch_website(
        self,
        base_url: str,
        max_pages: int = 100,
        url_pattern: str | None = None,
        use_sitemap: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[Document]:
        """
        Fetch all pages from a website.

        Args:
            base_url: The base URL to start from (e.g., https://ui.shadcn.com/docs)
            max_pages: Maximum number of pages to fetch
            url_pattern: Regex pattern to filter URLs (e.g., r'/docs/' to only get docs pages)
            use_sitemap: Try to find and use sitemap.xml first
            progress_callback: Optional callback(url, current, total) for progress updates

        Returns:
            List of Documents fetched from the website
        """
        parsed_base = urlparse(base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

        urls_to_fetch: list[str] = []

        # Try to find sitemap
        if use_sitemap:
            sitemap_urls = [
                f"{base_domain}/sitemap.xml",
                f"{base_domain}/sitemap_index.xml",
                f"{base_domain}/sitemap-0.xml",
            ]

            for sitemap_url in sitemap_urls:
                try:
                    found_urls = self.fetch_sitemap(sitemap_url)
                    if found_urls:
                        urls_to_fetch = found_urls
                        break
                except Exception:
                    continue

        # If no sitemap, crawl from base URL
        if not urls_to_fetch:
            urls_to_fetch = self._crawl_links(base_url, base_domain, max_pages * 2)

        # Filter URLs by pattern if provided
        if url_pattern:
            pattern = re.compile(url_pattern)
            urls_to_fetch = [u for u in urls_to_fetch if pattern.search(u)]

        # Also filter to only include URLs under the base_url path
        base_path = parsed_base.path.rstrip('/')
        if base_path:
            urls_to_fetch = [u for u in urls_to_fetch if base_path in u]

        # Deduplicate and limit
        urls_to_fetch = list(dict.fromkeys(urls_to_fetch))[:max_pages]

        # Fetch all pages
        documents: list[Document] = []
        for i, url in enumerate(urls_to_fetch):
            if progress_callback:
                progress_callback(url, i + 1, len(urls_to_fetch))

            try:
                doc = self.fetch_webpage(url)
                # Skip empty or very short pages
                if len(doc.content.strip()) > 100:
                    documents.append(doc)
            except Exception:
                continue

        return documents

    def _crawl_links(
        self,
        start_url: str,
        base_domain: str,
        max_urls: int,
    ) -> list[str]:
        """Crawl a website by following links."""
        visited: set[str] = set()
        to_visit: list[str] = [start_url]
        found_urls: list[str] = []

        while to_visit and len(found_urls) < max_urls:
            url = to_visit.pop(0)

            if url in visited:
                continue

            visited.add(url)

            try:
                response = self.client.get(url)
                response.raise_for_status()
            except Exception:
                continue

            found_urls.append(url)

            # Parse and find links
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]

                # Convert relative URLs to absolute
                if href.startswith("/"):
                    href = urljoin(base_domain, href)
                elif not href.startswith("http"):
                    href = urljoin(url, href)

                # Only follow links on the same domain
                if not href.startswith(base_domain):
                    continue

                # Skip anchors, query params variations, and non-HTML
                href = href.split("#")[0].split("?")[0]

                # Skip common non-content paths
                skip_patterns = [
                    r'\.(css|js|png|jpg|jpeg|gif|svg|ico|pdf|zip|tar|gz)$',
                    r'/api/',
                    r'/static/',
                    r'/assets/',
                    r'/_next/',
                ]
                if any(re.search(p, href, re.I) for p in skip_patterns):
                    continue

                if href not in visited and href not in to_visit:
                    to_visit.append(href)

        return found_urls

    def fetch_confluence(self, url: str) -> Document:
        """
        Fetch content from a Confluence page.

        Requires CONFLUENCE_EMAIL and CONFLUENCE_TOKEN env vars,
        or pass them to ContentFetcher constructor.
        """
        if not self.confluence_email or not self.confluence_token:
            raise ValueError(
                "Confluence authentication required. Set CONFLUENCE_EMAIL and "
                "CONFLUENCE_TOKEN environment variables, or pass them to ContentFetcher."
            )

        # Extract base URL and page ID from the Confluence URL
        base_url, page_id, space_key = self._parse_confluence_url(url)

        # Create auth header
        auth_string = f"{self.confluence_email}:{self.confluence_token}"
        auth_bytes = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_bytes}",
            "Accept": "application/json",
        }

        # Fetch page content via REST API
        api_url = f"{base_url}/rest/api/content/{page_id}"
        params = {"expand": "body.storage,version,space"}

        response = self.client.get(api_url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # Extract content and metadata
        title = data.get("title", "Untitled")
        storage_content = data.get("body", {}).get("storage", {}).get("value", "")
        space_name = data.get("space", {}).get("name", space_key)
        version = data.get("version", {}).get("number", 1)

        # Convert Confluence storage format (HTML-like) to markdown
        content = self._confluence_to_markdown(storage_content)

        return Document(
            id=self._generate_id(url),
            source=url,
            source_type=SourceType.CONFLUENCE,
            title=title,
            content=content,
            metadata={
                "confluence_page_id": page_id,
                "space_key": space_key,
                "space_name": space_name,
                "version": version,
                "base_url": base_url,
            },
        )

    def fetch_confluence_space(
        self,
        base_url: str,
        space_key: str,
        limit: int = 100,
    ) -> list[Document]:
        """
        Fetch all pages from a Confluence space.

        Args:
            base_url: Confluence base URL (e.g., https://company.atlassian.net/wiki)
            space_key: Space key (e.g., "DOCS", "ENG")
            limit: Maximum pages to fetch (default: 100)
        """
        if not self.confluence_email or not self.confluence_token:
            raise ValueError(
                "Confluence authentication required. Set CONFLUENCE_EMAIL and "
                "CONFLUENCE_TOKEN environment variables."
            )

        auth_string = f"{self.confluence_email}:{self.confluence_token}"
        auth_bytes = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_bytes}",
            "Accept": "application/json",
        }

        documents: list[Document] = []
        start = 0
        page_size = 25

        while len(documents) < limit:
            api_url = f"{base_url}/rest/api/content"
            params = {
                "spaceKey": space_key,
                "type": "page",
                "expand": "body.storage,version",
                "start": start,
                "limit": min(page_size, limit - len(documents)),
            }

            response = self.client.get(api_url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            for page in results:
                title = page.get("title", "Untitled")
                page_id = page.get("id", "")
                storage_content = page.get("body", {}).get("storage", {}).get("value", "")
                version = page.get("version", {}).get("number", 1)

                content = self._confluence_to_markdown(storage_content)
                page_url = f"{base_url}/spaces/{space_key}/pages/{page_id}"

                doc = Document(
                    id=self._generate_id(page_url),
                    source=page_url,
                    source_type=SourceType.CONFLUENCE,
                    title=title,
                    content=content,
                    metadata={
                        "confluence_page_id": page_id,
                        "space_key": space_key,
                        "version": version,
                        "base_url": base_url,
                    },
                )
                documents.append(doc)

            start += len(results)

            # Check if there are more pages
            if len(results) < page_size:
                break

        return documents

    def _parse_confluence_url(self, url: str) -> tuple[str, str, str]:
        """Parse a Confluence URL to extract base URL, page ID, and space key."""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}/wiki"

        # Handle different Confluence URL formats
        path = parsed.path

        # Format: /wiki/spaces/SPACE/pages/123456/Page+Title
        space_match = re.search(r"/spaces/([^/]+)/pages/(\d+)", path)
        if space_match:
            return base_url, space_match.group(2), space_match.group(1)

        # Format: /wiki/pages/viewpage.action?pageId=123456
        if "pageId=" in parsed.query:
            page_id_match = re.search(r"pageId=(\d+)", parsed.query)
            if page_id_match:
                space_match = re.search(r"spaceKey=([^&]+)", parsed.query)
                space_key = space_match.group(1) if space_match else ""
                return base_url, page_id_match.group(1), space_key

        # Format: /confluence/display/SPACE/Page+Title (older format)
        if "/confluence/" in path:
            base_url = f"{parsed.scheme}://{parsed.netloc}/confluence"
            display_match = re.search(r"/display/([^/]+)/", path)
            if display_match:
                # Need to look up page ID by title - for now raise error
                raise ValueError(
                    f"Cannot extract page ID from URL: {url}. "
                    "Please use a URL with the page ID."
                )

        raise ValueError(f"Cannot parse Confluence URL: {url}")

    def _confluence_to_markdown(self, storage_content: str) -> str:
        """Convert Confluence storage format to markdown."""
        if not storage_content:
            return ""

        soup = BeautifulSoup(storage_content, "html.parser")

        # Handle Confluence-specific elements before converting

        # Convert Confluence code blocks
        for code_block in soup.find_all("ac:structured-macro", {"ac:name": "code"}):
            language = ""
            code_content = ""

            # Get language parameter
            lang_param = code_block.find("ac:parameter", {"ac:name": "language"})
            if lang_param:
                language = lang_param.get_text()

            # Get code content
            plain_text = code_block.find("ac:plain-text-body")
            if plain_text:
                code_content = plain_text.get_text()

            # Replace with markdown code block
            new_code = soup.new_tag("pre")
            new_code.string = f"```{language}\n{code_content}\n```"
            code_block.replace_with(new_code)

        # Convert Confluence info/warning/note panels
        for panel in soup.find_all("ac:structured-macro", {"ac:name": ["info", "warning", "note", "tip"]}):
            panel_type = panel.get("ac:name", "note")
            body = panel.find("ac:rich-text-body")
            if body:
                content = body.get_text()
                blockquote = soup.new_tag("blockquote")
                blockquote.string = f"**{panel_type.upper()}**: {content}"
                panel.replace_with(blockquote)

        # Convert Confluence status macros
        for status in soup.find_all("ac:structured-macro", {"ac:name": "status"}):
            title_param = status.find("ac:parameter", {"ac:name": "title"})
            if title_param:
                status_text = title_param.get_text()
                span = soup.new_tag("span")
                span.string = f"[{status_text}]"
                status.replace_with(span)

        # Remove other Confluence macros we can't convert
        for macro in soup.find_all("ac:structured-macro"):
            macro.decompose()

        # Remove Confluence-specific attributes
        for tag in soup.find_all(True):
            for attr in list(tag.attrs.keys()):
                if attr.startswith("ac:") or attr.startswith("ri:"):
                    del tag[attr]

        # Convert to markdown
        content = md(str(soup), heading_style="ATX", code_language="")

        return content

    def fetch_github_repo(
        self,
        repo: str,
        branch: str | None = None,
        path: str = "",
        extensions: list[str] | None = None,
    ) -> list[Document]:
        """
        Fetch all matching files from a GitHub repository using gitingest.

        Args:
            repo: Repository in format "owner/repo"
            branch: Branch name (default: auto-detect from repo's default branch)
            path: Path within the repo to start from
            extensions: List of file extensions to fetch (default: [".md", ".mdx"])
        
        Note:
            Set GITHUB_TOKEN env var for higher rate limits (5,000/hour vs 60/hour).
            gitingest handles authentication automatically from the environment.
        """
        if extensions is None:
            extensions = [".md", ".mdx"]
        
        # Build the GitHub URL for gitingest
        if path:
            # Use tree URL for subdirectory
            branch_ref = branch or "main"
            url = f"https://github.com/{repo}/tree/{branch_ref}/{path}"
        else:
            url = f"https://github.com/{repo}"
        
        # Convert extensions to include patterns (e.g., ["*.md", "*.mdx"])
        include_patterns = [f"*{ext}" for ext in extensions]
        
        try:
            summary, tree, content = ingest(
                url,
                include_patterns=include_patterns,
                # Exclude common non-documentation files
                exclude_patterns=[
                    "node_modules/*",
                    "*.lock",
                    "dist/*",
                    "build/*",
                    ".git/*",
                ],
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                raise ValueError(
                    "GitHub API rate limit exceeded. Set GITHUB_TOKEN for higher limits:\n"
                    "  1. Go to https://github.com/settings/tokens\n"
                    "  2. Generate a new token (classic) with 'public_repo' scope\n"
                    "  3. Add to .env: GITHUB_TOKEN=ghp_your_token_here\n\n"
                    "This increases rate limits from 60/hour to 5,000/hour."
                ) from e
            elif "not found" in error_msg or "404" in error_msg:
                raise ValueError(f"Repository not found: {repo}") from e
            raise
        
        # Parse gitingest content into individual documents
        documents = self._parse_gitingest_content(content, repo, branch or "main")
        
        return documents

    def _parse_gitingest_content(
        self,
        content: str,
        repo: str,
        branch: str,
    ) -> list[Document]:
        """Parse gitingest content output into individual Document objects."""
        documents: list[Document] = []
        
        # gitingest format:
        # ================================================
        # FILE: path/to/file.md
        # ================================================
        # <file content>
        
        file_separator = "=" * 48
        parts = content.split(file_separator)
        
        current_path = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check if this is a FILE: header
            if part.startswith("FILE:"):
                current_path = part.replace("FILE:", "").strip()
            elif current_path:
                # This is file content
                file_content = part.strip()
                if file_content:
                    source = f"https://github.com/{repo}/blob/{branch}/{current_path}"
                    title = self._extract_markdown_title(file_content) or Path(current_path).stem
                    
                    doc = Document(
                        id=self._generate_id(source),
                        source=source,
                        source_type=SourceType.GITHUB,
                        title=title,
                        content=file_content,
                        metadata={
                            "repo": repo,
                            "branch": branch,
                            "path": current_path,
                        },
                    )
                    documents.append(doc)
                current_path = None
        
        return documents

    def _convert_to_raw_github_url(self, url: str) -> str:
        """Convert a GitHub URL to its raw content URL."""
        if "raw.githubusercontent.com" in url:
            return url

        # Handle github.com URLs
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 4:
            raise ValueError(f"Invalid GitHub URL: {url}")

        owner, repo = path_parts[0], path_parts[1]

        # Handle blob URLs
        if "blob" in path_parts:
            blob_idx = path_parts.index("blob")
            branch = path_parts[blob_idx + 1]
            file_path = "/".join(path_parts[blob_idx + 2 :])
        else:
            # Assume it's a direct file reference
            branch = path_parts[2] if len(path_parts) > 2 else "main"
            file_path = "/".join(path_parts[3:]) if len(path_parts) > 3 else ""

        return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    def _extract_repo_info(self, url: str) -> str:
        """Extract owner/repo from a GitHub URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        return ""

    def _extract_markdown_title(self, content: str) -> str | None:
        """Extract the first H1 heading from markdown content."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _title_from_url(self, url: str) -> str:
        """Generate a title from a URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if path:
            # Get the last meaningful part of the path
            parts = [p for p in path.split("/") if p]
            if parts:
                return parts[-1].replace("-", " ").replace("_", " ").title()
        return parsed.netloc

