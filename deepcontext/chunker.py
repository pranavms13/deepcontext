"""Semantic chunking with intelligent metadata extraction."""

import hashlib
import re
from dataclasses import dataclass

from chonkie import SemanticChunker as ChonkieSemanticChunker

from deepcontext.models import Chunk, Document


@dataclass
class ChunkConfig:
    """Configuration for chunking."""

    max_chunk_size: int = 2000  # Max characters/tokens per chunk
    min_chunk_size: int = 200   # Min characters for a chunk to be meaningful
    overlap_size: int = 100     # Overlap between chunks for context
    threshold: float = 0.7      # Similarity threshold for semantic chunking (0-1)
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # Model for semantic embeddings (768 dims)


class DocumentChunker:
    """
    Smart document chunker that creates Context7-style chunks.
    
    Strategy:
    1. Keep code blocks together with their preceding context
    2. Create chunks around logical sections (headings)
    3. Generate descriptive titles and descriptions
    """

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Chunk a document into meaningful, self-contained pieces.
        """
        content = document.content
        doc_title = self._clean_doc_title(document.title)
        
        # Extract all sections with their code blocks
        sections = self._extract_sections_with_code(content)
        
        chunks: list[Chunk] = []
        
        for section in sections:
            # Skip very small sections or navigation-like content
            if len(section["content"]) < self.config.min_chunk_size:
                continue
            if self._is_navigation_content(section["content"]):
                continue
                
            # Create chunk(s) from this section
            section_chunks = self._create_section_chunks(
                section, document, doc_title
            )
            chunks.extend(section_chunks)
        
        return chunks

    def _clean_doc_title(self, title: str | None) -> str:
        """Clean up document title."""
        if not title:
            return "Document"
        # Remove common suffixes
        title = re.sub(r'\s*[-|–—]\s*(shadcn/ui|Docs|Documentation).*$', '', title, flags=re.I)
        return title.strip()

    def _extract_sections_with_code(self, content: str) -> list[dict]:
        """
        Extract logical sections, keeping code blocks with their context.
        """
        sections = []
        lines = content.split('\n')
        
        current_section = {
            "title": "Introduction",
            "level": 0,
            "lines": [],
            "code_blocks": [],
        }
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save current section if it has content
                if current_section["lines"] or current_section["code_blocks"]:
                    current_section["content"] = '\n'.join(current_section["lines"])
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": heading_match.group(2).strip(),
                    "level": len(heading_match.group(1)),
                    "lines": [line],
                    "code_blocks": [],
                }
                i += 1
                continue
            
            # Check for code block start
            if line.startswith('```'):
                language = line[3:].strip()
                code_lines = [line]
                i += 1
                
                # Collect code block
                while i < len(lines) and not lines[i].startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                
                if i < len(lines):
                    code_lines.append(lines[i])  # closing ```
                    i += 1
                
                code_content = '\n'.join(code_lines[1:-1]) if len(code_lines) > 2 else ""
                current_section["code_blocks"].append({
                    "language": language,
                    "code": code_content,
                    "full": '\n'.join(code_lines),
                })
                current_section["lines"].extend(code_lines)
                continue
            
            current_section["lines"].append(line)
            i += 1
        
        # Don't forget last section
        if current_section["lines"] or current_section["code_blocks"]:
            current_section["content"] = '\n'.join(current_section["lines"])
            sections.append(current_section)
        
        return sections

    def _is_navigation_content(self, content: str) -> bool:
        """Check if content is mostly navigation/links."""
        # Count links vs total content
        links = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
        link_text_len = sum(len(t) for t in links)
        clean_text = re.sub(r'\[([^\]]+)\]\([^)]+\)', '', content)
        clean_text = re.sub(r'[#*\-\n\s]', '', clean_text)
        
        if len(clean_text) < 50:
            return True
        if link_text_len > len(clean_text) * 0.5:
            return True
        return False

    def _create_section_chunks(
        self, 
        section: dict, 
        document: Document, 
        doc_title: str
    ) -> list[Chunk]:
        """Create chunks from a section."""
        chunks = []
        content = section["content"]
        section_title = section["title"]
        code_blocks = section["code_blocks"]
        
        # If section is small enough, make it one chunk
        if len(content) <= self.config.max_chunk_size:
            chunk = self._make_chunk(
                content=content,
                section_title=section_title,
                doc_title=doc_title,
                document=document,
                code_blocks=code_blocks,
                chunk_index=0,
            )
            if chunk:
                chunks.append(chunk)
        else:
            # Split into multiple chunks, trying to keep code blocks intact
            chunk_contents = self._split_section(content, code_blocks)
            for i, (chunk_content, chunk_codes) in enumerate(chunk_contents):
                chunk = self._make_chunk(
                    content=chunk_content,
                    section_title=section_title,
                    doc_title=doc_title,
                    document=document,
                    code_blocks=chunk_codes,
                    chunk_index=i,
                )
                if chunk:
                    chunks.append(chunk)
        
        return chunks

    def _split_section(
        self, 
        content: str, 
        code_blocks: list[dict]
    ) -> list[tuple[str, list[dict]]]:
        """Split a large section into smaller chunks."""
        # Simple strategy: split by paragraphs, keeping code blocks together
        result = []
        
        # Split by double newlines (paragraphs)
        parts = re.split(r'\n\n+', content)
        
        current_chunk = []
        current_codes = []
        current_size = 0
        
        for part in parts:
            part_size = len(part)
            
            # Check if this part contains a code block
            part_has_code = '```' in part
            
            # If adding this would exceed limit, save current and start new
            if current_size + part_size > self.config.max_chunk_size and current_chunk:
                result.append(('\n\n'.join(current_chunk), current_codes))
                current_chunk = []
                current_codes = []
                current_size = 0
            
            current_chunk.append(part)
            current_size += part_size
            
            # Track code blocks in this part
            if part_has_code:
                for cb in code_blocks:
                    if cb["full"] in part or cb["code"] in part:
                        if cb not in current_codes:
                            current_codes.append(cb)
        
        if current_chunk:
            result.append(('\n\n'.join(current_chunk), current_codes))
        
        return result

    def _make_chunk(
        self,
        content: str,
        section_title: str,
        doc_title: str,
        document: Document,
        code_blocks: list[dict],
        chunk_index: int,
    ) -> Chunk | None:
        """Create a Chunk object with good title and description."""
        # Skip if content is too small or just whitespace
        clean_content = content.strip()
        if len(clean_content) < self.config.min_chunk_size:
            return None
        
        # Generate title
        title = self._generate_title(section_title, doc_title, content, chunk_index)
        
        # Generate description
        description = self._generate_description(content, code_blocks)
        
        # Extract code as list of strings
        code_list = [cb["code"] for cb in code_blocks if cb["code"]]
        
        # Determine primary language
        languages = [cb["language"] for cb in code_blocks if cb["language"]]
        language = languages[0] if languages else None
        
        # Generate ID
        chunk_id = hashlib.sha256(
            f"{document.id}:{section_title}:{chunk_index}:{content[:100]}".encode()
        ).hexdigest()[:16]
        
        return Chunk(
            id=chunk_id,
            document_id=document.id,
            source=document.source,
            source_type=document.source_type,
            title=title,
            description=description,
            content=clean_content,
            code_blocks=code_list,
            language=language,
            token_count=len(content.split()),
            metadata={
                **document.metadata,
                "section_title": section_title,
                "document_title": doc_title,
            },
        )

    def _generate_title(
        self, 
        section_title: str, 
        doc_title: str, 
        content: str,
        chunk_index: int
    ) -> str:
        """Generate a descriptive title for the chunk."""
        title = section_title
        
        # Always include document title if it's different from section title
        # This ensures "Collapsed" becomes "Breadcrumb - Collapsed"
        # and "Installation" becomes "Button - Installation"
        if doc_title and doc_title.lower() != section_title.lower():
            # Check if doc_title is already in section_title
            if doc_title.lower() not in section_title.lower():
                title = f"{doc_title} - {section_title}"
        
        # Add part number if this is a split chunk
        if chunk_index > 0:
            title = f"{title} (Part {chunk_index + 1})"
        
        return title

    def _generate_description(self, content: str, code_blocks: list[dict]) -> str:
        """Generate a meaningful description for the chunk."""
        # Remove code blocks for text analysis
        text = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Get meaningful sentences
        sentences = re.split(r'[.!?]\s+', text)
        meaningful = [
            s.strip() for s in sentences 
            if len(s.strip()) > 20 
            and not s.strip().startswith('Copy')
            and not re.match(r'^[\*\-\|]', s.strip())
        ]
        
        if meaningful:
            # Take first 2-3 meaningful sentences
            desc = '. '.join(meaningful[:3])
            desc = re.sub(r'\s+', ' ', desc).strip()
            if not desc.endswith('.'):
                desc += '.'
            if len(desc) > 500:
                desc = desc[:497] + '...'
            return desc
        
        # Fall back to describing code
        if code_blocks:
            languages = list(set(cb["language"] for cb in code_blocks if cb["language"]))
            if languages:
                return f"Code example in {', '.join(languages)}."
            return f"Contains {len(code_blocks)} code example(s)."
        
        return "Documentation section."


class SemanticChunker:
    """
    Semantic chunker that uses chonkie's embedding-based chunking.
    
    Groups semantically similar text together using embeddings,
    which is more intelligent than simple paragraph/heading splitting.
    """

    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()
        self._chunker = ChonkieSemanticChunker(
            embedding_model=self.config.embedding_model,
            threshold=self.config.threshold,
            chunk_size=self.config.max_chunk_size,
            min_sentences_per_chunk=1,
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Chunk a document using semantic similarity.
        """
        content = document.content
        doc_title = self._clean_doc_title(document.title)
        
        # Use chonkie's semantic chunker
        chonkie_chunks = self._chunker.chunk(content)
        
        chunks: list[Chunk] = []
        for i, chonkie_chunk in enumerate(chonkie_chunks):
            chunk_text = chonkie_chunk.text.strip()
            
            # Skip very small chunks
            if len(chunk_text) < self.config.min_chunk_size:
                continue
            
            # Extract code blocks from this chunk
            code_blocks = self._extract_code_blocks(chunk_text)
            
            # Generate title from first heading or first line
            title = self._generate_title(chunk_text, doc_title, i)
            
            # Generate description
            description = self._generate_description(chunk_text, code_blocks)
            
            # Determine language from code blocks
            languages = [cb["language"] for cb in code_blocks if cb["language"]]
            language = languages[0] if languages else None
            
            # Generate ID
            chunk_id = hashlib.sha256(
                f"{document.id}:semantic:{i}:{chunk_text[:100]}".encode()
            ).hexdigest()[:16]
            
            chunks.append(Chunk(
                id=chunk_id,
                document_id=document.id,
                source=document.source,
                source_type=document.source_type,
                title=title,
                description=description,
                content=chunk_text,
                code_blocks=[cb["code"] for cb in code_blocks if cb["code"]],
                language=language,
                token_count=chonkie_chunk.token_count,
                metadata={
                    **document.metadata,
                    "document_title": doc_title,
                    "chunk_index": i,
                },
            ))
        
        return chunks

    def _clean_doc_title(self, title: str | None) -> str:
        """Clean up document title."""
        if not title:
            return "Document"
        title = re.sub(r'\s*[-|–—]\s*(shadcn/ui|Docs|Documentation).*$', '', title, flags=re.I)
        return title.strip()

    def _extract_code_blocks(self, content: str) -> list[dict]:
        """Extract code blocks from content."""
        code_blocks = []
        pattern = r'```(\w*)\n(.*?)```'
        for match in re.finditer(pattern, content, re.DOTALL):
            code_blocks.append({
                "language": match.group(1) or None,
                "code": match.group(2).strip(),
            })
        return code_blocks

    def _generate_title(self, content: str, doc_title: str, chunk_index: int) -> str:
        """Generate a title for the chunk."""
        # Try to find a heading in the content
        heading_match = re.search(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        if heading_match:
            section_title = heading_match.group(1).strip()
            if doc_title and doc_title.lower() != section_title.lower():
                if doc_title.lower() not in section_title.lower():
                    return f"{doc_title} - {section_title}"
            return section_title
        
        # Fall back to first meaningful line
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('```'):
                # Clean markdown formatting
                title = re.sub(r'[*_`#]', '', line)[:60]
                if doc_title:
                    return f"{doc_title} - {title}"
                return title
        
        return f"{doc_title} (Part {chunk_index + 1})"

    def _generate_description(self, content: str, code_blocks: list[dict]) -> str:
        """Generate a meaningful description for the chunk."""
        # Remove code blocks for text analysis
        text = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        # Remove markdown formatting
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Get meaningful sentences
        sentences = re.split(r'[.!?]\s+', text)
        meaningful = [
            s.strip() for s in sentences 
            if len(s.strip()) > 20 
            and not s.strip().startswith('Copy')
            and not re.match(r'^[\*\-\|]', s.strip())
        ]
        
        if meaningful:
            desc = '. '.join(meaningful[:3])
            desc = re.sub(r'\s+', ' ', desc).strip()
            if not desc.endswith('.'):
                desc += '.'
            if len(desc) > 500:
                desc = desc[:497] + '...'
            return desc
        
        if code_blocks:
            languages = list(set(cb["language"] for cb in code_blocks if cb["language"]))
            if languages:
                return f"Code example in {', '.join(languages)}."
            return f"Contains {len(code_blocks)} code example(s)."
        
        return "Documentation section."


# Keep backwards compatibility - MarkdownCodeChunker uses the heading-aware DocumentChunker
MarkdownCodeChunker = DocumentChunker
