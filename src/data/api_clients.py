"""
Free Academic API Clients for Research Paper Discovery
Handles ArXiv, PubMed, and bioRxiv with rate limiting and caching
"""

import requests
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any, Any, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
from urllib.parse import quote_plus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """Standardized research paper data structure"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    categories: List[str]
    url: str
    source: str  # 'arxiv', 'pubmed', 'biorxiv'
    doi: Optional[str] = None
    journal: Optional[str] = None
    citations: Optional[int] = None

class CacheManager:
    """Simple file-based cache for API responses"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, source: str) -> str:
        """Generate cache key from query and source"""
        combined = f"{source}_{query}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, source: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Get cached response if it exists and is not expired"""
        cache_key = self._get_cache_key(query, source)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        # Check if cache is expired
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for query: {query}")
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                logger.info(f"Cache hit for query: {query}")
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, query: str, source: str, data: Dict) -> None:
        """Cache the response data"""
        cache_key = self._get_cache_key(query, source)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached response for query: {query}")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last_call < min_interval:
            wait_time = min_interval - time_since_last_call
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        self.last_call_time = time.time()

class ArXivClient:
    """
    ArXiv API client with rate limiting and caching
    Free API with no authentication required
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.base_url = "http://export.arxiv.org/api/query"
        self.cache = cache_manager
        self.rate_limiter = RateLimiter(calls_per_second=0.5)  # Conservative rate limit
    
    def search_papers(self, query: str, max_results: int = 20, category: str = None) -> List[ResearchPaper]:
        """
        Search ArXiv papers by query
        
        Args:
            query: Search term (e.g., "deep learning cancer detection")
            max_results: Maximum number of papers to return
            category: ArXiv category filter (e.g., "cs.CV", "cs.LG")
        """
        # Check cache first
        cache_key = f"{query}_{max_results}_{category}"
        cached_result = self.cache.get(cache_key, "arxiv")
        if cached_result:
            return [ResearchPaper(**paper) for paper in cached_result]
        
        # Build search query
        search_query = f"all:{quote_plus(query)}"
        if category:
            search_query += f"+AND+cat:{category}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            logger.info(f"Searching ArXiv for: {query}")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.text)
            
            # Cache the results
            papers_dict = [paper.__dict__ for paper in papers]
            self.cache.set(cache_key, "arxiv", papers_dict)
            
            logger.info(f"Found {len(papers)} papers from ArXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ResearchPaper]:
        """Parse ArXiv XML response into ResearchPaper objects"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', namespace):
                # Extract paper information
                paper_id = entry.find('atom:id', namespace).text.split('/')[-1]
                title = entry.find('atom:title', namespace).text.strip()
                abstract = entry.find('atom:summary', namespace).text.strip()
                published = entry.find('atom:published', namespace).text
                
                # Extract authors
                authors = []
                for author in entry.findall('atom:author', namespace):
                    name = author.find('atom:name', namespace).text
                    authors.append(name)
                
                # Extract categories
                categories = []
                for category in entry.findall('atom:category', namespace):
                    categories.append(category.get('term'))
                
                # Get URL
                pdf_url = None
                for link in entry.findall('atom:link', namespace):
                    if link.get('title') == 'pdf':
                        pdf_url = link.get('href')
                        break
                
                paper = ResearchPaper(
                    id=paper_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_date=published,
                    categories=categories,
                    url=pdf_url or f"https://arxiv.org/abs/{paper_id}",
                    source="arxiv"
                )
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error parsing ArXiv response: {e}")
        
        return papers

class PubMedClient:
    """
    PubMed API client for medical/life science papers
    Free API with abstracts (full papers require subscription)
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.cache = cache_manager
        self.rate_limiter = RateLimiter(calls_per_second=0.33)  # 3 requests per second max
    
    def search_papers(self, query: str, max_results: int = 20) -> List[ResearchPaper]:
        """
        Search PubMed papers by query
        
        Args:
            query: Search term (e.g., "machine learning medical imaging")
            max_results: Maximum number of papers to return
        """
        # Check cache first
        cache_key = f"{query}_{max_results}"
        cached_result = self.cache.get(cache_key, "pubmed")
        if cached_result:
            return [ResearchPaper(**paper) for paper in cached_result]
        
        try:
            # Step 1: Search for paper IDs
            paper_ids = self._search_pubmed_ids(query, max_results)
            if not paper_ids:
                return []
            
            # Step 2: Fetch paper details
            papers = self._fetch_paper_details(paper_ids)
            
            # Cache the results
            papers_dict = [paper.__dict__ for paper in papers]
            self.cache.set(cache_key, "pubmed", papers_dict)
            
            logger.info(f"Found {len(papers)} papers from PubMed")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def _search_pubmed_ids(self, query: str, max_results: int) -> List[str]:
        """Search PubMed for paper IDs"""
        self.rate_limiter.wait_if_needed()
        
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        response = requests.get(search_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('esearchresult', {}).get('idlist', [])
    
    def _fetch_paper_details(self, paper_ids: List[str]) -> List[ResearchPaper]:
        """Fetch detailed information for paper IDs"""
        if not paper_ids:
            return []
        
        self.rate_limiter.wait_if_needed()
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(paper_ids),
            'retmode': 'xml'
        }
        
        response = requests.get(fetch_url, params=params, timeout=30)
        response.raise_for_status()
        
        return self._parse_pubmed_response(response.text)
    
    def _parse_pubmed_response(self, xml_content: str) -> List[ResearchPaper]:
        """Parse PubMed XML response into ResearchPaper objects"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                # Extract basic info
                pmid = article.find('.//PMID').text
                
                # Title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else "No title available"
                
                # Abstract
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                
                # Authors
                authors = []
                for author in article.findall('.//Author'):
                    lastname = author.find('.//LastName')
                    firstname = author.find('.//ForeName')
                    if lastname is not None and firstname is not None:
                        authors.append(f"{firstname.text} {lastname.text}")
                
                # Publication date
                pub_date = article.find('.//PubDate')
                year = pub_date.find('.//Year')
                year_text = year.text if year is not None else "Unknown"
                
                # Journal
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
                
                paper = ResearchPaper(
                    id=pmid,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_date=year_text,
                    categories=["medical"],  # PubMed is medical focused
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source="pubmed",
                    journal=journal
                )
                papers.append(paper)
                
        except Exception as e:
            logger.error(f"Error parsing PubMed response: {e}")
        
        return papers

class BioRxivClient:
    """
    bioRxiv API client for biology preprints
    Free API for preprint papers
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.base_url = "https://api.biorxiv.org/details"
        self.cache = cache_manager
        self.rate_limiter = RateLimiter(calls_per_second=1.0)
    
    def search_papers(self, query: str, max_results: int = 20) -> List[ResearchPaper]:
        """
        Search bioRxiv papers (simplified - they don't have great search API)
        This is a basic implementation that searches recent papers
        """
        # Check cache first
        cache_key = f"{query}_{max_results}"
        cached_result = self.cache.get(cache_key, "biorxiv")
        if cached_result:
            return [ResearchPaper(**paper) for paper in cached_result]
        
        try:
            # bioRxiv API is limited, so we'll get recent papers and filter
            papers = self._fetch_recent_papers(max_results * 2)  # Fetch more to filter
            
            # Simple keyword filtering
            query_words = query.lower().split()
            filtered_papers = []
            
            for paper in papers:
                paper_text = f"{paper.title} {paper.abstract}".lower()
                if any(word in paper_text for word in query_words):
                    filtered_papers.append(paper)
                    if len(filtered_papers) >= max_results:
                        break
            
            # Cache the results
            papers_dict = [paper.__dict__ for paper in filtered_papers]
            self.cache.set(cache_key, "biorxiv", papers_dict)
            
            logger.info(f"Found {len(filtered_papers)} papers from bioRxiv")
            return filtered_papers
            
        except Exception as e:
            logger.error(f"Error searching bioRxiv: {e}")
            return []
    
    def _fetch_recent_papers(self, count: int = 50) -> List[ResearchPaper]:
        """Fetch recent papers from bioRxiv"""
        self.rate_limiter.wait_if_needed()
        
        # Get recent papers (last 30 days)
        import datetime
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/biorxiv/{start_date}/{end_date}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('collection', [])[:count]:
                paper = ResearchPaper(
                    id=item.get('doi', ''),
                    title=item.get('title', ''),
                    authors=item.get('authors', '').split(';'),
                    abstract=item.get('abstract', ''),
                    published_date=item.get('date', ''),
                    categories=['biology'],
                    url=f"https://www.biorxiv.org/content/{item.get('doi', '')}",
                    source="biorxiv",
                    doi=item.get('doi', '')
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching bioRxiv papers: {e}")
            return []

class AcademicAPIManager:
    """
    Unified manager for all academic API clients
    Provides single interface for multi-source paper search
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache = CacheManager(cache_dir)
        self.arxiv = ArXivClient(self.cache)
        self.pubmed = PubMedClient(self.cache)
        self.biorxiv = BioRxivClient(self.cache)
    
    def search_all_sources(self, query: str, max_results_per_source: int = 10) -> Dict[str, List[ResearchPaper]]:
        """
        Search all available sources for papers
        
        Returns:
            Dictionary with source names as keys and paper lists as values
        """
        results = {}
        
        # Search ArXiv
        try:
            results['arxiv'] = self.arxiv.search_papers(query, max_results_per_source)
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            results['arxiv'] = []
        
        # Search PubMed
        try:
            results['pubmed'] = self.pubmed.search_papers(query, max_results_per_source)
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            results['pubmed'] = []
        
        # Search bioRxiv
        try:
            results['biorxiv'] = self.biorxiv.search_papers(query, max_results_per_source)
        except Exception as e:
            logger.error(f"bioRxiv search failed: {e}")
            results['biorxiv'] = []
        
        return results
    
    def search_by_domain(self, query: str, domain: str, max_results: int = 20) -> List[ResearchPaper]:
        """
        Search specific domain/source
        
        Args:
            query: Search query
            domain: 'computer_science', 'medical', 'biology', etc.
            max_results: Maximum papers to return
        """
        if domain in ['computer_science', 'ai', 'machine_learning']:
            return self.arxiv.search_papers(query, max_results, category='cs.LG')
        elif domain in ['medical', 'medicine', 'clinical']:
            return self.pubmed.search_papers(query, max_results)
        elif domain in ['biology', 'life_sciences']:
            return self.biorxiv.search_papers(query, max_results)
        else:
            # Default: search ArXiv without category filter
            return self.arxiv.search_papers(query, max_results)

# Example usage and testing
if __name__ == "__main__":
    # Test the API clients
    api_manager = AcademicAPIManager()
    
    # Test search
    query = "deep learning cancer detection"
    print(f"Searching for: {query}")
    
    results = api_manager.search_all_sources(query, max_results_per_source=5)
    
    for source, papers in results.items():
        print(f"\n{source.upper()} Results ({len(papers)} papers):")
        for paper in papers[:3]:  # Show first 3
            print(f"  - {paper.title}")
            print(f"    Authors: {', '.join(paper.authors[:3])}")
            print(f"    Abstract: {paper.abstract[:100]}...")
            print()