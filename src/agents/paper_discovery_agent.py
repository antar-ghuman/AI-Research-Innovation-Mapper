"""
Paper Discovery Agent - Specialized AI agent for finding relevant research papers
Uses LangChain framework with free LLM (Groq) and vector search capabilities
OPTIMIZED VERSION with rate limiting and caching
"""

import os
import sys
from typing import List, Dict, Optional, Any, Type
import logging
from dataclasses import dataclass
import json
import time
import hashlib
import random

# LangChain imports
from langchain.tools import BaseTool
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
# from langchain.schema import AgentAction, AgentFinish
from langchain_core.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field

# Free LLM import
try:
    from groq import Groq
except ImportError:
    print("Warning: Groq not installed. Install with: pip install groq")
    Groq = None

# Import our components
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from data.api_clients import AcademicAPIManager, ResearchPaper
    from vector_store.chroma_client import ChromaVectorStore, SimilarityResult
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the correct directory and install dependencies")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaperDiscoveryResult:
    """Result from paper discovery agent"""
    query: str
    papers_found: List[ResearchPaper]
    similar_papers: List[SimilarityResult]
    domain_insights: Dict[str, Any]
    research_gaps: List[str]
    agent_reasoning: List[str]

class GroqLLMWrapper:
    """Enhanced Groq LLM wrapper with aggressive rate limiting and caching"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        if not Groq:
            raise ImportError("Groq package not installed")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        
        # Rate limiting
        self.last_call_time = 0
        self.min_interval = 3.0  # 3 seconds between calls (20 calls/minute max)
        self.call_count = 0
        self.hour_start = time.time()
        
        # Simple in-memory cache
        self.cache: Dict[str, str] = {}
        self.max_cache_size = 100
    
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for prompt"""
        cache_data = {
            "prompt": prompt[:500],  # Only first 500 chars for key
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 500)
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke with aggressive rate limiting and caching"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(prompt, **kwargs)
            if cache_key in self.cache:
                logger.info("Using cached response")
                return self.cache[cache_key]
            
            # Rate limiting
            current_time = time.time()
            
            # Reset hourly counter
            if current_time - self.hour_start > 3600:
                self.call_count = 0
                self.hour_start = current_time
            
            # Check hourly limit (max 30 calls per hour)
            if self.call_count >= 30:
                logger.warning("Hourly API limit reached, returning fallback response")
                return self._get_fallback_response(prompt)
            
            # Enforce minimum interval between calls
            time_since_last = current_time - self.last_call_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last + random.uniform(0.1, 0.5)
                logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            # Make API call
            self.call_count += 1
            self.last_call_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt[:1500]}],  # Truncate long prompts
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 400),  # Reduced token limit
                timeout=20
            )
            
            result = response.choices[0].message.content
            
            # Cache the response
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.cache.keys())[:20]
                for key in oldest_keys:
                    del self.cache[key]
            
            self.cache[cache_key] = result
            logger.info(f"API call successful (count: {self.call_count}/30)")
            return result
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Provide intelligent fallback when API is unavailable"""
        prompt_lower = prompt.lower()
        
        # Gap analysis fallback
        if "gap" in prompt_lower or ("research" in prompt_lower and "context" in prompt_lower):
            return """
            Research Gaps Identified:
            1. Limited interdisciplinary collaboration in this research area
            2. Underexplored technique transfer opportunities from related domains
            3. Need for domain-specific adaptation methods and validation studies
            
            Potential Research Opportunities:
            1. Cross-domain methodology development
            2. Comparative analysis of existing approaches
            3. Integration of complementary techniques from related fields
            """
        
        return "Analysis completed with limited API access. Please try again later for detailed insights."
    
    def __call__(self, prompt: str, **kwargs) -> str:
        return self.invoke(prompt, **kwargs)

# Input schemas for tools
class PaperSearchInput(BaseModel):
    query: str = Field(description="Search query for academic papers")

class SimilaritySearchInput(BaseModel):
    query: str = Field(description="Query for semantic similarity search")

class DomainAnalysisInput(BaseModel):
    query: str = Field(description="Research area or query to analyze")

class GapAnalysisInput(BaseModel):
    context: str = Field(description="Research context for gap analysis")

class PaperSearchTool(BaseTool):
    """LangChain tool for searching academic papers"""
    
    name: str = "paper_search"
    description: str = "Search for academic papers using query. Input should be a search query string."
    args_schema: Type[BaseModel] = PaperSearchInput
    
    # Store the API manager as a class variable that won't conflict with Pydantic
    _api_manager: Optional[AcademicAPIManager] = None
    
    def __init__(self, api_manager: AcademicAPIManager, **kwargs):
        super().__init__(**kwargs)
        PaperSearchTool._api_manager = api_manager
    
    def _run(self, query: str) -> str:
        """Search for papers and return formatted results"""
        try:
            if not PaperSearchTool._api_manager:
                return "Error: API manager not initialized"
                
            results = PaperSearchTool._api_manager.search_all_sources(query, max_results_per_source=5)
            
            formatted_results = []
            for source, papers in results.items():
                if papers:
                    formatted_results.append(f"\n{source.upper()} Results:")
                    for paper in papers[:3]:  # Limit for readability
                        formatted_results.append(f"- {paper.title}")
                        formatted_results.append(f"  Authors: {', '.join(paper.authors[:2])}")
                        formatted_results.append(f"  Abstract: {paper.abstract[:150]}...")
            
            return "\n".join(formatted_results) if formatted_results else "No papers found"
            
        except Exception as e:
            return f"Error searching papers: {str(e)}"

class SimilaritySearchTool(BaseTool):
    """LangChain tool for semantic similarity search"""
    
    name: str = "similarity_search"
    description: str = "Find papers similar to a query using semantic search. Input should be a research description."
    args_schema: Type[BaseModel] = SimilaritySearchInput
    
    _vector_store: Optional[ChromaVectorStore] = None
    
    def __init__(self, vector_store: ChromaVectorStore, **kwargs):
        super().__init__(**kwargs)
        SimilaritySearchTool._vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Perform similarity search and return results"""
        try:
            if not SimilaritySearchTool._vector_store:
                return "Error: Vector store not initialized"
                
            results = SimilaritySearchTool._vector_store.search_similar_papers(query, n_results=5)
            
            if not results:
                return "No similar papers found in vector database"
            
            formatted_results = ["Similar Papers Found:"]
            for result in results:
                formatted_results.append(f"- {result.paper.title}")
                formatted_results.append(f"  Similarity: {result.similarity_score:.3f}")
                formatted_results.append(f"  Domain: {result.paper.categories}")
                formatted_results.append(f"  Cross-domain: {result.cross_domain}")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error in similarity search: {str(e)}"

class DomainAnalysisTool(BaseTool):
    """LangChain tool for analyzing research domains"""
    
    name: str = "domain_analysis"
    description: str = "Analyze the research domain and identify key characteristics. Input should be a research area or query."
    args_schema: Type[BaseModel] = DomainAnalysisInput
    
    _vector_store: Optional[ChromaVectorStore] = None
    
    def __init__(self, vector_store: ChromaVectorStore, **kwargs):
        super().__init__(**kwargs)
        DomainAnalysisTool._vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Analyze domain characteristics"""
        try:
            if not DomainAnalysisTool._vector_store:
                return "Error: Vector store not initialized"
                
            # Get vector store statistics
            stats = DomainAnalysisTool._vector_store.get_collection_stats()
            
            # Infer domain from query
            domain = self._infer_domain(query)
            
            analysis = [
                f"Domain Analysis for: {query}",
                f"Inferred domain: {domain}",
                f"Available papers by domain: {stats.get('papers_by_domain', {})}",
                f"Available techniques by domain: {stats.get('techniques_by_domain', {})}"
            ]
            
            # Add domain-specific insights
            if domain in stats.get('papers_by_domain', {}):
                paper_count = stats['papers_by_domain'][domain]
                analysis.append(f"Found {paper_count} papers in {domain} domain")
            else:
                analysis.append(f"Limited data available for {domain} domain")
            
            return "\n".join(analysis)
            
        except Exception as e:
            return f"Error in domain analysis: {str(e)}"
    
    def _infer_domain(self, query: str) -> str:
        """Simple domain inference from query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["medical", "clinical", "patient", "disease"]):
            return "medical"
        elif any(term in query_lower for term in ["computer", "ai", "machine learning", "neural"]):
            return "computer_science"
        elif any(term in query_lower for term in ["biology", "genetic", "protein", "cell"]):
            return "biology"
        elif any(term in query_lower for term in ["astronomy", "star", "galaxy", "cosmic"]):
            return "astronomy"
        elif any(term in query_lower for term in ["chemistry", "chemical", "molecule"]):
            return "chemistry"
        else:
            return "general"

class ResearchGapAnalysisTool(BaseTool):
    """LangChain tool for identifying research gaps"""
    
    name: str = "gap_analysis"
    description: str = "Identify potential research gaps and opportunities. Input should be a research area description."
    args_schema: Type[BaseModel] = GapAnalysisInput
    
    _llm: Optional[GroqLLMWrapper] = None
    
    def __init__(self, llm: GroqLLMWrapper, **kwargs):
        super().__init__(**kwargs)
        ResearchGapAnalysisTool._llm = llm
    
    def _run(self, context: str) -> str:
        """Analyze research gaps using LLM with rate limiting"""
        try:
            if not ResearchGapAnalysisTool._llm:
                return "Error: LLM not initialized"
            
            # Simplified gap analysis prompt to reduce token usage
            gap_analysis_prompt = f"""
            Research context: {context[:800]}
            
            Identify 3 key research gaps and opportunities. Be concise.
            Format as:
            1. Gap name: Brief description
            2. Gap name: Brief description
            3. Gap name: Brief description
            """
            
            response = ResearchGapAnalysisTool._llm.invoke(gap_analysis_prompt, max_tokens=300)
            return response
            
        except Exception as e:
            return f"Error in gap analysis: {str(e)}"

class PaperDiscoveryAgent:
    """
    Optimized Paper Discovery Agent with rate limiting and better search
    """
    
    def __init__(self, groq_api_key: str, vector_store: ChromaVectorStore = None, api_manager: AcademicAPIManager = None):
        """
        Initialize the Paper Discovery Agent
        
        Args:
            groq_api_key: API key for Groq LLM
            vector_store: ChromaDB vector store instance
            api_manager: Academic API manager instance
        """
        # Initialize LLM
        self.llm = GroqLLMWrapper(groq_api_key)
        
        # Initialize components
        self.vector_store = vector_store or ChromaVectorStore()
        self.api_manager = api_manager or AcademicAPIManager()
        
        # Initialize tools
        self.paper_search_tool = PaperSearchTool(self.api_manager)
        self.similarity_tool = SimilaritySearchTool(self.vector_store)
        self.domain_tool = DomainAnalysisTool(self.vector_store)
        self.gap_tool = ResearchGapAnalysisTool(self.llm)
        
        logger.info("Paper Discovery Agent initialized with optimized rate limiting")
    
    def discover_papers(self, research_query: str, max_papers: int = 8) -> PaperDiscoveryResult:
        """
        Enhanced paper discovery with better search and minimal API usage
        
        Args:
            research_query: The research question or topic
            max_papers: Maximum number of papers to discover
            
        Returns:
            PaperDiscoveryResult with discovered papers and analysis
        """
        logger.info(f"Starting optimized paper discovery for: {research_query}")
        
        reasoning_steps = []
        
        try:
            # Step 1: Enhanced search with multiple query variants
            reasoning_steps.append("Generating enhanced search queries...")
            search_variants = self._generate_search_variants(research_query)
            
            all_papers = []
            for i, search_term in enumerate(search_variants[:3]):  # Try max 3 variants
                try:
                    reasoning_steps.append(f"Searching with variant {i+1}: '{search_term}'")
                    results = self.api_manager.search_all_sources(search_term, max_results_per_source=3)
                    
                    for source, papers in results.items():
                        all_papers.extend(papers)
                        
                except Exception as e:
                    logger.warning(f"Search failed for '{search_term}': {e}")
                    continue
            
            # Step 2: Filter and deduplicate papers
            reasoning_steps.append("Filtering and deduplicating results...")
            filtered_papers = self._filter_relevant_papers(all_papers, research_query)
            
            # Step 3: Add papers to vector store if we found new ones
            if filtered_papers:
                reasoning_steps.append("Adding new papers to vector database...")
                self.vector_store.add_papers(filtered_papers)
            
            # Step 4: Semantic similarity search (no API call)
            reasoning_steps.append("Performing semantic similarity search...")
            similar_papers = self.vector_store.search_similar_papers(research_query, n_results=5)
            
            # Step 5: Simple domain analysis (no API call)
            reasoning_steps.append("Analyzing research domain...")
            domain_stats = self.vector_store.get_collection_stats()
            
            # Step 6: Research gap analysis (ONE API call only)
            reasoning_steps.append("Identifying research gaps...")
            context = f"Research query: {research_query}\nFound {len(filtered_papers)} papers\nDomains: {domain_stats.get('papers_by_domain', {})}"
            gap_analysis_result = self.gap_tool._run(context)
            gap_analysis = self._parse_gap_analysis(gap_analysis_result)
            
            # Step 7: Compile results
            result = PaperDiscoveryResult(
                query=research_query,
                papers_found=filtered_papers[:max_papers],
                similar_papers=similar_papers,
                domain_insights={
                    "total_papers": domain_stats.get("total_papers", 0),
                    "papers_by_domain": domain_stats.get("papers_by_domain", {}),
                    "techniques_available": domain_stats.get("total_techniques", 0),
                    "search_variants_tried": len(search_variants)
                },
                research_gaps=gap_analysis.get("gaps", []),
                agent_reasoning=reasoning_steps
            )
            
            logger.info(f"Paper discovery completed. Found {len(filtered_papers)} papers, {len(similar_papers)} similar papers")
            return result
            
        except Exception as e:
            logger.error(f"Error in paper discovery: {e}")
            return PaperDiscoveryResult(
                query=research_query,
                papers_found=[],
                similar_papers=[],
                domain_insights={},
                research_gaps=["Analysis failed due to system limitations"],
                agent_reasoning=[f"Error occurred: {str(e)}"]
            )
    
    def _generate_search_variants(self, query: str) -> List[str]:
        """Generate better search term variants"""
        base_terms = query.lower().split()
        
        variants = [
            query,  # Original query
        ]
        
        # Add domain-specific variants based on query content
        query_lower = query.lower()
        
        if "speech" in query_lower and "therapy" in query_lower:
            variants.extend([
                "speech language therapy technology",
                "speech disorder treatment AI",
                "language intervention therapy",
                "speech pathology computational methods"
            ])
        elif "medical" in query_lower and "ai" in query_lower:
            variants.extend([
                "artificial intelligence healthcare",
                "machine learning medical diagnosis",
                "AI clinical applications"
            ])
        elif "cancer" in query_lower and "detection" in query_lower:
            variants.extend([
                "cancer screening AI",
                "tumor detection machine learning",
                "oncology artificial intelligence"
            ])
        else:
            # Generic variants
            if len(base_terms) >= 3:
                variants.extend([
                    " ".join(base_terms[:3]),  # First 3 words
                    f"{base_terms[0]} {base_terms[-1]}",  # First and last word
                ])
        
        return variants[:4]  # Limit to 4 variants max
    
    def _filter_relevant_papers(self, papers: List[ResearchPaper], query: str) -> List[ResearchPaper]:
        """Filter papers for relevance and remove duplicates"""
        if not papers:
            return []
        
        # Remove duplicates by title
        seen_titles = set()
        unique_papers = []
        
        query_terms = set(query.lower().split())
        
        for paper in papers:
            # Skip duplicates
            if paper.title.lower() in seen_titles:
                continue
            seen_titles.add(paper.title.lower())
            
            # Basic relevance check
            paper_text = (paper.title + " " + paper.abstract).lower()
            paper_terms = set(paper_text.split())
            
            # Check for at least some overlap with query terms
            overlap = len(query_terms.intersection(paper_terms))
            if overlap > 0 or len(query_terms) == 1:  # Be lenient for single-term queries
                unique_papers.append(paper)
        
        # Sort by relevance (basic keyword matching)
        def relevance_score(paper):
            paper_text = (paper.title + " " + paper.abstract).lower()
            score = 0
            for term in query_terms:
                if term in paper_text:
                    score += paper_text.count(term)
            return score
        
        unique_papers.sort(key=relevance_score, reverse=True)
        return unique_papers
    
    def _parse_gap_analysis(self, gap_text: str) -> Dict[str, List[str]]:
        """Parse gap analysis text into structured format"""
        try:
            # Simple parsing of gap analysis
            lines = gap_text.split('\n')
            gaps = []
            
            for line in lines:
                line = line.strip()
                # Look for numbered items or bullet points
                if (line.startswith(('1.', '2.', '3.', '-', '•')) or 
                    'gap' in line.lower() and len(line) > 10):
                    # Clean up the line
                    clean_line = line
                    for prefix in ['1.', '2.', '3.', '-', '•']:
                        if clean_line.startswith(prefix):
                            clean_line = clean_line[len(prefix):].strip()
                            break
                    
                    if len(clean_line) > 5:
                        gaps.append(clean_line)
            
            # Fallback gaps if parsing failed
            if not gaps:
                gaps = [
                    "Limited interdisciplinary research in this area",
                    "Underexplored technique transfer opportunities", 
                    "Need for domain-specific validation studies"
                ]
            
            return {"gaps": gaps[:5]}  # Limit to 5 gaps
                
        except Exception as e:
            logger.error(f"Error parsing gap analysis: {e}")
            return {"gaps": ["Gap analysis parsing failed"]}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information about the agent"""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "agent_type": "Paper Discovery Agent",
            "llm_model": self.llm.model,
            "api_calls_this_hour": self.llm.call_count,
            "cache_size": len(self.llm.cache),
            "tools_available": ["paper_search", "similarity_search", "domain_analysis", "gap_analysis"],
            "vector_store_papers": vector_stats.get("total_papers", 0),
            "vector_store_techniques": vector_stats.get("total_techniques", 0),
            "status": "ready with rate limiting"
        }

# Example usage and testing
def main():
    """Main function for running the Paper Discovery Agent standalone"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Please set GROQ_API_KEY environment variable")
        print("Get free API key from: https://console.groq.com/")
        exit(1)
    
    # Initialize agent
    print("Initializing Optimized Paper Discovery Agent...")
    agent = PaperDiscoveryAgent(groq_api_key)
    
    # Test discovery
    research_query = "speech therapy using AI technology"
    print(f"Testing paper discovery for: {research_query}")
    
    result = agent.discover_papers(research_query, max_papers=6)
    
    # Display results
    print(f"\n=== PAPER DISCOVERY RESULTS ===")
    print(f"Query: {result.query}")
    print(f"Papers found: {len(result.papers_found)}")
    print(f"Similar papers: {len(result.similar_papers)}")
    
    print(f"\n=== AGENT REASONING ===")
    for i, step in enumerate(result.agent_reasoning, 1):
        print(f"{i}. {step}")
    
    print(f"\n=== DISCOVERED PAPERS ===")
    for i, paper in enumerate(result.papers_found[:5], 1):
        print(f"{i}. {paper.title}")
        print(f"   Source: {paper.source}")
        print(f"   Authors: {', '.join(paper.authors[:2])}")
        print()
    
    print(f"\n=== RESEARCH GAPS IDENTIFIED ===")
    for gap in result.research_gaps:
        print(f"- {gap}")
    
    print(f"\n=== AGENT STATUS ===")
    status = agent.get_agent_status()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
