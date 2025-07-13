"""
ChromaDB Vector Store Client for Research Paper Embeddings
Handles paper storage, cross-domain similarity search, and technique extraction
"""

import sys

# Fix for Streamlit Cloud SQLite issue
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

# Now import ChromaDB
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any, Any, Tuple, Any
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import json
import os
from datetime import datetime
import hashlib

# Import our data types
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from data.api_clients import ResearchPaper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    """Result from similarity search with metadata"""
    paper: ResearchPaper
    similarity_score: float
    cross_domain: bool
    technique_match: Optional[str] = None
    reasoning: Optional[str] = None

@dataclass
class TechniqueExtraction:
    """Extracted technique from a research paper"""
    technique_name: str
    description: str
    domain: str
    paper_id: str
    confidence_score: float
    context: str  # Surrounding text where technique was found

class EmbeddingGenerator:
    """Generates embeddings using free sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator with free model
        
        Args:
            model_name: HuggingFace model name (free models)
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def generate_paper_embedding(self, paper: ResearchPaper) -> np.ndarray:
        """Generate embedding for research paper"""
        # Combine title and abstract for better representation
        text = f"{paper.title}. {paper.abstract}"
        
        # Handle empty abstracts
        if not paper.abstract or paper.abstract.strip() == "":
            text = paper.title
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for paper {paper.id}: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for arbitrary text"""
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return np.zeros(self.embedding_dim)

class TechniqueExtractor:
    """Extracts research techniques and methods from papers"""
    
    def __init__(self):
        # Common technique keywords across domains
        self.technique_patterns = {
            "machine_learning": [
                "neural network", "deep learning", "convolutional", "transformer",
                "gradient descent", "backpropagation", "lstm", "gru", "attention",
                "reinforcement learning", "supervised learning", "unsupervised",
                "classification", "regression", "clustering", "dimensionality reduction"
            ],
            "image_processing": [
                "image segmentation", "feature extraction", "edge detection",
                "histogram equalization", "fourier transform", "wavelet transform",
                "morphological operations", "filter", "convolution", "gaussian blur"
            ],
            "signal_processing": [
                "fft", "frequency domain", "time series", "spectral analysis",
                "noise reduction", "digital filter", "sampling", "quantization"
            ],
            "optimization": [
                "genetic algorithm", "simulated annealing", "particle swarm",
                "hill climbing", "gradient ascent", "linear programming",
                "constraint optimization", "multi-objective optimization"
            ],
            "statistics": [
                "regression analysis", "hypothesis testing", "anova", "chi-square",
                "correlation", "bayesian", "monte carlo", "bootstrap", "cross-validation"
            ]
        }
    
    def extract_techniques(self, paper: ResearchPaper) -> List[TechniqueExtraction]:
        """Extract techniques mentioned in the paper"""
        text = f"{paper.title} {paper.abstract}".lower()
        techniques = []
        
        for domain, patterns in self.technique_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    # Find context around the technique
                    start_idx = text.find(pattern)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(text), start_idx + len(pattern) + 50)
                    context = text[context_start:context_end]
                    
                    # Calculate confidence based on context
                    confidence = self._calculate_confidence(pattern, context, paper)
                    
                    technique = TechniqueExtraction(
                        technique_name=pattern,
                        description=f"Technique found in {paper.source} paper",
                        domain=domain,
                        paper_id=paper.id,
                        confidence_score=confidence,
                        context=context
                    )
                    techniques.append(technique)
        
        return techniques
    
    def _calculate_confidence(self, technique: str, context: str, paper: ResearchPaper) -> float:
        """Calculate confidence score for technique extraction"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if technique is in title
        if technique in paper.title.lower():
            confidence += 0.3
        
        # Higher confidence if mentioned multiple times
        count = paper.abstract.lower().count(technique)
        confidence += min(0.2, count * 0.05)
        
        # Adjust based on paper source
        if paper.source == "arxiv":
            confidence += 0.1  # ArXiv papers are usually more technical
        
        return min(1.0, confidence)

class ChromaVectorStore:
    """ChromaDB vector store for research papers with cross-domain search"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Directory to persist vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize technique extractor
        self.technique_extractor = TechniqueExtractor()
        
        # Create collections for different purposes
        self.papers_collection = self._get_or_create_collection("research_papers")
        self.techniques_collection = self._get_or_create_collection("techniques")
        
        logger.info(f"ChromaDB initialized with {len(self.papers_collection.get()['ids'])} papers")
    
    def _get_or_create_collection(self, name: str):
        """Get existing collection or create new one"""
        try:
            return self.client.get_collection(name=name)
        except Exception:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
    
    def add_papers(self, papers: List[ResearchPaper]) -> None:
        """Add research papers to vector store"""
        if not papers:
            return
        
        logger.info(f"Adding {len(papers)} papers to vector store")
        
        # Prepare data for batch insertion
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for paper in papers:
            try:
                # Generate embedding
                embedding = self.embedding_generator.generate_paper_embedding(paper)
                
                # Prepare document text
                document = f"{paper.title}. {paper.abstract}"
                
                # Prepare metadata
                metadata = {
                    "title": paper.title,
                    "authors": json.dumps(paper.authors),
                    "source": paper.source,
                    "published_date": paper.published_date,
                    "categories": json.dumps(paper.categories),
                    "url": paper.url,
                    "domain": self._classify_domain(paper),
                    "added_date": datetime.now().isoformat()
                }
                
                # Add optional fields
                if paper.doi:
                    metadata["doi"] = paper.doi
                if paper.journal:
                    metadata["journal"] = paper.journal
                
                # Create unique ID
                paper_id = f"{paper.source}_{paper.id}"
                
                embeddings.append(embedding.tolist())
                documents.append(document)
                metadatas.append(metadata)
                ids.append(paper_id)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.id}: {e}")
                continue
        
        # Batch insert to ChromaDB
        if embeddings:
            try:
                self.papers_collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully added {len(embeddings)} papers")
                
                # Extract and store techniques
                self._extract_and_store_techniques(papers)
                
            except Exception as e:
                logger.error(f"Error adding papers to ChromaDB: {e}")
    
    def _classify_domain(self, paper: ResearchPaper) -> str:
        """Classify paper into research domain"""
        title_abstract = f"{paper.title} {paper.abstract}".lower()
        
        # Domain classification based on content
        if paper.source == "pubmed":
            return "medical"
        elif paper.source == "biorxiv":
            return "biology"
        elif any(cat.startswith("cs.") for cat in paper.categories):
            if any(cat in ["cs.CV", "cs.LG", "cs.AI"] for cat in paper.categories):
                return "computer_science"
            return "computer_science"
        elif any(term in title_abstract for term in ["astronomy", "astrophysics", "galaxy", "star"]):
            return "astronomy"
        elif any(term in title_abstract for term in ["chemistry", "chemical", "molecule"]):
            return "chemistry"
        elif any(term in title_abstract for term in ["physics", "quantum", "particle"]):
            return "physics"
        else:
            return "general"
    
    def _extract_and_store_techniques(self, papers: List[ResearchPaper]) -> None:
        """Extract techniques from papers and store them separately"""
        logger.info("Extracting techniques from papers")
        
        technique_embeddings = []
        technique_documents = []
        technique_metadatas = []
        technique_ids = []
        
        for paper in papers:
            techniques = self.technique_extractor.extract_techniques(paper)
            
            for technique in techniques:
                # Generate embedding for technique
                technique_text = f"{technique.technique_name}: {technique.description}"
                embedding = self.embedding_generator.generate_text_embedding(technique_text)
                
                # Create unique technique ID
                technique_id = hashlib.md5(
                    f"{technique.technique_name}_{technique.domain}_{paper.id}".encode()
                ).hexdigest()
                
                metadata = {
                    "technique_name": technique.technique_name,
                    "domain": technique.domain,
                    "paper_id": technique.paper_id,
                    "paper_source": paper.source,
                    "confidence_score": technique.confidence_score,
                    "context": technique.context,
                    "paper_title": paper.title
                }
                
                technique_embeddings.append(embedding.tolist())
                technique_documents.append(technique_text)
                technique_metadatas.append(metadata)
                technique_ids.append(technique_id)
        
        # Store techniques in separate collection
        if technique_embeddings:
            try:
                self.techniques_collection.add(
                    embeddings=technique_embeddings,
                    documents=technique_documents,
                    metadatas=technique_metadatas,
                    ids=technique_ids
                )
                logger.info(f"Stored {len(technique_embeddings)} techniques")
            except Exception as e:
                logger.error(f"Error storing techniques: {e}")
    
    def search_similar_papers(
        self, 
        query: str, 
        n_results: int = 10,
        domain_filter: Optional[str] = None,
        exclude_domain: Optional[str] = None
    ) -> List[SimilarityResult]:
        """
        Search for similar papers using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            domain_filter: Only return papers from this domain
            exclude_domain: Exclude papers from this domain
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_text_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if domain_filter:
                where_clause["domain"] = domain_filter
            elif exclude_domain:
                where_clause = {"domain": {"$ne": exclude_domain}}
            
            # Search ChromaDB
            results = self.papers_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Convert to SimilarityResult objects
            similarity_results = []
            
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Convert distance to similarity score (ChromaDB returns distances)
                similarity_score = 1.0 - distance
                
                # Create ResearchPaper object from metadata
                paper = ResearchPaper(
                    id=results['ids'][0][i].split('_', 1)[1],  # Remove source prefix
                    title=metadata['title'],
                    authors=json.loads(metadata['authors']),
                    abstract=results['documents'][0][i].split('. ', 1)[1] if '. ' in results['documents'][0][i] else "",
                    published_date=metadata['published_date'],
                    categories=json.loads(metadata['categories']),
                    url=metadata['url'],
                    source=metadata['source'],
                    doi=metadata.get('doi'),
                    journal=metadata.get('journal')
                )
                
                # Determine if this is cross-domain
                query_domain = self._infer_domain_from_query(query)
                is_cross_domain = query_domain != metadata['domain'] if query_domain else False
                
                result = SimilarityResult(
                    paper=paper,
                    similarity_score=similarity_score,
                    cross_domain=is_cross_domain,
                    reasoning=f"Similarity: {similarity_score:.3f}, Domain: {metadata['domain']}"
                )
                similarity_results.append(result)
            
            logger.info(f"Found {len(similarity_results)} similar papers")
            return similarity_results
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {e}")
            return []
    
    def find_cross_domain_techniques(
        self, 
        query: str, 
        target_domain: str,
        n_results: int = 10
    ) -> List[SimilarityResult]:
        """
        Find techniques from OTHER domains that might apply to the query
        
        Args:
            query: Research problem description
            target_domain: Domain of the research problem
            n_results: Number of cross-domain techniques to find
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_text_embedding(query)
            
            # Search techniques collection, excluding target domain
            results = self.techniques_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results * 2,  # Get more to filter out target domain
                where={"domain": {"$ne": target_domain}}
            )
            
            cross_domain_results = []
            
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                similarity_score = 1.0 - distance
                
                # Create dummy paper object for technique
                paper = ResearchPaper(
                    id=metadata['paper_id'],
                    title=metadata['paper_title'],
                    authors=[],
                    abstract=metadata['context'],
                    published_date="",
                    categories=[metadata['domain']],
                    url="",
                    source=metadata['paper_source']
                )
                
                result = SimilarityResult(
                    paper=paper,
                    similarity_score=similarity_score,
                    cross_domain=True,
                    technique_match=metadata['technique_name'],
                    reasoning=f"Technique '{metadata['technique_name']}' from {metadata['domain']} domain (confidence: {metadata['confidence_score']:.2f})"
                )
                
                cross_domain_results.append(result)
                
                if len(cross_domain_results) >= n_results:
                    break
            
            logger.info(f"Found {len(cross_domain_results)} cross-domain techniques")
            return cross_domain_results
            
        except Exception as e:
            logger.error(f"Error finding cross-domain techniques: {e}")
            return []
    
    def _infer_domain_from_query(self, query: str) -> Optional[str]:
        """Infer research domain from query text"""
        query_lower = query.lower()
        
        domain_keywords = {
            "computer_science": ["ai", "machine learning", "deep learning", "neural network", "algorithm"],
            "medical": ["medical", "clinical", "patient", "disease", "treatment", "diagnosis"],
            "biology": ["biology", "genetic", "protein", "cell", "organism", "evolution"],
            "astronomy": ["astronomy", "star", "galaxy", "universe", "telescope", "cosmic"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "compound"],
            "physics": ["physics", "quantum", "particle", "energy", "force", "wave"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        papers_data = self.papers_collection.get()
        techniques_data = self.techniques_collection.get()
        
        # Count papers by domain
        domain_counts = {}
        for metadata in papers_data['metadatas']:
            domain = metadata.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Count techniques by domain
        technique_domain_counts = {}
        for metadata in techniques_data['metadatas']:
            domain = metadata.get('domain', 'unknown')
            technique_domain_counts[domain] = technique_domain_counts.get(domain, 0) + 1
        
        return {
            "total_papers": len(papers_data['ids']),
            "total_techniques": len(techniques_data['ids']),
            "papers_by_domain": domain_counts,
            "techniques_by_domain": technique_domain_counts,
            "embedding_dimension": self.embedding_generator.embedding_dim
        }
    
    def clear_collections(self) -> None:
        """Clear all collections (for testing/reset)"""
        try:
            self.client.delete_collection("research_papers")
            self.client.delete_collection("techniques")
            
            # Recreate collections
            self.papers_collection = self._get_or_create_collection("research_papers")
            self.techniques_collection = self._get_or_create_collection("techniques")
            
            logger.info("Collections cleared and recreated")
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the vector store
    from data.api_clients import AcademicAPIManager
    
    # Initialize components
    vector_store = ChromaVectorStore()
    api_manager = AcademicAPIManager()
    
    # Test with sample papers
    print("Testing ChromaDB Vector Store...")
    
    # Search for some papers
    query = "deep learning medical imaging"
    print(f"Searching for papers: {query}")
    
    papers_results = api_manager.search_all_sources(query, max_results_per_source=3)
    all_papers = []
    for source, papers in papers_results.items():
        all_papers.extend(papers)
    
    if all_papers:
        print(f"Found {len(all_papers)} papers, adding to vector store...")
        vector_store.add_papers(all_papers)
        
        # Test similarity search
        print("\nTesting similarity search...")
        similar_papers = vector_store.search_similar_papers(
            "neural networks for cancer detection", 
            n_results=5
        )
        
        for result in similar_papers:
            print(f"  - {result.paper.title}")
            print(f"    Similarity: {result.similarity_score:.3f}")
            print(f"    Cross-domain: {result.cross_domain}")
            print(f"    Source: {result.paper.source}")
            print()
        
        # Test cross-domain technique search
        print("Testing cross-domain technique search...")
        cross_domain = vector_store.find_cross_domain_techniques(
            "image classification for medical diagnosis",
            target_domain="medical",
            n_results=5
        )
        
        for result in cross_domain:
            print(f"  - Technique: {result.technique_match}")
            print(f"    From: {result.paper.categories}")
            print(f"    Reasoning: {result.reasoning}")
            print()
        
        # Show stats
        print("Vector store statistics:")
        stats = vector_store.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("No papers found to test with")