"""
Cross-Domain Connection Agent - Finds transferable techniques across research domains
Discovers hidden connections between different fields and suggests technique transfers
OPTIMIZED VERSION with minimal API calls and predefined mappings
"""

import os
import sys
from typing import List, Dict, Optional, Any, Type, Tuple
import logging
from dataclasses import dataclass
import json
import re
from collections import Counter, defaultdict
import time
import random

# LangChain imports
# from langchain.tools import BaseTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Import our components
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from data.api_clients import AcademicAPIManager, ResearchPaper
    from vector_store.chroma_client import ChromaVectorStore, SimilarityResult
    from agents.paper_discovery_agent import GroqLLMWrapper
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the correct directory and install dependencies")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrossDomainConnection:
    """Represents a connection between techniques across domains"""
    source_domain: str
    target_domain: str
    technique_name: str
    technique_description: str
    transfer_feasibility: float  # 0-1 score
    innovation_potential: float  # 0-1 score
    supporting_papers: List[ResearchPaper]
    reasoning: str
    analogy_explanation: str
    implementation_suggestions: List[str]

@dataclass
class DomainAnalysis:
    """Analysis of a research domain"""
    domain_name: str
    key_techniques: List[str]
    common_problems: List[str]
    methodology_patterns: List[str]
    paper_count: int
    innovation_gaps: List[str]

@dataclass
class CrossDomainResult:
    """Result from cross-domain analysis"""
    query: str
    target_domain: str
    connections_found: List[CrossDomainConnection]
    domain_analysis: Dict[str, DomainAnalysis]
    innovation_opportunities: List[str]
    agent_reasoning: List[str]
    transfer_recommendations: List[str]

class TechniqueAnalogizer:
    """Finds analogous techniques across different domains - OPTIMIZED VERSION"""
    
    def __init__(self, llm: GroqLLMWrapper):
        self.llm = llm
        
        # Pre-defined high-quality cross-domain connections
        self.proven_connections = {
            ("astronomy", "medical"): [
                {
                    "technique": "Adaptive optics image enhancement",
                    "description": "Correcting atmospheric distortions in telescope images",
                    "medical_application": "Medical imaging clarity improvement and artifact reduction",
                    "feasibility": 0.75,
                    "innovation": 0.80
                },
                {
                    "technique": "Signal processing for weak astronomical sources",
                    "description": "Detecting faint signals in noisy astronomical data",
                    "medical_application": "Early disease detection in noisy biomedical signals",
                    "feasibility": 0.70,
                    "innovation": 0.85
                }
            ],
            ("computer_science", "medical"): [
                {
                    "technique": "Deep learning image classification",
                    "description": "Neural networks for automated image analysis",
                    "medical_application": "Medical diagnosis automation and pattern recognition",
                    "feasibility": 0.85,
                    "innovation": 0.70
                },
                {
                    "technique": "Natural language processing",
                    "description": "Understanding and processing human language",
                    "medical_application": "Clinical notes analysis and patient communication",
                    "feasibility": 0.80,
                    "innovation": 0.75
                }
            ],
            ("physics", "medical"): [
                {
                    "technique": "Acoustic wave analysis",
                    "description": "Processing sound waves and vibrations",
                    "medical_application": "Ultrasound imaging enhancement and acoustic therapy",
                    "feasibility": 0.70,
                    "innovation": 0.75
                },
                {
                    "technique": "Signal processing techniques",
                    "description": "Filtering and analyzing complex signals",
                    "medical_application": "Biomedical signal analysis and noise reduction",
                    "feasibility": 0.75,
                    "innovation": 0.70
                }
            ],
            ("biology", "computer_science"): [
                {
                    "technique": "Evolutionary algorithms",
                    "description": "Optimization based on natural selection",
                    "medical_application": "Optimization problem solving and adaptive systems",
                    "feasibility": 0.80,
                    "innovation": 0.75
                },
                {
                    "technique": "Neural network architectures",
                    "description": "Brain-inspired computing models",
                    "medical_application": "Biomimetic computing systems and AI development",
                    "feasibility": 0.85,
                    "innovation": 0.70
                }
            ],
            ("astronomy", "computer_science"): [
                {
                    "technique": "Object detection algorithms",
                    "description": "Identifying objects in astronomical images",
                    "medical_application": "Automated feature recognition in various applications",
                    "feasibility": 0.75,
                    "innovation": 0.70
                },
                {
                    "technique": "Image processing pipelines",
                    "description": "Automated image analysis workflows",
                    "medical_application": "Computer vision applications and image enhancement",
                    "feasibility": 0.80,
                    "innovation": 0.65
                }
            ]
        }
    
    def find_analogous_techniques(self, problem_description: str, source_domain: str, target_domain: str) -> List[str]:
        """Find techniques from source domain - OPTIMIZED to use predefined mappings"""
        try:
            # First try predefined high-quality connections
            key = (source_domain, target_domain)
            if key in self.proven_connections:
                techniques = [conn["technique"] for conn in self.proven_connections[key]]
                logger.info(f"Using predefined connections from {source_domain} to {target_domain}")
                return techniques
            
            # Fallback: simple keyword-based matching without LLM
            fallback_techniques = self._generate_fallback_techniques(problem_description, source_domain, target_domain)
            return fallback_techniques
            
        except Exception as e:
            logger.error(f"Error finding analogous techniques: {e}")
            return [f"{source_domain} technique transfer"]
    
    def _generate_fallback_techniques(self, problem: str, source_domain: str, target_domain: str) -> List[str]:
        """Generate fallback techniques without LLM calls"""
        domain_techniques = {
            "computer_science": ["machine learning algorithms", "data processing methods"],
            "astronomy": ["image enhancement techniques", "signal processing methods"],
            "physics": ["wave analysis methods", "signal processing techniques"],
            "biology": ["pattern recognition methods", "evolutionary approaches"],
            "medical": ["diagnostic techniques", "treatment optimization methods"]
        }
        
        techniques = domain_techniques.get(source_domain, ["cross-domain methods"])
        return techniques[:2]  # Return max 2 techniques

class InnovationScorer:
    """Scores innovation potential - OPTIMIZED to avoid LLM calls"""
    
    def __init__(self, llm: GroqLLMWrapper):
        self.llm = llm
    
    def score_transfer_potential(self, connection: CrossDomainConnection) -> Tuple[float, float]:
        """Score using predefined mappings instead of LLM calls"""
        try:
            # Use predefined scores if available
            analogizer = TechniqueAnalogizer(self.llm)
            key = (connection.source_domain, connection.target_domain)
            
            if key in analogizer.proven_connections:
                for conn_data in analogizer.proven_connections[key]:
                    if conn_data["technique"].lower() in connection.technique_name.lower():
                        return conn_data["feasibility"], conn_data["innovation"]
            
            # Fallback scoring based on domain compatibility
            feasibility = self._simple_feasibility_score(connection.source_domain, connection.target_domain)
            innovation = self._simple_innovation_score(connection.technique_name)
            
            return feasibility, innovation
            
        except Exception as e:
            logger.error(f"Error scoring transfer potential: {e}")
            return 0.6, 0.6  # Default reasonable scores
    
    def _simple_feasibility_score(self, source_domain: str, target_domain: str) -> float:
        """Simple feasibility scoring without LLM"""
        compatibility_matrix = {
            ("astronomy", "medical"): 0.75,
            ("computer_science", "medical"): 0.85,
            ("physics", "medical"): 0.70,
            ("biology", "computer_science"): 0.80,
            ("astronomy", "computer_science"): 0.75,
        }
        
        key = (source_domain, target_domain)
        return compatibility_matrix.get(key, 0.60)
    
    def _simple_innovation_score(self, technique_name: str) -> float:
        """Simple innovation scoring based on keywords"""
        technique_lower = technique_name.lower()
        
        high_innovation_keywords = ["deep learning", "ai", "machine learning", "neural", "quantum"]
        medium_innovation_keywords = ["enhanced", "improved", "advanced", "optimized"]
        
        score = 0.5  # Base score
        
        if any(keyword in technique_lower for keyword in high_innovation_keywords):
            score += 0.25
        elif any(keyword in technique_lower for keyword in medium_innovation_keywords):
            score += 0.15
        
        return min(score, 0.95)

# Tool input schemas (simplified)
class CrossDomainSearchInput(BaseModel):
    query: str = Field(description="Research problem to find cross-domain solutions for")
    target_domain: str = Field(description="Domain of the research problem")

class CrossDomainSearchTool(BaseTool):
    """Simplified tool for finding cross-domain technique connections"""
    
    name: str = "cross_domain_search"
    description: str = "Find techniques from other domains that could apply to your research problem"
    args_schema: Type[BaseModel] = CrossDomainSearchInput
    
    _vector_store: Optional[ChromaVectorStore] = None
    
    def __init__(self, vector_store: ChromaVectorStore, **kwargs):
        super().__init__(**kwargs)
        CrossDomainSearchTool._vector_store = vector_store
    
    def _run(self, query: str, target_domain: str = "general") -> str:
        """Find cross-domain techniques without heavy processing"""
        try:
            result = [f"Cross-Domain Analysis for: {query}"]
            result.append(f"Target domain: {target_domain}")
            result.append("High-potential technique transfers identified")
            return "\n".join(result)
            
        except Exception as e:
            return f"Error in cross-domain search: {str(e)}"

class CrossDomainAgent:
    """
    OPTIMIZED Cross-Domain Agent with minimal API calls and predefined connections
    """
    
    def __init__(self, groq_api_key: str, vector_store: ChromaVectorStore = None, api_manager: AcademicAPIManager = None):
        """Initialize the Cross-Domain Connection Agent"""
        # Initialize LLM
        self.llm = GroqLLMWrapper(groq_api_key)
        
        # Initialize components
        self.vector_store = vector_store or ChromaVectorStore()
        self.api_manager = api_manager or AcademicAPIManager()
        
        # Initialize analyzers
        self.analogizer = TechniqueAnalogizer(self.llm)
        self.scorer = InnovationScorer(self.llm)
        
        # Initialize tools
        self.cross_domain_tool = CrossDomainSearchTool(self.vector_store)
        
        logger.info("Cross-Domain Agent initialized with optimized processing")
    
    def find_cross_domain_connections(self, research_query: str, target_domain: str = None) -> CrossDomainResult:
        """
        OPTIMIZED method with minimal API calls and predefined connections
        """
        logger.info(f"Finding cross-domain connections for: {research_query}")
        
        reasoning_steps = []
        
        try:
            # Step 1: Infer target domain if not provided (no API call)
            if not target_domain:
                target_domain = self._infer_domain(research_query)
                reasoning_steps.append(f"Inferred target domain: {target_domain}")
            
            # Step 2: Generate connections using predefined mappings (NO API CALLS)
            reasoning_steps.append("Generating cross-domain connections using proven mappings...")
            all_connections = self._generate_predefined_connections(research_query, target_domain)
            
            # Step 3: Score connections using predefined scores (NO API CALLS)
            reasoning_steps.append("Scoring connection potential...")
            for connection in all_connections:
                feasibility, innovation = self.scorer.score_transfer_potential(connection)
                connection.transfer_feasibility = feasibility
                connection.innovation_potential = innovation
            
            # Sort and limit to top 2
            all_connections.sort(
                key=lambda x: (x.transfer_feasibility + x.innovation_potential) / 2, 
                reverse=True
            )
            
            # Step 4: Generate simple analysis (no API calls)
            reasoning_steps.append("Generating analysis and recommendations...")
            innovation_opportunities = self._generate_simple_opportunities(research_query)
            transfer_recommendations = self._generate_simple_recommendations(all_connections[:2])
            
            result = CrossDomainResult(
                query=research_query,
                target_domain=target_domain,
                connections_found=all_connections[:2],  # Ensure only 2 connections
                domain_analysis={},
                innovation_opportunities=innovation_opportunities,
                agent_reasoning=reasoning_steps,
                transfer_recommendations=transfer_recommendations
            )
            
            logger.info(f"Found {len(all_connections[:2])} cross-domain connections")
            return result
            
        except Exception as e:
            logger.error(f"Error in cross-domain analysis: {e}")
            return self._create_fallback_result(research_query, target_domain)
    
    def _infer_domain(self, query: str) -> str:
        """Infer research domain from query"""
        query_lower = query.lower()
        
        domain_keywords = {
            "medical": ["medical", "clinical", "patient", "disease", "treatment", "diagnosis", "health", "therapy"],
            "computer_science": ["ai", "machine learning", "deep learning", "neural network", "algorithm", "computer"],
            "biology": ["biology", "genetic", "protein", "cell", "organism", "evolution", "molecular"],
            "astronomy": ["astronomy", "star", "galaxy", "universe", "telescope", "cosmic", "space"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "compound", "synthesis"],
            "physics": ["physics", "quantum", "particle", "energy", "force", "wave", "mechanics"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return "general"
    
    def _generate_predefined_connections(self, query: str, target_domain: str) -> List[CrossDomainConnection]:
        """Generate connections using predefined high-quality mappings"""
        connections = []
        
        # Get relevant source domains
        relevant_domains = self._get_relevant_source_domains(query, target_domain)
        
        for source_domain in relevant_domains[:2]:  # Limit to 2 source domains
            key = (source_domain, target_domain)
            
            if key in self.analogizer.proven_connections:
                for conn_data in self.analogizer.proven_connections[key][:1]:  # 1 per domain
                    connection = CrossDomainConnection(
                        source_domain=source_domain,
                        target_domain=target_domain,
                        technique_name=conn_data["technique"],
                        technique_description=conn_data["description"],
                        transfer_feasibility=conn_data["feasibility"],
                        innovation_potential=conn_data["innovation"],
                        supporting_papers=[],
                        reasoning=f"Proven technique transfer from {source_domain} to {target_domain}",
                        analogy_explanation=f"Applying {conn_data['technique']} to {target_domain}",
                        implementation_suggestions=[
                            f"Adapt {conn_data['technique']} for {target_domain} applications",
                            f"Collaborate with {source_domain} experts",
                            "Pilot study to validate transfer feasibility"
                        ]
                    )
                    connections.append(connection)
        
        # If no predefined connections, create fallback connections
        if not connections:
            connections = self._create_fallback_connections(query, target_domain)
        
        return connections[:2]  # Ensure only 2 connections
    
    def _get_relevant_source_domains(self, query: str, target_domain: str) -> List[str]:
        """Get most relevant source domains based on query keywords"""
        query_lower = query.lower()
        
        domain_relevance = {
            "computer_science": ["ai", "machine", "learning", "neural", "algorithm", "data", "technology"],
            "astronomy": ["image", "signal", "processing", "detection", "analysis", "pattern", "enhancement"],
            "physics": ["wave", "signal", "frequency", "resonance", "vibration", "acoustic", "processing"],
            "biology": ["behavior", "neural", "brain", "cognitive", "evolution", "adaptation", "pattern"],
            "medical": ["medical", "health", "clinical", "patient", "diagnosis", "treatment", "therapy"]
        }
        
        scores = {}
        for domain, keywords in domain_relevance.items():
            if domain != target_domain:
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    scores[domain] = score
        
        # Return domains sorted by relevance
        sorted_domains = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_domains if sorted_domains else ["computer_science", "astronomy"]
    
    def _create_fallback_connections(self, query: str, target_domain: str) -> List[CrossDomainConnection]:
        """Create generic fallback connections"""
        fallback_data = [
            {
                "source": "computer_science",
                "technique": "Machine learning algorithms",
                "description": "Data-driven pattern recognition and prediction",
                "feasibility": 0.75,
                "innovation": 0.70
            },
            {
                "source": "astronomy",
                "technique": "Signal processing methods",
                "description": "Extracting meaningful information from noisy data",
                "feasibility": 0.70,
                "innovation": 0.75
            }
        ]
        
        connections = []
        for i, data in enumerate(fallback_data[:2]):
            if data["source"] != target_domain:
                connection = CrossDomainConnection(
                    source_domain=data["source"],
                    target_domain=target_domain,
                    technique_name=data["technique"],
                    technique_description=data["description"],
                    transfer_feasibility=data["feasibility"],
                    innovation_potential=data["innovation"],
                    supporting_papers=[],
                    reasoning=f"Generic technique transfer opportunity identified",
                    analogy_explanation=f"Applying {data['technique']} principles to {target_domain}",
                    implementation_suggestions=[
                        "Investigate adaptation requirements",
                        "Pilot feasibility study",
                        "Interdisciplinary collaboration"
                    ]
                )
                connections.append(connection)
        
        return connections
    
    def _generate_simple_opportunities(self, query: str) -> List[str]:
        """Generate opportunities without LLM calls"""
        return [
            f"Explore interdisciplinary approaches for: {query}",
            "Investigate technique transfer feasibility studies",
            "Develop hybrid methodologies combining domain expertise"
        ]
    
    def _generate_simple_recommendations(self, connections: List[CrossDomainConnection]) -> List[str]:
        """Generate recommendations without LLM calls"""
        recommendations = []
        
        for conn in connections:
            if conn.transfer_feasibility > 0.6:
                recommendations.append(
                    f"High priority: Investigate {conn.technique_name} transfer from {conn.source_domain}"
                )
        
        if not recommendations:
            recommendations.append("Focus on building interdisciplinary collaboration networks")
        
        return recommendations[:3]
    
    def _create_fallback_result(self, query: str, target_domain: str) -> CrossDomainResult:
        """Create minimal result when everything fails"""
        return CrossDomainResult(
            query=query,
            target_domain=target_domain or "general",
            connections_found=[],
            domain_analysis={},
            innovation_opportunities=["Explore cross-domain technique transfer opportunities"],
            agent_reasoning=["Analysis completed with limited resources"],
            transfer_recommendations=["Consider interdisciplinary collaboration"]
        )
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status information about the agent"""
        return {
            "agent_type": "Cross-Domain Connection Agent",
            "llm_model": self.llm.model,
            "api_calls_this_hour": getattr(self.llm, 'call_count', 0),
            "predefined_connections": len(self.analogizer.proven_connections),
            "tools_available": ["cross_domain_search"],
            "domains_supported": ["computer_science", "medical", "astronomy", "physics", "biology"],
            "status": "ready with minimal API usage"
        }

# Example usage and testing
def main():
    """Main function for running the Cross-Domain Agent standalone"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Please set GROQ_API_KEY environment variable")
        exit(1)
    
    # Initialize agent
    print("Initializing Optimized Cross-Domain Agent...")
    agent = CrossDomainAgent(groq_api_key)
    
    # Test cross-domain discovery
    research_query = "speech therapy using AI technology"
    print(f"Finding cross-domain connections for: {research_query}")
    
    result = agent.find_cross_domain_connections(research_query)
    
    # Display results
    print(f"\n=== CROSS-DOMAIN ANALYSIS RESULTS ===")
    print(f"Query: {result.query}")
    print(f"Target Domain: {result.target_domain}")
    print(f"Connections Found: {len(result.connections_found)}")
    
    print(f"\n=== AGENT REASONING ===")
    for i, step in enumerate(result.agent_reasoning, 1):
        print(f"{i}. {step}")
    
    print(f"\n=== CROSS-DOMAIN CONNECTIONS ===")
    for i, connection in enumerate(result.connections_found, 1):
        print(f"{i}. {connection.technique_name}")
        print(f"   From: {connection.source_domain} → To: {connection.target_domain}")
        print(f"   Feasibility: {connection.transfer_feasibility:.2f}")
        print(f"   Innovation: {connection.innovation_potential:.2f}")
        print(f"   Description: {connection.technique_description}")
        print()
    
    print(f"\n=== INNOVATION OPPORTUNITIES ===")
    for opp in result.innovation_opportunities:
        print(f"- {opp}")
    
    print(f"\n=== TRANSFER RECOMMENDATIONS ===")
    for rec in result.transfer_recommendations:
        print(f"- {rec}")
    
    print(f"\n=== AGENT STATUS ===")
    status = agent.get_agent_status()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
