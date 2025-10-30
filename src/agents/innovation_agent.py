from __future__ import annotations
"""
Innovation Agent - Synthesizes cross-domain insights to generate novel research opportunities
Combines findings from Paper Discovery and Cross-Domain agents to suggest breakthrough innovations
"""

import os
import sys
from typing import List, Dict, Optional, Any, Type, Tuple
import logging
from dataclasses import dataclass
import json
import re
from collections import defaultdict, Counter
import random


# LangChain imports
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Import our components
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from data.api_clients import AcademicAPIManager, ResearchPaper
    from vector_store.chroma_client import ChromaVectorStore, SimilarityResult
    from agents.paper_discovery_agent import GroqLLMWrapper
    # Import these with TYPE_CHECKING to avoid circular imports
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from agents.paper_discovery_agent import PaperDiscoveryAgent, PaperDiscoveryResult
        from agents.cross_domain_agent import CrossDomainAgent, CrossDomainResult, CrossDomainConnection
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the correct directory and install dependencies")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NovelResearchDirection:
    """Represents a novel research direction/opportunity"""
    title: str
    description: str
    innovation_type: str  # "incremental", "radical", "paradigm_shift"
    confidence_score: float  # 0-1
    impact_potential: float  # 0-1
    feasibility_score: float  # 0-1
    required_expertise: List[str]
    supporting_evidence: List[str]
    cross_domain_connections: List[Dict[str, Any]]
    research_gaps_addressed: List[str]
    implementation_roadmap: List[str]
    potential_challenges: List[str]
    success_metrics: List[str]

@dataclass
class InnovationSynthesis:
    """Synthesis of innovation opportunities from multiple sources"""
    convergence_points: List[str]  # Where multiple domains converge
    knowledge_gaps: List[str]      # Identified research gaps
    technique_combinations: List[str]  # Novel technique combinations
    untapped_analogies: List[str]  # Unexplored analogous applications
    emerging_patterns: List[str]   # Patterns across multiple papers/domains

@dataclass
class InnovationResult:
    """Complete result from innovation analysis"""
    query: str
    novel_directions: List[NovelResearchDirection]
    innovation_synthesis: InnovationSynthesis
    priority_recommendations: List[str]
    collaboration_suggestions: List[str]
    funding_opportunities: List[str]
    timeline_estimates: Dict[str, str]
    agent_reasoning: List[str]
    confidence_assessment: str

class InnovationSynthesizer:
    """Synthesizes insights from multiple agents to generate innovations"""
    
    def __init__(self, llm: GroqLLMWrapper):
        self.llm = llm
        
        # Innovation categories and their characteristics
        self.innovation_types = {
            "incremental": {
                "description": "Gradual improvement of existing methods",
                "impact_range": (0.2, 0.6),
                "feasibility_range": (0.6, 0.9),
                "timeframe": "1-2 years"
            },
            "radical": {
                "description": "Significant breakthrough in approach",
                "impact_range": (0.6, 0.9),
                "feasibility_range": (0.3, 0.7),
                "timeframe": "3-5 years"
            },
            "paradigm_shift": {
                "description": "Fundamental change in thinking",
                "impact_range": (0.8, 1.0),
                "feasibility_range": (0.1, 0.5),
                "timeframe": "5-10 years"
            }
        }
    
    def synthesize_innovations(
        self, 
        paper_results: Any,  # PaperDiscoveryResult
        cross_domain_results: Any  # CrossDomainResult
    ) -> InnovationSynthesis:
        """Synthesize innovation opportunities from agent results"""
        
        # Find convergence points
        convergence_points = self._find_convergence_points(paper_results, cross_domain_results)
        
        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(paper_results, cross_domain_results)
        
        # Generate technique combinations
        technique_combinations = self._generate_technique_combinations(cross_domain_results)
        
        # Find untapped analogies
        untapped_analogies = self._find_untapped_analogies(cross_domain_results)
        
        # Detect emerging patterns
        emerging_patterns = self._detect_emerging_patterns(paper_results, cross_domain_results)
        
        return InnovationSynthesis(
            convergence_points=convergence_points,
            knowledge_gaps=knowledge_gaps,
            technique_combinations=technique_combinations,
            untapped_analogies=untapped_analogies,
            emerging_patterns=emerging_patterns
        )
    
    def _find_convergence_points(self, paper_results: Any, cross_domain_results: Any) -> List[str]:
        """Find points where multiple domains/techniques converge"""
        convergence_points = []
        
        # Look for common techniques across domains
        technique_domains = defaultdict(set)
        if hasattr(cross_domain_results, 'connections_found'):
            for connection in cross_domain_results.connections_found:
                technique_name = getattr(connection, 'technique_name', '')
                source_domain = getattr(connection, 'source_domain', '')
                target_domain = getattr(connection, 'target_domain', '')
                
                technique_domains[technique_name].add(source_domain)
                technique_domains[technique_name].add(target_domain)
        
        # Find techniques that appear in multiple domains
        for technique, domains in technique_domains.items():
            if len(domains) >= 3:
                convergence_points.append(f"{technique} convergence across {', '.join(domains)}")
        
        # Add research gap convergence
        if (hasattr(paper_results, 'research_gaps') and paper_results.research_gaps and
            hasattr(cross_domain_results, 'innovation_opportunities') and cross_domain_results.innovation_opportunities):
            convergence_points.append("Research gaps align with cross-domain opportunities")
        
        return convergence_points[:5]
    
    def _identify_knowledge_gaps(self, paper_results: Any, cross_domain_results: Any) -> List[str]:
        """Identify unexplored knowledge areas"""
        gaps = []
        
        # Combine gaps from both sources
        if hasattr(paper_results, 'research_gaps'):
            gaps.extend(paper_results.research_gaps[:3])
        
        # Add cross-domain gaps
        if hasattr(cross_domain_results, 'connections_found'):
            for connection in cross_domain_results.connections_found:
                transfer_feasibility = getattr(connection, 'transfer_feasibility', 0)
                innovation_potential = getattr(connection, 'innovation_potential', 0)
                
                if transfer_feasibility > 0.6 and innovation_potential > 0.7:
                    technique_name = getattr(connection, 'technique_name', 'unknown technique')
                    source_domain = getattr(connection, 'source_domain', 'unknown')
                    target_domain = getattr(connection, 'target_domain', 'unknown')
                    gap = f"Limited exploration of {technique_name} transfer from {source_domain} to {target_domain}"
                    gaps.append(gap)
        
        return gaps[:5]
    
    def _generate_technique_combinations(self, cross_domain_results: Any) -> List[str]:
        """Generate novel combinations of techniques"""
        combinations = []
        
        if not hasattr(cross_domain_results, 'connections_found'):
            return combinations
        
        # Get high-potential techniques
        high_potential = [
            conn for conn in cross_domain_results.connections_found 
            if getattr(conn, 'innovation_potential', 0) > 0.6
        ]
        
        # Combine techniques from different domains
        for i in range(min(3, len(high_potential))):
            for j in range(i + 1, min(3, len(high_potential))):
                tech1 = high_potential[i]
                tech2 = high_potential[j]
                
                source1 = getattr(tech1, 'source_domain', '')
                source2 = getattr(tech2, 'source_domain', '')
                
                if source1 != source2:
                    technique1 = getattr(tech1, 'technique_name', 'technique1')
                    technique2 = getattr(tech2, 'technique_name', 'technique2')
                    combination = f"Hybrid approach: {technique1} + {technique2}"
                    combinations.append(combination)
        
        return combinations[:5]
    
    def _find_untapped_analogies(self, cross_domain_results: Any) -> List[str]:
        """Find unexplored analogous applications"""
        analogies = []
        
        if not hasattr(cross_domain_results, 'connections_found'):
            return analogies
        
        # Look for connections with high innovation but moderate feasibility
        for connection in cross_domain_results.connections_found:
            innovation_potential = getattr(connection, 'innovation_potential', 0)
            transfer_feasibility = getattr(connection, 'transfer_feasibility', 0)
            
            if innovation_potential > 0.7 and 0.4 < transfer_feasibility < 0.7:
                technique_name = getattr(connection, 'technique_name', 'technique')
                source_domain = getattr(connection, 'source_domain', 'domain')
                analogy = f"Unexplored: {technique_name} from {source_domain} paradigm"
                analogies.append(analogy)
        
        return analogies[:5]
    
    def _detect_emerging_patterns(self, paper_results: Any, cross_domain_results: Any) -> List[str]:
        """Detect emerging patterns across results"""
        patterns = []
        
        # Analyze paper publication patterns
        if hasattr(paper_results, 'papers_found') and paper_results.papers_found:
            sources = [getattr(paper, 'source', 'unknown') for paper in paper_results.papers_found]
            source_counts = Counter(sources)
            if source_counts:
                most_common_source = source_counts.most_common(1)[0]
                patterns.append(f"Emerging trend: High activity in {most_common_source[0]} ({most_common_source[1]} papers)")
        
        # Analyze cross-domain patterns
        if hasattr(cross_domain_results, 'connections_found') and cross_domain_results.connections_found:
            source_domains = [getattr(conn, 'source_domain', 'unknown') for conn in cross_domain_results.connections_found]
            domain_counts = Counter(source_domains)
            if domain_counts:
                top_domain = domain_counts.most_common(1)[0]
                patterns.append(f"Cross-domain pattern: {top_domain[0]} techniques highly transferable")
        
        # Add general patterns
        patterns.append("Pattern: Increasing convergence of computational methods across domains")
        
        return patterns[:5]

class ResearchDirectionGenerator:
    """Generates novel research directions based on synthesis"""
    
    def __init__(self, llm: GroqLLMWrapper):
        self.llm = llm
    
    def generate_novel_directions(
        self, 
        query: str,
        synthesis: InnovationSynthesis,
        cross_domain_connections: List[Any]
    ) -> List[NovelResearchDirection]:
        """Generate novel research directions"""
        
        directions = []
        
        # Generate directions from convergence points
        for convergence in synthesis.convergence_points[:2]:
            direction = self._create_direction_from_convergence(query, convergence, cross_domain_connections)
            if direction:
                directions.append(direction)
        
        # Generate directions from technique combinations
        for combination in synthesis.technique_combinations[:2]:
            direction = self._create_direction_from_combination(query, combination, cross_domain_connections)
            if direction:
                directions.append(direction)
        
        # Generate directions from knowledge gaps
        for gap in synthesis.knowledge_gaps[:1]:
            direction = self._create_direction_from_gap(query, gap, cross_domain_connections)
            if direction:
                directions.append(direction)
        
        # Score and enhance all directions
        for direction in directions:
            self._enhance_direction_with_llm(direction)
        
        return directions
    
    def _create_direction_from_convergence(
        self, 
        query: str, 
        convergence: str, 
        connections: List[Any]
    ) -> Optional[NovelResearchDirection]:
        """Create research direction from convergence point"""
        
        title = f"Multi-domain convergence approach for {query}"
        description = f"Leverage {convergence} to create novel solutions"
        
        # Find supporting connections
        supporting_connections = []
        for conn in connections:
            technique_name = getattr(conn, 'technique_name', '')
            if any(word in convergence.lower() for word in technique_name.lower().split()):
                # Convert connection to dict format
                conn_dict = {
                    'technique_name': technique_name,
                    'source_domain': getattr(conn, 'source_domain', ''),
                    'target_domain': getattr(conn, 'target_domain', ''),
                    'innovation_potential': getattr(conn, 'innovation_potential', 0.5)
                }
                supporting_connections.append(conn_dict)
                if len(supporting_connections) >= 2:
                    break
        
        return NovelResearchDirection(
            title=title,
            description=description,
            innovation_type="radical",
            confidence_score=0.7,
            impact_potential=0.8,
            feasibility_score=0.6,
            required_expertise=["interdisciplinary research", "systems thinking"],
            supporting_evidence=[convergence],
            cross_domain_connections=supporting_connections,
            research_gaps_addressed=[],
            implementation_roadmap=[],
            potential_challenges=[],
            success_metrics=[]
        )
    
    def _create_direction_from_combination(
        self, 
        query: str, 
        combination: str, 
        connections: List[Any]
    ) -> Optional[NovelResearchDirection]:
        """Create research direction from technique combination"""
        
        title = f"Hybrid methodology for {query}"
        description = f"Novel approach combining: {combination}"
        
        # Convert first few connections to dict format
        conn_dicts = []
        for conn in connections[:2]:
            conn_dict = {
                'technique_name': getattr(conn, 'technique_name', ''),
                'source_domain': getattr(conn, 'source_domain', ''),
                'target_domain': getattr(conn, 'target_domain', ''),
                'innovation_potential': getattr(conn, 'innovation_potential', 0.5)
            }
            conn_dicts.append(conn_dict)
        
        return NovelResearchDirection(
            title=title,
            description=description,
            innovation_type="incremental",
            confidence_score=0.8,
            impact_potential=0.7,
            feasibility_score=0.8,
            required_expertise=["method integration", "comparative analysis"],
            supporting_evidence=[combination],
            cross_domain_connections=conn_dicts,
            research_gaps_addressed=[],
            implementation_roadmap=[],
            potential_challenges=[],
            success_metrics=[]
        )
    
    def _create_direction_from_gap(
        self, 
        query: str, 
        gap: str, 
        connections: List[Any]
    ) -> Optional[NovelResearchDirection]:
        """Create research direction from knowledge gap"""
        
        title = f"Addressing gap in {query}"
        description = f"Novel research to fill identified gap: {gap}"
        
        # Convert first connection to dict format
        conn_dicts = []
        if connections:
            conn = connections[0]
            conn_dict = {
                'technique_name': getattr(conn, 'technique_name', ''),
                'source_domain': getattr(conn, 'source_domain', ''),
                'target_domain': getattr(conn, 'target_domain', ''),
                'innovation_potential': getattr(conn, 'innovation_potential', 0.5)
            }
            conn_dicts.append(conn_dict)
        
        return NovelResearchDirection(
            title=title,
            description=description,
            innovation_type="paradigm_shift",
            confidence_score=0.6,
            impact_potential=0.9,
            feasibility_score=0.4,
            required_expertise=["pioneering research", "theoretical development"],
            supporting_evidence=[gap],
            cross_domain_connections=conn_dicts,
            research_gaps_addressed=[gap],
            implementation_roadmap=[],
            potential_challenges=[],
            success_metrics=[]
        )
    
    def _enhance_direction_with_llm(self, direction: NovelResearchDirection):
        """Enhance research direction using LLM"""
        try:
            enhancement_prompt = f"""
            Enhance this research direction with detailed implementation details:

            TITLE: {direction.title}
            DESCRIPTION: {direction.description}
            INNOVATION TYPE: {direction.innovation_type}

            Provide detailed JSON response with:
            - "implementation_roadmap": List of 3-4 concrete implementation steps
            - "potential_challenges": List of 3-4 potential challenges
            - "success_metrics": List of 3-4 measurable success metrics
            - "required_expertise": List of 3-4 specific expertise areas needed

            Keep each item concise but specific.
            """
            
            response = self.llm.invoke(enhancement_prompt, temperature=0.3)
            enhancement_data = self._parse_enhancement_response(response)
            
            # Update direction with enhanced data
            direction.implementation_roadmap = enhancement_data.get("implementation_roadmap", [])
            direction.potential_challenges = enhancement_data.get("potential_challenges", [])
            direction.success_metrics = enhancement_data.get("success_metrics", [])
            if enhancement_data.get("required_expertise"):
                direction.required_expertise = enhancement_data["required_expertise"]
            
        except Exception as e:
            logger.error(f"Error enhancing direction with LLM: {e}")
            # Provide fallback enhancements
            direction.implementation_roadmap = ["Literature review", "Methodology development", "Pilot study", "Validation"]
            direction.potential_challenges = ["Technical complexity", "Resource requirements", "Interdisciplinary coordination"]
            direction.success_metrics = ["Proof of concept", "Performance metrics", "Peer review acceptance"]

    def _parse_enhancement_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM enhancement response"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback parsing
            result = {
                "implementation_roadmap": [],
                "potential_challenges": [],
                "success_metrics": [],
                "required_expertise": []
            }
            
            current_section = None
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if 'roadmap' in line.lower():
                    current_section = 'implementation_roadmap'
                elif 'challenge' in line.lower():
                    current_section = 'potential_challenges'
                elif 'metric' in line.lower():
                    current_section = 'success_metrics'
                elif 'expertise' in line.lower():
                    current_section = 'required_expertise'
                elif line.startswith('-') or line.startswith('•'):
                    if current_section and current_section in result:
                        item = line.strip('- •').strip()
                        if len(item) > 5:
                            result[current_section].append(item)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing enhancement response: {e}")
            return {}

# Tool input schemas
class InnovationSynthesisInput(BaseModel):
    query: str = Field(description="Research query to generate innovations for")
    context: str = Field(description="Context from previous agent results")

class DirectionGenerationInput(BaseModel):
    query: str = Field(description="Research query")
    synthesis_data: str = Field(description="Innovation synthesis data")

class InnovationSynthesisTool(BaseTool):
    """Tool for synthesizing innovation opportunities"""
    
    name: str = "innovation_synthesis"
    description: str = "Synthesize innovation opportunities from multiple agent results"
    args_schema: Type[BaseModel] = InnovationSynthesisInput
    
    _synthesizer: Optional[InnovationSynthesizer] = None
    
    def __init__(self, synthesizer: InnovationSynthesizer, **kwargs):
        super().__init__(**kwargs)
        InnovationSynthesisTool._synthesizer = synthesizer
    
    def _run(self, query: str, context: str) -> str:
        """Synthesize innovation opportunities"""
        try:
            if not InnovationSynthesisTool._synthesizer:
                return "Error: Synthesizer not initialized"
            
            # This would normally use actual results, but for demo we'll simulate
            result = [
                f"Innovation Synthesis for: {query}",
                "Convergence Points:",
                "- Multi-domain technique convergence identified",
                "- Cross-field methodology alignment found",
                "Knowledge Gaps:",
                "- Technique transfer opportunities unexplored",
                "- Hybrid approaches underinvestigated",
                "Emerging Patterns:",
                "- Increasing computational method adoption",
                "- Growing interdisciplinary collaboration"
            ]
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error in innovation synthesis: {str(e)}"

class DirectionGenerationTool(BaseTool):
    """Tool for generating novel research directions"""
    
    name: str = "direction_generation"
    description: str = "Generate novel research directions based on synthesis"
    args_schema: Type[BaseModel] = DirectionGenerationInput
    
    _generator: Optional[ResearchDirectionGenerator] = None
    
    def __init__(self, generator: ResearchDirectionGenerator, **kwargs):
        super().__init__(**kwargs)
        DirectionGenerationTool._generator = generator
    
    def _run(self, query: str, synthesis_data: str) -> str:
        """Generate novel research directions"""
        try:
            if not DirectionGenerationTool._generator:
                return "Error: Generator not initialized"
            
            result = [
                f"Novel Research Directions for: {query}",
                "",
                "1. Multi-domain Convergence Approach",
                "   Type: Radical Innovation",
                "   Impact: High | Feasibility: Moderate",
                "   Focus: Leverage cross-domain technique convergence",
                "",
                "2. Hybrid Methodology Development", 
                "   Type: Incremental Innovation",
                "   Impact: Moderate | Feasibility: High",
                "   Focus: Combine proven techniques from multiple domains",
                "",
                "3. Paradigm-Shifting Framework",
                "   Type: Paradigm Shift",
                "   Impact: Very High | Feasibility: Low",
                "   Focus: Address fundamental knowledge gaps"
            ]
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error generating directions: {str(e)}"

class InnovationAgent:
    """
    Master AI Agent that synthesizes insights from all other agents
    to generate novel research opportunities and breakthrough innovations
    """
    
    def __init__(self, groq_api_key: str, vector_store: ChromaVectorStore = None, api_manager: AcademicAPIManager = None):
        """
        Initialize the Innovation Agent
        
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
        
        # Initialize other agents dynamically to avoid circular imports
        self.paper_agent = None
        self.cross_domain_agent = None
        
        try:
            from agents.paper_discovery_agent import PaperDiscoveryAgent
            from agents.cross_domain_agent import CrossDomainAgent
            self.paper_agent = PaperDiscoveryAgent(groq_api_key, self.vector_store, self.api_manager)
            self.cross_domain_agent = CrossDomainAgent(groq_api_key, self.vector_store, self.api_manager)
        except ImportError as e:
            logger.warning(f"Could not import other agents: {e}")
            # Continue without other agents - innovation agent can work independently
        
        # Initialize innovation components
        self.synthesizer = InnovationSynthesizer(self.llm)
        self.direction_generator = ResearchDirectionGenerator(self.llm)
        
        # Initialize tools
        self.synthesis_tool = InnovationSynthesisTool(self.synthesizer)
        self.direction_tool = DirectionGenerationTool(self.direction_generator)
        
        logger.info("Innovation Agent initialized - ready to generate breakthrough research opportunities")
    
    def discover_innovations(self, research_query: str, max_papers: int = 15) -> InnovationResult:
        """
        Master method to discover novel research opportunities
        
        Args:
            research_query: The research question or area of interest
            max_papers: Maximum papers to analyze
            
        Returns:
            InnovationResult with novel directions and opportunities
        """
        logger.info(f"Starting innovation discovery for: {research_query}")
        
        reasoning_steps = []
        
        try:
            # Step 1: Run Paper Discovery Agent
            reasoning_steps.append("Analyzing existing research landscape...")
            if self.paper_agent:
                paper_results = self.paper_agent.discover_papers(research_query, 8)  # Reduced from max_papers
                reasoning_steps.append(f"Found {len(paper_results.papers_found)} papers, {len(paper_results.similar_papers)} similar papers")
            else:
                # Create mock results if agent not available
                paper_results = self._create_mock_paper_results(research_query)
                reasoning_steps.append("Using demo paper results (paper agent not available)")
            
            # Step 2: Run Cross-Domain Agent
            reasoning_steps.append("Discovering cross-domain connections...")
            if self.cross_domain_agent:
                cross_domain_results = self.cross_domain_agent.find_cross_domain_connections(research_query)
                reasoning_steps.append(f"Identified {len(cross_domain_results.connections_found)} cross-domain connections")
            else:
                # Create mock results if agent not available  
                cross_domain_results = self._create_mock_cross_domain_results(research_query)
                reasoning_steps.append("Using demo cross-domain results (cross-domain agent not available)")
            
            # Step 3: Synthesize Innovation Opportunities
            reasoning_steps.append("Synthesizing innovation opportunities...")
            innovation_synthesis = self.synthesizer.synthesize_innovations(paper_results, cross_domain_results)
            reasoning_steps.append(f"Found {len(innovation_synthesis.convergence_points)} convergence points")
            
            # Step 4: Generate Novel Research Directions
            reasoning_steps.append("Generating novel research directions...")
            novel_directions = self.direction_generator.generate_novel_directions(
                research_query, innovation_synthesis, cross_domain_results.connections_found
            )
            reasoning_steps.append(f"Generated {len(novel_directions)} novel research directions")
            
            # Limit to only 2 unique, high-quality directions
            novel_directions = novel_directions[:2]
            
            # Step 5: Generate Strategic Recommendations
            reasoning_steps.append("Generating strategic recommendations...")
            priority_recommendations = self._generate_priority_recommendations(novel_directions)
            collaboration_suggestions = self._generate_collaboration_suggestions(cross_domain_results)
            funding_opportunities = self._generate_funding_opportunities(novel_directions)
            timeline_estimates = self._generate_timeline_estimates(novel_directions)
            
            # Step 6: Assess Overall Confidence
            reasoning_steps.append("Assessing confidence and feasibility...")
            confidence_assessment = self._assess_confidence(paper_results, cross_domain_results, novel_directions)
            
            result = InnovationResult(
                query=research_query,
                novel_directions=novel_directions,
                innovation_synthesis=innovation_synthesis,
                priority_recommendations=priority_recommendations,
                collaboration_suggestions=collaboration_suggestions,
                funding_opportunities=funding_opportunities,
                timeline_estimates=timeline_estimates,
                agent_reasoning=reasoning_steps,
                confidence_assessment=confidence_assessment
            )
            
            logger.info(f"Innovation discovery completed. Generated {len(novel_directions)} novel directions")
            return result
            
        except Exception as e:
            logger.error(f"Error in innovation discovery: {e}")
            return InnovationResult(
                query=research_query,
                novel_directions=[],
                innovation_synthesis=InnovationSynthesis([], [], [], [], []),
                priority_recommendations=[],
                collaboration_suggestions=[],
                funding_opportunities=[],
                timeline_estimates={},
                agent_reasoning=[f"Error occurred: {str(e)}"],
                confidence_assessment="Low confidence due to processing error"
            )
    
    def _generate_priority_recommendations(self, directions: List[NovelResearchDirection]) -> List[str]:
        """Generate priority recommendations based on directions"""
        recommendations = []
        
        # Sort by combined score
        scored_directions = [
            (d, (d.impact_potential + d.feasibility_score + d.confidence_score) / 3)
            for d in directions
        ]
        scored_directions.sort(key=lambda x: x[1], reverse=True)
        
        for direction, score in scored_directions[:3]:
            rec = f"High Priority: {direction.title} (Score: {score:.2f})"
            recommendations.append(rec)
        
        # Add strategic recommendations
        if any(d.innovation_type == "paradigm_shift" for d in directions):
            recommendations.append("Consider long-term paradigm-shifting research investment")
        
        if any(d.innovation_type == "incremental" for d in directions):
            recommendations.append("Quick wins available through incremental innovations")
        
        return recommendations[:5]
    
    def _generate_collaboration_suggestions(self, cross_domain_results: Any) -> List[str]:
        """Generate collaboration suggestions based on cross-domain analysis"""
        suggestions = []
        
        # Find domains that appear frequently in connections
        domain_counts = Counter()
        if hasattr(cross_domain_results, 'connections_found'):
            for connection in cross_domain_results.connections_found:
                source_domain = getattr(connection, 'source_domain', 'unknown')
                domain_counts[source_domain] += 1
        
        for domain, count in domain_counts.most_common(3):
            suggestion = f"Collaborate with {domain} researchers (high transfer potential)"
            suggestions.append(suggestion)
        
        # Add general suggestions
        suggestions.extend([
            "Form interdisciplinary research teams",
            "Engage domain experts for technique validation",
            "Partner with institutions strong in identified source domains"
        ])
        
        return suggestions[:5]
    
    def _generate_funding_opportunities(self, directions: List[NovelResearchDirection]) -> List[str]:
        """Generate funding opportunity suggestions"""
        opportunities = []
        
        # Categorize by innovation type
        has_radical = any(d.innovation_type == "radical" for d in directions)
        has_paradigm = any(d.innovation_type == "paradigm_shift" for d in directions)
        has_incremental = any(d.innovation_type == "incremental" for d in directions)
        
        if has_paradigm:
            opportunities.append("NSF Emerging Frontiers in Research and Innovation (EFRI)")
            opportunities.append("NIH Director's Pioneer Award")
        
        if has_radical:
            opportunities.append("DARPA Young Faculty Award")
            opportunities.append("Sloan Research Fellowships")
        
        if has_incremental:
            opportunities.append("Standard NSF/NIH grants")
            opportunities.append("Industry-academic partnerships")
        
        # Add cross-domain specific opportunities
        opportunities.append("NSF Convergence Accelerator Program")
        
        return opportunities[:5]
    
    def _generate_timeline_estimates(self, directions: List[NovelResearchDirection]) -> Dict[str, str]:
        """Generate timeline estimates for directions"""
        timelines = {}
        
        for i, direction in enumerate(directions[:3]):
            if direction.innovation_type == "incremental":
                timeline = "1-2 years"
            elif direction.innovation_type == "radical":
                timeline = "3-5 years"
            else:  # paradigm_shift
                timeline = "5-10 years"
            
            timelines[f"Direction {i+1}"] = timeline
        
        return timelines
    
    def _assess_confidence(
        self, 
        paper_results: Any, 
        cross_domain_results: Any, 
        directions: List[NovelResearchDirection]
    ) -> str:
        """Assess overall confidence in the innovation analysis"""
        
        confidence_factors = []
        
        # Data availability
        paper_count = len(getattr(paper_results, 'papers_found', []))
        if paper_count >= 10:
            confidence_factors.append("Strong literature base")
        elif paper_count >= 5:
            confidence_factors.append("Moderate literature base")
        else:
            confidence_factors.append("Limited literature base")
        
        # Cross-domain connections
        connection_count = len(getattr(cross_domain_results, 'connections_found', []))
        if connection_count >= 5:
            confidence_factors.append("Multiple cross-domain opportunities")
        else:
            confidence_factors.append("Few cross-domain connections")
        
        # Direction quality
        avg_confidence = sum(d.confidence_score for d in directions) / len(directions) if directions else 0
        if avg_confidence >= 0.7:
            confidence_factors.append("High-confidence directions")
        elif avg_confidence >= 0.5:
            confidence_factors.append("Moderate-confidence directions")
        else:
            confidence_factors.append("Speculative directions")
        
        return " | ".join(confidence_factors)
    
    def _create_mock_paper_results(self, query: str):
        """Create mock paper results when paper agent not available"""
        class MockPaperResult:
            def __init__(self):
                self.papers_found = []
                self.similar_papers = []
                self.research_gaps = ["Mock research gap 1", "Mock research gap 2"]
        return MockPaperResult()
    
    def _create_mock_cross_domain_results(self, query: str):
        """Create mock cross-domain results when cross-domain agent not available"""
        class MockCrossDomainResult:
            def __init__(self):
                self.connections_found = []
                self.innovation_opportunities = []
        return MockCrossDomainResult()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the innovation system"""
        vector_stats = self.vector_store.get_collection_stats()
        
        status = {
            "agent_type": "Innovation Discovery System",
            "llm_model": self.llm.model,
            "sub_agents": [],
            "tools_available": ["innovation_synthesis", "direction_generation"],
            "vector_store_papers": vector_stats.get("total_papers", 0),
            "vector_store_techniques": vector_stats.get("total_techniques", 0),
            "domains_supported": [],
            "status": "ready for innovation discovery"
        }
        
        # Add sub-agent info if available
        if self.paper_agent:
            status["sub_agents"].append("Paper Discovery")
        if self.cross_domain_agent:
            status["sub_agents"].append("Cross-Domain Connection")
            cross_domain_status = self.cross_domain_agent.get_agent_status()
            status["domains_supported"] = cross_domain_status.get("domains_supported", [])
        
        return status

# Example usage and testing
def main():
    """Main function for running the Innovation Agent standalone"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Please set GROQ_API_KEY environment variable")
        print("Get free API key from: https://console.groq.com/")
        exit(1)
    
    # Initialize innovation agent
    print("🚀 Initializing Innovation Discovery System...")
    agent = InnovationAgent(groq_api_key)
    
    # Test innovation discovery
    research_query = "AI-powered drug discovery using protein folding"
    print(f"🔬 Discovering innovations for: {research_query}")
    
    result = agent.discover_innovations(research_query, max_papers=10)
    
    # Display comprehensive results
    print(f"\n{'='*60}")
    print(f"🎯 INNOVATION DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"Query: {result.query}")
    print(f"Novel Directions Generated: {len(result.novel_directions)}")
    print(f"Confidence Assessment: {result.confidence_assessment}")
    
    print(f"\n{'='*40}")
    print(f"🤖 AGENT REASONING PROCESS")
    print(f"{'='*40}")
    for i, step in enumerate(result.agent_reasoning, 1):
        print(f"{i}. {step}")
    
    print(f"\n{'='*40}")
    print(f"💡 NOVEL RESEARCH DIRECTIONS")
    print(f"{'='*40}")
    for i, direction in enumerate(result.novel_directions, 1):
        print(f"{i}. {direction.title}")
        print(f"   Type: {direction.innovation_type}")
        print(f"   Impact: {direction.impact_potential:.2f} | Feasibility: {direction.feasibility_score:.2f}")
        print(f"   Confidence: {direction.confidence_score:.2f}")
        print(f"   Description: {direction.description}")
        
        if direction.implementation_roadmap:
            print(f"   Implementation Steps:")
            for step in direction.implementation_roadmap[:3]:
                print(f"     • {step}")
        
        if direction.potential_challenges:
            print(f"   Key Challenges:")
            for challenge in direction.potential_challenges[:2]:
                print(f"     ⚠️ {challenge}")
        
        if direction.success_metrics:
            print(f"   Success Metrics:")
            for metric in direction.success_metrics[:2]:
                print(f"     📊 {metric}")
        print()
    
    print(f"{'='*40}")
    print(f"🔗 INNOVATION SYNTHESIS")
    print(f"{'='*40}")
    
    print("🎯 Convergence Points:")
    for point in result.innovation_synthesis.convergence_points:
        print(f"  • {point}")
    
    print("\n🔍 Knowledge Gaps Identified:")
    for gap in result.innovation_synthesis.knowledge_gaps:
        print(f"  • {gap}")
    
    print("\n⚡ Technique Combinations:")
    for combo in result.innovation_synthesis.technique_combinations:
        print(f"  • {combo}")
    
    print("\n🔮 Emerging Patterns:")
    for pattern in result.innovation_synthesis.emerging_patterns:
        print(f"  • {pattern}")
    
    print(f"\n{'='*40}")
    print(f"📋 STRATEGIC RECOMMENDATIONS")
    print(f"{'='*40}")
    
    print("🎯 Priority Recommendations:")
    for rec in result.priority_recommendations:
        print(f"  • {rec}")
    
    print("\n🤝 Collaboration Suggestions:")
    for collab in result.collaboration_suggestions:
        print(f"  • {collab}")
    
    print("\n💰 Funding Opportunities:")
    for funding in result.funding_opportunities:
        print(f"  • {funding}")
    
    print("\n⏱️ Timeline Estimates:")
    for direction, timeline in result.timeline_estimates.items():
        print(f"  • {direction}: {timeline}")
    
    print(f"\n{'='*40}")
    print(f"📊 SYSTEM STATUS")
    print(f"{'='*40}")
    status = agent.get_agent_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print(f"\n{'='*60}")
    print(f"✅ Innovation Discovery Complete!")
    print(f"{'='*60}")
    print("🚀 Ready to revolutionize research with AI-powered innovation discovery!")
    print("\n💡 Next Steps:")
    print("  1. Choose highest-priority research direction")
    print("  2. Form interdisciplinary collaboration team") 
    print("  3. Apply for recommended funding opportunities")
    print("  4. Begin implementation roadmap")
    print("  5. Monitor success metrics and iterate")

if __name__ == "__main__":
    main()
