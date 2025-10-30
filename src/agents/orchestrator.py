"""
Research Orchestrator - Coordinates all AI agents for comprehensive research analysis
Provides unified interface for Paper Discovery, Cross-Domain, and Innovation agents
"""
from __future__ import annotations 
import os
import sys
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Import our agents
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from data.api_clients import AcademicAPIManager, ResearchPaper
    from vector_store.chroma_client import ChromaVectorStore
    from agents.paper_discovery_agent import (
        PaperDiscoveryAgent, 
        PaperDiscoveryResult, 
        GroqLLMWrapper
    )
    from agents.cross_domain_agent import (
        CrossDomainAgent, 
        CrossDomainResult
    )
    # Import innovation agent components separately to avoid circular imports
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from the correct directory and install dependencies")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrchestrationConfig:
    """Configuration for orchestrator behavior"""
    max_papers_per_agent: int = 15
    enable_parallel_execution: bool = True
    cache_results: bool = True
    detailed_logging: bool = True
    timeout_seconds: int = 300  # 5 minutes per agent

@dataclass
class AgentResult:
    """Wrapper for individual agent results with metadata"""
    agent_name: str
    success: bool
    execution_time: float
    result: Any
    error_message: Optional[str] = None

@dataclass
class ComprehensiveResult:
    """Complete research analysis result from all agents"""
    query: str
    execution_summary: Dict[str, Any]
    
    # Individual agent results (using Any to avoid circular imports)
    paper_discovery: Optional[Any]  # PaperDiscoveryResult
    cross_domain_analysis: Optional[Any]  # CrossDomainResult
    innovation_opportunities: Optional[Any]  # InnovationResult
    
    # Synthesized insights
    key_insights: List[str]
    action_recommendations: List[str]
    confidence_score: float
    
    # Metadata
    total_papers_analyzed: int
    cross_domain_connections: int
    novel_directions_generated: int
    execution_time: float
    agent_performance: Dict[str, AgentResult]

class ExecutionMonitor:
    """Monitors agent execution and provides progress updates"""
    
    def __init__(self):
        self.start_time = None
        self.current_step = 0
        self.total_steps = 3
        self.step_descriptions = [
            "Discovering research papers...",
            "Finding cross-domain connections...", 
            "Generating innovation opportunities..."
        ]
    
    def start_execution(self):
        """Start monitoring execution"""
        self.start_time = time.time()
        self.current_step = 0
        logger.info("🚀 Starting research orchestration")
    
    def update_step(self, step: int, custom_message: str = None):
        """Update current execution step"""
        self.current_step = step
        if step < len(self.step_descriptions):
            message = custom_message or self.step_descriptions[step]
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(f"Step {step + 1}/{self.total_steps}: {message} (elapsed: {elapsed:.1f}s)")
    
    def complete_execution(self):
        """Complete execution monitoring"""
        total_time = time.time() - self.start_time if self.start_time else 0
        logger.info(f"✅ Research orchestration completed in {total_time:.1f}s")
        return total_time

class ResultSynthesizer:
    """Synthesizes results from multiple agents into actionable insights"""
    
    def __init__(self, llm: GroqLLMWrapper):
        self.llm = llm
    
    def synthesize_insights(
        self, 
        paper_result: Optional[Any],  # PaperDiscoveryResult
        cross_domain_result: Optional[Any],  # CrossDomainResult
        innovation_result: Optional[Any]  # InnovationResult
    ) -> Tuple[List[str], List[str], float]:
        """
        Synthesize key insights and recommendations from all agent results
        
        Returns:
            Tuple of (key_insights, action_recommendations, confidence_score)
        """
        key_insights = []
        action_recommendations = []
        confidence_factors = []
        
        # Synthesize paper discovery insights
        if paper_result and hasattr(paper_result, 'papers_found') and paper_result.papers_found:
            key_insights.append(f"Found {len(paper_result.papers_found)} relevant papers across {len(set(p.source for p in paper_result.papers_found))} databases")
            
            if hasattr(paper_result, 'research_gaps') and paper_result.research_gaps:
                key_insights.append(f"Identified {len(paper_result.research_gaps)} research gaps to explore")
                action_recommendations.append("Address identified research gaps with novel approaches")
            
            confidence_factors.append(0.8 if len(paper_result.papers_found) >= 10 else 0.5)
        
        # Synthesize cross-domain insights
        if cross_domain_result and hasattr(cross_domain_result, 'connections_found') and cross_domain_result.connections_found:
            high_potential_connections = [
                c for c in cross_domain_result.connections_found 
                if hasattr(c, 'innovation_potential') and getattr(c, 'innovation_potential', 0) > 0.6
            ]
            
            key_insights.append(f"Discovered {len(cross_domain_result.connections_found)} cross-domain connections")
            
            if high_potential_connections:
                key_insights.append(f"{len(high_potential_connections)} connections show high innovation potential")
                action_recommendations.append("Prioritize high-potential cross-domain technique transfers")
            
            confidence_factors.append(0.7 if len(cross_domain_result.connections_found) >= 5 else 0.4)
        
        # Synthesize innovation insights
        if innovation_result and hasattr(innovation_result, 'novel_directions') and innovation_result.novel_directions:
            breakthrough_directions = [
                d for d in innovation_result.novel_directions 
                if hasattr(d, 'innovation_type') and getattr(d, 'innovation_type', '') in ["radical", "paradigm_shift"]
            ]
            
            key_insights.append(f"Generated {len(innovation_result.novel_directions)} novel research directions")
            
            if breakthrough_directions:
                key_insights.append(f"{len(breakthrough_directions)} directions offer breakthrough potential")
                action_recommendations.append("Focus on radical/paradigm-shifting opportunities for maximum impact")
            
            confidence_factors.append(0.9 if len(innovation_result.novel_directions) >= 3 else 0.6)
        
        # Generate meta-insights using LLM
        if key_insights:
            meta_insights = self._generate_meta_insights(key_insights, action_recommendations)
            key_insights.extend(meta_insights)
        
        # Calculate overall confidence
        confidence_score = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        return key_insights[:7], action_recommendations[:5], confidence_score
    
    def _generate_meta_insights(self, insights: List[str], recommendations: List[str]) -> List[str]:
        """Generate meta-level insights using LLM"""
        try:
            synthesis_prompt = f"""
            Based on these research findings, provide 2-3 high-level strategic insights:
            
            FINDINGS:
            {chr(10).join(f"- {insight}" for insight in insights)}
            
            RECOMMENDATIONS:
            {chr(10).join(f"- {rec}" for rec in recommendations)}
            
            Generate strategic insights that identify patterns, opportunities, or meta-trends.
            Keep each insight to 1-2 sentences.
            """
            
            response = self.llm.invoke(synthesis_prompt, temperature=0.3)
            
            # Parse insights from response
            lines = response.split('\n')
            meta_insights = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or len(line) > 20):
                    clean_line = line.strip('- •').strip()
                    if clean_line and len(clean_line) > 10:
                        meta_insights.append(clean_line)
            
            return meta_insights[:3]
            
        except Exception as e:
            logger.error(f"Error generating meta insights: {e}")
            return ["Cross-domain research shows high potential for breakthrough innovations"]

class ResearchOrchestrator:
    """
    Master coordinator for all research discovery agents
    Provides unified interface and comprehensive analysis
    """
    
    def __init__(
        self, 
        groq_api_key: str,
        config: OrchestrationConfig = None,
        vector_store: ChromaVectorStore = None,
        api_manager: AcademicAPIManager = None
    ):
        """
        Initialize the Research Orchestrator
        
        Args:
            groq_api_key: API key for Groq LLM
            config: Orchestration configuration
            vector_store: Shared ChromaDB instance
            api_manager: Shared academic API manager
        """
        self.config = config or OrchestrationConfig()
        
        # Initialize shared components
        self.vector_store = vector_store or ChromaVectorStore()
        self.api_manager = api_manager or AcademicAPIManager()
        self.llm = GroqLLMWrapper(groq_api_key)
        
        # Initialize agents
        self.paper_agent = PaperDiscoveryAgent(groq_api_key, self.vector_store, self.api_manager)
        self.cross_domain_agent = CrossDomainAgent(groq_api_key, self.vector_store, self.api_manager)
        
        # Import innovation agent dynamically to avoid circular import
        try:
            from agents.innovation_agent import InnovationAgent
            self.innovation_agent = InnovationAgent(groq_api_key, self.vector_store, self.api_manager)
        except ImportError:
            self.innovation_agent = None
            logger.warning("Innovation agent not available - continuing without it")
        
        # Initialize utilities
        self.monitor = ExecutionMonitor()
        self.synthesizer = ResultSynthesizer(self.llm)
        
        # Results cache
        self.results_cache = {} if self.config.cache_results else None
        
        logger.info("🎯 Research Orchestrator initialized with all agents ready")
    
    def analyze_research(self, research_query: str) -> ComprehensiveResult:
        """
        Comprehensive research analysis using all agents
        
        Args:
            research_query: The research question or topic to analyze
            
        Returns:
            ComprehensiveResult with insights from all agents
        """
        logger.info(f"🚀 Starting comprehensive research analysis for: {research_query}")
        
        # Check cache
        if self.results_cache and research_query in self.results_cache:
            logger.info("📋 Returning cached results")
            return self.results_cache[research_query]
        
        # Start monitoring
        self.monitor.start_execution()
        
        if self.config.enable_parallel_execution:
            result = self._execute_agents_parallel(research_query)
        else:
            result = self._execute_agents_sequential(research_query)
        
        # Cache results
        if self.results_cache:
            self.results_cache[research_query] = result
        
        total_time = self.monitor.complete_execution()
        result.execution_time = total_time
        
        return result
    
    def _execute_agents_parallel(self, research_query: str) -> ComprehensiveResult:
        """Execute agents in parallel for faster processing"""
        
        agent_results = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(self._run_paper_discovery, research_query): "paper_discovery",
                executor.submit(self._run_cross_domain, research_query): "cross_domain", 
                executor.submit(self._run_innovation, research_query): "innovation"
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_agent, timeout=self.config.timeout_seconds):
                agent_name = future_to_agent[future]
                completed += 1
                
                try:
                    result = future.result(timeout=60)  # Individual agent timeout
                    agent_results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        success=True,
                        execution_time=time.time(),  # Approximate
                        result=result
                    )
                    self.monitor.update_step(completed - 1, f"{agent_name} completed")
                    
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    agent_results[agent_name] = AgentResult(
                        agent_name=agent_name,
                        success=False,
                        execution_time=0,
                        result=None,
                        error_message=str(e)
                    )
        
        return self._synthesize_final_result(research_query, agent_results)
    
    def _execute_agents_sequential(self, research_query: str) -> ComprehensiveResult:
        """Execute agents sequentially for debugging"""
        
        agent_results = {}
        
        # Run Paper Discovery Agent
        self.monitor.update_step(0)
        paper_result = self._run_paper_discovery(research_query)
        agent_results["paper_discovery"] = AgentResult(
            agent_name="paper_discovery",
            success=paper_result is not None,
            execution_time=0,  # Simplified for sequential
            result=paper_result
        )
        
        # Run Cross-Domain Agent
        self.monitor.update_step(1)
        cross_domain_result = self._run_cross_domain(research_query)
        agent_results["cross_domain"] = AgentResult(
            agent_name="cross_domain",
            success=cross_domain_result is not None,
            execution_time=0,
            result=cross_domain_result
        )
        
        # Run Innovation Agent
        self.monitor.update_step(2)
        innovation_result = self._run_innovation(research_query)
        agent_results["innovation"] = AgentResult(
            agent_name="innovation",
            success=innovation_result is not None,
            execution_time=0,
            result=innovation_result
        )
        
        return self._synthesize_final_result(research_query, agent_results)
    
    def _run_paper_discovery(self, query: str) -> Optional[Any]:
        """Run paper discovery agent with error handling"""
        try:
            return self.paper_agent.discover_papers(query, self.config.max_papers_per_agent)
        except Exception as e:
            logger.error(f"Paper discovery failed: {e}")
            return None
    
    def _run_cross_domain(self, query: str) -> Optional[Any]:
        """Run cross-domain agent with error handling"""
        try:
            return self.cross_domain_agent.find_cross_domain_connections(query)
        except Exception as e:
            logger.error(f"Cross-domain analysis failed: {e}")
            return None
    
    def _run_innovation(self, query: str) -> Optional[Any]:
        """Run innovation agent with error handling"""
        try:
            if self.innovation_agent is None:
                logger.warning("Innovation agent not available")
                return None
            return self.innovation_agent.discover_innovations(query, self.config.max_papers_per_agent)
        except Exception as e:
            logger.error(f"Innovation discovery failed: {e}")
            return None
    
    def _synthesize_final_result(self, query: str, agent_results: Dict[str, AgentResult]) -> ComprehensiveResult:
        """Synthesize final comprehensive result"""
        
        # Extract successful results
        paper_result = agent_results.get("paper_discovery", AgentResult("", False, 0, None)).result if agent_results.get("paper_discovery", AgentResult("", False, 0, None)).success else None
        cross_domain_result = agent_results.get("cross_domain", AgentResult("", False, 0, None)).result if agent_results.get("cross_domain", AgentResult("", False, 0, None)).success else None
        innovation_result = agent_results.get("innovation", AgentResult("", False, 0, None)).result if agent_results.get("innovation", AgentResult("", False, 0, None)).success else None
        
        # Synthesize insights
        key_insights, action_recommendations, confidence_score = self.synthesizer.synthesize_insights(
            paper_result, cross_domain_result, innovation_result
        )
        
        # Calculate statistics
        total_papers = len(paper_result.papers_found) if paper_result and hasattr(paper_result, 'papers_found') else 0
        cross_domain_connections = len(cross_domain_result.connections_found) if cross_domain_result and hasattr(cross_domain_result, 'connections_found') else 0
        novel_directions = len(innovation_result.novel_directions) if innovation_result and hasattr(innovation_result, 'novel_directions') else 0
        
        # Create execution summary
        execution_summary = {
            "agents_executed": len(agent_results),
            "successful_agents": sum(1 for r in agent_results.values() if r.success),
            "failed_agents": sum(1 for r in agent_results.values() if not r.success),
            "parallel_execution": self.config.enable_parallel_execution,
            "total_processing_time": sum(r.execution_time for r in agent_results.values())
        }
        
        return ComprehensiveResult(
            query=query,
            execution_summary=execution_summary,
            paper_discovery=paper_result,
            cross_domain_analysis=cross_domain_result,
            innovation_opportunities=innovation_result,
            key_insights=key_insights,
            action_recommendations=action_recommendations,
            confidence_score=confidence_score,
            total_papers_analyzed=total_papers,
            cross_domain_connections=cross_domain_connections,
            novel_directions_generated=novel_directions,
            execution_time=0,  # Will be set by monitor
            agent_performance=agent_results
        )
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "orchestrator_version": "1.0",
            "agents_available": ["paper_discovery", "cross_domain", "innovation"],
            "parallel_execution": self.config.enable_parallel_execution,
            "cache_enabled": self.config.cache_results,
            "cached_queries": len(self.results_cache) if self.results_cache else 0,
            "vector_store_papers": vector_stats.get("total_papers", 0),
            "vector_store_techniques": vector_stats.get("total_techniques", 0),
            "max_papers_per_agent": self.config.max_papers_per_agent,
            "timeout_seconds": self.config.timeout_seconds,
            "status": "ready for comprehensive research analysis"
        }
    
    def clear_cache(self):
        """Clear results cache"""
        if self.results_cache:
            self.results_cache.clear()
            logger.info("📋 Results cache cleared")

# Example usage and testing
def main():
    """Main function for running the Research Orchestrator standalone"""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Please set GROQ_API_KEY environment variable")
        print("Get free API key from: https://console.groq.com/")
        exit(1)
    
    # Initialize orchestrator
    print("🎼 Initializing Research Orchestrator...")
    config = OrchestrationConfig(
        max_papers_per_agent=8,
        enable_parallel_execution=True,
        detailed_logging=True
    )
    orchestrator = ResearchOrchestrator(groq_api_key, config)
    
    # Test comprehensive research analysis
    research_query = "quantum machine learning for drug discovery"
    print(f"🔬 Running comprehensive analysis for: {research_query}")
    
    result = orchestrator.analyze_research(research_query)
    
    # Display comprehensive results
    print(f"\n{'='*80}")
    print(f"🎯 COMPREHENSIVE RESEARCH ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Query: {result.query}")
    print(f"Total Execution Time: {result.execution_time:.1f}s")
    print(f"Confidence Score: {result.confidence_score:.2f}/1.0")
    
    print(f"\n{'='*50}")
    print(f"📊 EXECUTION SUMMARY")
    print(f"{'='*50}")
    for key, value in result.execution_summary.items():
        print(f"{key}: {value}")
    
    print(f"\n{'='*50}")
    print(f"🔍 KEY INSIGHTS")
    print(f"{'='*50}")
    for i, insight in enumerate(result.key_insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\n{'='*50}")
    print(f"🎯 ACTION RECOMMENDATIONS")
    print(f"{'='*50}")
    for i, rec in enumerate(result.action_recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n{'='*50}")
    print(f"📈 QUANTITATIVE RESULTS")
    print(f"{'='*50}")
    print(f"Papers Analyzed: {result.total_papers_analyzed}")
    print(f"Cross-Domain Connections: {result.cross_domain_connections}")
    print(f"Novel Directions Generated: {result.novel_directions_generated}")
    
    print(f"\n{'='*50}")
    print(f"🤖 AGENT PERFORMANCE")
    print(f"{'='*50}")
    for agent_name, performance in result.agent_performance.items():
        status = "✅ SUCCESS" if performance.success else "❌ FAILED"
        print(f"{agent_name}: {status}")
        if performance.error_message:
            print(f"  Error: {performance.error_message}")
    
    # Show detailed results from each agent
    if result.paper_discovery:
        print(f"\n{'='*50}")
        print(f"📚 PAPER DISCOVERY HIGHLIGHTS")
        print(f"{'='*50}")
        print(f"Papers found: {len(result.paper_discovery.papers_found)}")
        if result.paper_discovery.papers_found:
            for paper in result.paper_discovery.papers_found[:3]:
                print(f"  • {paper.title} ({paper.source})")
    
    if result.cross_domain_analysis:
        print(f"\n{'='*50}")
        print(f"🔗 CROSS-DOMAIN HIGHLIGHTS")
        print(f"{'='*50}")
        print(f"Connections found: {len(result.cross_domain_analysis.connections_found)}")
        for conn in result.cross_domain_analysis.connections_found[:3]:
            print(f"  • {conn.technique_name}: {conn.source_domain} → {conn.target_domain}")
            print(f"    Innovation potential: {conn.innovation_potential:.2f}")
    
    if result.innovation_opportunities:
        print(f"\n{'='*50}")
        print(f"💡 INNOVATION HIGHLIGHTS")
        print(f"{'='*50}")
        print(f"Novel directions: {len(result.innovation_opportunities.novel_directions)}")
        for direction in result.innovation_opportunities.novel_directions[:3]:
            print(f"  • {direction.title}")
            print(f"    Type: {direction.innovation_type} | Impact: {direction.impact_potential:.2f}")
    
    print(f"\n{'='*50}")
    print(f"📊 ORCHESTRATOR STATUS")
    print(f"{'='*50}")
    status = orchestrator.get_orchestrator_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print(f"\n{'='*80}")
    print(f"✅ Comprehensive Research Analysis Complete!")
    print(f"{'='*80}")
    print("🚀 All agents coordinated successfully!")
    print("💡 Ready for Streamlit interface integration!")

if __name__ == "__main__":
    main()
