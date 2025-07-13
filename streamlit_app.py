"""
AI Research Innovation Mapper - Streamlit Web Application
Interactive interface for discovering cross-domain research innovations
"""

import streamlit as st
import sys
import os
from typing import Dict, List, Any, Optional
import time
import json
import traceback
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import networkx as nx
import numpy as np
import logging

# Set up logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import our system components
try:
    from utils.config import get_config, validate_system, get_api_keys_status
    from utils.logging_config import init_logging, get_component_logger
    from utils.demo_data import get_demo_data
    from agents.orchestrator import ResearchOrchestrator, OrchestrationConfig
    from data.api_clients import AcademicAPIManager
    from vector_store.chroma_client import ChromaVectorStore
        
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🔬 AI Research Innovation Mapper",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .innovation-card {
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .connection-badge {
        background: #FF6B6B;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .demo-scenario {
        border-left: 4px solid #45B7D1;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 5px;
        color: #333;
    }
    
    .demo-scenario h4 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .demo-scenario p {
        color: #555;
        margin-bottom: 0.3rem;
    }
    
    .demo-scenario code {
        background: #e9ecef;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        color: #d63384;
        font-family: monospace;
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = get_config()
        self.demo_data = get_demo_data()
        self.orchestrator = None
        self.logger = None
        
        # Initialize session state
        self._init_session_state()
        
        # Initialize logging
        self._init_logging()
        
        # Initialize orchestrator if API key available
        self._init_orchestrator()
    
    def _init_session_state(self):
        """Initialize Streamlit session state"""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = False
        
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = "checking"
    
    def _init_logging(self):
        """Initialize logging system"""
        try:
            init_logging(use_json=False, console_level="INFO")
            self.logger = get_component_logger("streamlit_ui")
            self.logger.log_user_interaction("app_startup")
        except Exception as e:
            st.warning(f"Logging initialization failed: {e}")
    
    def _init_orchestrator(self):
        """Initialize research orchestrator"""
        try:
            api_status = get_api_keys_status()
            
            if api_status.get("groq_api_key", False):
                orchestrator_config = OrchestrationConfig(
                    max_papers_per_agent=8,  # Fixed moderate value
                    enable_parallel_execution=self.config.agents.enable_parallel_execution,
                    cache_results=self.config.agents.enable_caching
                )
                
                self.orchestrator = ResearchOrchestrator(
                    groq_api_key=self.config.api.groq_api_key,
                    config=orchestrator_config
                )
                
                st.session_state.system_status = "ready"
                if self.logger:
                    self.logger.log_user_interaction("orchestrator_initialized")
            else:
                st.session_state.system_status = "missing_api_key"
                
        except Exception as e:
            # If tensor error or other initialization issue, enable demo mode
            logger.warning(f"Orchestrator initialization failed: {e}")
            if "tensor" in str(e).lower() or "device" in str(e).lower():
                st.session_state.system_status = "demo_mode"
                st.session_state.demo_mode = True
                logger.info("🎭 Enabling demo mode due to tensor/device issues")
            else:
                st.session_state.system_status = "error"
            
            if self.logger:
                self.logger.log_agent_error("orchestrator", e, "initialization")
    
    def run(self):
        """Main application entry point"""
        self._render_header()
        
        # Check system status
        if st.session_state.system_status == "missing_api_key":
            self._render_setup_page()
            return
        elif st.session_state.system_status == "error":
            self._render_error_page()
            return
        elif st.session_state.system_status == "demo_mode":
            st.info("🎭 **Demo Mode Active**: System running with simulated data due to initialization issues. Full functionality available with proper setup.")
            st.session_state.demo_mode = True
        
        # Main application
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🔬 AI Research Innovation Mapper</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Discover breakthrough research opportunities by connecting insights across domains
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_setup_page(self):
        """Render setup page for missing API keys"""
        st.warning("🔧 **Setup Required**")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### Welcome to AI Research Innovation Mapper!
            
            To get started, you need to set up your API keys:
            
            #### Required API Key:
            - **Groq API Key**: Free tier available at [console.groq.com](https://console.groq.com/)
            
            #### Setup Instructions:
            1. Get your free Groq API key
            2. Create a `.env` file in the project root
            3. Add: `GROQ_API_KEY=your_key_here`
            4. Restart the application
            
            #### Or set environment variable:
            ```bash
            export GROQ_API_KEY="your_key_here"
            ```
            """)
            
            # Demo mode option
            st.markdown("---")
            st.markdown("### 🎭 Try Demo Mode")
            st.markdown("Experience the interface with pre-loaded demo data (limited functionality)")
            
            if st.button("🚀 Launch Demo Mode", type="primary"):
                st.session_state.demo_mode = True
                st.rerun()
    
    def _render_error_page(self):
        """Render error page"""
        st.error("❌ **System Error**")
        st.markdown("There was an error initializing the system. Please check the logs and try again.")
        
        if st.button("🔄 Retry Initialization"):
            st.session_state.system_status = "checking"
            st.rerun()
    
    def _render_sidebar(self):
        """Render application sidebar"""
        with st.sidebar:
            st.markdown("## ⚙️ Settings")
            
            # Minimal settings - removed max_papers slider
            st.session_state.enable_parallel = st.checkbox("Parallel execution", True)
            st.session_state.show_debug = st.checkbox("Show debug info", False)
            
            # Clear results button
            st.markdown("---")
            if st.button("🧹 Clear Results"):
                st.session_state.analysis_results = None
                st.session_state.current_query = ""
                st.rerun()
    
    def _render_main_content(self):
        """Render main application content"""
        # Query input section
        self._render_query_input()
        
        # Results section
        if st.session_state.analysis_results:
            self._render_results()
        else:
            self._render_welcome_content()
    
    def _render_query_input(self):
        """Render query input section"""
        st.markdown("## 🔍 Research Query")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Enter your research question or topic:",
                value=st.session_state.current_query,
                placeholder="e.g., 'AI for cancer detection using medical imaging'",
                help="Describe your research interest - the AI will find cross-domain connections and innovations"
            )
            st.session_state.current_query = query
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            analyze_button = st.button("🚀 Analyze", type="primary", disabled=not query.strip())
        
        # Simple suggestions
        if not query.strip():
            st.markdown("**Example queries:** deep learning medical imaging, quantum computing drug discovery, whale song speech therapy")
        
        # Run analysis
        if analyze_button and query.strip():
            self._run_analysis(query)
    
    def _run_analysis(self, query: str):
        """Run research analysis"""
        if self.logger:
            self.logger.log_user_interaction("analysis_start", {"query": query})
        
        # Demo mode handling
        if st.session_state.demo_mode or not self.orchestrator:
            self._run_demo_analysis(query)
            return
        
        # Real analysis
        try:
            with st.spinner("🤖 AI agents are analyzing your query..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                status_text.text("🔍 Searching research papers...")
                progress_bar.progress(20)
                time.sleep(1)
                
                status_text.text("🔗 Finding cross-domain connections...")
                progress_bar.progress(50)
                time.sleep(1)
                
                status_text.text("💡 Generating innovation opportunities...")
                progress_bar.progress(80)
                
                # Run actual analysis
                result = self.orchestrator.analyze_research(query)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state.analysis_results = result
                st.session_state.analysis_history.append({
                    "query": query,
                    "timestamp": datetime.now(),
                    "summary": f"{result.total_papers_analyzed} papers, {result.cross_domain_connections} connections"
                })
                
                if self.logger:
                    self.logger.log_user_interaction("analysis_complete", {
                        "query": query,
                        "papers_found": result.total_papers_analyzed,
                        "connections": result.cross_domain_connections
                    })
                
                st.success("🎉 Analysis complete! Scroll down to see results.")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            if self.logger:
                self.logger.log_agent_error("streamlit_analysis", e, query)
            
            if st.session_state.show_debug:
                st.text(traceback.format_exc())
    
    def _run_demo_analysis(self, query: str):
        """Run demo analysis with simulated results"""
        with st.spinner("🎭 Generating demo results..."):
            progress_bar = st.progress(0)
            
            # Simulate analysis steps
            for i, step in enumerate(["Searching papers", "Finding connections", "Generating innovations"], 1):
                st.text(f"🤖 {step}...")
                time.sleep(1)
                progress_bar.progress(i * 33)
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            
            # Create demo results
            demo_result = self._create_demo_result(query)
            st.session_state.analysis_results = demo_result
            
            st.success("🎭 Demo analysis complete!")
            st.rerun()
    
    def _create_demo_result(self, query: str):
        """Create realistic demo results"""
        demo_data = self.demo_data  # Store reference for inner class
        
        class DemoResult:
            def __init__(self, query):
                self.query = query
                self.total_papers_analyzed = 12
                self.cross_domain_connections = 2  # Limited to 2 as intended
                self.novel_directions_generated = 2  # Also limited to 2
                self.confidence_score = 0.85
                self.execution_time = 45.2
                
                # Sample data
                self.key_insights = [
                    f"Found {self.total_papers_analyzed} relevant papers across multiple domains",
                    f"Discovered {self.cross_domain_connections} high-potential cross-domain connections",
                    "Strongest technique transfer potential from astronomy domain",
                    "2 breakthrough innovation opportunities identified"
                ]
                
                self.action_recommendations = [
                    "Prioritize astronomy image processing techniques for medical applications",
                    "Establish collaboration with computer vision researchers",
                    "Focus on radical innovation opportunities for maximum impact",
                    "Apply for NSF Convergence Accelerator funding"
                ]
                
                # Mock paper discovery results
                self.paper_discovery = type('obj', (), {
                    'papers_found': demo_data.get_sample_papers(count=5),
                    'research_gaps': ["Limited cross-domain technique exploration", "Underutilized transfer learning"]
                })()
                
                # Mock cross-domain results - LIMIT TO 2 CONNECTIONS
                sample_connections = demo_data.get_sample_connections(count=2)  # Changed from 6 to 2
                self.cross_domain_analysis = type('obj', (), {
                    'connections_found': sample_connections,
                    'target_domain': "medical"  # Default domain
                })()
                
                # Mock innovation results - LIMIT TO 2 DIRECTIONS
                self.innovation_opportunities = type('obj', (), {
                    'novel_directions': demo_data.get_sample_innovations(count=2),  # Changed from 3 to 2
                    'priority_recommendations': self.action_recommendations[:2]
                })()
        
        return DemoResult(query)
    
    def _infer_domain(self, query: str) -> str:
        """Simple domain inference for demo"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["medical", "cancer", "disease"]):
            return "medical"
        elif any(word in query_lower for word in ["ai", "machine learning", "neural"]):
            return "computer_science"
        elif any(word in query_lower for word in ["quantum", "physics"]):
            return "physics"
        else:
            return "general"
    
    def _render_welcome_content(self):
        """Render welcome content when no results"""
        st.markdown("## 🔬 AI Research Innovation Mapper")
        
        st.markdown("""
        ### How It Works
        
        **Multi-Agent AI System** analyzes research across multiple domains:
        - **Paper Discovery Agent**: Searches ArXiv, PubMed, and bioRxiv
        - **Cross-Domain Agent**: Finds technique transfers between fields  
        - **Innovation Agent**: Generates breakthrough research opportunities
        
        **Cross-Domain Connections** discovered:
        - Astronomy image processing → Medical imaging
        - Whale song analysis → Speech therapy
        - Quantum computing → Drug discovery
        
        Enter your research query above to discover hidden connections and innovation opportunities.
        """)
        
        # Simple metrics if orchestrator available
        if self.orchestrator:
            stats = self.demo_data.get_statistics()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Research Domains", stats['domains_covered'])
            with col2:
                st.metric("Sample Papers", stats['total_sample_papers'])
            with col3:
                st.metric("Demo Scenarios", stats['total_demo_scenarios'])
    
    def _render_results(self):
        """Render analysis results"""
        result = st.session_state.analysis_results
        
        # Results header
        st.markdown("## 🎉 Analysis Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📚 Papers Analyzed", 
                result.total_papers_analyzed,
                help="Total research papers found and analyzed"
            )
        
        with col2:
            st.metric(
                "🔗 Cross-Domain Connections", 
                result.cross_domain_connections,
                help="Technique transfers found between domains"
            )
        
        with col3:
            st.metric(
                "💡 Novel Directions", 
                result.novel_directions_generated,
                help="Breakthrough research opportunities identified"
            )
        
        with col4:
            st.metric(
                "🎯 Confidence Score", 
                f"{result.confidence_score:.1%}",
                help="System confidence in the analysis"
            )
        
        # Tabs for different result sections  
        tab1, tab2, tab3 = st.tabs([
            "🔍 Key Insights", 
            "📚 Papers Found", 
            "🔗 Cross-Domain"
        ])
        
        with tab1:
            self._render_insights_tab(result)
        
        with tab2:
            self._render_papers_tab(result)
        
        with tab3:
            self._render_cross_domain_tab(result)
            
            # Add innovations to the cross-domain tab to save space
            if hasattr(result, 'innovation_opportunities') and result.innovation_opportunities.novel_directions:
                st.markdown("---")
                st.markdown("### 💡 Innovation Opportunities")
                
                directions = result.innovation_opportunities.novel_directions[:2]  # Only show 2
                
                for i, direction in enumerate(directions, 1):
                    if isinstance(direction, dict):
                        title = direction.get('title', f'Innovation Direction {i}')
                        description = direction.get('description', 'No description available')
                        innovation_type = direction.get('innovation_type', 'unknown')
                        impact = direction.get('impact_potential', 0.5)
                        feasibility = direction.get('feasibility_score', 0.5)
                    else:
                        title = getattr(direction, 'title', f'Innovation Direction {i}')
                        description = getattr(direction, 'description', 'No description available')
                        innovation_type = getattr(direction, 'innovation_type', 'unknown')
                        impact = getattr(direction, 'impact_potential', 0.5)
                        feasibility = getattr(direction, 'feasibility_score', 0.5)
                    
                    with st.expander(f"💡 {title}", expanded=(i == 1)):
                        type_colors = {'incremental': '🟢', 'radical': '🟡', 'paradigm_shift': '🔴'}
                        type_icon = type_colors.get(innovation_type, '⚪')
                        
                        st.markdown(f"**Type**: {type_icon} {innovation_type.replace('_', ' ').title()}")
                        st.markdown(f"**Description**: {description}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Impact", f"{impact:.1%}")
                        with col2:
                            st.metric("Feasibility", f"{feasibility:.1%}")
    
    def _render_insights_tab(self, result):
        """Render key insights tab"""
        st.markdown("### 🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💡 **Discoveries**")
            for insight in result.key_insights:
                st.markdown(f"• {insight}")
        
        with col2:
            st.markdown("#### 🎯 **Recommended Actions**")
            for action in result.action_recommendations:
                st.markdown(f"• {action}")
        
        # Execution summary
        st.markdown("---")
        st.markdown("### ⚡ Execution Summary")
        
        exec_col1, exec_col2, exec_col3 = st.columns(3)
        
        with exec_col1:
            st.markdown(f"**Total Time**: {result.execution_time:.1f}s")
        
        with exec_col2:
            st.markdown(f"**Query**: {result.query}")
        
        with exec_col3:
            st.markdown(f"**Confidence**: {result.confidence_score:.1%}")
    
    def _render_papers_tab(self, result):
        """Render papers found tab"""
        st.markdown("### 📚 Research Papers Found")
        
        if hasattr(result, 'paper_discovery') and result.paper_discovery.papers_found:
            papers = result.paper_discovery.papers_found
            
            # Papers by source
            sources = {}
            for paper in papers:
                if paper.source not in sources:
                    sources[paper.source] = []
                sources[paper.source].append(paper)
            
            for source, source_papers in sources.items():
                st.markdown(f"#### 📖 {source.upper()} ({len(source_papers)} papers)")
                
                for i, paper in enumerate(source_papers, 1):
                    with st.expander(f"{i}. {paper.title}"):
                        st.markdown(f"**Authors**: {', '.join(paper.authors[:3])}")
                        st.markdown(f"**Published**: {paper.published_date}")
                        st.markdown(f"**Categories**: {', '.join(paper.categories)}")
                        st.markdown(f"**Abstract**: {paper.abstract[:300]}...")
                        st.markdown(f"**URL**: [{paper.url}]({paper.url})")
            
            # Research gaps
            if hasattr(result.paper_discovery, 'research_gaps') and result.paper_discovery.research_gaps:
                st.markdown("---")
                st.markdown("#### 🔍 **Research Gaps Identified**")
                for gap in result.paper_discovery.research_gaps:
                    st.markdown(f"• {gap}")
        else:
            st.info("No papers found or demo mode active. Try a different query or check API connectivity.")
    
    def _render_cross_domain_tab(self, result):
        """Render cross-domain connections tab - LIMITED TO 2 CONNECTIONS"""
        st.markdown("### 🔗 Cross-Domain Connections")
        
        if hasattr(result, 'cross_domain_analysis') and result.cross_domain_analysis.connections_found:
            connections = result.cross_domain_analysis.connections_found[:2]  # LIMIT TO 2 CONNECTIONS
            
            for i, connection in enumerate(connections, 1):
                st.markdown(f"#### 🔗 Connection {i}")
                
                # Handle both dict and object formats
                if isinstance(connection, dict):
                    source_domain = connection.get('source_domain', 'Unknown')
                    target_domain = connection.get('target_domain', 'Unknown')
                    technique = connection.get('technique_name', 'Unknown technique')
                    feasibility = connection.get('transfer_feasibility', 0.5)
                    innovation = connection.get('innovation_potential', 0.5)
                    reasoning = connection.get('reasoning', 'No reasoning provided')
                else:
                    source_domain = getattr(connection, 'source_domain', 'Unknown')
                    target_domain = getattr(connection, 'target_domain', 'Unknown')
                    technique = getattr(connection, 'technique_name', 'Unknown technique')
                    feasibility = getattr(connection, 'transfer_feasibility', 0.5)
                    innovation = getattr(connection, 'innovation_potential', 0.5)
                    reasoning = getattr(connection, 'reasoning', 'No reasoning provided')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #4ECDC4;
                        border-radius: 10px;
                        padding: 1rem;
                        margin: 1rem 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    ">
                        <h5 style="color: white; margin-bottom: 0.5rem;">{technique}</h5>
                        <p style="color: #f0f0f0;"><strong>From:</strong> <span style="background: #FF6B6B; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">{source_domain}</span></p>
                        <p style="color: #f0f0f0;"><strong>To:</strong> <span style="background: #4ECDC4; color: white; padding: 0.2rem 0.5rem; border-radius: 15px; font-size: 0.8rem;">{target_domain}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Feasibility**: {feasibility:.1%}")
                    st.progress(feasibility)
                    st.markdown(f"**Innovation Potential**: {innovation:.1%}")
                    st.progress(innovation)
                
                st.markdown(f"**Reasoning**: {reasoning}")
                st.markdown("---")
        else:
            st.info("No cross-domain connections found. Try a more specific query or check system status.")

# Global app instance
app = StreamlitApp()

def main():
    """Main application entry point"""
    try:
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        if st.button("🔄 Restart Application"):
            st.rerun()

if __name__ == "__main__":
    main()