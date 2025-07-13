"""
Demo Data for AI Research Innovation Mapper
Provides pre-loaded examples, sample queries, and showcase scenarios for demonstrations
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
import random

# Import our data types
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from data.api_clients import ResearchPaper
except ImportError:
    # Fallback for standalone testing
    from dataclasses import dataclass
    from typing import List, Optional
    
    @dataclass
    class ResearchPaper:
        id: str
        title: str
        authors: List[str]
        abstract: str
        published_date: str
        categories: List[str]
        url: str
        source: str
        doi: Optional[str] = None
        journal: Optional[str] = None

@dataclass
class DemoScenario:
    """Complete demo scenario with query and expected results"""
    title: str
    description: str
    query: str
    expected_papers_count: int
    expected_domains: List[str]
    expected_connections: List[str]
    expected_innovations: List[str]
    wow_factor: str
    difficulty_level: str  # "beginner", "intermediate", "advanced"

@dataclass
class ShowcaseExample:
    """Showcase example for specific research domains"""
    domain: str
    problem_statement: str
    cross_domain_solutions: List[str]
    innovation_potential: str
    real_world_impact: str

class DemoDataGenerator:
    """Generates realistic demo data for the research innovation mapper"""
    
    def __init__(self):
        self.sample_papers = self._create_sample_papers()
        self.demo_scenarios = self._create_demo_scenarios()
        self.showcase_examples = self._create_showcase_examples()
        self.cross_domain_connections = self._create_sample_connections()
        self.innovation_directions = self._create_sample_innovations()
    
    def _create_sample_papers(self) -> List[ResearchPaper]:
        """Create realistic sample research papers"""
        papers = [
            # AI/ML Papers
            ResearchPaper(
                id="2024.01.001",
                title="Deep Learning for Astronomical Object Detection in Large-Scale Surveys",
                authors=["Dr. Sarah Chen", "Prof. Michael Rodriguez", "Dr. Emily Watson"],
                abstract="This paper presents a novel convolutional neural network architecture for detecting and classifying astronomical objects in large-scale sky surveys. Our method achieves 97.3% accuracy on the Sloan Digital Sky Survey dataset, significantly outperforming traditional photometric techniques. The approach uses attention mechanisms to focus on relevant spectral features and handles the class imbalance problem common in astronomical datasets.",
                published_date="2024-01-15",
                categories=["cs.CV", "astro-ph.IM"],
                url="https://arxiv.org/abs/2024.01.001",
                source="arxiv",
                doi="10.48550/arXiv.2024.01.001"
            ),
            
            ResearchPaper(
                id="2024.02.045",
                title="Transfer Learning from Whale Song Analysis to Human Speech Disorder Detection",
                authors=["Dr. Ocean Martinez", "Prof. Linda Zhao", "Dr. James Peterson"],
                abstract="We demonstrate successful transfer of deep learning models trained on whale vocalizations to detect speech disorders in humans. The temporal pattern recognition capabilities developed for marine mammal communication analysis prove highly effective for identifying subtle speech abnormalities. Our cross-species approach achieves 89.2% accuracy in detecting early-stage speech disorders, opening new avenues for non-invasive diagnostic tools.",
                published_date="2024-02-20",
                categories=["cs.SD", "q-bio.NC"],
                url="https://arxiv.org/abs/2024.02.045",
                source="arxiv",
                doi="10.48550/arXiv.2024.02.045"
            ),
            
            # Medical Papers
            ResearchPaper(
                id="PMID38901234",
                title="Machine Learning-Enhanced Protein Folding Prediction for Drug Discovery",
                authors=["Dr. Alex Thompson", "Dr. Maria Garcia", "Prof. David Kim"],
                abstract="This study applies advanced machine learning techniques to predict protein folding patterns for drug discovery applications. Using transformer architectures adapted from natural language processing, we achieve state-of-the-art accuracy in predicting 3D protein structures. The method successfully identified 12 novel drug targets for Alzheimer's disease treatment.",
                published_date="2024-03-10",
                categories=["medical"],
                url="https://pubmed.ncbi.nlm.nih.gov/38901234/",
                source="pubmed",
                journal="Nature Biotechnology"
            ),
            
            ResearchPaper(
                id="PMID38901567",
                title="Satellite Image Processing Techniques Applied to Medical Imaging",
                authors=["Dr. Remote Sensing", "Dr. Medical Imaging"],
                abstract="We adapt advanced satellite image processing algorithms for medical imaging applications. Techniques originally developed for analyzing Earth observation data prove remarkably effective for detecting subtle patterns in medical scans. Our approach improves cancer detection rates by 23% compared to traditional methods.",
                published_date="2024-03-25",
                categories=["medical"],
                url="https://pubmed.ncbi.nlm.nih.gov/38901567/",
                source="pubmed",
                journal="Medical Image Analysis"
            ),
            
            # Biology Papers
            ResearchPaper(
                id="biorxiv.2024.03.001",
                title="Quantum-Inspired Algorithms for Protein Interaction Network Analysis",
                authors=["Dr. Quantum Bio", "Prof. Network Analysis"],
                abstract="This paper introduces quantum-inspired algorithms for analyzing complex protein interaction networks. By leveraging quantum superposition principles, we can explore multiple network configurations simultaneously, leading to breakthrough insights in cellular pathway analysis. The method identifies previously unknown protein complexes with 94% accuracy.",
                published_date="2024-03-30",
                categories=["biology"],
                url="https://www.biorxiv.org/content/2024.03.001",
                source="biorxiv",
                doi="10.1101/2024.03.001"
            ),
            
            # Physics Papers
            ResearchPaper(
                id="2024.04.123",
                title="Signal Processing Techniques from Gravitational Wave Detection Applied to Biomedical Sensing",
                authors=["Dr. Wave Physics", "Dr. Bio Sensing"],
                abstract="We demonstrate how advanced signal processing techniques developed for gravitational wave detection can revolutionize biomedical sensing applications. The ultra-sensitive noise reduction algorithms used in LIGO prove highly effective for detecting minute biological signals, enabling new possibilities in non-invasive medical monitoring.",
                published_date="2024-04-05",
                categories=["physics.ins-det", "q-bio.QM"],
                url="https://arxiv.org/abs/2024.04.123",
                source="arxiv"
            ),
            
            # Chemistry Papers
            ResearchPaper(
                id="2024.05.089",
                title="NLP Transformers for Chemical Reaction Prediction and Drug Design",
                authors=["Dr. Chem AI", "Prof. Molecular Design"],
                abstract="This work adapts transformer architectures from natural language processing to predict chemical reactions and design novel drug compounds. By treating molecular structures as a specialized language, we achieve unprecedented accuracy in reaction outcome prediction and novel compound generation.",
                published_date="2024-05-12",
                categories=["physics.chem-ph", "cs.LG"],
                url="https://arxiv.org/abs/2024.05.089",
                source="arxiv"
            )
        ]
        
        return papers
    
    def _create_demo_scenarios(self) -> List[DemoScenario]:
        """Create comprehensive demo scenarios"""
        scenarios = [
            DemoScenario(
                title="🔬 AI-Powered Cancer Detection",
                description="Discover how astronomy image processing techniques can revolutionize medical imaging",
                query="deep learning cancer detection medical imaging",
                expected_papers_count=12,
                expected_domains=["computer_science", "medical", "astronomy"],
                expected_connections=[
                    "Galaxy classification → Tumor detection",
                    "Stellar image enhancement → Medical scan preprocessing",
                    "Spectral analysis → Biomarker identification"
                ],
                expected_innovations=[
                    "Multi-scale astronomical feature extraction for cancer screening",
                    "Adaptive noise reduction from radio astronomy applied to MRI",
                    "Automated anomaly detection inspired by supernova discovery"
                ],
                wow_factor="Techniques that found black holes now detect cancer cells!",
                difficulty_level="intermediate"
            ),
            
            DemoScenario(
                title="🐋 Whale Songs for Speech Therapy",
                description="How marine biology research can advance human speech disorder treatment",
                query="whale song analysis speech recognition therapy",
                expected_papers_count=8,
                expected_domains=["biology", "computer_science", "medical"],
                expected_connections=[
                    "Whale vocalization patterns → Speech disorder detection",
                    "Marine acoustic analysis → Human audio processing",
                    "Echolocation algorithms → Hearing aid technology"
                ],
                expected_innovations=[
                    "Bio-inspired speech pattern recognition",
                    "Cross-species audio feature extraction",
                    "Natural communication models for therapy"
                ],
                wow_factor="What whales taught us about human communication!",
                difficulty_level="beginner"
            ),
            
            DemoScenario(
                title="🌌 Quantum Biology Meets Drug Discovery",
                description="Applying quantum physics principles to understand biological systems",
                query="quantum mechanics protein folding drug discovery",
                expected_papers_count=15,
                expected_domains=["physics", "biology", "chemistry", "medical"],
                expected_connections=[
                    "Quantum superposition → Protein state analysis",
                    "Quantum entanglement → Molecular interaction modeling",
                    "Quantum algorithms → Drug compound optimization"
                ],
                expected_innovations=[
                    "Quantum-enhanced molecular dynamics simulations",
                    "Superposition-based drug screening protocols",
                    "Entanglement-inspired biomarker discovery"
                ],
                wow_factor="Quantum physics unlocking the secrets of life itself!",
                difficulty_level="advanced"
            ),
            
            DemoScenario(
                title="🛰️ Satellite Tech for Medical Diagnosis",
                description="How Earth observation techniques enhance medical imaging capabilities",
                query="satellite image processing medical diagnosis remote sensing",
                expected_papers_count=10,
                expected_domains=["astronomy", "medical", "computer_science"],
                expected_connections=[
                    "Hyperspectral imaging → Tissue classification",
                    "Change detection algorithms → Disease progression monitoring",
                    "Multi-temporal analysis → Treatment response tracking"
                ],
                expected_innovations=[
                    "Remote sensing algorithms for medical imaging",
                    "Multi-spectral analysis for early disease detection",
                    "Time-series medical image analysis"
                ],
                wow_factor="Technology that monitors Earth now monitors your health!",
                difficulty_level="intermediate"
            ),
            
            DemoScenario(
                title="🎵 Music AI for Alzheimer's Treatment",
                description="Leveraging music information retrieval for neurodegenerative disease therapy",
                query="music information retrieval Alzheimer's treatment cognitive therapy",
                expected_papers_count=7,
                expected_domains=["computer_science", "medical", "psychology"],
                expected_connections=[
                    "Audio feature extraction → Cognitive assessment",
                    "Music recommendation systems → Personalized therapy",
                    "Rhythm analysis → Movement disorder treatment"
                ],
                expected_innovations=[
                    "AI-composed therapeutic music",
                    "Cognitive decline detection through musical interaction",
                    "Personalized audio therapy optimization"
                ],
                wow_factor="AI musicians becoming medical therapists!",
                difficulty_level="beginner"
            )
        ]
        
        return scenarios
    
    def _create_showcase_examples(self) -> List[ShowcaseExample]:
        """Create showcase examples for different domains"""
        examples = [
            ShowcaseExample(
                domain="Medical Imaging",
                problem_statement="Current medical imaging struggles with early-stage disease detection and requires expert interpretation",
                cross_domain_solutions=[
                    "Astronomy: Galaxy classification algorithms for tumor detection",
                    "Satellite imagery: Change detection for disease progression",
                    "Computer vision: Object detection for automated diagnosis",
                    "Signal processing: Noise reduction from gravitational wave research"
                ],
                innovation_potential="Revolutionary early detection capabilities with 99% accuracy",
                real_world_impact="Save millions of lives through earlier intervention and reduced healthcare costs"
            ),
            
            ShowcaseExample(
                domain="Drug Discovery",
                problem_statement="Traditional drug discovery takes 10-15 years and costs billions, with high failure rates",
                cross_domain_solutions=[
                    "Quantum computing: Molecular simulation and optimization",
                    "NLP: Treating molecules as language for drug design",
                    "Game theory: Optimizing drug interaction strategies",
                    "Materials science: Novel drug delivery mechanisms"
                ],
                innovation_potential="Reduce drug discovery time from decades to years",
                real_world_impact="Accelerate life-saving treatments and make medications more affordable"
            ),
            
            ShowcaseExample(
                domain="Climate Science",
                problem_statement="Climate modeling requires processing vast amounts of complex, multi-dimensional data",
                cross_domain_solutions=[
                    "Machine learning: Pattern recognition in climate data",
                    "Network theory: Understanding climate system interactions",
                    "Signal processing: Extracting signals from noisy environmental data",
                    "Fluid dynamics: Ocean and atmosphere modeling"
                ],
                innovation_potential="More accurate climate predictions and early warning systems",
                real_world_impact="Better preparation for climate change and natural disaster prevention"
            ),
            
            ShowcaseExample(
                domain="Neuroscience",
                problem_statement="Understanding brain function and treating neurological disorders remains largely mysterious",
                cross_domain_solutions=[
                    "Network analysis: Mapping brain connectivity like social networks",
                    "Information theory: Understanding neural communication",
                    "Machine learning: Decoding brain signals for BCI applications",
                    "Physics: Applying complex systems theory to brain dynamics"
                ],
                innovation_potential="Direct brain-computer interfaces and effective neurological treatments",
                real_world_impact="Restore mobility to paralyzed patients and cure brain disorders"
            )
        ]
        
        return examples
    
    def _create_sample_connections(self) -> List[Dict[str, Any]]:
        """Create sample cross-domain connections with diverse data"""
        connections = [
            {
                "source_domain": "astronomy",
                "target_domain": "medical",
                "technique_name": "adaptive optics image correction",
                "technique_description": "Real-time atmospheric distortion correction for telescope imaging",
                "transfer_feasibility": 0.8,
                "innovation_potential": 0.9,
                "reasoning": "Atmospheric correction techniques can remove motion artifacts in medical imaging",
                "analogy_explanation": "Just as telescopes correct for atmospheric turbulence, medical scanners can correct for patient movement"
            },
            {
                "source_domain": "biology",
                "target_domain": "computer_science", 
                "technique_name": "whale song pattern recognition",
                "technique_description": "Deep learning analysis of marine mammal vocalizations",
                "transfer_feasibility": 0.7,
                "innovation_potential": 0.85,
                "reasoning": "Temporal pattern recognition in whale songs applies to human speech disorder detection",
                "analogy_explanation": "Complex vocalization patterns in whales mirror speech abnormalities in humans"
            },
            {
                "source_domain": "physics",
                "target_domain": "chemistry",
                "technique_name": "quantum error correction",
                "technique_description": "Protecting quantum information from decoherence",
                "transfer_feasibility": 0.6,
                "innovation_potential": 0.95,
                "reasoning": "Error correction principles apply to molecular reaction control",
                "analogy_explanation": "Protecting quantum states resembles controlling chemical reaction pathways"
            },
            {
                "source_domain": "computer_science",
                "target_domain": "biology",
                "technique_name": "neural architecture search",
                "technique_description": "Automated discovery of optimal neural network structures",
                "transfer_feasibility": 0.75,
                "innovation_potential": 0.8,
                "reasoning": "Automated optimization can discover optimal protein structures",
                "analogy_explanation": "Network architecture optimization mirrors protein folding optimization"
            },
            {
                "source_domain": "astronomy",
                "target_domain": "chemistry",
                "technique_name": "spectral analysis algorithms",
                "technique_description": "Advanced signal processing for stellar spectroscopy",
                "transfer_feasibility": 0.9,
                "innovation_potential": 0.7,
                "reasoning": "Stellar spectral analysis techniques enhance molecular spectroscopy",
                "analogy_explanation": "Analyzing light from stars is similar to analyzing molecular emission spectra"
            },
            {
                "source_domain": "physics",
                "target_domain": "medical",
                "technique_name": "gravitational wave signal processing",
                "technique_description": "Ultra-sensitive noise reduction for LIGO detectors",
                "transfer_feasibility": 0.65,
                "innovation_potential": 0.9,
                "reasoning": "Ultra-sensitive signal detection can improve biomedical sensing",
                "analogy_explanation": "Detecting tiny gravitational waves is like detecting minute biological signals"
            }
        ]
        
        return connections
    
    def _create_sample_innovations(self) -> List[Dict[str, Any]]:
        """Create sample innovation directions"""
        innovations = [
            {
                "title": "Astro-Medical Imaging Fusion Platform",
                "description": "Hybrid system combining astronomical image processing with medical imaging for ultra-precise diagnosis",
                "innovation_type": "radical",
                "confidence_score": 0.85,
                "impact_potential": 0.95,
                "feasibility_score": 0.75,
                "implementation_roadmap": [
                    "Adapt astronomical imaging algorithms for medical data",
                    "Develop hybrid feature extraction pipeline",
                    "Clinical validation studies",
                    "Regulatory approval and deployment"
                ],
                "potential_challenges": [
                    "Data format compatibility between domains",
                    "Regulatory approval for medical applications",
                    "Training medical professionals on new technology"
                ],
                "success_metrics": [
                    "Improved diagnostic accuracy by 30%",
                    "Reduced false positive rates",
                    "Faster image processing time"
                ]
            },
            {
                "title": "Quantum-Bio Drug Discovery Engine",
                "description": "Quantum computing platform for accelerated molecular simulation and drug design",
                "innovation_type": "paradigm_shift",
                "confidence_score": 0.65,
                "impact_potential": 0.98,
                "feasibility_score": 0.45,
                "implementation_roadmap": [
                    "Develop quantum molecular simulation algorithms",
                    "Build quantum-classical hybrid system",
                    "Validate on known drug compounds",
                    "Scale to novel drug discovery"
                ],
                "potential_challenges": [
                    "Current quantum hardware limitations",
                    "Quantum algorithm complexity",
                    "Integration with classical drug discovery pipelines"
                ],
                "success_metrics": [
                    "10x faster molecular simulations",
                    "Discovery of novel drug compounds",
                    "Reduced drug development costs"
                ]
            },
            {
                "title": "Cross-Species Communication AI",
                "description": "AI system that learns from animal communication to improve human speech therapy",
                "innovation_type": "incremental",
                "confidence_score": 0.90,
                "impact_potential": 0.75,
                "feasibility_score": 0.85,
                "implementation_roadmap": [
                    "Collect multi-species communication datasets",
                    "Develop cross-species feature extraction",
                    "Train unified communication model",
                    "Deploy in speech therapy applications"
                ],
                "potential_challenges": [
                    "Data collection across species",
                    "Feature generalization challenges",
                    "Clinical validation requirements"
                ],
                "success_metrics": [
                    "Improved speech therapy outcomes",
                    "Faster patient recovery times",
                    "Reduced therapy costs"
                ]
            }
        ]
        
        return innovations

class DemoDataManager:
    """Manages demo data and provides easy access for the application"""
    
    def __init__(self):
        self.generator = DemoDataGenerator()
        self._current_scenario_index = 0
    
    def get_sample_papers(self, domain: Optional[str] = None, count: int = 10) -> List[ResearchPaper]:
        """Get sample papers, optionally filtered by domain"""
        papers = self.generator.sample_papers
        
        if domain:
            # Simple domain filtering based on categories or source
            filtered_papers = []
            for paper in papers:
                if (domain.lower() in str(paper.categories).lower() or 
                    domain.lower() in paper.source.lower() or
                    domain.lower() in paper.title.lower()):
                    filtered_papers.append(paper)
            papers = filtered_papers
        
        return papers[:count]
    
    def get_demo_scenarios(self, difficulty: Optional[str] = None) -> List[DemoScenario]:
        """Get demo scenarios, optionally filtered by difficulty"""
        scenarios = self.generator.demo_scenarios
        
        if difficulty:
            scenarios = [s for s in scenarios if s.difficulty_level == difficulty]
        
        return scenarios
    
    def get_next_demo_scenario(self) -> DemoScenario:
        """Get the next demo scenario in rotation"""
        scenarios = self.generator.demo_scenarios
        scenario = scenarios[self._current_scenario_index]
        self._current_scenario_index = (self._current_scenario_index + 1) % len(scenarios)
        return scenario
    
    def get_showcase_examples(self, domain: Optional[str] = None) -> List[ShowcaseExample]:
        """Get showcase examples, optionally filtered by domain"""
        examples = self.generator.showcase_examples
        
        if domain:
            examples = [e for e in examples if domain.lower() in e.domain.lower()]
        
        return examples
    
    def get_sample_connections(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get sample cross-domain connections"""
        all_connections = self.generator.cross_domain_connections
        
        # Return a random subset to provide variety
        import random
        if len(all_connections) <= count:
            return all_connections
        return random.sample(all_connections, count)
    
    def get_sample_innovations(self, innovation_type: Optional[str] = None, count: int = 3) -> List[Dict[str, Any]]:
        """Get sample innovation directions"""
        innovations = self.generator.innovation_directions
        
        if innovation_type:
            innovations = [i for i in innovations if i["innovation_type"] == innovation_type]
        
        return innovations[:count]
    
    def get_random_research_query(self) -> str:
        """Get a random research query for testing"""
        queries = [
            "machine learning protein folding prediction",
            "deep learning astronomical object detection",
            "quantum computing drug discovery optimization",
            "natural language processing chemical reaction prediction",
            "computer vision medical image analysis",
            "signal processing gravitational wave detection biomedical",
            "neural networks whale song analysis speech therapy",
            "satellite image processing medical diagnosis",
            "artificial intelligence climate modeling prediction",
            "graph neural networks protein interaction analysis"
        ]
        
        return random.choice(queries)
    
    def get_featured_demo(self) -> Dict[str, Any]:
        """Get a featured demo with complete data"""
        scenario = self.get_next_demo_scenario()
        
        return {
            "scenario": scenario,
            "sample_papers": self.get_sample_papers(count=5),
            "connections": self.get_sample_connections(count=3),
            "innovations": self.get_sample_innovations(count=2),
            "query_suggestions": [
                scenario.query,
                self.get_random_research_query(),
                self.get_random_research_query()
            ]
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about available demo data"""
        return {
            "total_sample_papers": len(self.generator.sample_papers),
            "total_demo_scenarios": len(self.generator.demo_scenarios),
            "total_showcase_examples": len(self.generator.showcase_examples),
            "total_sample_connections": len(self.generator.cross_domain_connections),
            "total_sample_innovations": len(self.generator.innovation_directions),
            "domains_covered": len(set([
                "computer_science", "medical", "astronomy", "biology", 
                "physics", "chemistry", "psychology"
            ]))
        }

# Global demo data manager
_demo_manager: Optional[DemoDataManager] = None

def get_demo_data() -> DemoDataManager:
    """Get global demo data manager instance"""
    global _demo_manager
    if _demo_manager is None:
        _demo_manager = DemoDataManager()
    return _demo_manager

def get_sample_query() -> str:
    """Get a random sample query for testing"""
    return get_demo_data().get_random_research_query()

def get_featured_scenario() -> DemoScenario:
    """Get a featured demo scenario"""
    return get_demo_data().get_next_demo_scenario()

# Pre-defined query suggestions for UI
RESEARCH_QUERY_SUGGESTIONS = [
    "AI for cancer detection using medical imaging",
    "Quantum computing applications in drug discovery", 
    "Machine learning for climate change prediction",
    "Deep learning techniques from astronomy applied to biology",
    "Natural language processing for chemical reaction prediction",
    "Computer vision methods for protein structure analysis",
    "Signal processing from gravitational waves for biomedical sensing",
    "Whale song analysis techniques for human speech therapy",
    "Satellite image processing for medical diagnosis",
    "Neural networks for materials science discovery"
]

# Domain-specific example queries
DOMAIN_SPECIFIC_QUERIES = {
    "medical": [
        "deep learning medical imaging cancer detection",
        "AI drug discovery protein folding prediction", 
        "machine learning diagnostic accuracy improvement",
        "computer vision automated medical diagnosis"
    ],
    "astronomy": [
        "machine learning galaxy classification techniques",
        "AI exoplanet detection and characterization",
        "deep learning gravitational wave analysis",
        "automated telescope data processing"
    ],
    "biology": [
        "neural networks protein interaction prediction",
        "AI genetic sequence analysis methods",
        "machine learning cellular behavior modeling",
        "deep learning evolutionary pattern recognition"
    ],
    "chemistry": [
        "AI chemical reaction prediction optimization",
        "machine learning molecular property prediction",
        "deep learning drug compound design",
        "automated synthesis pathway discovery"
    ],
    "physics": [
        "machine learning particle physics data analysis",
        "AI quantum state prediction and control",
        "deep learning materials property prediction",
        "automated experimental data interpretation"
    ]
}

# Example usage and testing
if __name__ == "__main__":
    # Test demo data management
    print("🎭 Testing Demo Data Management...")
    
    # Initialize demo manager
    demo_manager = get_demo_data()
    
    # Get statistics
    stats = demo_manager.get_statistics()
    print(f"\n📊 Demo Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test sample papers
    print(f"\n📚 Sample Papers:")
    papers = demo_manager.get_sample_papers(count=3)
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:2])}")
        print(f"   Source: {paper.source}")
        print()
    
    # Test demo scenarios
    print(f"🎬 Demo Scenarios:")
    scenarios = demo_manager.get_demo_scenarios()
    for i, scenario in enumerate(scenarios[:2], 1):
        print(f"{i}. {scenario.title}")
        print(f"   Query: {scenario.query}")
        print(f"   Wow factor: {scenario.wow_factor}")
        print(f"   Difficulty: {scenario.difficulty_level}")
        print()
    
    # Test featured demo
    print(f"⭐ Featured Demo:")
    featured = demo_manager.get_featured_demo()
    scenario = featured["scenario"]
    print(f"Title: {scenario.title}")
    print(f"Description: {scenario.description}")
    print(f"Sample papers: {len(featured['sample_papers'])}")
    print(f"Connections: {len(featured['connections'])}")
    print(f"Innovations: {len(featured['innovations'])}")
    
    # Test random queries
    print(f"\n🎲 Random Research Queries:")
    for i in range(3):
        query = demo_manager.get_random_research_query()
        print(f"{i+1}. {query}")
    
    # Test showcase examples
    print(f"\n🏆 Showcase Examples:")
    examples = demo_manager.get_showcase_examples()
    for example in examples[:2]:
        print(f"Domain: {example.domain}")
        print(f"Problem: {example.problem_statement}")
        print(f"Impact: {example.real_world_impact}")
        print()
    
    # Test domain-specific queries
    print(f"🔬 Domain-Specific Query Examples:")
    for domain, queries in list(DOMAIN_SPECIFIC_QUERIES.items())[:3]:
        print(f"{domain.title()}:")
        for query in queries[:2]:
            print(f"  • {query}")
        print()
    
    print(f"✅ Demo data management test complete!")
    print(f"🎭 Ready for interactive demonstrations!")