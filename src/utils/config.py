"""
Configuration Management for AI Research Innovation Mapper
Handles environment variables, API keys, and system settings
"""

import os
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from dotenv import load_dotenv

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    # LLM Configuration
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-70b-versatile"
    groq_temperature: float = 0.7
    groq_max_tokens: int = 1000
    
    # Academic APIs (all free)
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    biorxiv_base_url: str = "https://api.biorxiv.org/details"
    
    # Rate limiting
    arxiv_rate_limit: float = 0.5  # requests per second
    pubmed_rate_limit: float = 0.33  # 3 requests per second max
    biorxiv_rate_limit: float = 1.0
    
    # Timeout settings
    api_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class VectorStoreConfig:
    """Configuration for ChromaDB vector store"""
    persist_directory: str = "./chroma_db"  # Move outside data/ to avoid Streamlit conflicts
    collection_name_papers: str = "research_papers"
    collection_name_techniques: str = "techniques"
    
    # Embedding configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda" if available
    
    # Search configuration
    similarity_threshold: float = 0.7
    max_results_default: int = 10
    distance_metric: str = "cosine"

@dataclass
class AgentConfig:
    """Configuration for AI agents"""
    # Paper Discovery Agent
    max_papers_per_source: int = 10
    enable_similarity_search: bool = True
    
    # Cross-Domain Agent
    max_cross_domain_results: int = 8
    min_transfer_feasibility: float = 0.3
    min_innovation_potential: float = 0.4
    
    # Innovation Agent
    max_novel_directions: int = 5
    confidence_threshold: float = 0.5
    
    # Orchestrator
    enable_parallel_execution: bool = True
    agent_timeout: int = 300  # 5 minutes
    enable_caching: bool = True
    max_cache_size: int = 100

@dataclass
class DataConfig:
    """Configuration for data storage and processing"""
    # Directory structure
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    papers_dir: str = "./data/papers"
    embeddings_dir: str = "./data/embeddings"
    logs_dir: str = "./logs"
    
    # Cache settings
    cache_expiry_hours: int = 24
    max_cache_files: int = 1000
    
    # Processing settings
    max_abstract_length: int = 2000
    min_abstract_length: int = 50
    max_title_length: int = 500

@dataclass
class UIConfig:
    """Configuration for Streamlit UI"""
    # App settings
    page_title: str = "AI Research Innovation Mapper"
    page_icon: str = "🔬"
    layout: str = "wide"
    
    # Display settings
    max_papers_display: int = 20
    max_connections_display: int = 15
    max_directions_display: int = 10
    
    # Performance settings
    enable_progress_bars: bool = True
    show_debug_info: bool = False
    auto_refresh_seconds: int = 30

@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "./logs/research_mapper.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Component-specific logging
    enable_agent_logging: bool = True
    enable_api_logging: bool = True
    enable_vector_logging: bool = True

@dataclass
class SystemConfig:
    """Main system configuration container"""
    api: APIConfig = field(default_factory=APIConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    environment: str = "development"  # development, production, testing
    debug_mode: bool = False
    
    def __post_init__(self):
        """Post-initialization setup"""
        self._create_directories()
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.data.data_dir,
            self.data.cache_dir,
            self.data.papers_dir,
            self.data.embeddings_dir,
            self.data.logs_dir,
            self.vector_store.persist_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration settings"""
        # Check required API keys
        if not self.api.groq_api_key:
            logger.warning("GROQ_API_KEY not set - some features may not work")
        
        # Validate rate limits
        if self.api.arxiv_rate_limit <= 0:
            raise ValueError("ArXiv rate limit must be positive")
        
        # Validate thresholds
        if not 0 <= self.agents.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_file: Optional[str] = None, env_file: Optional[str] = ".env"):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional JSON config file path
            env_file: Environment file path
        """
        self.config_file = config_file
        self.env_file = env_file
        self._config: Optional[SystemConfig] = None
        
        # Load environment variables
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
    
    def load_config(self) -> SystemConfig:
        """Load and return system configuration"""
        if self._config is None:
            self._config = self._build_config()
        return self._config
    
    def _build_config(self) -> SystemConfig:
        """Build configuration from various sources"""
        # Start with defaults
        config = SystemConfig()
        
        # Override with environment variables
        self._load_from_environment(config)
        
        # Override with config file if provided
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file(config)
        
        return config
    
    def _load_from_environment(self, config: SystemConfig):
        """Load configuration from environment variables"""
        # API Configuration
        config.api.groq_api_key = os.getenv("GROQ_API_KEY")
        
        if os.getenv("GROQ_MODEL"):
            config.api.groq_model = os.getenv("GROQ_MODEL")
        
        if os.getenv("GROQ_TEMPERATURE"):
            config.api.groq_temperature = float(os.getenv("GROQ_TEMPERATURE"))
        
        # Data directories
        if os.getenv("DATA_DIR"):
            config.data.data_dir = os.getenv("DATA_DIR")
        
        if os.getenv("CACHE_DIR"):
            config.data.cache_dir = os.getenv("CACHE_DIR")
        
        # Vector store
        if os.getenv("CHROMA_PERSIST_DIR"):
            config.vector_store.persist_directory = os.getenv("CHROMA_PERSIST_DIR")
        
        if os.getenv("EMBEDDING_MODEL"):
            config.vector_store.embedding_model = os.getenv("EMBEDDING_MODEL")
        
        # Agent settings
        if os.getenv("MAX_PAPERS_PER_SOURCE"):
            config.agents.max_papers_per_source = int(os.getenv("MAX_PAPERS_PER_SOURCE"))
        
        if os.getenv("ENABLE_PARALLEL_EXECUTION"):
            config.agents.enable_parallel_execution = os.getenv("ENABLE_PARALLEL_EXECUTION").lower() == "true"
        
        # Environment
        if os.getenv("ENVIRONMENT"):
            config.environment = os.getenv("ENVIRONMENT")
        
        if os.getenv("DEBUG_MODE"):
            config.debug_mode = os.getenv("DEBUG_MODE").lower() == "true"
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            config.logging.log_level = os.getenv("LOG_LEVEL")
    
    def _load_from_file(self, config: SystemConfig):
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values
            self._update_config_from_dict(config, file_config)
            logger.info(f"Loaded configuration from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file {self.config_file}: {e}")
    
    def _update_config_from_dict(self, config: SystemConfig, config_dict: Dict[str, Any]):
        """Update config object from dictionary"""
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def save_config(self, output_file: str):
        """Save current configuration to file"""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        config_dict = self._config_to_dict(self._config)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        result = {}
        
        for field_name in ['api', 'vector_store', 'agents', 'data', 'ui', 'logging']:
            if hasattr(config, field_name):
                section = getattr(config, field_name)
                result[field_name] = {}
                
                for attr_name in dir(section):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(section, attr_name)
                        if not callable(attr_value):
                            result[field_name][attr_name] = attr_value
        
        # Add top-level fields
        result['environment'] = config.environment
        result['debug_mode'] = config.debug_mode
        
        return result
    
    def get_api_keys_status(self) -> Dict[str, bool]:
        """Check status of API keys"""
        config = self.load_config()
        
        return {
            "groq_api_key": bool(config.api.groq_api_key),
            "all_required_keys": bool(config.api.groq_api_key)
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate complete system setup"""
        config = self.load_config()
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check API keys
        if not config.api.groq_api_key:
            validation_result["errors"].append("GROQ_API_KEY is required")
            validation_result["valid"] = False
        
        # Check directories
        required_dirs = [
            config.data.data_dir,
            config.vector_store.persist_directory
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                validation_result["warnings"].append(f"Directory {directory} will be created")
        
        # Check write permissions
        try:
            test_file = os.path.join(config.data.data_dir, ".test_write")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            validation_result["errors"].append(f"No write permission in data directory: {e}")
            validation_result["valid"] = False
        
        # Info about configuration
        validation_result["info"].extend([
            f"Environment: {config.environment}",
            f"Debug mode: {config.debug_mode}",
            f"Parallel execution: {config.agents.enable_parallel_execution}",
            f"Caching enabled: {config.agents.enable_caching}"
        ])
        
        return validation_result

# Global configuration instance
_config_manager: Optional[ConfigManager] = None

def get_config() -> SystemConfig:
    """Get global configuration instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.load_config()

def init_config(config_file: Optional[str] = None, env_file: Optional[str] = ".env") -> SystemConfig:
    """Initialize configuration with custom files"""
    global _config_manager
    _config_manager = ConfigManager(config_file, env_file)
    return _config_manager.load_config()

def validate_system() -> Dict[str, Any]:
    """Validate complete system setup"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.validate_setup()

def get_api_keys_status() -> Dict[str, bool]:
    """Get status of API keys"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.get_api_keys_status()

# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "environment": "development",
    "debug_mode": True,
    "logging": {
        "log_level": "DEBUG",
        "enable_agent_logging": True
    },
    "agents": {
        "enable_caching": True,
        "max_papers_per_source": 5
    }
}

PRODUCTION_CONFIG = {
    "environment": "production", 
    "debug_mode": False,
    "logging": {
        "log_level": "INFO",
        "enable_agent_logging": False
    },
    "agents": {
        "enable_caching": True,
        "max_papers_per_source": 15
    }
}

TESTING_CONFIG = {
    "environment": "testing",
    "debug_mode": True,
    "logging": {
        "log_level": "WARNING"
    },
    "agents": {
        "enable_caching": False,
        "max_papers_per_source": 3,
        "agent_timeout": 60
    }
}

# Example usage and testing
if __name__ == "__main__":
    # Test configuration management
    print("🔧 Testing Configuration Management...")
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # Load configuration
    config = config_manager.load_config()
    
    print(f"✅ Configuration loaded successfully")
    print(f"Environment: {config.environment}")
    print(f"Debug mode: {config.debug_mode}")
    
    # Check API keys status
    api_status = config_manager.get_api_keys_status()
    print(f"\n🔑 API Keys Status:")
    for key, status in api_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {key}: {status_icon}")
    
    # Validate system
    validation = config_manager.validate_setup()
    print(f"\n🔍 System Validation:")
    print(f"Valid: {'✅' if validation['valid'] else '❌'}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️ {warning}")
    
    if validation['info']:
        print("Info:")
        for info in validation['info']:
            print(f"  ℹ️ {info}")
    
    # Test configuration sections
    print(f"\n📋 Configuration Summary:")
    print(f"API Config:")
    print(f"  Groq Model: {config.api.groq_model}")
    print(f"  Rate Limits: ArXiv={config.api.arxiv_rate_limit}, PubMed={config.api.pubmed_rate_limit}")
    
    print(f"Vector Store:")
    print(f"  Directory: {config.vector_store.persist_directory}")
    print(f"  Embedding Model: {config.vector_store.embedding_model}")
    
    print(f"Agents:")
    print(f"  Max Papers: {config.agents.max_papers_per_source}")
    print(f"  Parallel Execution: {config.agents.enable_parallel_execution}")
    print(f"  Caching: {config.agents.enable_caching}")
    
    # Test saving configuration
    try:
        output_file = "./config_example.json"
        config_manager.save_config(output_file)
        print(f"\n💾 Configuration saved to {output_file}")
        
        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        print(f"❌ Error saving config: {e}")
    
    print(f"\n✅ Configuration management test complete!")