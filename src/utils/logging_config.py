"""
Logging Configuration for AI Research Innovation Mapper
Provides comprehensive logging setup with structured logging, multiple handlers, and component-specific loggers
"""

import os
import sys
import logging
import logging.handlers
import time
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import json
import traceback
from dataclasses import dataclass
from enum import Enum

# Import config if available
try:
    from .config import get_config, LoggingConfig
except ImportError:
    # Fallback if config not available
    @dataclass
    class LoggingConfig:
        log_level: str = "INFO"
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_file: str = "./logs/research_mapper.log"
        max_log_size: int = 10 * 1024 * 1024
        backup_count: int = 5
        enable_agent_logging: bool = True
        enable_api_logging: bool = True
        enable_vector_logging: bool = True
    
    def get_config():
        class Config:
            logging = LoggingConfig()
        return Config()

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ComponentLogger:
    """Specialized logger for specific system components"""
    
    def __init__(self, component_name: str, base_logger: logging.Logger):
        self.component_name = component_name
        self.logger = base_logger.getChild(component_name)
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_agent_start(self, agent_name: str, query: str):
        """Log agent execution start"""
        self.logger.info(f"🤖 {agent_name} starting analysis", extra={
            "event_type": "agent_start",
            "agent_name": agent_name,
            "query": query,
            "session_id": self._session_id
        })
    
    def log_agent_complete(self, agent_name: str, execution_time: float, results_count: int):
        """Log agent execution completion"""
        self.logger.info(f"✅ {agent_name} completed in {execution_time:.2f}s", extra={
            "event_type": "agent_complete",
            "agent_name": agent_name,
            "execution_time": execution_time,
            "results_count": results_count,
            "session_id": self._session_id
        })
    
    def log_agent_error(self, agent_name: str, error: Exception, context: str = ""):
        """Log agent execution error"""
        self.logger.error(f"❌ {agent_name} failed: {str(error)}", extra={
            "event_type": "agent_error",
            "agent_name": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "session_id": self._session_id
        })
    
    def log_api_call(self, api_name: str, endpoint: str, params: Dict[str, Any] = None):
        """Log API call"""
        self.logger.debug(f"📡 API call to {api_name}", extra={
            "event_type": "api_call",
            "api_name": api_name,
            "endpoint": endpoint,
            "params": params or {},
            "session_id": self._session_id
        })
    
    def log_api_response(self, api_name: str, status_code: int, response_size: int, duration: float):
        """Log API response"""
        self.logger.debug(f"📡 API response from {api_name}: {status_code}", extra={
            "event_type": "api_response",
            "api_name": api_name,
            "status_code": status_code,
            "response_size": response_size,
            "duration": duration,
            "session_id": self._session_id
        })
    
    def log_vector_operation(self, operation: str, collection: str, count: int, duration: float):
        """Log vector database operations"""
        self.logger.debug(f"🗄️ Vector {operation} on {collection}: {count} items", extra={
            "event_type": "vector_operation",
            "operation": operation,
            "collection": collection,
            "count": count,
            "duration": duration,
            "session_id": self._session_id
        })
    
    def log_innovation_discovery(self, query: str, directions_count: int, confidence_score: float):
        """Log innovation discovery results"""
        self.logger.info(f"💡 Innovation discovery: {directions_count} directions found", extra={
            "event_type": "innovation_discovery",
            "query": query,
            "directions_count": directions_count,
            "confidence_score": confidence_score,
            "session_id": self._session_id
        })
    
    def log_cross_domain_connection(self, source_domain: str, target_domain: str, technique: str, score: float):
        """Log cross-domain connection discovery"""
        self.logger.info(f"🔗 Cross-domain connection: {technique} ({source_domain} → {target_domain})", extra={
            "event_type": "cross_domain_connection",
            "source_domain": source_domain,
            "target_domain": target_domain,
            "technique": technique,
            "score": score,
            "session_id": self._session_id
        })
    
    def log_user_interaction(self, action: str, details: Dict[str, Any] = None):
        """Log user interface interactions"""
        self.logger.info(f"👤 User action: {action}", extra={
            "event_type": "user_interaction",
            "action": action,
            "details": details or {},
            "session_id": self._session_id
        })

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output option"""
    
    def __init__(self, use_json: bool = False, include_extra: bool = True):
        self.use_json = use_json
        self.include_extra = include_extra
        
        if use_json:
            super().__init__()
        else:
            format_string = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
            super().__init__(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record):
        if self.use_json:
            return self._format_json(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record):
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if self.include_extra and hasattr(record, 'session_id'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName',
                              'processName', 'process', 'getMessage', 'exc_info', 'exc_text',
                              'stack_info', 'message']:
                    log_entry[key] = value
        
        return json.dumps(log_entry)
    
    def _format_text(self, record):
        """Format log record as readable text"""
        formatted = super().format(record)
        
        # Add emoji and extra formatting for specific event types
        if hasattr(record, 'event_type'):
            event_type = record.event_type
            if event_type == "agent_start":
                formatted = f"🚀 {formatted}"
            elif event_type == "agent_complete":
                formatted = f"✅ {formatted}"
            elif event_type == "agent_error":
                formatted = f"❌ {formatted}"
            elif event_type == "api_call":
                formatted = f"📡 {formatted}"
            elif event_type == "vector_operation":
                formatted = f"🗄️ {formatted}"
            elif event_type == "innovation_discovery":
                formatted = f"💡 {formatted}"
            elif event_type == "cross_domain_connection":
                formatted = f"🔗 {formatted}"
            elif event_type == "user_interaction":
                formatted = f"👤 {formatted}"
        
        return formatted

class LoggingManager:
    """Central logging management for the entire system"""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or get_config().logging
        self._loggers: Dict[str, ComponentLogger] = {}
        self._initialized = False
        
        # Create logs directory
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self, use_json: bool = False, console_level: Optional[str] = None):
        """Setup comprehensive logging system"""
        if self._initialized:
            return
        
        # Configure root logger
        root_logger = logging.getLogger("research_mapper")
        root_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_level = console_level or self.config.log_level
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = StructuredFormatter(use_json=False, include_extra=True)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.config.log_file,
            maxBytes=self.config.max_log_size,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all levels
        file_formatter = StructuredFormatter(use_json=use_json, include_extra=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error-only file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.config.log_file.replace('.log', '_errors.log'),
            maxBytes=self.config.max_log_size // 2,
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        
        # JSON structured log handler (optional)
        if use_json:
            json_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file.replace('.log', '_structured.jsonl'),
                maxBytes=self.config.max_log_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.INFO)
            json_formatter = StructuredFormatter(use_json=True, include_extra=True)
            json_handler.setFormatter(json_formatter)
            root_logger.addHandler(json_handler)
        
        # Configure component-specific loggers
        self._configure_component_loggers(root_logger)
        
        self._initialized = True
        root_logger.info("🔧 Logging system initialized")
    
    def _configure_component_loggers(self, root_logger: logging.Logger):
        """Configure loggers for specific components"""
        components = [
            "orchestrator",
            "paper_discovery", 
            "cross_domain",
            "innovation",
            "vector_store",
            "api_client",
            "streamlit_ui"
        ]
        
        for component in components:
            component_logger = ComponentLogger(component, root_logger)
            self._loggers[component] = component_logger
        
        # Configure third-party library loggers
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    def get_logger(self, component: str) -> ComponentLogger:
        """Get logger for specific component"""
        if not self._initialized:
            self.setup_logging()
        
        if component not in self._loggers:
            root_logger = logging.getLogger("research_mapper")
            self._loggers[component] = ComponentLogger(component, root_logger)
        
        return self._loggers[component]
    
    def log_system_startup(self, config_summary: Dict[str, Any]):
        """Log system startup information"""
        logger = self.get_logger("system")
        logger.logger.info("🚀 AI Research Innovation Mapper starting up", extra={
            "event_type": "system_startup",
            "config": config_summary
        })
    
    def log_system_shutdown(self):
        """Log system shutdown"""
        logger = self.get_logger("system")
        logger.logger.info("🛑 AI Research Innovation Mapper shutting down", extra={
            "event_type": "system_shutdown"
        })
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        log_file_path = Path(self.config.log_file)
        error_log_path = Path(self.config.log_file.replace('.log', '_errors.log'))
        
        stats = {
            "log_file": str(log_file_path),
            "log_file_exists": log_file_path.exists(),
            "log_file_size": log_file_path.stat().st_size if log_file_path.exists() else 0,
            "error_log_exists": error_log_path.exists(),
            "error_log_size": error_log_path.stat().st_size if error_log_path.exists() else 0,
            "active_loggers": len(self._loggers),
            "log_level": self.config.log_level
        }
        
        return stats
    
    def tail_logs(self, lines: int = 50, level: Optional[str] = None) -> List[str]:
        """Get recent log entries"""
        log_file_path = Path(self.config.log_file)
        
        if not log_file_path.exists():
            return ["Log file not found"]
        
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            # Filter by level if specified
            if level:
                level_upper = level.upper()
                filtered_lines = [line for line in all_lines if level_upper in line]
                return filtered_lines[-lines:] if filtered_lines else ["No log entries found for level"]
            
            return all_lines[-lines:]
            
        except Exception as e:
            return [f"Error reading log file: {str(e)}"]

# Performance monitoring decorator
def log_performance(component: str, operation: str):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_component_logger(component)
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                logger.logger.debug(f"⏱️ {operation} completed in {duration:.3f}s", extra={
                    "event_type": "performance",
                    "operation": operation,
                    "duration": duration,
                    "success": True
                })
                
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_agent_error(component, e, f"During {operation}")
                
                logger.logger.debug(f"⏱️ {operation} failed after {duration:.3f}s", extra={
                    "event_type": "performance",
                    "operation": operation,
                    "duration": duration,
                    "success": False
                })
                
                raise
        
        return wrapper
    return decorator

# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None

def init_logging(use_json: bool = False, console_level: Optional[str] = None, config: Optional[LoggingConfig] = None):
    """Initialize global logging system"""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    _logging_manager.setup_logging(use_json, console_level)
    return _logging_manager

def get_component_logger(component: str) -> ComponentLogger:
    """Get logger for specific component"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
        _logging_manager.setup_logging()
    return _logging_manager.get_logger(component)

def get_logging_stats() -> Dict[str, Any]:
    """Get logging system statistics"""
    global _logging_manager
    if _logging_manager is None:
        return {"error": "Logging not initialized"}
    return _logging_manager.get_log_stats()

def tail_logs(lines: int = 50, level: Optional[str] = None) -> List[str]:
    """Get recent log entries"""
    global _logging_manager
    if _logging_manager is None:
        return ["Logging not initialized"]
    return _logging_manager.tail_logs(lines, level)

# Context manager for logging sessions
class LoggingSession:
    """Context manager for logging user sessions"""
    
    def __init__(self, session_type: str, user_id: Optional[str] = None):
        self.session_type = session_type
        self.user_id = user_id
        self.start_time = None
        self.logger = get_component_logger("session")
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.logger.info(f"📅 Starting {self.session_type} session", extra={
            "event_type": "session_start",
            "session_type": self.session_type,
            "user_id": self.user_id
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.logger.info(f"✅ {self.session_type} session completed", extra={
                "event_type": "session_end",
                "session_type": self.session_type,
                "duration": duration,
                "success": True
            })
        else:
            self.logger.logger.error(f"❌ {self.session_type} session failed", extra={
                "event_type": "session_end",
                "session_type": self.session_type,
                "duration": duration,
                "success": False,
                "error_type": exc_type.__name__,
                "error_message": str(exc_val)
            })

# Example usage and testing
if __name__ == "__main__":
    # Test logging configuration
    print("📝 Testing Logging Configuration...")
    
    # Initialize logging
    logging_manager = init_logging(use_json=False, console_level="INFO")
    
    # Test component loggers
    orchestrator_logger = get_component_logger("orchestrator")
    api_logger = get_component_logger("api_client")
    vector_logger = get_component_logger("vector_store")
    
    # Test different log types
    print("\n🧪 Testing different log types...")
    
    # Agent logging
    orchestrator_logger.log_agent_start("TestAgent", "test query")
    orchestrator_logger.log_agent_complete("TestAgent", 2.5, 10)
    
    # API logging
    api_logger.log_api_call("ArXiv", "/query", {"search": "test"})
    api_logger.log_api_response("ArXiv", 200, 1024, 1.2)
    
    # Vector operations
    vector_logger.log_vector_operation("insert", "papers", 5, 0.8)
    
    # Innovation discovery
    orchestrator_logger.log_innovation_discovery("test query", 3, 0.85)
    
    # Cross-domain connections
    orchestrator_logger.log_cross_domain_connection("astronomy", "medical", "image processing", 0.75)
    
    # Error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        orchestrator_logger.log_agent_error("TestAgent", e, "Testing error logging")
    
    # Test session logging
    print("\n📅 Testing session logging...")
    with LoggingSession("test_session", "test_user"):
        time.sleep(0.1)  # Simulate some work
        print("Session work completed")
    
    # Test performance logging
    @log_performance("test_component", "test_operation")
    def test_function():
        time.sleep(0.1)
        return "test result"
    
    print("\n⏱️ Testing performance logging...")
    result = test_function()
    
    # Get logging stats
    print("\n📊 Logging Statistics:")
    stats = get_logging_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show recent logs
    print("\n📋 Recent log entries:")
    recent_logs = tail_logs(10)
    for log_line in recent_logs[-5:]:  # Show last 5 lines
        print(f"  {log_line.strip()}")
    
    print("\n✅ Logging configuration test complete!")
    print(f"📁 Log files created in: {logging_manager.config.log_file}")