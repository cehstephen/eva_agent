import os
import json
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    BUSINESS_ANALYST = "business_analyst"
    SQL_CREATOR = "sql_creator"
    SQL_VALIDATOR = "sql_validator"
    COORDINATOR_EXECUTOR = "coordinator_executor"

@dataclass
class AgentResponse:
    """Standard response format for all agents"""
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict] = None
    timestamp: datetime = None
    attempt_number: int = 1
    error_details: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class TaskRequest:
    """Standard request format for tasks"""
    user_input: str
    context: Optional[Dict] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

class DatabaseLogger:
    """Handles all database logging operations"""
    
    def __init__(self, db_path: str = "agent_framework.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # User requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    request_id TEXT UNIQUE NOT NULL,
                    user_input TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'processing',
                    final_result TEXT,
                    error_message TEXT,
                    processing_duration_seconds REAL
                )
            """)
            
            # Agent executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    attempt_number INTEGER DEFAULT 1,
                    input_data TEXT,
                    output_data TEXT,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    error_details TEXT,
                    execution_duration_seconds REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (request_id) REFERENCES user_requests (request_id)
                )
            """)
            
            # System events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    session_id TEXT,
                    request_id TEXT,
                    description TEXT,
                    data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def log_user_request(self, session_id: str, request_id: str, user_input: str) -> None:
        """Log a new user request"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_requests (session_id, request_id, user_input)
                VALUES (?, ?, ?)
            """, (session_id, request_id, user_input))
            conn.commit()
    
    def update_user_request(self, request_id: str, status: str, final_result: str = None, 
                           error_message: str = None, duration: float = None) -> None:
        """Update user request with final status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE user_requests 
                SET status = ?, final_result = ?, error_message = ?, processing_duration_seconds = ?
                WHERE request_id = ?
            """, (status, final_result, error_message, duration, request_id))
            conn.commit()
    
    def log_agent_execution(self, request_id: str, session_id: str, agent_name: str, 
                           agent_type: str, attempt_number: int, input_data: Any, 
                           response: AgentResponse, duration: float) -> None:
        """Log an agent execution"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_executions (
                    request_id, session_id, agent_name, agent_type, attempt_number,
                    input_data, output_data, success, error_message, error_details,
                    execution_duration_seconds, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_id, session_id, agent_name, agent_type, attempt_number,
                json.dumps(str(input_data)[:1000]), 
                json.dumps(str(response.data)[:1000]) if response.data else None,
                response.success, response.message, response.error_details,
                duration, json.dumps(response.metadata) if response.metadata else None
            ))
            conn.commit()
    
    def log_system_event(self, event_type: str, event_name: str, description: str,
                        session_id: str = None, request_id: str = None, data: Any = None) -> None:
        """Log a system event"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_events (event_type, event_name, session_id, request_id, description, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_type, event_name, session_id, request_id, description, 
                  json.dumps(data) if data else None))
            conn.commit()
    
    def get_request_history(self, session_id: str = None, limit: int = 100) -> List[Dict]:
        """Get request history"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if session_id:
                cursor.execute("""
                    SELECT * FROM user_requests 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM user_requests 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_agent_executions(self, request_id: str) -> List[Dict]:
        """Get all agent executions for a request"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM agent_executions 
                WHERE request_id = ? 
                ORDER BY timestamp ASC
            """, (request_id,))
            
            return [dict(row) for row in cursor.fetchall()]

class RetryableAgent:
    """Wrapper class that adds retry logic to any agent"""
    
    def __init__(self, agent: 'BaseAgent', max_retries: int = 2, db_logger: DatabaseLogger = None):
        self.agent = agent
        self.max_retries = max_retries
        self.db_logger = db_logger
    
    async def process_with_retry(self, request: TaskRequest) -> AgentResponse:
        """Process request with retry logic"""
        last_response = None
        
        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt
            start_time = datetime.now()
            
            try:
                # Log attempt start
                if self.db_logger:
                    self.db_logger.log_system_event(
                        "agent_attempt", 
                        f"{self.agent.name}_attempt_{attempt}",
                        f"Starting attempt {attempt} for {self.agent.name}",
                        request.session_id,
                        request.request_id
                    )
                
                logger.info(f"Agent {self.agent.name} - Attempt {attempt}/{self.max_retries + 1}")
                
                response = await self.agent.process(request)
                response.attempt_number = attempt
                
                duration = (datetime.now() - start_time).total_seconds()
                
                # Log the execution
                if self.db_logger:
                    self.db_logger.log_agent_execution(
                        request.request_id, request.session_id,
                        self.agent.name, self.agent.agent_type.value,
                        attempt, request.user_input, response, duration
                    )
                
                if response.success:
                    logger.info(f"Agent {self.agent.name} succeeded on attempt {attempt}")
                    return response
                else:
                    logger.warning(f"Agent {self.agent.name} failed on attempt {attempt}: {response.message}")
                    last_response = response
                    
                    if attempt <= self.max_retries:
                        # Add delay before retry (exponential backoff)
                        delay = 2 ** (attempt - 1)
                        logger.info(f"Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                error_msg = f"Exception in agent {self.agent.name}, attempt {attempt}: {str(e)}"
                logger.error(error_msg)
                
                last_response = AgentResponse(
                    success=False,
                    data=None,
                    message=f"Agent execution failed: {str(e)}",
                    error_details=str(e),
                    attempt_number=attempt
                )
                
                # Log the failed execution
                if self.db_logger:
                    self.db_logger.log_agent_execution(
                        request.request_id, request.session_id,
                        self.agent.name, self.agent.agent_type.value,
                        attempt, request.user_input, last_response, duration
                    )
                
                if attempt <= self.max_retries:
                    delay = 2 ** (attempt - 1)
                    await asyncio.sleep(delay)
        
        # All attempts failed
        final_response = last_response or AgentResponse(
            success=False,
            data=None,
            message=f"Agent {self.agent.name} failed after {self.max_retries + 1} attempts",
            attempt_number=self.max_retries + 1
        )
        
        logger.error(f"Agent {self.agent.name} failed after all retry attempts")
        return final_response

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
    
    async def generate_response(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        # Note: In a real implementation, you would use the OpenAI client here
        # This is a mock implementation for demonstration
        logger.info(f"Generating response with {self.model}")
        
        # Mock response - replace with actual OpenAI API call
        mock_responses = {
            "business_analyst": """
            {
                "business_objective": "Calculate daily sales metrics",
                "required_data_elements": ["customer_count", "total_sales", "transaction_date"],
                "expected_output_format": "Count and sum aggregation",
                "filters_constraints": "Filter by today's date",
                "structured_sql_prompt": "Create SQL to count unique customers and sum sales for today"
            }
            """,
            "sql_creator": """
            SELECT 
                COUNT(DISTINCT customer_id) as customer_count,
                SUM(sale_amount) as total_sales
            FROM sales 
            WHERE DATE(transaction_date) = CURRENT_DATE
            """,
            "sql_validator": """
            CORRECTED_SQL: SELECT 
                COUNT(DISTINCT customer_id) as customer_count,
                SUM(sale_amount) as total_sales
            FROM sales 
            WHERE DATE(transaction_date) = CURRENT_DATE
            CHANGES: No changes needed - SQL is syntactically correct
            """
        }
        
        # Simple logic to return appropriate mock response
        if "business" in prompt.lower() or "analyze" in prompt.lower():
            return mock_responses["business_analyst"]
        elif "create" in prompt.lower() and "sql" in prompt.lower():
            return mock_responses["sql_creator"]
        elif "validate" in prompt.lower() or "correct" in prompt.lower():
            return mock_responses["sql_validator"]
        else:
            return f"Mock response for prompt: {prompt[:50]}..."

class BaseAgent(ABC):
    """Base class for all agents in the framework"""
    
    def __init__(self, name: str, llm_provider: LLMProvider, agent_type: AgentType):
        self.name = name
        self.llm_provider = llm_provider
        self.agent_type = agent_type
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    async def process(self, request: TaskRequest) -> AgentResponse:
        """Process a task request and return a response"""
        pass
    
    async def _generate_llm_response(self, prompt: str, **kwargs) -> str:
        """Generate response using the LLM provider"""
        try:
            return await self.llm_provider.generate_response(
                prompt=prompt,
                system_prompt=self.system_prompt,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

class BusinessAnalystAgent(BaseAgent):
    """Agent responsible for analyzing business requirements and creating structured prompts"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("Business Analyst", llm_provider, AgentType.BUSINESS_ANALYST)
    
    def _get_system_prompt(self) -> str:
        return """You are a Business Analyst specializing in translating business requirements into technical specifications.
        
Your role is to:
1. Analyze user requests and understand their business intent
2. Extract key requirements and constraints
3. Create structured prompts for SQL generation
4. Identify the type of analysis needed (reporting, aggregation, filtering, etc.)

Always provide clear, structured output that includes:
- Business objective
- Required data elements
- Expected output format
- Any constraints or filters needed

Focus on understanding what the user wants to achieve from a business perspective."""
    
    async def process(self, request: TaskRequest) -> AgentResponse:
        """Analyze business requirements and create structured prompt"""
        try:
            analysis_prompt = f"""
            Analyze this business request and create a structured prompt for SQL generation:
            
            User Request: {request.user_input}
            
            Please provide:
            1. Business Objective
            2. Required Data Elements
            3. Expected Output Format
            4. Filters/Constraints
            5. Structured SQL Prompt
            
            Format your response as JSON.
            """
            
            response = await self._generate_llm_response(analysis_prompt)
            
            return AgentResponse(
                success=True,
                data=response,
                message="Business analysis completed successfully",
                metadata={"agent": self.name, "type": "business_analysis"}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Business analysis failed: {str(e)}",
                error_details=str(e),
                metadata={"agent": self.name, "error": str(e)}
            )

class SQLCreatorAgent(BaseAgent):
    """Agent specialized in creating SQL queries from business requirements"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("SQL Creator", llm_provider, AgentType.SQL_CREATOR)
    
    def _get_system_prompt(self) -> str:
        return """You are an expert SQL developer with deep knowledge of database design and query optimization.
        
Your role is to:
1. Convert business requirements into efficient SQL queries
2. Use appropriate SQL functions and clauses
3. Follow SQL best practices and conventions
4. Create queries that are readable and maintainable
5. Consider performance implications

Guidelines:
- Use proper table aliases
- Include appropriate WHERE clauses for filtering
- Use GROUP BY and aggregation functions when needed
- Handle potential NULL values
- Use JOINS efficiently
- Comment complex logic

Always return only valid SQL code without additional formatting."""
    
    async def process(self, request: TaskRequest) -> AgentResponse:
        """Create SQL query from business requirements"""
        try:
            sql_prompt = f"""
            Create a SQL query based on this business analysis:
            
            {request.user_input}
            
            Return only the SQL query without any additional text or formatting.
            Make sure the query is syntactically correct and follows best practices.
            """
            
            sql_query = await self._generate_llm_response(sql_prompt)
            
            return AgentResponse(
                success=True,
                data=sql_query.strip(),
                message="SQL query created successfully",
                metadata={"agent": self.name, "type": "sql_generation"}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"SQL creation failed: {str(e)}",
                error_details=str(e),
                metadata={"agent": self.name, "error": str(e)}
            )

class SQLValidatorAgent(BaseAgent):
    """Agent responsible for validating and correcting SQL queries"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("SQL Validator", llm_provider, AgentType.SQL_VALIDATOR)
    
    def _get_system_prompt(self) -> str:
        return """You are a senior SQL database administrator and query optimization expert.
        
Your role is to:
1. Validate SQL syntax and logic
2. Identify and correct errors
3. Optimize query performance
4. Ensure best practices are followed
5. Check for potential security issues

Common issues to check:
- Syntax errors
- Missing table aliases
- Incorrect JOIN conditions
- Missing GROUP BY clauses
- Ambiguous column references
- Performance anti-patterns
- SQL injection vulnerabilities

If the SQL is correct, return it as-is. If there are issues, provide the corrected version with a brief explanation of changes made."""
    
    async def process(self, request: TaskRequest) -> AgentResponse:
        """Validate and correct SQL query"""
        try:
            validation_prompt = f"""
            Validate and correct this SQL query if needed:
            
            {request.user_input}
            
            If the query is correct, return it unchanged.
            If there are issues, return the corrected query and list the changes made.
            
            Format your response as:
            CORRECTED_SQL: [the corrected SQL query]
            CHANGES: [list of changes made, or "No changes needed"]
            """
            
            response = await self._generate_llm_response(validation_prompt)
            
            return AgentResponse(
                success=True,
                data=response,
                message="SQL validation completed",
                metadata={"agent": self.name, "type": "sql_validation"}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"SQL validation failed: {str(e)}",
                error_details=str(e),
                metadata={"agent": self.name, "error": str(e)}
            )

class CoordinatorExecutorAgent(BaseAgent):
    """Agent that coordinates the workflow and executes SQL queries"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("Coordinator/Executor", llm_provider, AgentType.COORDINATOR_EXECUTOR)
        self.databricks_enabled = os.getenv('DATABRICKS_ENABLED', 'false').lower() == 'true'
        self.databricks_config = self._get_databricks_config()
    
    def _get_system_prompt(self) -> str:
        return """You are a system coordinator responsible for orchestrating the workflow and executing queries.
        
Your role is to:
1. Coordinate the flow between different agents
2. Execute SQL queries when appropriate
3. Handle errors and retries
4. Provide final results to users

You ensure the entire pipeline works smoothly and efficiently."""
    
    def _get_databricks_config(self) -> Dict:
        """Get Databricks configuration from environment variables"""
        return {
            'server_hostname': os.getenv('DATABRICKS_SERVER_HOSTNAME'),
            'http_path': os.getenv('DATABRICKS_HTTP_PATH'),
            'access_token': os.getenv('DATABRICKS_ACCESS_TOKEN')
        }
    
    async def execute_sql(self, sql_query: str) -> AgentResponse:
        """Execute SQL query against Databricks if enabled"""
        if not self.databricks_enabled:
            return AgentResponse(
                success=True,
                data=sql_query,
                message="Databricks execution disabled. Returning validated SQL query.",
                metadata={"execution_mode": "disabled"}
            )
        
        try:
            # Note: In a real implementation, you would use the Databricks connector here
            # Example: from databricks import sql
            logger.info("Executing SQL query against Databricks")
            
            # Mock execution - replace with actual Databricks execution
            mock_result = {
                "query": sql_query,
                "rows_affected": 42,
                "execution_time": "0.15s",
                "status": "completed",
                "results": [
                    {"customer_count": 15, "total_sales": 2500.00}
                ]
            }
            
            return AgentResponse(
                success=True,
                data=mock_result,
                message="SQL query executed successfully against Databricks",
                metadata={"execution_mode": "databricks", "config": self.databricks_config}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=sql_query,
                message=f"Databricks execution failed: {str(e)}. Returning SQL query.",
                error_details=str(e),
                metadata={"execution_mode": "failed", "error": str(e)}
            )
    
    async def process(self, request: TaskRequest) -> AgentResponse:
        """Coordinate the entire workflow"""
        try:
            # This method handles coordination logic
            # In practice, this would orchestrate calls to other agents
            return AgentResponse(
                success=True,
                data="Coordination completed",
                message="Workflow coordinated successfully",
                metadata={"agent": self.name, "type": "coordination"}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Coordination failed: {str(e)}",
                error_details=str(e),
                metadata={"agent": self.name, "error": str(e)}
            )


class MultiAgentFramework:
    """Main framework class that orchestrates all agents"""
    
    def __init__(self, llm_provider: LLMProvider, db_path: str = "agent_framework.db", max_retries: int = 2):
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        self.db_logger = DatabaseLogger(db_path)
        self.agents = self._initialize_agents()
        self.retryable_agents = self._wrap_agents_with_retry()
        self.session_history = {}
        
        # Log framework initialization
        self.db_logger.log_system_event(
            "framework", "initialization", 
            f"MultiAgentFramework initialized with {len(self.agents)} agents"
        )
    
    def _initialize_agents(self) -> Dict[AgentType, BaseAgent]:
        """Initialize all agents"""
        return {
            AgentType.BUSINESS_ANALYST: BusinessAnalystAgent(self.llm_provider),
            AgentType.SQL_CREATOR: SQLCreatorAgent(self.llm_provider),
            AgentType.SQL_VALIDATOR: SQLValidatorAgent(self.llm_provider),
            AgentType.COORDINATOR_EXECUTOR: CoordinatorExecutorAgent(self.llm_provider)
        }
    
    def _wrap_agents_with_retry(self) -> Dict[AgentType, RetryableAgent]:
        """Wrap all agents with retry logic"""
        return {
            agent_type: RetryableAgent(agent, self.max_retries, self.db_logger)
            for agent_type, agent in self.agents.items()
        }
    
    async def process_request(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user request through the entire agent pipeline"""
        session_id = session_id or f"session_{datetime.now().timestamp()}"
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Log the user request
        self.db_logger.log_user_request(session_id, request_id, user_input)
        self.db_logger.log_system_event(
            "request", "started",
            f"Processing user request: {user_input[:100]}...",
            session_id, request_id
        )
        
        try:
            # Step 1: Business Analysis
            logger.info("Step 1: Business Analysis")
            ba_request = TaskRequest(
                user_input=user_input, 
                session_id=session_id, 
                request_id=request_id
            )
            ba_response = await self.retryable_agents[AgentType.BUSINESS_ANALYST].process_with_retry(ba_request)
            
            if not ba_response.success:
                error_msg = "Business analysis failed after all retry attempts"
                self.db_logger.update_user_request(
                    request_id, "failed", error_message=error_msg,
                    duration=(datetime.now() - start_time).total_seconds()
                )
                return self._create_error_response(error_msg, ba_response, request_id, session_id)
            
            # Step 2: SQL Creation
            logger.info("Step 2: SQL Creation")
            sql_request = TaskRequest(
                user_input=ba_response.data, 
                session_id=session_id, 
                request_id=request_id
            )
            sql_response = await self.retryable_agents[AgentType.SQL_CREATOR].process_with_retry(sql_request)
            
            if not sql_response.success:
                error_msg = "SQL creation failed after all retry attempts"
                self.db_logger.update_user_request(
                    request_id, "failed", error_message=error_msg,
                    duration=(datetime.now() - start_time).total_seconds()
                )
                return self._create_error_response(error_msg, sql_response, request_id, session_id)
            
            # Step 3: SQL Validation
            logger.info("Step 3: SQL Validation")
            val_request = TaskRequest(
                user_input=sql_response.data, 
                session_id=session_id, 
                request_id=request_id
            )
            val_response = await self.retryable_agents[AgentType.SQL_VALIDATOR].process_with_retry(val_request)
            
            if not val_response.success:
                error_msg = "SQL validation failed after all retry attempts"
                self.db_logger.update_user_request(
                    request_id, "failed", error_message=error_msg,
                    duration=(datetime.now() - start_time).total_seconds()
                )
                return self._create_error_response(error_msg, val_response, request_id, session_id)
            
            # Extract corrected SQL from validation response
            validated_sql = self._extract_sql_from_validation(val_response.data)
            
            # Step 4: Execution
            logger.info("Step 4: SQL Execution")
            coordinator = self.agents[AgentType.COORDINATOR_EXECUTOR]
            exec_start_time = datetime.now()
            exec_response = await coordinator.execute_sql(validated_sql)
            exec_duration = (datetime.now() - exec_start_time).total_seconds()
            
            # Log the execution manually since it's not going through retry wrapper
            self.db_logger.log_agent_execution(
                request_id, session_id, coordinator.name, 
                coordinator.agent_type.value, 1, validated_sql, 
                exec_response, exec_duration
            )
            
            total_duration = (datetime.now() - start_time).total_seconds()
            
            # Store session history
            pipeline_results = {
                'business_analysis': ba_response,
                'sql_creation': sql_response,
                'sql_validation': val_response,
                'execution': exec_response
            }
            
            self.session_history[session_id] = {
                'user_input': user_input,
                'request_id': request_id,
                'pipeline_results': pipeline_results,
                'timestamp': datetime.now(),
                'duration': total_duration
            }
            
            # Update user request with success
            final_result = {
                'final_sql': validated_sql,
                'execution_result': exec_response.data
            }
            self.db_logger.update_user_request(
                request_id, "completed", 
                final_result=json.dumps(final_result),
                duration=total_duration
            )
            
            # Log successful completion
            self.db_logger.log_system_event(
                "request", "completed",
                f"Successfully processed request in {total_duration:.2f}s",
                session_id, request_id, final_result
            )
            
            return {
                'success': True,
                'session_id': session_id,
                'request_id': request_id,
                'final_sql': validated_sql,
                'execution_result': exec_response.data,
                'pipeline_results': {
                    'business_analysis': ba_response.data,
                    'sql_creation': sql_response.data,
                    'sql_validation': val_response.data,
                    'execution': exec_response.data
                },
                'metadata': {
                    'databricks_enabled': coordinator.databricks_enabled,
                    'processing_time': total_duration,
                    'retry_counts': {
                        'business_analyst': ba_response.attempt_number,
                        'sql_creator': sql_response.attempt_number,
                        'sql_validator': val_response.attempt_number
                    }
                }
            }
            
        except Exception as e:
            total_duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Framework processing error: {str(e)}"
            
            logger.error(error_msg)
            
            # Update user request with error
            self.db_logger.update_user_request(
                request_id, "error", error_message=error_msg, duration=total_duration
            )
            
            # Log system error
            self.db_logger.log_system_event(
                "error", "framework_exception",
                error_msg, session_id, request_id, {"exception": str(e)}
            )
            
            return {
                'success': False,
                'error': error_msg,
                'session_id': session_id,
                'request_id': request_id,
                'timestamp': datetime.now(),
                'duration': total_duration
            }
    
    def _extract_sql_from_validation(self, validation_response: str) -> str:
        """Extract SQL from validation response"""
        try:
            # Look for CORRECTED_SQL: pattern
            if "CORRECTED_SQL:" in validation_response:
                lines = validation_response.split('\n')
                for line in lines:
                    if line.strip().startswith("CORRECTED_SQL:"):
                        return line.replace("CORRECTED_SQL:", "").strip()
            
            # If no pattern found, return the whole response
            return validation_response.strip()
            
        except Exception:
            return validation_response
    
    def _create_error_response(self, message: str, agent_response: AgentResponse, 
                              request_id: str, session_id: str) -> Dict[str, Any]:
        """Create standardized error response"""
        self.db_logger.log_system_event(
            "error", "agent_failure",
            f"{message}: {agent_response.message}",
            session_id, request_id,
            {"agent_response": asdict(agent_response)}
        )
        
        return {
            'success': False,
            'error': message,
            'details': agent_response.message,
            'request_id': request_id,
            'session_id': session_id,
            'attempt_number': agent_response.attempt_number,
            'timestamp': datetime.now()
        }
    
    def get_session_history(self, session_id: str) -> Optional[Dict]:
        """Get session history"""
        return self.session_history.get(session_id)
    
    def add_custom_agent(self, agent_type: AgentType, agent: BaseAgent):
        """Add a custom agent to the framework"""
        self.agents[agent_type] = agent
        logger.info(f"Added custom agent: {agent.name}")

# Example usage and demo
async def main():
    """Example usage of the Multi-Agent Framework"""
    
    # Initialize LLM provider (replace with actual API key)
    llm_provider = OpenAIProvider(api_key="your-openai-api-key")
    
    # Initialize framework
    framework = MultiAgentFramework(llm_provider)
    
    # Example request
    user_request = "I need a way to determine how many customers purchased something in my shop today and what the total sale was"
    
    print("Processing request:", user_request)
    print("-" * 50)
    
    # Process the request
    result = await framework.process_request(user_request)
    
    if result['success']:
        print("✅ Request processed successfully!")
        print(f"Session ID: {result['session_id']}")
        print(f"Final SQL: {result['final_sql']}")
        print(f"Execution Result: {result['execution_result']}")
    else:
        print("❌ Request processing failed:")
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
