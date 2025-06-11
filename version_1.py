import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

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
            logger.info("Executing SQL query against Databricks")
            
            # Mock execution - replace with actual Databricks execution
            mock_result = {
                "query": sql_query,
                "rows_affected": 42,
                "execution_time": "0.15s",
                "status": "completed"
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
                metadata={"agent": self.name, "error": str(e)}
            )

class MultiAgentFramework:
    """Main framework class that orchestrates all agents"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.agents = self._initialize_agents()
        self.session_history = {}
    
    def _initialize_agents(self) -> Dict[AgentType, BaseAgent]:
        """Initialize all agents"""
        return {
            AgentType.BUSINESS_ANALYST: BusinessAnalystAgent(self.llm_provider),
            AgentType.SQL_CREATOR: SQLCreatorAgent(self.llm_provider),
            AgentType.SQL_VALIDATOR: SQLValidatorAgent(self.llm_provider),
            AgentType.COORDINATOR_EXECUTOR: CoordinatorExecutorAgent(self.llm_provider)
        }
    
    async def process_request(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """Process a user request through the entire agent pipeline"""
        session_id = session_id or f"session_{datetime.now().timestamp()}"
        
        try:
            # Step 1: Business Analysis
            logger.info("Step 1: Business Analysis")
            ba_request = TaskRequest(user_input=user_input, session_id=session_id)
            ba_response = await self.agents[AgentType.BUSINESS_ANALYST].process(ba_request)
            
            if not ba_response.success:
                return self._create_error_response("Business analysis failed", ba_response)
            
            # Step 2: SQL Creation
            logger.info("Step 2: SQL Creation")
            sql_request = TaskRequest(user_input=ba_response.data, session_id=session_id)
            sql_response = await self.agents[AgentType.SQL_CREATOR].process(sql_request)
            
            if not sql_response.success:
                return self._create_error_response("SQL creation failed", sql_response)
            
            # Step 3: SQL Validation
            logger.info("Step 3: SQL Validation")
            val_request = TaskRequest(user_input=sql_response.data, session_id=session_id)
            val_response = await self.agents[AgentType.SQL_VALIDATOR].process(val_request)
            
            if not val_response.success:
                return self._create_error_response("SQL validation failed", val_response)
            
            # Extract corrected SQL from validation response
            validated_sql = self._extract_sql_from_validation(val_response.data)
            
            # Step 4: Execution
            logger.info("Step 4: SQL Execution")
            coordinator = self.agents[AgentType.COORDINATOR_EXECUTOR]
            exec_response = await coordinator.execute_sql(validated_sql)
            
            # Store session history
            self.session_history[session_id] = {
                'user_input': user_input,
                'business_analysis': ba_response,
                'sql_creation': sql_response,
                'sql_validation': val_response,
                'execution': exec_response,
                'timestamp': datetime.now()
            }
            
            return {
                'success': True,
                'session_id': session_id,
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
                    'processing_time': datetime.now()
                }
            }
            
        except Exception as e:
            logger.error(f"Framework processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'timestamp': datetime.now()
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
    
    def _create_error_response(self, message: str, agent_response: AgentResponse) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': message,
            'details': agent_response.message,
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
