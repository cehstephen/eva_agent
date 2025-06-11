import asyncio
import sys
sys.path.append('.') 

from multi_agent import MultiAgentFramework, OpenAIProvider

async def run():
    llm_provider = OpenAIProvider(api_key="th0000002354_your-api-key")
    framework = MultiAgentFramework(llm_provider)
    result = await framework.process_request("I need a way to determine how many customers purchased something in my shop today and what the total sale was")
    print(result)

asyncio.run(run())
