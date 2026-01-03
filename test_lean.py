import asyncio
from kimina_client import AsyncKiminaClient

client = AsyncKiminaClient(api_url="http://localhost:12332")
response = asyncio.run(client.check("#check Nat"))
print(response.results[0].response["messages"])
