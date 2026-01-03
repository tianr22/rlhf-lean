from openai import OpenAI, AsyncOpenAI


class Agent:
    def __init__(self) -> None:
        pass

    def prove(self, problem: str) -> str:
        raise NotImplementedError


class LocalAgent(Agent):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int):
        super().__init__()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def prove(self, problem_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": problem_prompt}],
            max_tokens=self.max_tokens,
            timeout=1000,
            temperature=0.6,
            top_p=0.95,
        )

        try:
            response_text = response.choices[0].message.content
        except Exception as e:
            response_text = f"{e}"

        return response_text
