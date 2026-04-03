import os
import time
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp   # ← added for async + mock support

# --- CONSTANTS & CONFIGURATION ---
MODEL_ID = "gemini-2.5-flash-preview-09-2025"
API_KEY = ""  # The environment provides the key at runtime (or leave empty for mock)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class LLMResponse:
    """
    Standardized response schema for orchestration.
    """
    text: str
    model_name: str
    processing_time: float


class LLMAdapter:
    """
    Unified interface for LLM interactions.
    Now supports BOTH your ngrok mock API AND real Gemini.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        mock_base_url: str = None,   # ← NEW: pass your ngrok URL here
        api_key: str = None
    ):
        
        self.model_id = model_id
        self.mock_base_url = mock_base_url.rstrip("/") if mock_base_url else None
        self.api_key = api_key or API_KEY
        self._session: Optional[aiohttp.ClientSession] = None

        if self.mock_base_url:
            self.base_url = f"{self.mock_base_url}/chat"
            self.model_id = "mock-vulnerable-llm-v2"
        else:
            self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
            
    async def _ensure_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _is_tool_abuse_prompt(self, prompt: str) -> bool:
        """
        Detect whether the prompt is attempting tool access.
        """
        tool_keywords = [
            "list tools",
            "available tools",
            "filesystem",
            "environment variables",
            "database credentials",
            "logs",
            "execute calculator",
            "internal files"
        ]
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in tool_keywords)
    
    def _simulate_tool_leak(self, prompt: str) -> str:
        """
        Simulate vulnerable tool execution responses.
        """

        prompt = prompt.lower()

        if "environment variables" in prompt:
            return "ENV_SECRET=FLAG{tool_abuse_demo}"

        if "filesystem" in prompt:
            return "File reader result: FLAG{tool_abuse_demo}"

        if "database credentials" in prompt:
            return "DB lookup result: FLAG{MOCK_DB_LEAK}"

        if "logs" in prompt:
            return "Log output: FLAG{tool_abuse_demo}"

        if "calculator" in prompt:
            return "Calculator says: FLAG{tool_abuse_demo}"

        if "tools" in prompt:
            return "Available tools: calculator, db_lookup, file_reader"

        return "Tool execution complete."

    async def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        CORE GENERATION LOGIC (unchanged style from your original)
        Now routes to mock or Gemini automatically.
        """
        start_time = time.time()
        
        retries = 1 if not self.mock_base_url else 5   # no need for retries on fast local mock
        
        # Local short-circuit: Detect tool-abuse attempts locally for Mock mode
        if self.mock_base_url and self._is_tool_abuse_prompt(prompt):
            return LLMResponse(
                text=self._simulate_tool_leak(prompt),
                model_name=self.model_id,
                processing_time=time.time() - start_time
            )

        

        for attempt in range(retries):
            try:
                if self.mock_base_url:
                    response_json = await self._make_mock_request(prompt)
                else:
                    response_json = await self._make_gemini_request(prompt, system_instruction)

                text = response_json['candidates'][0]['content']['parts'][0]['text']
                return LLMResponse(
                    text=text,
                    model_name=self.model_id,
                    processing_time=time.time() - start_time
                )
            except Exception as e:
                logging.warning(f"LLM Adapter request failed (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                continue

        return None

    async def _make_mock_request(self, prompt: str) -> Dict[str, Any]:
        session = await self._ensure_session()
        async with session.post(
            self.base_url,
            json={"message": prompt},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _make_gemini_request(self, prompt: str, system_instruction: Optional[str]) -> Dict[str, Any]:
        session = await self._ensure_session()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        async with session.post(
            self.base_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self):
        """Cleanup the session (called automatically by orchestrator)."""
        if self._session:
            await self._session.close()
            self._session = None


# --- EXECUTION (unchanged) ---
if __name__ == "__main__":
    async def main():
        adapter = LLMAdapter()  # or LLMAdapter(mock_base_url="https://abc.ngrok.io")
        result = await adapter.generate_content("Hello, system check.")
        if result:
            print(f"Response: {result.text}")

    asyncio.run(main())
