import os
import time
import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# --- CONSTANTS & CONFIGURATION ---
MODEL_ID = "gemini-2.5-flash-preview-09-2025"
API_KEY = "" # The environment provides the key at runtime

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
    Handles payloads and mandatory exponential backoff.
    """

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={API_KEY}"

    async def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        CORE GENERATION LOGIC
        ---------------------
        Implements exponential backoff: 1s, 2s, 4s, 8s, 16s.
        """
        start_time = time.time()
        
        # Construct the standardized payload
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        # Exponential Backoff Retry Loop
        retries = 5
        for attempt in range(retries):
            try:
            
                # This is a  representation of the required request flow
                response = await self._make_request(payload)
                
                if response and 'candidates' in response:
                    text = response['candidates'][0]['content']['parts'][0]['text']
                    return LLMResponse(
                        text=text,
                        model_name=self.model_id,
                        processing_time=time.time() - start_time
                    )
            except Exception:
                if attempt < retries - 1:
                    wait_time = 2 ** attempt # 1, 2, 4, 8, 16
                    await asyncio.sleep(wait_time)
                continue
        
        return None

    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to handle the raw API request.
        """
        # This uses the environment's specific networking capabilities
        # Typically handled via a global fetch or specialized library
        import urllib.request
        
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))

# --- EXECUTION ---
if __name__ == "__main__":
    async def main():
        adapter = LLMAdapter()
        result = await adapter.generate_content("Hello, system check.")
        if result:
            print(f"Response: {result.text}")

    asyncio.run(main())