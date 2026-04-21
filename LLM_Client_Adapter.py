import os
import time
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import aiohttp

# --- CONSTANTS & CONFIGURATION ---
MODEL_ID = "gemini-2.5-flash-preview-09-2025"
API_KEY = os.getenv("TARGET_API_KEY", "")  # Pull from environment or leave empty
USER_ID = os.getenv("TARGET_USER_ID", "red-team-1")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class LLMResponse:
    """Standardized response schema for orchestration."""
    text: str
    model_name: str
    processing_time: float
    decision: Optional[str] = None  # Captured from backend decision records

class LLMAdapter:
    """
    Unified interface for LLM interactions.
    Returns None if the backend (Mock or Gemini) is unreachable.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        mock_base_url: str = None,
        api_key: str = None,
        user_id: str = None
    ):
        
        self.model_id = model_id
        self.mock_base_url = mock_base_url.rstrip("/") if mock_base_url else None
        self.api_key = api_key or API_KEY
        self.user_id = user_id or USER_ID
        self.total_requests = 0  # Physical HTTP request counter
        self._session: Optional[aiohttp.ClientSession] = None

        if self.mock_base_url:
            # If the user provides a full path (contains /chat), use it exactly.
            # Otherwise, append /chat/ for the mock API.
            if "/chat" in self.mock_base_url.lower():
                self.base_url = self.mock_base_url
            else:
                self.base_url = f"{self.mock_base_url.rstrip('/')}/chat"
            self.model_id = "mock-vulnerable-llm-v2"
        else:
            # If no API key is provided and no mock is set, this URL will be invalid, 
            # but we handle the actual failure in the generate_content loop.
            self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={self.api_key}"
            
    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def generate_content(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None
    ) -> Optional[LLMResponse]:
        """
        Routes to mock or Gemini. Returns None on any failure to trigger 
        Orchestrator fault tolerance.
        """
        # Safety check: if no connection parameters exist, fail immediately
        if not self.mock_base_url and not self.api_key:
            return None
            
        start_time = time.time()

        # Retries: 5 for Gemini, 1 for Target API (to ensure 1:1 request matching)
        max_retries = 5 if not self.mock_base_url else 1

        for attempt in range(max_retries):
            try:
                if self.mock_base_url:
                    response_json = await self._make_mock_request(prompt)
                else:
                    response_json = await self._make_gemini_request(prompt, system_instruction)

                if not response_json:
                    continue

                # --- ROBUST PARSING LOGIC ---
                text = ""
                decision = response_json.get('decision')  # Extract the backend decision record
                
                # 1. Try Gemini Format
                if 'candidates' in response_json:
                    text = response_json['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text', "")
                
                # 2. Try OpenAI Format (Common in many APIs)
                elif 'choices' in response_json:
                    text = response_json['choices'][0].get('message', {}).get('content', "")
                
                # 3. Try Simple Key Fallbacks
                else:
                    text = response_json.get('text') or response_json.get('response') or response_json.get('message', "")

                if text or text == "":
                    return LLMResponse(
                        text=str(text),
                        model_name=self.model_id,
                        processing_time=time.time() - start_time,
                        decision=decision
                    )
                
            except aiohttp.ClientResponseError as e:
                if e.status in [403, 422, 401]:
                    # Cloudflare often uses 403 for WAF blocks. 422 is used by some LLM gateways.
                    source = "Cloudflare/WAF" if e.status == 403 else "Target Gateway"
                    error_msg = f"SECURITY_BLOCK: Request rejected by {source} ({e.status})."
                    return LLMResponse(
                        text=error_msg,
                        model_name=self.model_id,
                        processing_time=time.time() - start_time
                    )
                elif e.status == 429:
                    # Rate limiting - backoff and retry if possible
                    error_msg = "RATE_LIMIT_REACHED: 429 Too Many Requests."
                    return LLMResponse(
                        text=error_msg,
                        model_name=self.model_id,
                        processing_time=time.time() - start_time
                    )
                print(f"[LLMAdapter] API Error {e.status}: {e.message}")
                # STOP retrying on ALL 4xx errors (Client errors/Security blocks)
                if 400 <= e.status < 500:
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as e:
                # Log the specific error to help debugging
                print(f"[LLMAdapter] Connection error on attempt {attempt + 1}: {type(e).__name__} - {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

        return None

    async def _make_mock_request(self, prompt: str) -> Dict[str, Any]:
        self.total_requests += 1
        session = await self._ensure_session()
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            # Critical: Ngrok and many WAF providers block the default aiohttp user agent.
            # Using a browser-like signature is mandatory for tunnel-based testing.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "ngrok-skip-browser-warning": "true"
        }
        
        async with session.post(
            self.base_url,
            # Matches target FastAPI schema (user_id/prompt) to prevent 422 validation errors
            json={
                "user_id": self.user_id,
                "prompt": prompt
            },
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15) # Shorter timeout for mocks
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _make_gemini_request(self, prompt: str, system_instruction: Optional[str]) -> Dict[str, Any]:
        self.total_requests += 1
        session = await self._ensure_session()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        async with session.post(
            self.base_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self):
        """Cleanup the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None