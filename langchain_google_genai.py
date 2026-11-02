"""
Compatibility shim for `langchain_google_genai.ChatGoogleGenerativeAI` using
Google Gemini 2.5 Flash via the official `google-generativeai` client.

This preserves the `.invoke(messages)` interface used by the agents and returns
an object with `.content`.
"""
import os
from typing import List, Any
import time
import random
import threading
from google.api_core import exceptions

# Global lock and state for throttling
api_call_lock = threading.Lock()
last_api_call_time = 0
MIN_CALL_INTERVAL = 2.0  # seconds (to stay under 30 requests per minute)

try:
    import google.generativeai as genai
except Exception:
    genai = None


class SimpleResponse:
    def __init__(self, content: str):
        self.content = content


class ChatGoogleGenerativeAI:
    """Minimal wrapper around Gemini for drop-in compatibility."""

    def __init__(self, model: str = None, google_api_key: str = None, temperature: float = 0.7):
        self.model = model or "models/gemini-2.5-flash"
        self.temperature = temperature
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        self._client_ok = False
        self._model = None

        if genai and self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                self._model = genai.GenerativeModel(self.model)
                self._client_ok = True
            except Exception:
                self._client_ok = False

    def _normalize_messages(self, messages: List[Any]) -> str:
        """Flatten a list of LC chat messages into a single prompt.

        We join system + human messages into a single text prompt for Gemini.
        """
        parts = []
        for m in messages:
            role = getattr(m, "role", None) or getattr(m, "type", None)
            content = getattr(m, "content", None) or getattr(m, "message", None)
            if isinstance(m, str):
                content = m
                role = "user"
            if content is None:
                continue
            if role in ("system", "ai", "assistant"):
                parts.append(f"System/Assistant:\n{content}\n")
            else:
                parts.append(f"User:\n{content}\n")
        return "\n".join(parts).strip()

    def invoke(self, messages: List[Any]):
        global last_api_call_time

        if not self.google_api_key:
            return SimpleResponse(content="[No LLM available] Gemini client not configured or missing GOOGLE_API_KEY.")

        max_retries = 3
        base_delay = 5  # seconds

        with api_call_lock:
            elapsed = time.time() - last_api_call_time
            if elapsed < MIN_CALL_INTERVAL:
                sleep_time = MIN_CALL_INTERVAL - elapsed
                print(f"Throttling API call. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            
            last_api_call_time = time.time()

        for attempt in range(max_retries):
            try:
                # Configure the client just-in-time for this call
                genai.configure(api_key=self.google_api_key)
                model = genai.GenerativeModel(self.model)
                
                prompt = self._normalize_messages(messages)
                
                # Generate content
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature
                    )
                )
                return SimpleResponse(content=response.text)
            except exceptions.ResourceExhausted as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Resource exhausted. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Returning resource exhausted error.")
                    return SimpleResponse(content="[No LLM available] Resource limit reached. Please try again later.")
            except exceptions.DeadlineExceeded:
                return SimpleResponse(content="[No LLM available] Request timed out. Please try again.")
            except Exception as e:
                # Use a logger if available, otherwise print
                print(f"Error invoking Gemini: {e}")
                return SimpleResponse(content=f"[No LLM available] Error: {e}")
