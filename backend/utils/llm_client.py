"""Groq LLM client with multi-key rate-limit handling."""
import os
import re
import time
import logging
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClient:
    MAX_TOTAL_WAIT: int = 120

    def __init__(self):
        self.clients: dict[str, Groq] = {}
        self.cooldown_until: dict[str, float] = {}

        key_names = ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
        for name in key_names:
            val = os.getenv(name, "").strip()
            if val and val != "PASTE_YOUR_NEW_KEY_1_HERE" \
                     and val != "PASTE_YOUR_NEW_KEY_2_HERE" \
                     and val != "PASTE_YOUR_NEW_KEY_3_HERE":
                self.clients[val] = Groq(api_key=val)
                self.cooldown_until[val] = 0.0

        if not self.clients:
            raise RuntimeError(
                "[LLMClient] No valid Groq API keys found. "
                "Check your .env file — GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3"
            )

        self.default_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        logger.info(f"[LLMClient] Initialized with {len(self.clients)} key(s). Model: {self.default_model}")

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        if model is None:
            model = self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()
        last_error: Optional[Exception] = None

        while time.time() - start_time < self.MAX_TOTAL_WAIT:
            keys_tried_this_pass = 0

            for key, client in self.clients.items():
                remaining = self.cooldown_until.get(key, 0.0) - time.time()
                if remaining > 0:
                    continue

                keys_tried_this_pass += 1

                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self.cooldown_until[key] = 0.0
                    return resp.choices[0].message.content

                except Exception as exc:
                    msg = str(exc).lower()
                    last_error = exc

                    if "per day" in msg or "daily" in msg or "quota" in msg:
                        logger.warning(f"[LLMClient] key ...{key[-6:]} hit DAILY limit. Marking dead for 23h.")
                        self.cooldown_until[key] = time.time() + 82_800
                        continue

                    wait = self._parse_retry_after(msg)
                    logger.warning(f"[LLMClient] key ...{key[-6:]} rate limited. Cooling down for {wait:.1f}s.")
                    self.cooldown_until[key] = time.time() + wait
                    continue

            if keys_tried_this_pass == 0:
                active_cooldowns = {
                    k: v for k, v in self.cooldown_until.items()
                    if v > time.time()
                }
                if not active_cooldowns:
                    time.sleep(1.0)
                    continue

                next_available_at = min(active_cooldowns.values())
                sleep_for = (next_available_at - time.time()) + 1.5
                budget_remaining = (start_time + self.MAX_TOTAL_WAIT) - time.time()
                if sleep_for > budget_remaining:
                    logger.warning(f"[LLMClient] Wait ({sleep_for:.1f}s) exceeds budget ({budget_remaining:.1f}s left). Giving up.")
                    break

                logger.info(f"[LLMClient] All keys cooling down. Sleeping {sleep_for:.1f}s for next available key...")
                time.sleep(sleep_for)

        raise RuntimeError(
            f"[LLMClient] All Groq keys exhausted after {self.MAX_TOTAL_WAIT}s. Last error: {last_error}"
        )

    def _parse_retry_after(self, error_message: str) -> float:
        msg = error_message.lower()
        match = re.search(r'(\d+)m\s*([\d.]+)s', msg)
        if match:
            return float(match.group(1)) * 60 + float(match.group(2)) + 2.0
        match = re.search(r'(?:try again in|retry in|wait)\s*([\d.]+)s', msg)
        if match:
            return float(match.group(1)) + 2.0
        match = re.search(r'retry after\s*(\d+)', msg)
        if match:
            return float(match.group(1)) + 2.0
        match = re.search(r'([\d.]+)s', msg)
        if match and any(w in msg for w in ["limit", "rate", "wait", "retry"]):
            return float(match.group(1)) + 2.0
        logger.debug(f"[LLMClient] Could not parse retry-after from: '{error_message[:120]}'")
        return 30.0

    def _all_keys_dead(self) -> bool:
        now = time.time()
        return all((self.cooldown_until.get(k, 0) - now) > 3600 for k in self.clients)

    def key_count(self) -> int:
        return len(self.clients)

    def available_key_count(self) -> int:
        now = time.time()
        return sum(1 for k in self.clients if self.cooldown_until.get(k, 0) <= now)
