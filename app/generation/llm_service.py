"""Grounded text generation through the current Google Gen AI SDK."""

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
FALLBACK_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GeminiService:
    """Send prompts to Gemini without retrieval or prompt-building logic."""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if not api_key or api_key == "your_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is missing. Create a .env file in the project "
                "root and set GEMINI_API_KEY to your API key."
            )

        self.model_name = model_name
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60000),
        )

    def generate(
        self,
        prompt: str,
        system_instruction: str,
    ) -> str:
        """Generate one concise answer from a prepared grounded prompt."""

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("prompt must not be empty")

        cleaned_system_instruction = system_instruction.strip()
        if not cleaned_system_instruction:
            raise ValueError("system_instruction must not be empty")

        try:
            response = self._generate_content(
                self.model_name,
                cleaned_prompt,
                cleaned_system_instruction,
            )
        except (errors.APIError, httpx.TimeoutException) as exc:
            if isinstance(exc, errors.APIError) and not (
                _is_temporarily_unavailable(exc)
            ):
                _raise_generation_error(exc)

            print(
                "Primary Gemini model temporarily unavailable; "
                "trying fallback model."
            )
            try:
                response = self._generate_content(
                    FALLBACK_GEMINI_MODEL,
                    cleaned_prompt,
                    cleaned_system_instruction,
                )
            except httpx.TimeoutException:
                _raise_temporarily_unavailable_error()
            except errors.APIError as fallback_exc:
                if _is_temporarily_unavailable(fallback_exc):
                    _raise_temporarily_unavailable_error()
                _raise_generation_error(fallback_exc)

        try:
            answer = (response.text or "").strip()
        except ValueError as exc:
            raise RuntimeError(
                "Gemini response did not contain usable text"
            ) from exc

        if not answer:
            raise RuntimeError("Gemini returned an empty text response")

        return answer

    def _generate_content(
        self,
        model_name: str,
        prompt: str,
        system_instruction: str,
    ):
        """Generate content with one model using the shared SDK configuration."""

        return self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )


def _is_temporarily_unavailable(exc: errors.APIError) -> bool:
    """Return whether Gemini reported the transient UNAVAILABLE condition."""

    status_code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    return status_code == 503 or (
        status_code is None and status == "UNAVAILABLE"
    )


def _raise_generation_error(exc: errors.APIError) -> None:
    """Raise the existing readable Gemini API error."""

    status_code = getattr(exc, "code", None)
    status_text = (
        f" API status: {status_code}." if status_code is not None else ""
    )
    raise RuntimeError(
        "Gemini generation failed. Check the API key, model access, "
        f"and network connection.{status_text}"
    ) from None


def _raise_temporarily_unavailable_error() -> None:
    """Raise a concise error after both transient attempts fail."""

    raise RuntimeError(
        "Gemini is temporarily unavailable. Please retry the request."
    ) from None
