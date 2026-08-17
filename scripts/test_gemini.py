"""Verify Gemini API connectivity without running document retrieval."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.llm_service import GeminiService  # noqa: E402


CONNECTIVITY_SYSTEM_INSTRUCTION = (
    "You are a connectivity test assistant. Respond briefly and plainly."
)
CONNECTIVITY_PROMPT = (
    "Reply with a short confirmation that the Gemini API connection works."
)


def main() -> None:
    """Send one minimal request to verify Gemini connectivity."""

    print("Gemini API connectivity test only; document retrieval is not used.")
    service = GeminiService()
    response = service.generate(
        prompt=CONNECTIVITY_PROMPT,
        system_instruction=CONNECTIVITY_SYSTEM_INSTRUCTION,
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
