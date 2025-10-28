import os
from typing import Optional

DEFAULT_LOCAL_MODEL = "llama3.1:latest"  # Ollama model tag
DEFAULT_HUB_MODEL = "google/flan-t5-base"
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _ollama_is_available(base_url: str) -> bool:
    """
    Check whether an Ollama server is reachable before wiring up the ChatOllama LLM.
    """
    try:
        import httpx
    except ImportError:
        return False

    health_url = base_url.rstrip("/") + "/api/version"
    try:
        response = httpx.get(health_url, timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False


def load_llm(
    model: Optional[str] = None,
    use_hf_hub: Optional[bool] = None,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
):
    """
    Load an LLM via Hugging Face Hub (if requested/available) or a local Ollama chat model.

    Selection logic:
    - If use_hf_hub is True OR HUGGINGFACEHUB_API_TOKEN is set, use HF Hub (HuggingFaceHub wrapper)
    - Else, use local Ollama (ChatOllama)
    """
    token_present = bool(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    backend_hub = bool(use_hf_hub) or token_present

    chosen_model = model or (DEFAULT_HUB_MODEL if backend_hub else DEFAULT_LOCAL_MODEL)

    if backend_hub:
        try:
            from langchain_community.llms import HuggingFaceHub
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required for the Hugging Face Hub backend."
            ) from exc
        return HuggingFaceHub(
            repo_id=chosen_model,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        )

    # Default local backend: try Ollama first (if the server is reachable),
    # otherwise fail fast with a helpful message so the user can start Ollama.
    if not _ollama_is_available(DEFAULT_OLLAMA_URL):
        raise RuntimeError(
            "Could not reach the Ollama server at "
            f"{DEFAULT_OLLAMA_URL!r}. Start Ollama (`ollama serve`) or pass "
            "`--hub` with a Hugging Face model."
        )

    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for the Ollama backend. "
            "Install it via `pip install langchain-ollama`."
        ) from exc

    return ChatOllama(
        model=chosen_model,
        temperature=temperature,
    )
