"""OpenAI client configuration for pydantic_ai with proxy support."""

import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ancestral_synth.config import settings


# Cache the http client and provider to reuse connections
_http_client: httpx.AsyncClient | None = None
_provider: OpenAIProvider | None = None


def get_http_client() -> httpx.AsyncClient:
    """Get a shared httpx client with SSL verification disabled for proxy support."""
    global _http_client
    if _http_client is None:
        # Disable SSL verification to work with proxy's self-signed certificate
        _http_client = httpx.AsyncClient(verify=False)
    return _http_client


def get_openai_provider() -> OpenAIProvider:
    """Get a shared OpenAI provider with custom HTTP client."""
    global _provider
    if _provider is None:
        _provider = OpenAIProvider(http_client=get_http_client())
    return _provider


def create_openai_model(model_name: str | None = None) -> OpenAIChatModel:
    """Create an OpenAI model configured for the proxy environment.

    Args:
        model_name: The model name (e.g., "gpt-4o-mini"). Defaults to settings.llm_model.

    Returns:
        An OpenAIChatModel configured with SSL verification disabled.
    """
    name = model_name or settings.llm_model
    return OpenAIChatModel(name, provider=get_openai_provider())
