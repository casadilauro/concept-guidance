"""Paper-faithful recipe generation and diversity analysis."""

EMBEDDING_MODEL = "openai/text-embedding-3-large"
EMBEDDING_BACKEND = "openrouter"
K_NEIGHBORS = 15
MODEL_NAME = "openai/gpt-oss-120b:nitro"

from .generation import GenerationSettings, generate_nonce, generate_prior_titles, generate_wide
