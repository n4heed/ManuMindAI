"""
Pluggable embedding and generation providers for the RAG pipeline.

EMBEDDING_PROVIDER / LLM_PROVIDER in settings.py select which implementation
is used. "local" is a dependency-free, deterministic fallback that works with
no API key and no network access, so the app stays functional end-to-end
before organizer-issued embedding/LLM access is available.

To switch to a real provider once a key is issued:
  1. Set EMBEDDING_PROVIDER / LLM_PROVIDER in .env (e.g. "openai", "anthropic").
  2. Set the matching API key in .env.
  3. Fill in the call inside OpenAIEmbeddingProvider.embed /
     OpenAILLMProvider.generate (or the Anthropic equivalents) below.
No other code needs to change - callers only use get_embedding_provider()
and get_llm_provider().
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

from django.conf import settings

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9-]*")

# Common English stopwords plus generic manual-boilerplate words that would
# otherwise inflate similarity between unrelated passages (see
# LocalEmbeddingProvider docstring and README "Retrieval threshold").
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being to of in on for with and or
    this that these those it its as at by from into over under above
    should shall must can could would will do does did not no
    what when where why who which how
    """.split()
)


def _stem(token: str) -> str:
    """
    Minimal, conservative suffix stripping (no external dependency). This is
    NOT a real stemmer (no Porter algorithm) - it only trims the handful of
    endings that commonly separate a question's wording from a manual's
    (e.g. "checked"/"check", "filters"/"filter") while avoiding short words
    and words ending "ss" (e.g. "process") where stripping "s" would be wrong.
    """
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic sparse bag-of-words "embedding" (no download, no network
    call): term frequencies over the stopword-filtered vocabulary, L2
    normalized, stored as a {token: weight} dict rather than a fixed-size
    dense vector.

    An earlier version used feature hashing into a fixed 256-dim vector,
    which was dropped after testing showed hash collisions produced
    unrelated passages scoring (0.44) *higher* than genuinely related ones
    (0.09) - see README "Retrieval threshold" for the measurements. Exact
    term matching has no collision noise, so unrelated text with zero
    shared vocabulary scores exactly 0.0.
    """

    name = "local"

    def embed(self, texts: list[str]) -> list[dict]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> dict:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm > 0:
            counts = {k: v / norm for k, v in counts.items()}
        return counts


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use the openai embedding provider.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [item.embedding for item in response.data]


def get_embedding_provider() -> EmbeddingProvider:
    provider = settings.EMBEDDING_PROVIDER
    if provider == "local":
        return LocalEmbeddingProvider()
    if provider == "openai":
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


def cosine_similarity(a, b) -> float:
    """Works for either dense vectors (list[float]) or sparse {token: weight} dicts."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LLMProvider(ABC):
    name: str
    is_deterministic_fallback: bool = False

    @abstractmethod
    def generate(self, question: str, context_chunks: list[dict]) -> str:
        """context_chunks: list of {"text": str, "document_id": str, "page_number": int}."""
        ...


class LocalLLMProvider(LLMProvider):
    """
    Deterministic, extractive "answer": returns the most relevant sentences
    from the retrieved evidence, without calling any external AI model.
    This is NOT an AI-generated answer - callers must label it as such
    (see Question.llm_provider / the "local" badge in the UI).
    """

    name = "local"
    is_deterministic_fallback = True

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        question_tokens = set(_tokenize(question))
        scored_sentences: list[tuple[float, str]] = []
        for chunk in context_chunks:
            # Split on sentence punctuation AND newlines - tabular manual
            # content (schedules, limit tables) often has no punctuation,
            # so splitting on newlines too keeps each table row separate
            # instead of one giant run-on match.
            candidates = re.split(r"(?<=[.!?])\s+|\n+", chunk["text"])
            for sentence in candidates:
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_tokens = set(_tokenize(sentence))
                overlap = len(question_tokens & sentence_tokens)
                if overlap:
                    scored_sentences.append((overlap, sentence))
        if not scored_sentences:
            # No sentence shares a keyword with the question; fall back to the
            # single best-ranked chunk the retriever already selected.
            best = context_chunks[0]["text"].strip()
            return best[:600]
        scored_sentences.sort(key=lambda s: s[0], reverse=True)
        top_sentences: list[str] = []
        for _, sentence in scored_sentences:
            if sentence not in top_sentences:  # chunk overlap can duplicate the same sentence
                top_sentences.append(sentence)
            if len(top_sentences) == 3:
                break
        return " ".join(top_sentences)


class OpenAILLMProvider(LLMProvider):
    name = "openai"

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use the openai LLM provider.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        context_text = "\n\n".join(
            f"[{c['document_id']} p.{c['page_number']}] {c['text']}" for c in context_chunks
        )
        prompt = (
            "Answer the maintenance question using ONLY the evidence below. "
            "If the evidence does not contain the answer, say so explicitly.\n\n"
            f"Evidence:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


class AnthropicLLMProvider(LLMProvider):
    name = "anthropic"

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot use the anthropic LLM provider.")
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        context_text = "\n\n".join(
            f"[{c['document_id']} p.{c['page_number']}] {c['text']}" for c in context_chunks
        )
        prompt = (
            "Answer the maintenance question using ONLY the evidence below. "
            "If the evidence does not contain the answer, say so explicitly.\n\n"
            f"Evidence:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


def get_llm_provider() -> LLMProvider:
    provider = settings.LLM_PROVIDER
    if provider == "local":
        return LocalLLMProvider()
    if provider == "openai":
        return OpenAILLMProvider()
    if provider == "anthropic":
        return AnthropicLLMProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
