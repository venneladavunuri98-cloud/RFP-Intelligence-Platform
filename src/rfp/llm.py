"""Shared LLM plumbing: lazy model construction, bounded retries, telemetry.

Every structured LLM call goes through invoke_structured(): schema failures
and API errors are retried, logged, and finally returned as errors (never
raised), so callers can route to the error state / human review.
"""
import time

from . import config, telemetry


def chat_model():
    from langchain.chat_models import init_chat_model
    if not config.MODEL:
        raise RuntimeError("LLM_MODEL is not set (see .env.example)")
    return init_chat_model(config.MODEL, temperature=0)


def is_daily_quota_error(e: Exception) -> bool:
    """Free-tier per-day quota (e.g. GenerateRequestsPerDay...): retrying cannot succeed."""
    return "PerDay" in str(e)


def invoke_structured(runnable_factory, messages, run_id, node):
    """Returns (parsed_model | None, errors). `runnable_factory` builds a
    with_structured_output(..., include_raw=True) runnable."""
    errors = []
    for attempt in range(1, config.EXTRACT_MAX_ATTEMPTS + 1):  # 1 try + 2 retries
        started = time.perf_counter()
        try:
            out = runnable_factory().invoke(messages)
        except Exception as e:  # API/network/rate-limit errors: retry, then fail closed
            telemetry.log_call(run_id, node, config.MODEL, None, started,
                               attempt=attempt, ok=False, error=type(e).__name__)
            errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
            if is_daily_quota_error(e):
                errors.append("daily quota exhausted: not retrying")
                break
            if attempt < config.EXTRACT_MAX_ATTEMPTS:
                time.sleep(config.RETRY_BACKOFF_S * attempt)
            continue
        parsed = out.get("parsed")
        telemetry.log_call(run_id, node, config.MODEL, out.get("raw"), started,
                           attempt=attempt, ok=parsed is not None)
        if parsed is not None:
            return parsed, errors
        errors.append(f"attempt {attempt}: schema validation failed: {out.get('parsing_error')}")
        finish = (getattr(out.get("raw"), "response_metadata", None) or {}).get("finish_reason")
        if finish == "MAX_TOKENS":  # deterministic: the same prompt would truncate again
            errors.append("output truncated at the model's max output tokens: not retrying")
            break
    return None, errors


def embedder():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)


EMBED_BATCH = 90          # free tier counts every text as one request (limit 100/min)
EMBED_PAUSE_S = 61.0      # wait between batches so a batch never exceeds the per-minute quota


def _embed(texts: list[str], task_type: str, sleep=time.sleep) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        if start:
            sleep(EMBED_PAUSE_S)
        vectors += embedder().embed_documents(texts[start:start + EMBED_BATCH], task_type=task_type,
                                              output_dimensionality=config.EMBED_DIM)
    return vectors


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_queries(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_QUERY")
