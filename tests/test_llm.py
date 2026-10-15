from rfp import config, llm


def test_embeddings_are_batched_under_the_per_minute_quota(monkeypatch):
    calls, sleeps = [], []

    class FakeEmbedder:
        def embed_documents(self, texts, task_type, output_dimensionality):
            calls.append((len(texts), task_type, output_dimensionality))
            return [[0.0] * output_dimensionality for _ in texts]
    monkeypatch.setattr(llm, "embedder", lambda: FakeEmbedder())
    out = llm._embed([f"t{i}" for i in range(200)], "RETRIEVAL_QUERY", sleep=sleeps.append)
    assert len(out) == 200
    assert [c[0] for c in calls] == [90, 90, 20]
    assert all(c[1:] == ("RETRIEVAL_QUERY", config.EMBED_DIM) for c in calls)
    assert sleeps == [llm.EMBED_PAUSE_S] * 2      # pause only between batches


def test_daily_quota_detection():
    assert llm.is_daily_quota_error(RuntimeError("quotaId: GenerateRequestsPerDayPerProjectPerModel"))
    assert not llm.is_daily_quota_error(RuntimeError("EmbedContentRequestsPerMinutePerUser"))
