"""State must survive checkpointing unchanged — including int page keys,
which the verifier looks up with citation["page_number"] (an int)."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from rfp.schemas import RFPExtraction
from rfp.state import GraphState

CIT = {"page_number": 1, "exact_quote": "Request for Proposal: Cloud Services"}
EXTRACTION = RFPExtraction.model_validate({
    "title": {"value": "Cloud Services", "citation": CIT},
    "requirements": [],
}).model_dump()

SAMPLE: GraphState = {
    "run_id": "run-test", "pdf_path": "x.pdf",
    "pages": {1: "page one text", 2: "page two text"},
    "redaction_stats": {"EMAIL": 1}, "risk_flags": [], "extraction": EXTRACTION,
    "status": "ok", "errors": [],
}


def test_serializer_roundtrip_preserves_values_and_int_keys():
    # Same serializer MemorySaver uses today and PostgresSaver uses in Stage 2.
    serde = JsonPlusSerializer()
    restored = serde.loads_typed(serde.dumps_typed(SAMPLE))
    assert restored == SAMPLE
    assert all(isinstance(k, int) for k in restored["pages"])


def test_state_survives_memorysaver_checkpoint():
    def load(state: GraphState):
        return {k: v for k, v in SAMPLE.items() if k not in ("run_id", "pdf_path")}

    g = StateGraph(GraphState)
    g.add_node("load", load)
    g.add_edge(START, "load")
    g.add_edge("load", END)
    graph = g.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"run_id": "run-test", "pdf_path": "x.pdf"}, cfg)
    saved = graph.get_state(cfg).values  # read back from the checkpointer
    assert saved == SAMPLE
    assert saved["pages"][1] == "page one text"
