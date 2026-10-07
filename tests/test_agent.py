from pathlib import Path

from verdictlens.agent import AgentOptions, _route_after_verify, build_agent_graph, run_agent
from verdictlens.verify import VerificationReport


def test_agent_runs_without_optional_model() -> None:
    result = run_agent("The Supreme Court allowed the appeal under Article 21.")
    assert result.final_summary
    assert result.verification.is_faithful


def test_agent_empty_input() -> None:
    result = run_agent("")
    assert result.final_summary == ""


def test_repair_route_is_bounded() -> None:
    options = AgentOptions(max_repairs=2)
    bad_report = VerificationReport(unsupported_entities=["foreign court"])
    assert _route_after_verify({"verification": bad_report, "repair_count": 0}, options) == "repair"
    assert _route_after_verify({"verification": bad_report, "repair_count": 2}, options) == "finalize"


def test_graph_builds_when_langgraph_is_installed(tmp_path: Path) -> None:
    graph = build_agent_graph(AgentOptions(trace_dir=tmp_path, run_id="test-run"))
    result = graph.invoke({"source": "The appeal was allowed."})
    assert result["final_summary"]
    assert (tmp_path / "test-run.jsonl").exists()
