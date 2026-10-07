"""LangGraph verify-and-repair summarization agent."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

from .abstractive import SummarizerConfig, cleanup_model, generate_summary
from .entities import extract_outcome_label
from .extractive import summarize_extractive
from .preprocess import split_sentences
from .verify import SentenceSupportFlag, VerificationReport, verify_summary

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentOptions:
    """Runtime options shared by graph nodes."""

    summarizer_config: SummarizerConfig | None = None
    use_model: bool = False
    p3_outcome: str | None = None
    max_repairs: int = 2
    trace_dir: Path = Path("outputs/traces")
    run_id: str = ""


class AgentState(TypedDict, total=False):
    source: str
    extracted_summary: str
    generated_summary: str
    current_summary: str
    verification: VerificationReport
    repair_count: int
    final_summary: str


@dataclass(frozen=True)
class AgentResult:
    extractive_summary: str
    generated_summary: str
    final_summary: str
    verification: VerificationReport
    repair_count: int = 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, VerificationReport):
        return asdict(value)
    if isinstance(value, SentenceSupportFlag):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _trace(options: AgentOptions, node: str, node_input: dict[str, Any], node_output: dict[str, Any]) -> None:
    """Append one node input/output event to the run trace."""
    options.trace_dir.mkdir(parents=True, exist_ok=True)
    run_id = options.run_id or "agent-run"
    path = options.trace_dir / f"{run_id}.jsonl"
    event = {"node": node, "input": _jsonable(node_input), "output": _jsonable(node_output)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _ollama_repair(sentence: str, evidence: list[str], reasons: list[str]) -> str:
    """Ask local Ollama to rewrite one sentence only, or return DELETE."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Install ollama to use the repair node") from exc
    prompt = (
        "Repair exactly one legal-summary sentence. Return only the replacement sentence or DELETE. "
        "You may use only the source evidence below; do not add facts, entities, or outcomes.\n"
        f"Flagged sentence: {sentence}\nReasons: {', '.join(reasons)}\n"
        f"Source evidence:\n- " + "\n- ".join(evidence)
    )
    response = ollama.chat(
        model="qwen2.5:7b-instruct",
        messages=[
            {"role": "system", "content": "You are a conservative legal-summary repair tool."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
    )
    content = response.get("message", {}).get("content", "") if isinstance(response, dict) else response.message.content
    return str(content).strip()


def _extract_node(state: AgentState, options: AgentOptions) -> AgentState:
    result = summarize_extractive(state["source"])
    _trace(options, "extract", {"source": state["source"]}, {"extractive_summary": result})
    return {"extracted_summary": result, "current_summary": result, "repair_count": 0}


def _generate_node(state: AgentState, options: AgentOptions) -> AgentState:
    summary = state["extracted_summary"]
    if options.use_model:
        summary = generate_summary(state["source"], options.summarizer_config)
    _trace(options, "generate", {"source": state["source"], "extractive_summary": state["extracted_summary"]}, {"generated_summary": summary})
    return {"generated_summary": summary, "current_summary": summary}


def _verify_node(state: AgentState, options: AgentOptions) -> AgentState:
    report = verify_summary(state["source"], state["current_summary"], p3_outcome=options.p3_outcome)
    _trace(options, "verify", {"summary": state["current_summary"]}, {"verification": report})
    return {"verification": report}


def _flag_for_report(summary: str, report: VerificationReport) -> SentenceSupportFlag | None:
    if report.sentence_flags:
        return report.sentence_flags[0]
    sentences = split_sentences(summary)
    for sentence in sentences:
        if any(entity.casefold() in sentence.casefold() for entity in report.unsupported_entities):
            return SentenceSupportFlag(sentence, ["unsupported_entity"], [])
    return None


def _repair_node(state: AgentState, options: AgentOptions) -> AgentState:
    report = state["verification"]
    summary = state["current_summary"]
    flag = _flag_for_report(summary, report)
    if flag is not None:
        cleanup_model(None)
        replacement = _ollama_repair(flag.sentence, flag.retrieved_source_sentences, flag.reasons)
        if replacement.casefold() == "delete":
            summary = summary.replace(flag.sentence, "").strip()
        elif replacement:
            summary = summary.replace(flag.sentence, replacement, 1)
    else:
        source_outcome = extract_outcome_label(state["source"], final_window=False)
        summary_outcome = extract_outcome_label(summary, final_window=False)
        if source_outcome and not summary_outcome:
            summary = f"{summary} The final disposition was {source_outcome}.".strip()
    repair_count = state.get("repair_count", 0) + 1
    _trace(options, "repair", {"summary": state["current_summary"], "verification": report}, {"summary": summary, "repair_count": repair_count})
    return {"current_summary": summary, "repair_count": repair_count}


def _route_after_verify(state: AgentState, options: AgentOptions) -> str:
    report = state["verification"]
    if report.is_faithful or state.get("repair_count", 0) >= options.max_repairs:
        return "finalize"
    return "repair"


def _finalize_node(state: AgentState, options: AgentOptions) -> AgentState:
    _trace(options, "finalize", {"summary": state["current_summary"], "verification": state["verification"]}, {"final_summary": state["current_summary"]})
    return {"final_summary": state["current_summary"]}


def build_agent_graph(options: AgentOptions | None = None) -> Any:
    """Build the extract -> generate -> verify -> repair -> verify -> finalize graph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("Install langgraph to build the agent graph") from exc
    options = options or AgentOptions(run_id=uuid.uuid4().hex)
    graph = StateGraph(AgentState)
    graph.add_node("extract", lambda state: _extract_node(state, options))
    graph.add_node("generate", lambda state: _generate_node(state, options))
    graph.add_node("verify", lambda state: _verify_node(state, options))
    graph.add_node("repair", lambda state: _repair_node(state, options))
    graph.add_node("finalize", lambda state: _finalize_node(state, options))
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", lambda state: _route_after_verify(state, options), {"repair": "repair", "finalize": "finalize"})
    graph.add_edge("repair", "verify")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(source: str, config: SummarizerConfig | None = None, use_model: bool = False) -> AgentResult:
    """Run the graph when available, with a deterministic fallback for minimal installs."""
    options = AgentOptions(summarizer_config=config, use_model=use_model, run_id=uuid.uuid4().hex)
    try:
        graph = build_agent_graph(options)
        state = graph.invoke({"source": source})
        return AgentResult(state.get("extracted_summary", ""), state.get("generated_summary", ""), state.get("final_summary", ""), state["verification"], state.get("repair_count", 0))
    except RuntimeError as exc:
        if "langgraph" not in str(exc):
            raise
        extracted = summarize_extractive(source)
        report = verify_summary(source, extracted, check_support=False)
        return AgentResult(extracted, extracted, extracted, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one judgment through the VerdictLens agent")
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=Path("outputs/traces"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/agent_result.json"))
    args = parser.parse_args()
    source = args.text_file.read_text(encoding="utf-8")
    options = AgentOptions(use_model=args.use_model, trace_dir=args.trace_dir, run_id=uuid.uuid4().hex)
    graph = build_agent_graph(options)
    state = graph.invoke({"source": source})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(state), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote agent result to %s", args.output)


if __name__ == "__main__":
    main()
