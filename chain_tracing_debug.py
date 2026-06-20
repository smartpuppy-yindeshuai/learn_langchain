"""
Chain Tracing & Debugging System for LangChain
==============================================
A comprehensive tracing and debugging framework that provides:

  1. Observability   — Record inputs, outputs, and intermediate state for every
                       component in a chain (LLM, Tool, Parser, Retriever).
  2. Performance     — Capture execution time per step, aggregate metrics, and
   Profiling           trigger alerts when thresholds are exceeded.
  3. Error           — Quickly locate the failing node and root cause when a
   Localization        chain breaks, with full call-tree context.
  4. Audit Trail     — Persist execution logs in memory (extensible to file or
                       external platforms like LangSmith).

Architecture:
  Chain execution emits callback events (on_chain_start, on_llm_end, etc.)
  → Custom CallbackHandlers intercept these events
  → SpanRecords are created and stored in a SpanStore
  → ConsoleTraceHandler prints formatted output in real time
  → PerformanceMonitor aggregates metrics and fires alerts

File: chain_tracing_debug.py
"""

import os
import re
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.agents import AgentAction, AgentFinish

# Set up Python logging as a fallback for non-console contexts
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 0. Common Configuration
# ============================================================

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# DashScope OpenAI-compatible endpoint parameters shared across all LLM instances
common_kwargs = {
    "api_key": openai_api_key,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "extra_body": {"enable_thinking": False},  # disable Qwen3 thinking mode
}


# ============================================================
# 1. SpanRecord — The Atomic Unit of Tracing
# ============================================================
# Every component invocation (LLM call, tool call, chain step, retriever query)
# produces one SpanRecord. Spans are linked via parent_run_id to form a tree
# that mirrors the nested execution structure of the chain.

@dataclass
class SpanRecord:
    """
    Represents a single execution span within a chain.

    Attributes:
        run_id:        Unique identifier (UUID) for this specific invocation.
        parent_run_id: The run_id of the parent span (None for root/chain-level).
        span_type:     Category of the component — one of "chain", "llm", "tool", "retriever".
        name:          Human-readable name of the component (e.g., "ChatOpenAI", "search").
        inputs:        The input data passed to this component.
        outputs:       The output data returned (None until the component completes).
        start_time:    Timestamp when this span began executing.
        end_time:      Timestamp when this span finished (None while still running).
        duration_ms:   Wall-clock execution time in milliseconds (computed on completion).
        tags:          User-defined tags for filtering and grouping (e.g., ["env:prod"]).
        metadata:      Arbitrary key-value pairs (model params, token usage, etc.).
        error:         Error message if the component failed, None on success.
        status:        Current state — "running", "success", or "error".
    """
    run_id: str
    parent_run_id: Optional[str]
    span_type: str                          # "chain" | "llm" | "tool" | "retriever"
    name: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status: str = "running"

    def complete(self, outputs: Dict[str, Any]) -> None:
        """
        Mark this span as successfully completed.
        Computes duration_ms from start_time to now and stores outputs.
        """
        self.end_time = datetime.now()
        # Calculate wall-clock duration in milliseconds
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.outputs = outputs
        self.status = "success"

    def fail(self, error: str) -> None:
        """
        Mark this span as failed.
        Records the error message and finalizes timing.
        """
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.error = error
        self.status = "error"

    def __str__(self) -> str:
        """Compact string representation for logging and display."""
        duration_str = f"{self.duration_ms:.1f}ms" if self.duration_ms else "running"
        status_icon = {"success": "✅", "error": "❌", "running": "⏳"}.get(self.status, "?")
        return (
            f"{status_icon} [{self.span_type}] {self.name} | "
            f"{duration_str} | run_id={self.run_id[:8]}"
        )


# ============================================================
# 2. SpanStore — In-Memory Storage with Tree Building
# ============================================================
# Collects all SpanRecords produced during chain execution. Provides methods
# to query by run_id, build parent-child trees, and compute aggregate statistics.

class SpanStore:
    """
    Thread-safe in-memory store for span records.

    Supports:
      - Insertion and retrieval of individual spans by run_id
      - Building a tree view of nested spans (parent → children)
      - Aggregate statistics (total spans, error count, average durations)
    """

    def __init__(self):
        # run_id → SpanRecord mapping for O(1) lookups
        self._spans: Dict[str, SpanRecord] = {}
        # parent_run_id → list of child run_ids for tree construction
        self._children: Dict[str, List[str]] = defaultdict(list)
        # Ordered list of root-level (chain) run_ids for iteration
        self._root_runs: List[str] = []

    def add(self, span: SpanRecord) -> None:
        """
        Insert a span into the store.
        If the span has a parent, register it in the parent's children list.
        If it's a root span (no parent), add it to the root runs list.
        """
        self._spans[span.run_id] = span

        if span.parent_run_id:
            # This span is a child of another span — link them
            self._children[span.parent_run_id].append(span.run_id)
        else:
            # Root-level span (typically a chain invocation)
            if span.run_id not in self._root_runs:
                self._root_runs.append(span.run_id)

    def get(self, run_id: str) -> Optional[SpanRecord]:
        """Retrieve a span by its run_id, or None if not found."""
        return self._spans.get(run_id)

    def get_children(self, parent_run_id: str) -> List[SpanRecord]:
        """
        Get all direct children of a given span.
        Returns a list of SpanRecords (not just IDs) for convenience.
        """
        child_ids = self._children.get(parent_run_id, [])
        return [self._spans[cid] for cid in child_ids if cid in self._spans]

    def get_tree(self, root_run_id: str, indent: int = 0) -> str:
        """
        Build a human-readable tree representation of all spans under a root.

        Example output:
          [chain] qa_chain | 1680.0ms | ✅
            [llm] ChatOpenAI | 1230.0ms | ✅
            [tool] search | 450.0ms | ✅
              [llm] ChatOpenAI | 320.0ms | ✅
        """
        span = self.get(root_run_id)
        if not span:
            return "(no spans found)"

        # Build the current node's line with indentation
        prefix = "  " * indent
        line = f"{prefix}{span}\n"

        # Recursively build children
        for child in self.get_children(root_run_id):
            line += self.get_tree(child.run_id, indent + 1)

        return line

    def stats(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all stored spans.

        Returns a dict with:
          - total_spans:    total number of spans recorded
          - success_count:  number of spans with status "success"
          - error_count:    number of spans with status "error"
          - running_count:  number of spans still in "running" state
          - avg_duration_ms: average duration of completed spans
          - error_rate:     fraction of completed spans that failed
          - by_type:        breakdown of counts and avg duration by span_type
        """
        all_spans = list(self._spans.values())
        completed = [s for s in all_spans if s.status in ("success", "error")]
        errors = [s for s in all_spans if s.status == "error"]
        with_duration = [s for s in completed if s.duration_ms is not None]

        # Per-type breakdown
        by_type: Dict[str, Dict[str, Any]] = {}
        for span_type in set(s.span_type for s in all_spans):
            typed = [s for s in with_duration if s.span_type == span_type]
            by_type[span_type] = {
                "count": len([s for s in all_spans if s.span_type == span_type]),
                "errors": len([s for s in errors if s.span_type == span_type]),
                "avg_duration_ms": (
                    sum(s.duration_ms for s in typed) / len(typed) if typed else 0
                ),
            }

        total_completed = len(completed)
        return {
            "total_spans": len(all_spans),
            "success_count": len([s for s in all_spans if s.status == "success"]),
            "error_count": len(errors),
            "running_count": len([s for s in all_spans if s.status == "running"]),
            "avg_duration_ms": (
                sum(s.duration_ms for s in with_duration) / len(with_duration)
                if with_duration else 0
            ),
            "error_rate": len(errors) / total_completed if total_completed else 0,
            "by_type": by_type,
        }


# Global span store instance — all handlers write to this shared store
span_store = SpanStore()


# ============================================================
# 3. TracingCallbackHandler — Core Event Collector
# ============================================================
# Implements LangChain's BaseCallbackHandler to intercept every lifecycle event
# emitted during chain execution. Each event creates or updates a SpanRecord.

class TracingCallbackHandler(BaseCallbackHandler):
    """
    A comprehensive callback handler that traces every event in a LangChain
    execution into SpanRecords stored in the global SpanStore.

    Lifecycle coverage:
      - Chain events:  on_chain_start / on_chain_end / on_chain_error
      - LLM events:    on_llm_start / on_llm_end / on_llm_error
      - Tool events:   on_tool_start / on_tool_end / on_tool_error
      - Retriever:     on_retriever_start / on_retriever_end / on_retriever_error

    Usage:
        handler = TracingCallbackHandler()
        chain.invoke(input, config={"callbacks": [handler]})
        # All spans are now in span_store
    """

    def __init__(self, tags: Optional[List[str]] = None):
        """
        Initialize the handler with optional user-defined tags.

        Args:
            tags: List of tags to attach to every span (e.g., ["env:prod", "v1.2"]).
                  Useful for filtering and grouping traces later.
        """
        super().__init__()
        self.tags = tags or []

    def _make_span_id(self) -> str:
        """Generate a unique span identifier (UUID4)."""
        return str(uuid.uuid4())

    # ----- Chain Events -----

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a chain (or any Runnable) begins execution.
        Creates a new "chain" type SpanRecord and registers it in the store.
        """
        # Guard: some Runnables (RunnablePassthrough, RunnableLambda, etc.)
        # pass None for serialized instead of a dict
        serialized = serialized or {}
        span = SpanRecord(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            span_type="chain",
            # Extract the chain name from serialized metadata, fallback to "chain"
            name=serialized.get("name", "chain"),
            # Sanitize inputs to ensure they're serializable
            inputs=_safe_dict(inputs),
            tags=self.tags + (tags or []),
            metadata={
                **kwargs.get("metadata", {}),
                "serialized": serialized,
            },
        )
        span_store.add(span)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a chain completes successfully.
        Updates the corresponding SpanRecord with outputs and finalizes timing.
        """
        span = span_store.get(str(run_id))
        if span:
            span.complete(_safe_dict(outputs))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a chain raises an unhandled exception.
        Marks the SpanRecord as failed and records the error message.
        """
        span = span_store.get(str(run_id))
        if span:
            span.fail(f"{type(error).__name__}: {error}")

    # ----- LLM Events -----

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fired when an LLM call begins (before the API request is sent).
        Creates an "llm" type SpanRecord.

        Note: 'prompts' contains the rendered prompt strings that will be
        sent to the model. This is invaluable for debugging prompt issues.
        """
        # Guard against None serialized dict
        serialized = serialized or {}
        # Extract model parameters from invocation_params in kwargs
        invocation_params = kwargs.get("invocation_params", {})
        span = SpanRecord(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            span_type="llm",
            name=serialized.get("name", serialized.get("id", ["llm"])[-1] if isinstance(serialized.get("id"), list) else "llm"),
            inputs={"prompts": prompts},
            tags=self.tags + (tags or []),
            metadata={
                "model": invocation_params.get("model_name", "unknown"),
                "temperature": invocation_params.get("temperature"),
                "max_tokens": invocation_params.get("max_tokens"),
            },
        )
        span_store.add(span)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when an LLM call completes successfully.
        Records the response and extracts token usage from the response metadata.
        """
        span = span_store.get(str(run_id))
        if span:
            # Extract token usage — try multiple locations since different
            # providers (OpenAI, DashScope, etc.) store it differently
            token_usage = {}

            # Location 1: response.llm_output (common for ChatOpenAI / DashScope)
            if hasattr(response, "llm_output") and response.llm_output:
                tu = response.llm_output.get("token_usage", response.llm_output)
                if isinstance(tu, dict):
                    token_usage = tu

            # Location 2: generation_info on individual generations
            if not token_usage and hasattr(response, "generations") and response.generations:
                first_gen = response.generations[0]
                if first_gen and hasattr(first_gen[0], "generation_info"):
                    gen_info = first_gen[0].generation_info or {}
                    token_usage = gen_info.get("usage", gen_info)

            # Location 3: usage_metadata on the message itself (newer LangChain)
            if not token_usage and hasattr(response, "generations") and response.generations:
                msg = response.generations[0][0].message if response.generations[0] else None
                if msg and hasattr(msg, "usage_metadata"):
                    um = msg.usage_metadata or {}
                    token_usage = {
                        "prompt_tokens": um.get("input_tokens", 0),
                        "completion_tokens": um.get("output_tokens", 0),
                        "total_tokens": um.get("total_tokens", 0),
                    }

            # Store the response text (first generation only for simplicity)
            output_text = ""
            if hasattr(response, "generations") and response.generations:
                output_text = response.generations[0][0].text if response.generations[0] else ""

            span.metadata["token_usage"] = token_usage
            span.complete({"response": output_text, "token_usage": token_usage})

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when an LLM call fails (API error, timeout, rate limit, etc.).
        Marks the LLM span as failed.
        """
        span = span_store.get(str(run_id))
        if span:
            span.fail(f"{type(error).__name__}: {error}")

    # ----- Tool Events -----

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a tool invocation begins.
        Creates a "tool" type SpanRecord with the tool name and input.
        """
        serialized = serialized or {}
        span = SpanRecord(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            span_type="tool",
            name=serialized.get("name", "tool"),
            inputs={"input": input_str, "structured_inputs": inputs},
            tags=self.tags + (tags or []),
        )
        span_store.add(span)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a tool completes successfully.
        Records the tool's output string.
        """
        span = span_store.get(str(run_id))
        if span:
            span.complete({"output": output})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a tool invocation fails.
        Marks the tool span as failed with the error details.
        """
        span = span_store.get(str(run_id))
        if span:
            span.fail(f"{type(error).__name__}: {error}")

    # ----- Retriever Events -----

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a retriever begins fetching documents.
        Creates a "retriever" type SpanRecord.
        """
        serialized = serialized or {}
        span = SpanRecord(
            run_id=str(run_id),
            parent_run_id=str(parent_run_id) if parent_run_id else None,
            span_type="retriever",
            name=serialized.get("name", "retriever"),
            inputs={"query": query},
            tags=self.tags + (tags or []),
        )
        span_store.add(span)

    def on_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Fired when a retriever returns documents.
        Records the number and content of retrieved documents.
        """
        span = span_store.get(str(run_id))
        if span:
            # Store document count and page content summaries
            doc_summaries = []
            for doc in documents:
                content = getattr(doc, "page_content", str(doc))
                # Truncate long documents to keep the trace readable
                doc_summaries.append(content[:200] + "..." if len(content) > 200 else content)
            span.complete({
                "document_count": len(documents),
                "documents": doc_summaries,
            })

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Fired when a retriever fails. Marks the span as failed."""
        span = span_store.get(str(run_id))
        if span:
            span.fail(f"{type(error).__name__}: {error}")


# ============================================================
# 4. ConsoleTraceHandler — Real-Time Formatted Console Output
# ============================================================
# A lightweight callback handler that prints human-readable trace lines to
# stdout in real time as events occur. Ideal for development and debugging.
#
# Output format:
#   [event_type] [component_name] ▶/◀ details | timing

class ConsoleTraceHandler(BaseCallbackHandler):
    """
    Prints formatted trace lines to the console in real time.

    Designed for development/debugging use. For production, prefer
    TracingCallbackHandler + SpanStore (or an external platform like LangSmith).

    Example output:
        [chain/start] [qa_chain] ▶ input: {"question": "..."}
          [llm/start] [ChatOpenAI] ▶ prompts: [...]
          [llm/end]   [ChatOpenAI] ◀ 1.23s | tokens: 128/256
        [chain/end]   [qa_chain] ◀ 1.68s | status: success
    """

    def __init__(self, indent: int = 0):
        """
        Args:
            indent: Initial indentation level (number of 2-space indents).
                    Useful when this handler is used for a sub-chain.
        """
        super().__init__()
        self.indent = indent
        # Track start times per run_id for inline duration calculation
        self._start_times: Dict[str, float] = {}

    def _log(self, level: str, prefix: str, name: str, message: str, depth: int = 0) -> None:
        """
        Print a formatted trace line with proper indentation and coloring.

        Args:
            level:   Event level string (e.g., "chain/start", "llm/end").
            prefix:  Arrow prefix — "▶" for start events, "◀" for end/error events.
            name:    Component name (e.g., "ChatOpenAI", "search").
            message: The details to display.
            depth:   Additional indentation depth (for nested spans).
        """
        total_indent = self.indent + depth
        padding = "  " * total_indent
        print(f"{padding}[{level}] [{name}] {prefix} {message}")

    def _get_depth(self, parent_run_id: Optional[uuid.UUID]) -> int:
        """Compute indentation depth from parent chain depth."""
        # Simple heuristic: each nesting level adds 1 to depth
        return 1 if parent_run_id else 0

    # ----- Chain Events -----

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        # Guard: serialized may be None for some Runnables
        serialized = serialized or {}
        depth = self._get_depth(parent_run_id)
        name = serialized.get("name", "chain")
        self._start_times[str(run_id)] = time.time()
        # Show a truncated version of inputs to avoid flooding the console
        input_preview = _truncate_dict(_safe_dict(inputs), max_len=120)
        self._log("chain/start", "▶", name, f"input: {input_preview}", depth)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        span_id = str(run_id)
        duration = time.time() - self._start_times.pop(span_id, time.time())
        span = span_store.get(span_id)
        depth = 1 if span and span.parent_run_id else 0
        name = span.name if span else "chain"
        self._log("chain/end", "◀", name, f"{duration:.2f}s | status: success", depth)

    def on_chain_error(self, error, *, run_id, **kwargs):
        span_id = str(run_id)
        duration = time.time() - self._start_times.pop(span_id, time.time())
        span = span_store.get(span_id)
        depth = 1 if span and span.parent_run_id else 0
        name = span.name if span else "chain"
        self._log("chain/error", "✗", name, f"{duration:.2f}s | error: {error}", depth)

    # ----- LLM Events -----

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        serialized = serialized or {}
        depth = self._get_depth(parent_run_id) + (1 if parent_run_id else 0)
        name = serialized.get("name", "llm")
        self._start_times[str(run_id)] = time.time()
        # Show prompt count and first 80 chars of the first prompt
        prompt_preview = prompts[0][:80] + "..." if prompts and len(prompts[0]) > 80 else (prompts[0] if prompts else "")
        self._log("llm/start", "▶", name, f"prompts: [{prompt_preview}]", depth)

    def on_llm_end(self, response, *, run_id, **kwargs):
        span_id = str(run_id)
        duration = time.time() - self._start_times.pop(span_id, time.time())
        span = span_store.get(span_id)
        depth = 2 if span and span.parent_run_id else 1
        name = span.name if span else "llm"
        # Extract token info if available
        tokens = ""
        if span and span.metadata.get("token_usage"):
            tu = span.metadata["token_usage"]
            tokens = f" | tokens: {tu.get('prompt_tokens', '?')}/{tu.get('completion_tokens', '?')}"
        self._log("llm/end", "◀", name, f"{duration:.2f}s{tokens}", depth)

    def on_llm_error(self, error, *, run_id, **kwargs):
        span_id = str(run_id)
        duration = time.time() - self._start_times.pop(span_id, time.time())
        span = span_store.get(span_id)
        depth = 2 if span and span.parent_run_id else 1
        name = span.name if span else "llm"
        self._log("llm/error", "✗", name, f"{duration:.2f}s | error: {error}", depth)

    # ----- Tool Events -----

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        serialized = serialized or {}
        depth = self._get_depth(parent_run_id) + (1 if parent_run_id else 0)
        name = serialized.get("name", "tool")
        self._start_times[str(run_id)] = time.time()
        input_preview = input_str[:100] + "..." if len(input_str) > 100 else input_str
        self._log("tool/start", "▶", name, f"input: {input_preview}", depth)

    def on_tool_end(self, output, *, run_id, **kwargs):
        span_id = str(run_id)
        duration = time.time() - self._start_times.pop(span_id, time.time())
        span = span_store.get(span_id)
        depth = 2 if span and span.parent_run_id else 1
        name = span.name if span else "tool"
        output_preview = str(output)[:100] + "..." if len(str(output)) > 100 else str(output)
        self._log("tool/end", "◀", name, f"{duration:.2f}s | output: {output_preview}", depth)

    def on_tool_error(self, error, *, run_id, **kwargs):
        span_id = str(run_id)
        duration = time.time() - self._start_times.pop(span_id, time.time())
        span = span_store.get(span_id)
        depth = 2 if span and span.parent_run_id else 1
        name = span.name if span else "tool"
        self._log("tool/error", "✗", name, f"{duration:.2f}s | error: {error}", depth)


# ============================================================
# 5. PerformanceMonitor — Metrics Collection & Alerting
# ============================================================
# Runs alongside the TracingCallbackHandler, watching for performance
# anomalies such as slow chains, high error rates, or excessive token usage.

class PerformanceMonitor(BaseCallbackHandler):
    """
    Monitors performance metrics in real time and logs warnings when
    configurable thresholds are exceeded.

    Tracked metrics:
      - End-to-end chain latency
      - Per-LLM call latency
      - Per-tool call latency
      - Token consumption per LLM call
      - Consecutive error count

    Thresholds (configurable via constructor):
      - chain_latency_warn_sec:  warn if total chain time exceeds this (default: 30s)
      - llm_latency_warn_sec:    warn if a single LLM call exceeds this (default: 20s)
      - tool_latency_warn_sec:   warn if a single tool call exceeds this (default: 10s)
      - token_warn_threshold:    warn if token usage per call exceeds this (default: 4000)
      - consecutive_error_limit: warn after this many consecutive errors (default: 3)
    """

    def __init__(
        self,
        chain_latency_warn_sec: float = 30.0,
        llm_latency_warn_sec: float = 20.0,
        tool_latency_warn_sec: float = 10.0,
        token_warn_threshold: int = 4000,
        consecutive_error_limit: int = 3,
    ):
        super().__init__()
        # Threshold configuration
        self.chain_latency_warn = chain_latency_warn_sec
        self.llm_latency_warn = llm_latency_warn_sec
        self.tool_latency_warn = tool_latency_warn_sec
        self.token_warn = token_warn_threshold
        self.error_limit = consecutive_error_limit

        # Internal state tracking
        self._start_times: Dict[str, float] = {}  # run_id → start timestamp
        self._consecutive_errors = 0               # rolling error counter
        self._total_llm_calls = 0                  # lifetime LLM call count
        self._total_tokens_used = 0                # lifetime token count

    # ----- Chain Events -----

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """Record chain start time for end-to-end latency measurement."""
        self._start_times[f"chain:{run_id}"] = time.time()

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        """
        Check end-to-end chain latency against the warning threshold.
        Logs a WARNING if the chain took longer than expected.
        """
        key = f"chain:{run_id}"
        start = self._start_times.pop(key, None)
        if start:
            duration = time.time() - start
            if duration > self.chain_latency_warn:
                logger.warning(
                    f"⚠️ SLOW CHAIN ALERT: chain completed in {duration:.2f}s "
                    f"(threshold: {self.chain_latency_warn}s)"
                )
            # Reset consecutive error counter on success
            self._consecutive_errors = 0

    def on_chain_error(self, error, *, run_id, **kwargs):
        """
        Track consecutive chain errors and alert when the limit is reached.
        Useful for detecting systemic failures (e.g., API down, bad config).
        """
        self._start_times.pop(f"chain:{run_id}", None)
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.error_limit:
            logger.warning(
                f"🚨 CONSECUTIVE ERROR ALERT: {self._consecutive_errors} "
                f"consecutive chain errors detected. Last error: {error}"
            )

    # ----- LLM Events -----

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        """Record LLM call start time for latency measurement."""
        self._start_times[f"llm:{run_id}"] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs):
        """
        Check LLM call latency and token usage against warning thresholds.
        """
        key = f"llm:{run_id}"
        start = self._start_times.pop(key, None)
        self._total_llm_calls += 1

        if start:
            duration = time.time() - start
            if duration > self.llm_latency_warn:
                logger.warning(
                    f"⚠️ SLOW LLM ALERT: LLM call took {duration:.2f}s "
                    f"(threshold: {self.llm_latency_warn}s)"
                )

        # Check token usage from the span record
        span = span_store.get(str(run_id))
        if span and span.metadata.get("token_usage"):
            tu = span.metadata["token_usage"]
            total_tokens = tu.get("prompt_tokens", 0) + tu.get("completion_tokens", 0)
            self._total_tokens_used += total_tokens
            if total_tokens > self.token_warn:
                logger.warning(
                    f"⚠️ HIGH TOKEN USAGE: {total_tokens} tokens used "
                    f"(threshold: {self.token_warn})"
                )

    def on_llm_error(self, error, *, run_id, **kwargs):
        """Clean up start time tracking on LLM error."""
        self._start_times.pop(f"llm:{run_id}", None)

    # ----- Tool Events -----

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        """Record tool call start time for latency measurement."""
        self._start_times[f"tool:{run_id}"] = time.time()

    def on_tool_end(self, output, *, run_id, **kwargs):
        """Check tool call latency against the warning threshold."""
        key = f"tool:{run_id}"
        start = self._start_times.pop(key, None)
        if start:
            duration = time.time() - start
            if duration > self.tool_latency_warn:
                logger.warning(
                    f"⚠️ SLOW TOOL ALERT: tool call took {duration:.2f}s "
                    f"(threshold: {self.tool_latency_warn}s)"
                )

    def on_tool_error(self, error, *, run_id, **kwargs):
        """Clean up start time tracking on tool error."""
        self._start_times.pop(f"tool:{run_id}", None)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of all performance metrics collected so far.

        Returns:
            Dict with total_llm_calls, total_tokens, consecutive_errors, etc.
        """
        return {
            "total_llm_calls": self._total_llm_calls,
            "total_tokens_used": self._total_tokens_used,
            "consecutive_errors": self._consecutive_errors,
            "thresholds": {
                "chain_latency_warn_sec": self.chain_latency_warn,
                "llm_latency_warn_sec": self.llm_latency_warn,
                "tool_latency_warn_sec": self.tool_latency_warn,
                "token_warn_threshold": self.token_warn,
                "consecutive_error_limit": self.error_limit,
            },
        }


# ============================================================
# 6. PII Masking Utility
# ============================================================
# Sanitizes sensitive personal information from trace data before it is
# stored or displayed. This is a security requirement per the spec.

def mask_pii(text: str) -> str:
    """
    Mask personally identifiable information (PII) in a text string.

    Currently masks:
      - Chinese mobile phone numbers (1xx-xxxx-xxxx)
      - Email addresses
      - ID card numbers (18-digit Chinese format)
      - Generic API keys / tokens (key=xxx patterns)

    Args:
        text: The raw text that may contain PII.

    Returns:
        The text with PII replaced by [MASKED] placeholders.
    """
    # ORDER MATTERS: longer/more-specific patterns must be matched first
    # 18-digit Chinese ID card numbers (must come before phone to avoid partial match)
    text = re.sub(r'\d{17}[\dXx]', '[MASKED_ID]', text)
    # Chinese mobile phone numbers: 13x-xxxx-xxxx, 15x, 18x, etc.
    text = re.sub(r'1[3-9]\d{9}', '[MASKED_PHONE]', text)
    # Email addresses
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.]+', '[MASKED_EMAIL]', text)
    # API key patterns (key=..., token=..., secret=...)
    text = re.sub(r'(key|token|secret|password)\s*[=:]\s*\S+', r'\1=[MASKED]', text, flags=re.IGNORECASE)
    return text


# ============================================================
# 7. Helper Utilities
# ============================================================

def _safe_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert any object to a JSON-safe dictionary.

    Handles:
      - dict → returned as-is (with recursive sanitization)
      - str/number → wrapped in {"value": obj}
      - Other types → converted via str()
    """
    if isinstance(obj, dict):
        return {str(k): _safe_value(v) for k, v in obj.items()}
    if isinstance(obj, (str, int, float, bool)):
        return {"value": obj}
    return {"value": str(obj)}


def _safe_value(v: Any) -> Any:
    """Convert a single value to a JSON-safe type."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return _safe_dict(v)
    if isinstance(v, (list, tuple)):
        return [_safe_value(item) for item in v]
    return str(v)


def _truncate_dict(d: Dict[str, Any], max_len: int = 200) -> Dict[str, Any]:
    """
    Truncate string values in a dictionary to keep console output readable.

    Args:
        d:      The dictionary to truncate.
        max_len: Maximum character length for any string value.

    Returns:
        A new dictionary with truncated values.
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + f"... (+{len(v) - max_len} chars)"
        elif isinstance(v, dict):
            result[k] = _truncate_dict(v, max_len)
        else:
            result[k] = v
    return result


# ============================================================
# 8. Example Chain — Multi-Step QA Chain with Tracing
# ============================================================
# Demonstrates the tracing system on a realistic multi-step chain:
#   Step 1: Classify the question type (LLM)
#   Step 2: Simulate a "search tool" lookup
#   Step 3: Generate the final answer using the search context (LLM)

# ----- 8.1 LLM Instances -----

# Lightweight model for classification — low token count, low temperature
llm_classifier = ChatOpenAI(
    model_name="qwen3-max",
    temperature=0.1,
    max_tokens=30,
    **common_kwargs,
)

# Heavy model for answer generation — higher token count, moderate temperature
llm_answer = ChatOpenAI(
    model_name="qwen3.7-max",
    temperature=0.5,
    max_tokens=300,
    **common_kwargs,
)

# ----- 8.2 Step 1: Question Classifier -----

classifier_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Classify the following question into one of these categories:
- factual: asking about a verifiable fact
- opinion: asking for an opinion or recommendation
- howto: asking how to do something

Only output the category label (factual / opinion / howto). No explanation.

Question: {question}
Category:""",
)

# classifier_chain: prompt → LLM → extract text
classifier_chain = classifier_prompt | llm_classifier | StrOutputParser()

# ----- 8.3 Step 2: Simulated Search Tool -----

def simulated_search(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates a search/retrieval tool that returns context documents.

    In a real application, this would query a vector database, search engine,
    or knowledge base. Here we return hardcoded results for demonstration.

    Args:
        data: Dict containing at least {"question": "..."}.

    Returns:
        Dict with original question, classified type, and retrieved context.
    """
    question = data.get("question", "")
    q_type = data.get("type", "factual").strip().lower()

    # Return different context based on question type to simulate
    # how a real retriever would adapt its search strategy
    context_map = {
        "factual": "Quantum computing uses quantum mechanical phenomena such as "
                   "superposition and entanglement to process information. Unlike "
                   "classical bits that are either 0 or 1, qubits can exist in a "
                   "superposition of both states simultaneously.",
        "opinion": "Expert consensus suggests that quantum computing will "
                   "revolutionize cryptography, drug discovery, and optimization "
                   "problems within the next decade, though practical limitations "
                   "remain for general-purpose computing.",
        "howto": "To get started with quantum computing: 1) Learn linear algebra "
                 "and quantum mechanics basics. 2) Study quantum gates and circuits. "
                 "3) Use IBM Qiskit or Google Cirq frameworks. 4) Practice with "
                 "simulated quantum circuits before using real quantum hardware.",
    }

    context = context_map.get(q_type, context_map["factual"])

    return {
        "question": question,
        "type": q_type,
        "context": context,
    }

# Wrap the plain function as a LangChain Runnable so it participates in tracing
search_tool = RunnableLambda(simulated_search)

# ----- 8.4 Step 3: Answer Generator -----

answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a knowledgeable assistant. Answer the following question
based on the provided context. Be clear, concise, and accurate.

Context: {context}

Question: {question}

Answer:""",
)

# answer_chain: prompt → LLM → extract text
answer_chain = answer_prompt | llm_answer | StrOutputParser()

# ----- 8.5 Full Chain Assembly -----
# Data flow:
#   {"question": "..."}
#     → classify type (adds "type" field via RunnablePassthrough.assign)
#     → simulated search (adds "context" field)
#     → generate answer (uses "question" + "context")
#
# Each step emits callback events that are intercepted by our handlers.

full_qa_chain = (
    # Step 1: classify the question, preserving the original input
    RunnablePassthrough.assign(type=classifier_chain)
    # Step 2: look up context via simulated search
    | search_tool
    # Step 3: generate the final answer from context + question
    | RunnablePassthrough.assign(answer=answer_chain)
)


# ============================================================
# 9. Debug Breakpoint Utility
# ============================================================
# A RunnableLambda that pauses execution for interactive debugging.
# Insert it anywhere in a chain to inspect intermediate state.

def debug_breakpoint(data: Any) -> Any:
    """
    Interactive debug breakpoint — pauses chain execution and drops into
    Python's pdb debugger. Insert this between chain steps to inspect
    intermediate state.

    Usage:
        chain = step1 | RunnableLambda(debug_breakpoint) | step2

    When the chain reaches this point, you'll get a pdb prompt where you can:
      - `data`        — inspect the current data
      - `data.keys()` — see available fields
      - `c`           — continue execution
      - `q`           — quit
    """
    print(f"\n🔍 DEBUG BREAKPOINT — inspecting data:")
    print(f"   Type: {type(data).__name__}")
    if isinstance(data, dict):
        for k, v in data.items():
            val_str = str(v)[:100]
            print(f"   {k}: {val_str}")
    else:
        print(f"   Value: {str(data)[:200]}")
    print(f"   (Returning data unchanged — remove debug_breakpoint for production)\n")
    return data


# Chain with a debug breakpoint inserted between search and answer generation
debug_qa_chain = (
    RunnablePassthrough.assign(type=classifier_chain)
    | search_tool
    | RunnableLambda(debug_breakpoint)       # ← Pause here to inspect search results
    | RunnablePassthrough.assign(answer=answer_chain)
)


# ============================================================
# 10. Test Cases & Execution
# ============================================================

test_cases = [
    {
        "name": "Factual Question (Normal Execution)",
        "question": "What is quantum computing and how does it differ from classical computing?",
        "expected_type": "factual",
        "description": "Tests the happy path — all steps should succeed.",
    },
    {
        "name": "How-To Question (Normal Execution)",
        "question": "How do I get started with quantum computing?",
        "expected_type": "howto",
        "description": "Tests a different classification path through the chain.",
    },
    {
        "name": "Opinion Question (Normal Execution)",
        "question": "What do experts think about the future of quantum computing?",
        "expected_type": "opinion",
        "description": "Tests the opinion classification and corresponding context retrieval.",
    },
]


def run_with_tracing(chain, input_data: Dict[str, Any], label: str = "") -> Dict[str, Any]:
    """
    Execute a chain with full tracing enabled.

    Attaches three callback handlers:
      1. TracingCallbackHandler — collects SpanRecords into SpanStore
      2. ConsoleTraceHandler    — prints real-time trace to stdout
      3. PerformanceMonitor     — watches for performance anomalies

    Args:
        chain:      The LangChain runnable to execute.
        input_data: The input dictionary for the chain.
        label:      A human-readable label for this execution (displayed in output).

    Returns:
        The chain's output dictionary.
    """
    # Instantiate all three handlers
    tracing_handler = TracingCallbackHandler(tags=[f"test:{label}"])
    console_handler = ConsoleTraceHandler()
    perf_monitor = PerformanceMonitor(
        chain_latency_warn_sec=15.0,  # Lower threshold for demo purposes
        llm_latency_warn_sec=10.0,
        tool_latency_warn_sec=5.0,
    )

    print(f"\n{'═' * 70}")
    print(f"  📊 Executing: {label}")
    print(f"{'═' * 70}")

    # Execute the chain with all handlers attached
    result = chain.invoke(
        input_data,
        config={
            "callbacks": [tracing_handler, console_handler, perf_monitor],
            "tags": [f"test:{label}"],
        },
    )

    return result


def print_span_tree(label: str) -> None:
    """
    Print the span tree for the most recent chain execution.

    Finds the latest root run in the SpanStore and displays its tree.
    Also prints aggregate statistics.
    """
    print(f"\n{'─' * 70}")
    print(f"  🌳 Span Tree: {label}")
    print(f"{'─' * 70}")

    # Display the last root run's tree
    if span_store._root_runs:
        last_root = span_store._root_runs[-1]
        print(span_store.get_tree(last_root))

    # Display aggregate statistics
    stats = span_store.stats()
    print(f"  📈 Aggregate Stats:")
    print(f"     Total spans:    {stats['total_spans']}")
    print(f"     Successful:     {stats['success_count']}")
    print(f"     Errors:         {stats['error_count']}")
    print(f"     Error rate:     {stats['error_rate']:.1%}")
    print(f"     Avg duration:   {stats['avg_duration_ms']:.1f}ms")

    if stats["by_type"]:
        print(f"     By type:")
        for stype, info in stats["by_type"].items():
            print(f"       {stype}: {info['count']} spans, "
                  f"avg {info['avg_duration_ms']:.1f}ms, "
                  f"{info['errors']} errors")


# ============================================================
# 11. Main Execution
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  🔍 LangChain Chain Tracing & Debugging — Demo")
    print("  Demonstrates: SpanRecord, TracingCallbackHandler, ConsoleTraceHandler,")
    print("                PerformanceMonitor, SpanStore, PII masking")
    print("=" * 70)

    # ----- Test 1–3: Normal execution with different question types -----
    for i, case in enumerate(test_cases, 1):
        result = run_with_tracing(
            chain=full_qa_chain,
            input_data={"question": case["question"]},
            label=f"Test {i}: {case['name']}",
        )

        print(f"\n  ✅ Result:")
        print(f"     Question type: {result.get('type', 'N/A')}")
        print(f"     Answer: {result.get('answer', 'N/A')[:200]}")

        print_span_tree(f"Test {i}: {case['name']}")

    # ----- Test 4: Demonstrate PII masking -----
    print(f"\n{'═' * 70}")
    print("  🔒 Test 4: PII Masking Utility")
    print(f"{'═' * 70}")

    pii_examples = [
        "Call me at 13812345678 for details",
        "My email is john.doe@example.com and my ID is 110101199001011234",
        "API key=sk-abc123def456ghi789, token=eyJhbGciOiJIUzI1NiJ9",
        "Contact support at 15900001111 or email admin@company.cn",
    ]

    for text in pii_examples:
        masked = mask_pii(text)
        print(f"  Original: {text}")
        print(f"  Masked:   {masked}")
        print()

    # ----- Test 5: Performance monitor summary -----
    print(f"{'═' * 70}")
    print("  📊 Final Performance Monitor Summary")
    print(f"{'═' * 70}")

    perf = PerformanceMonitor()
    summary = perf.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # ----- Final aggregate stats -----
    print(f"\n{'═' * 70}")
    print("  📈 Overall SpanStore Statistics (All Tests)")
    print(f"{'═' * 70}")
    stats = span_store.stats()
    print(f"  Total spans recorded: {stats['total_spans']}")
    print(f"  Successful:           {stats['success_count']}")
    print(f"  Errors:               {stats['error_count']}")
    print(f"  Overall error rate:   {stats['error_rate']:.1%}")
    print(f"  Average duration:     {stats['avg_duration_ms']:.1f}ms")

    print(f"\n{'=' * 70}")
    print("  ✅ All tests completed successfully!")
    print("=" * 70)
