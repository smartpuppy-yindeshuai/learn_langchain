"""
LangSmith Flow Observer — Task Flow Observability & Reliability
===============================================================
Provides end-to-end observability for LangChain-powered task flows via
the LangSmith SDK. Three core capabilities:

  1. Data Flow Tracing   — capture every step (prompts, model calls, tool
                           invocations, outputs) as a nested trace tree.
  2. Model Performance    — record per-step latency, token usage, cost,
                           and throughput; aggregate across runs.
  3. Execution Error Log  — auto-capture exceptions, stack traces, and
                           categorize errors for trend analysis.

Usage:
    from flow_observer import FlowObserver

    observer = FlowObserver(project_name="my-project")
    observer.enable_tracing()          # turn on auto-tracing for LCEL chains

    with observer.observe("my-run"):   # context-manager for ad-hoc spans
        result = my_chain.invoke(...)

    observer.print_recent_errors()     # query & display recent failures
"""

import os
import functools
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Optional, Sequence

from langsmith import Client, traceable, tracing_context
from langsmith.schemas import Run


# ============================================================================
# Error Category Taxonomy
# ============================================================================
# Maps substring patterns found in error messages to human-readable category
# labels. Used by FlowObserver.categorize_error() to bucket failures for
# trend analysis. Add or override entries via FlowObserver.register_error_pattern().

_DEFAULT_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("timeout",           "timeout"),           # network / API timeouts
    ("timed out",         "timeout"),
    ("rate limit",        "rate_limit"),        # HTTP 429 or quota exceeded
    ("rate_limit",        "rate_limit"),
    ("too many requests", "rate_limit"),
    ("json",              "parse_error"),       # JSON decoding failures
    ("parse",             "parse_error"),
    ("decode",            "parse_error"),
    ("validation",        "validation_error"),  # Pydantic / schema errors
    ("valueerror",        "validation_error"),  # Python ValueError
    ("typeerror",         "validation_error"),  # Python TypeError
    ("invalid",           "validation_error"),
    ("auth",              "auth_error"),         # authentication / authorization
    ("api key",           "auth_error"),
    ("forbidden",         "auth_error"),
    ("not found",         "not_found"),         # 404 / missing resources
    ("connection",        "connection_error"),  # network-level failures
    ("refused",           "connection_error"),
    ("unavailable",       "connection_error"),
]


# ============================================================================
# Data Models
# ============================================================================
# Lightweight value objects returned by the observer's query helpers.

@dataclass
class ErrorRecord:
    """
    Represents a single execution error captured from a LangSmith run.

    Attributes:
        run_id:      Unique identifier of the failed run.
        trace_id:    Trace (root run) this error belongs to.
        name:        Human-readable name of the span that failed.
        run_type:    Type of run (e.g. "chain", "llm", "tool", "retriever").
        error_msg:   The raw error message / stack trace.
        category:    Classified error category (e.g. "timeout", "rate_limit").
        start_time:  When the run started (UTC).
        end_time:    When the run ended (UTC).
        tags:        Any tags attached to the run.
        metadata:    Any metadata attached to the run.
        app_url:     Direct link to the run in the LangSmith UI.
    """
    run_id: str
    trace_id: str
    name: str
    run_type: str
    error_msg: str
    category: str
    start_time: datetime
    end_time: Optional[datetime]
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    app_url: Optional[str] = None


@dataclass
class RunMetrics:
    """
    Performance metrics for a single run.

    Attributes:
        run_id:            Unique identifier of the run.
        name:              Name of the span.
        run_type:          Type of run (chain / llm / tool / retriever).
        latency:           Wall-clock duration in seconds.
        prompt_tokens:     Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        total_tokens:      Total token consumption.
        total_cost:        Estimated cost in USD (if available).
        status:            "success" or "error".
        start_time:        When the run started (UTC).
    """
    run_id: str
    name: str
    run_type: str
    latency: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    total_cost: Optional[str]
    status: str
    start_time: datetime


@dataclass
class AggregateMetrics:
    """
    Aggregated performance statistics across multiple runs.

    Attributes:
        total_runs:     Number of runs included.
        success_count:  Runs that completed without error.
        error_count:    Runs that ended with an error.
        avg_latency:    Mean wall-clock latency in seconds.
        p50_latency:    Median latency.
        p95_latency:    95th-percentile latency.
        total_tokens:   Sum of all tokens consumed.
        total_cost:     Sum of estimated costs (if available).
        error_categories: Breakdown of errors by category with counts.
    """
    total_runs: int
    success_count: int
    error_count: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    total_tokens: int
    total_cost: str
    error_categories: dict[str, int]


# ============================================================================
# FlowObserver — Main Observability Interface
# ============================================================================

class FlowObserver:
    """
    High-level observability wrapper around the LangSmith SDK.

    Provides:
      - One-call tracing enablement for all LangChain LCEL chains.
      - @traced decorator and observe() context manager for custom code.
      - Helpers to query run metrics, errors, and aggregate statistics.
      - Error categorization for trend analysis.

    Args:
        project_name: LangSmith project to send traces to. If the project
                      does not exist, it will be created automatically.
        api_key:      LangSmith API key. Defaults to LANGSMITH_API_KEY env var.
        endpoint:     LangSmith API endpoint. Defaults to the public cloud.
    """

    def __init__(
        self,
        project_name: str = "default",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        self.project_name = project_name

        # Resolve API key from argument or environment variable.
        # LANGSMITH_API_KEY and LANGCHAIN_API_KEY are both recognized
        # by the LangSmith SDK; we check both for flexibility.
        self.api_key = api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")

        # Check LANGSMITH_ENDPOINT first (the newer convention), then
        # fall back to LANGCHAIN_ENDPOINT, then the public US cloud.
        # For APAC region users, set LANGSMITH_ENDPOINT to
        # https://apac.api.smith.langchain.com before initializing.
        self.endpoint = (
            endpoint
            or os.getenv("LANGSMITH_ENDPOINT")
            or os.getenv("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com"
        )

        # Initialize the LangSmith API client for programmatic queries.
        # This client is used for reading runs, listing errors, etc.
        # It is NOT used for writing traces — that is handled by the
        # environment variables set in enable_tracing().
        self._client = Client(
            api_key=self.api_key,
            api_url=self.endpoint,
        )

        # Error classification patterns (mutable, user-extensible).
        self._error_patterns: list[tuple[str, str]] = list(_DEFAULT_ERROR_PATTERNS)

        # Track whether tracing has been enabled to give clear errors.
        self._tracing_enabled = False

    # ------------------------------------------------------------------
    # Tracing Configuration
    # ------------------------------------------------------------------

    def enable_tracing(self) -> None:
        """
        Enable automatic LangSmith tracing for all LangChain LCEL chains.

        This sets the environment variables that LangChain's callback
        system reads to auto-instrument every .invoke() / .stream() call:

          LANGCHAIN_TRACING_V2=true   → activates the LangSmith tracer
          LANGCHAIN_PROJECT=<name>    → project name for grouping
          LANGCHAIN_API_KEY=<key>     → authentication

        After calling this method, any LangChain chain you invoke will
        automatically create a trace tree in LangSmith.

        Note: This modifies os.environ for the current process. To scope
        tracing to a specific block of code, use observe() instead.
        """
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = self.project_name

        if self.api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.api_key

        # Set the endpoint so the LangChain SDK sends traces to the
        # correct region (e.g. APAC: https://apac.api.smith.langchain.com).
        os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
        os.environ["LANGSMITH_ENDPOINT"] = self.endpoint

        self._tracing_enabled = True
        print(f"✅ LangSmith tracing enabled → project: \"{self.project_name}\", endpoint: {self.endpoint}")

    def disable_tracing(self) -> None:
        """
        Disable automatic tracing by removing the environment variables.

        Useful in notebooks or long-running processes where you want to
        toggle tracing on and off between experiments.
        """
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_PROJECT", None)
        os.environ.pop("LANGCHAIN_ENDPOINT", None)
        os.environ.pop("LANGSMITH_ENDPOINT", None)
        self._tracing_enabled = False
        print("⏹️  LangSmith tracing disabled")

    # ------------------------------------------------------------------
    # Decorator: @traced
    # ------------------------------------------------------------------

    def traced(
        self,
        name: Optional[str] = None,
        run_type: str = "chain",
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Callable:
        """
        Decorator that wraps a function in a LangSmith trace span.

        Use this to trace custom Python functions that are NOT LangChain
        runnables — e.g. data preprocessing, post-processing, database
        lookups, or any auxiliary logic in your pipeline.

        Args:
            name:     Display name for the span. Defaults to the function name.
            run_type: LangSmith run type ("chain", "tool", "retriever", etc.).
            metadata: Key-value pairs attached to the span for filtering.
            tags:     String tags for grouping and filtering.

        Returns:
            A decorator that wraps the target function.

        Example:
            @observer.traced(name="preprocess", metadata={"stage": "input"})
            def clean_input(text: str) -> str:
                return text.strip().lower()
        """
        def decorator(func: Callable) -> Callable:
            # LangSmith's @traceable decorator handles span creation,
            # input/output capture, and error recording automatically.
            return traceable(
                name=name or func.__name__,
                run_type=run_type,
                metadata=metadata or {},
                tags=tags or [],
            )(func)

        return decorator

    # ------------------------------------------------------------------
    # Context Manager: observe()
    # ------------------------------------------------------------------

    @contextmanager
    def observe(
        self,
        run_name: str = "observation",
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Generator[None, None, None]:
        """
        Context manager that creates a tracing scope for a block of code.

        All LangChain calls and @traced functions invoked within this block
        will be grouped under a single parent trace in LangSmith. This is
        ideal for tracing an entire workflow execution or a test scenario.

        Unlike enable_tracing() which sets global env vars, observe() uses
        LangSmith's tracing_context() to scope tracing to the with-block.

        Args:
            run_name: Name shown in the LangSmith UI for this trace.
            metadata: Key-value pairs to attach (e.g. {"test_case": "smoke"}).
            tags:     Tags for filtering (e.g. ["production", "v2"]).

        Example:
            with observer.observe("feedback-pipeline", metadata={"env": "staging"}):
                result = full_chain.invoke({"feedback": "..."})
                post_process(result)
        """
        # tracing_context() is LangSmith's scoped tracing mechanism.
        # It creates a parent run that groups all child spans created
        # within the with-block, then finalizes the trace on exit.
        with tracing_context(
            project_name=self.project_name,
            metadata=metadata or {},
            tags=tags or [],
            client=self._client,
        ):
            yield

    # ------------------------------------------------------------------
    # Metadata Annotation
    # ------------------------------------------------------------------

    def annotate_run(
        self,
        run_id: str,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Add metadata or tags to an existing run after it has completed.

        Useful for retroactively labeling runs — e.g. after you've
        determined the feedback type or detected an anomaly.

        Args:
            run_id:   ID of the run to annotate.
            metadata: Key-value pairs to merge into existing metadata.
            tags:     Tags to append to existing tags.

        Note: The LangSmith API's update endpoint is used internally.
        """
        update_payload: dict[str, Any] = {}
        if metadata:
            update_payload["extra"] = {"metadata": metadata}
        if tags:
            update_payload["tags"] = tags

        self._client.update_run(run_id, **update_payload)
        print(f"🏷️  Annotated run {run_id} with metadata={metadata}, tags={tags}")

    # ------------------------------------------------------------------
    # Metrics Query: Single Run
    # ------------------------------------------------------------------

    def get_run_metrics(self, run_id: str) -> RunMetrics:
        """
        Retrieve performance metrics for a specific run.

        Fetches the run from LangSmith and extracts latency, token usage,
        cost, and status into a clean RunMetrics dataclass.

        Args:
            run_id: The UUID of the run to inspect.

        Returns:
            A RunMetrics object with the run's performance data.

        Raises:
            LangSmithError: If the API key lacks permission to read runs,
                            or the run_id does not exist.
        """
        try:
            run = self._client.read_run(run_id)
            return self._run_to_metrics(run)
        except Exception as e:
            print(f"⚠️  Failed to fetch run {run_id}: {e}")
            raise

    def get_trace_metrics(self, trace_id: str) -> list[RunMetrics]:
        """
        Retrieve metrics for every span in a trace tree.

        A trace is a tree of runs (parent → children). This fetches all
        runs under the given trace and returns their metrics as a flat list,
        ordered by start time.

        Args:
            trace_id: The trace (root run) UUID.

        Returns:
            List of RunMetrics, one per span in the trace.
            Returns an empty list if the API call fails.
        """
        try:
            runs = self._client.list_runs(trace_id=trace_id)
            metrics = [self._run_to_metrics(r) for r in runs]
            # Sort chronologically so the output mirrors execution order.
            metrics.sort(key=lambda m: m.start_time)
            return metrics
        except Exception as e:
            print(f"⚠️  Failed to fetch trace {trace_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Metrics Query: Aggregated
    # ------------------------------------------------------------------

    def get_aggregate_metrics(
        self,
        hours: int = 24,
        run_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> AggregateMetrics:
        """
        Compute aggregate performance statistics over recent runs.

        Fetches all runs in the project from the last N hours and computes:
          - success / error counts
          - mean, p50, p95 latency
          - total token consumption and cost
          - error category breakdown

        Args:
            hours:    Look-back window in hours (default: 24).
            run_type: Filter by run type ("chain", "llm", "tool"). None = all.
            tags:     Filter by tags. None = all.

        Returns:
            An AggregateMetrics summary.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Build the filter expression for tag matching.
        # LangSmith uses a simple query syntax: has(tags, "value")
        tag_filter = None
        if tags:
            clauses = [f'has(tags, "{t}")' for t in tags]
            tag_filter = " && ".join(clauses)

        try:
            runs = list(self._client.list_runs(
                project_name=self.project_name,
                run_type=run_type,
                start_time=since,
                filter=tag_filter,
                is_root=True,  # only top-level runs to avoid double-counting
            ))
        except Exception as e:
            print(f"⚠️  Failed to query runs from LangSmith: {e}")
            print("   This may be due to API key permissions or network issues.")
            print("   Returning empty metrics. Tracing still works — check the LangSmith UI.")
            return AggregateMetrics(
                total_runs=0, success_count=0, error_count=0,
                avg_latency=0.0, p50_latency=0.0, p95_latency=0.0,
                total_tokens=0, total_cost="0", error_categories={},
            )

        return self._compute_aggregates(runs)
    # ------------------------------------------------------------------
    # Error Query & Categorization
    # ------------------------------------------------------------------

    def get_recent_errors(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> list[ErrorRecord]:
        """
        Fetch recent failed runs and classify them by error category.

        Queries LangSmith for runs with error=True, then runs each error
        message through the pattern matcher to assign a category label.

        Args:
            hours: Look-back window in hours.
            limit: Maximum number of errors to return.

        Returns:
            List of ErrorRecord objects, sorted newest-first.
        """
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        # error=True filters to only failed runs in the LangSmith API.
        try:
            error_runs = list(self._client.list_runs(
                project_name=self.project_name,
                error=True,
                start_time=since,
                limit=limit,
            ))
        except Exception as e:
            print(f"⚠️  Failed to query errors from LangSmith: {e}")
            print("   This may be due to API key permissions. Check the LangSmith UI for errors.")
            return []

        records = []
        for run in error_runs:
            category = self.categorize_error(run.error or "")
            records.append(ErrorRecord(
                run_id=str(run.id),
                trace_id=str(run.trace_id),
                name=run.name,
                run_type=run.run_type,
                error_msg=(run.error or "")[:500],  # truncate long traces
                category=category,
                start_time=run.start_time,
                end_time=run.end_time,
                tags=run.tags or [],
                metadata=run.metadata or {},
                app_url=run.app_path,
            ))

        return records

    def categorize_error(self, error_msg: str) -> str:
        """
        Classify an error message into a category using pattern matching.

        Iterates through registered patterns and returns the first match.
        Falls back to "unknown" if no pattern matches.

        Args:
            error_msg: The raw error message or stack trace.

        Returns:
            Category string (e.g. "timeout", "rate_limit", "parse_error").
        """
        lower_msg = error_msg.lower()
        for pattern, category in self._error_patterns:
            if pattern in lower_msg:
                return category
        return "unknown"

    def register_error_pattern(self, pattern: str, category: str) -> None:
        """
        Add a custom error classification rule.

        Allows you to extend the default taxonomy with application-specific
        error types — e.g. "vector_store" for RAG retrieval failures.

        Args:
            pattern:  Substring to search for (case-insensitive).
            category: Category label to assign when the pattern matches.

        Example:
            observer.register_error_pattern("embedding", "embedding_error")
        """
        self._error_patterns.append((pattern.lower(), category))

    def get_error_summary(self, hours: int = 24) -> dict[str, int]:
        """
        Get a breakdown of error categories with counts.

        A convenience wrapper around get_recent_errors() that groups
        errors by category and returns a frequency map.

        Args:
            hours: Look-back window.

        Returns:
            Dict mapping category → count, e.g. {"timeout": 3, "parse_error": 1}.
        """
        errors = self.get_recent_errors(hours=hours)
        summary: dict[str, int] = {}
        for err in errors:
            summary[err.category] = summary.get(err.category, 0) + 1
        return summary

    # ------------------------------------------------------------------
    # Display Helpers
    # ------------------------------------------------------------------

    def print_recent_errors(self, hours: int = 24, limit: int = 10) -> None:
        """
        Pretty-print recent errors to stdout for quick inspection.

        Useful during development and debugging to see what's failing
        without switching to the LangSmith web UI.
        """
        errors = self.get_recent_errors(hours=hours, limit=limit)

        if not errors:
            print(f"✅ No errors in the last {hours}h")
            return

        print(f"\n{'=' * 70}")
        print(f"❌ Recent Errors (last {hours}h, showing {len(errors)})")
        print(f"{'=' * 70}")

        for i, err in enumerate(errors, 1):
            print(f"\n{'─' * 70}")
            print(f"  {i}. [{err.category}] {err.name}")
            print(f"     Run ID:   {err.run_id}")
            print(f"     Type:     {err.run_type}")
            print(f"     Time:     {err.start_time.isoformat()}")
            if err.tags:
                print(f"     Tags:     {', '.join(err.tags)}")
            # Show first 200 chars of the error message
            preview = err.error_msg[:200].replace("\n", " ")
            print(f"     Error:    {preview}...")

        print(f"\n{'=' * 70}")

    def print_metrics_table(self, metrics: list[RunMetrics]) -> None:
        """
        Pretty-print a list of RunMetrics as a formatted table.
        """
        if not metrics:
            print("No runs found.")
            return

        print(f"\n{'=' * 90}")
        print(f"  {'Name':<30} {'Type':<10} {'Latency':>10} {'Tokens':>10} {'Status':<10}")
        print(f"{'─' * 90}")

        for m in metrics:
            latency_str = f"{m.latency:.2f}s" if m.latency is not None else "n/a"
            tokens_str = str(m.total_tokens) if m.total_tokens else "n/a"
            print(f"  {m.name:<30} {m.run_type:<10} {latency_str:>10} {tokens_str:>10} {m.status:<10}")

        print(f"{'=' * 90}")

    def print_aggregate_summary(self, agg: AggregateMetrics) -> None:
        """
        Pretty-print an AggregateMetrics summary.
        """
        print(f"\n{'=' * 60}")
        print(f"  📊 Aggregate Metrics")
        print(f"{'=' * 60}")
        print(f"  Total runs:     {agg.total_runs}")
        print(f"  Successes:      {agg.success_count}")
        print(f"  Errors:         {agg.error_count}")
        print(f"  Avg latency:    {agg.avg_latency:.2f}s")
        print(f"  P50 latency:    {agg.p50_latency:.2f}s")
        print(f"  P95 latency:    {agg.p95_latency:.2f}s")
        print(f"  Total tokens:   {agg.total_tokens}")
        print(f"  Total cost:     {agg.total_cost}")

        if agg.error_categories:
            print(f"\n  Error Breakdown:")
            for cat, count in sorted(agg.error_categories.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

        print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_to_metrics(run: Run) -> RunMetrics:
        """Convert a LangSmith Run object into a RunMetrics dataclass."""
        # Compute latency from timestamps if the SDK didn't provide it.
        latency = None
        if run.latency is not None:
            latency = run.latency
        elif run.end_time and run.start_time:
            latency = (run.end_time - run.start_time).total_seconds()

        return RunMetrics(
            run_id=str(run.id),
            name=run.name,
            run_type=run.run_type,
            latency=latency,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            total_tokens=run.total_tokens,
            total_cost=str(run.total_cost) if run.total_cost else None,
            status="error" if run.error else "success",
            start_time=run.start_time,
        )

    def _compute_aggregates(self, runs: list[Run]) -> AggregateMetrics:
        """Compute aggregate statistics from a list of Run objects."""
        if not runs:
            return AggregateMetrics(
                total_runs=0, success_count=0, error_count=0,
                avg_latency=0.0, p50_latency=0.0, p95_latency=0.0,
                total_tokens=0, total_cost="0", error_categories={},
            )

        # Separate successes and errors.
        successes = [r for r in runs if not r.error]
        errors = [r for r in runs if r.error]

        # Collect latencies for percentile computation.
        latencies = []
        for r in runs:
            if r.latency is not None:
                latencies.append(r.latency)
            elif r.end_time and r.start_time:
                latencies.append((r.end_time - r.start_time).total_seconds())

        latencies.sort()

        def percentile(data: list[float], p: int) -> float:
            """Compute the p-th percentile of a sorted list."""
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[f]
            return data[f] + (k - f) * (data[c] - data[f])

        # Sum token usage across all runs.
        total_tokens = sum(r.total_tokens or 0 for r in runs)

        # Sum cost if available (Decimal → float for aggregation).
        total_cost = sum(float(r.total_cost or 0) for r in runs)

        # Build error category breakdown.
        error_cats: dict[str, int] = {}
        for r in errors:
            cat = self.categorize_error(r.error or "")
            error_cats[cat] = error_cats.get(cat, 0) + 1

        return AggregateMetrics(
            total_runs=len(runs),
            success_count=len(successes),
            error_count=len(errors),
            avg_latency=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency=percentile(latencies, 50),
            p95_latency=percentile(latencies, 95),
            total_tokens=total_tokens,
            total_cost=f"{total_cost:.4f}",
            error_categories=error_cats,
        )
