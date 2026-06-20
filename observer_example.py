"""
LangSmith Observer — Integration Example
=========================================
Demonstrates how to add full observability to existing LangChain workflows
using the FlowObserver wrapper around the LangSmith SDK.

This file shows three integration patterns:

  Pattern 1 — Global tracing (zero code changes to existing chains)
  Pattern 2 — Scoped tracing with observe() context manager
  Pattern 3 — Custom function tracing with @traced decorator

After running, visit https://smith.langchain.com to see the trace trees.

Prerequisites:
  - LANGSMITH_API_KEY environment variable (already configured)
  - OPENAI_API_KEY environment variable (for DashScope/Qwen models)
  - pip install langsmith langchain-openai langchain-core
"""

import os
import time

# Import the observer module we just built.
from flow_observer import FlowObserver

# Import existing LangChain components from the project.
# These are the same chains used in the original examples —
# no modifications needed for tracing to work.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


# ============================================================================
# 0. Initialize the FlowObserver
# ============================================================================
# The observer wraps the LangSmith SDK client and provides convenient
# helpers for tracing, metrics, and error analysis.
#
# project_name groups all traces under a single project in the LangSmith UI.
# Each project acts as a namespace — traces from different experiments or
# environments stay separate.

observer = FlowObserver(
    project_name="demo-project",
    endpoint="https://apac.api.smith.langchain.com",
)


# ============================================================================
# 1. Pattern: Global Tracing (Zero Code Changes)
# ============================================================================
# By calling enable_tracing(), we set environment variables that LangChain
# reads automatically. After this point, EVERY LangChain chain invocation
# in this process will create a trace in LangSmith — no changes to the
# chain code itself.
#
# This is the simplest integration path: add one line to your entry point.

observer.enable_tracing()


# ============================================================================
# 2. Build a Simple Chain (same pattern as sequential_chain_example.py)
# ============================================================================
# We build a lightweight 2-step chain for demonstration:
#   Step 1: topic → generate a title
#   Step 2: title → write a short summary
#
# Because tracing is enabled, both LLM calls will appear as child spans
# under a single parent trace in LangSmith.

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="qwen3-max",
    temperature=0.5,
    max_tokens=150,
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False},
)

parser = StrOutputParser()

# Step 1: Generate a title from a topic
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a catchy article title for this topic. Output only the title.\nTopic: {topic}",
)
title_chain = title_prompt | llm | parser

# Step 2: Write a short summary from the title
summary_prompt = PromptTemplate(
    input_variables=["title"],
    template="Write a 50-word summary for an article titled: {title}\nOutput only the summary.",
)
summary_chain = summary_prompt | llm | parser

# Assemble the sequential chain
simple_chain = (
    title_chain
    | RunnableLambda(lambda title: {"title": title})
    | summary_chain
)


# ============================================================================
# 3. Pattern: Scoped Tracing with observe()
# ============================================================================
# observe() creates a named parent trace that groups all spans within the
# with-block. This is ideal for:
#   - Labeling a specific experiment or test case
#   - Attaching metadata (e.g. environment, version, test name)
#   - Keeping traces organized when running multiple experiments

print("=" * 60)
print("🔍 Pattern 3: Scoped tracing with observe()")
print("=" * 60)

# Run the chain inside an observe() block.
# The metadata and tags will be visible in the LangSmith UI, making it
# easy to filter and compare runs later.
with observer.observe(
    run_name="simple-chain-demo",
    metadata={"experiment": "scoped-tracing", "model": "qwen3-max"},
    tags=["demo", "sequential"],
):
    print("\n📝 Invoking chain with topic: 'Renewable Energy'")
    result = simple_chain.invoke({"topic": "Renewable Energy"})
    print(f"\n✅ Result:\n{result}")


# ============================================================================
# 4. Pattern: @traced Decorator for Custom Functions
# ============================================================================
# Real pipelines often include custom Python logic that isn't a LangChain
# runnable — data validation, API calls, database lookups, etc.
#
# The @traced decorator wraps these functions in a LangSmith span so they
# appear alongside LLM calls in the trace tree. This gives you visibility
# into the FULL data flow, not just the model calls.

# Use observer.traced() to decorate a custom preprocessing function.
# The span will show up as a "tool" type in LangSmith, with the input
# text and the cleaned output captured automatically.
@observer.traced(
    name="validate_input",
    run_type="tool",
    metadata={"stage": "preprocessing"},
    tags=["validation"],
)
def validate_input(text: str) -> str:
    """
    Validate and sanitize user input before passing to the LLM.

    This simulates a real-world preprocessing step — e.g. stripping
    HTML, checking length limits, or filtering profanity.
    """
    # Simulate some validation logic.
    cleaned = text.strip()
    if len(cleaned) > 500:
        cleaned = cleaned[:500]
    if not cleaned:
        raise ValueError("Input text cannot be empty")
    return cleaned


# Another traced function: post-processing the LLM output.
@observer.traced(
    name="format_output",
    run_type="tool",
    metadata={"stage": "postprocessing"},
)
def format_output(raw_text: str) -> dict:
    """
    Format the raw LLM output into a structured response.

    In production this might add metadata, compute word counts,
    or transform the text for an API response.
    """
    words = raw_text.split()
    return {
        "text": raw_text,
        "word_count": len(words),
        "char_count": len(raw_text),
    }


# Build a chain that includes the custom traced functions.
# RunnableLambda wraps our plain functions as LangChain runnables,
# but the @traced decorator ensures they get their own spans.
full_pipeline = (
    RunnableLambda(validate_input)
    | RunnableLambda(lambda text: {"topic": text})
    | title_chain
    | RunnableLambda(lambda title: {"title": title})
    | summary_chain
    | RunnableLambda(format_output)
)

print(f"\n{'=' * 60}")
print("🔍 Pattern 4: Full pipeline with @traced custom functions")
print("=" * 60)

with observer.observe(
    run_name="full-pipeline-demo",
    metadata={"experiment": "traced-decorator", "version": "v1"},
    tags=["demo", "full-pipeline"],
):
    print("\n📝 Invoking full pipeline...")
    pipeline_result = full_pipeline.invoke("  Machine Learning in Healthcare  ")
    print(f"\n✅ Pipeline result:")
    print(f"   Text:        {pipeline_result['text'][:100]}...")
    print(f"   Word count:  {pipeline_result['word_count']}")
    print(f"   Char count:  {pipeline_result['char_count']}")


# ============================================================================
# 5. Deliberate Error — Demonstrate Error Capture
# ============================================================================
# To show how error logging works, we invoke the chain with input that
# triggers a validation error. The error will be:
#   - Captured automatically by LangSmith (stack trace + span context)
#   - Classifiable via observer.categorize_error()
#   - Queryable via observer.get_recent_errors()

print(f"\n{'=' * 60}")
print("🔍 Pattern 5: Deliberate error for error capture demo")
print("=" * 60)

try:
    with observer.observe(
        run_name="error-demo",
        metadata={"experiment": "error-capture"},
        tags=["demo", "error-test"],
    ):
        # This will raise ValueError("Input text cannot be empty")
        # because validate_input rejects empty strings.
        print("\n📝 Invoking pipeline with empty input (should fail)...")
        full_pipeline.invoke("   ")
except ValueError as e:
    print(f"\n❌ Caught expected error: {e}")
    print("   (This error has been recorded in LangSmith with full context)")


# ============================================================================
# 6. Query Metrics & Errors
# ============================================================================
# After running some traces, we can query LangSmith for metrics and errors.
#
# Note: There is a short delay between when a run completes and when it
# becomes queryable via the API. We sleep briefly to ensure the runs
# we just created are indexed.

print(f"\n{'=' * 60}")
print("📊 Querying metrics and errors...")
print("=" * 60)

# Wait a few seconds for LangSmith to index the runs.
print("\n⏳ Waiting for traces to be indexed...")
time.sleep(5)

# --- 6.1 Aggregate Metrics ---
# Fetch aggregate stats for all runs in the last hour.
print("\n--- Aggregate Metrics (last 1 hour) ---")
agg = observer.get_aggregate_metrics(hours=1)
observer.print_aggregate_summary(agg)

# --- 6.2 Recent Errors ---
# Show any errors that occurred in the last hour.
# Our deliberate error above should appear here.
print("\n--- Recent Errors (last 1 hour) ---")
observer.print_recent_errors(hours=1, limit=5)

# --- 6.3 Error Category Summary ---
# Show the breakdown of error types.
print("\n--- Error Category Summary ---")
summary = observer.get_error_summary(hours=1)
if summary:
    for category, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {category}: {count}")
else:
    print("  No errors found (they may not be indexed yet)")


# ============================================================================
# 7. Disable Tracing (Cleanup)
# ============================================================================
# Optionally disable tracing when done. In a production app you'd typically
# leave tracing enabled, but this is useful for:
#   - Jupyter notebooks where you toggle between experiments
#   - Test suites where only specific tests should be traced
#   - Cost control (each traced run sends data to LangSmith)

observer.disable_tracing()

print(f"\n{'=' * 60}")
print("✅ Observer demo complete!")
print(f"🔗 View traces at: https://smith.langchain.com")
print("=" * 60)
