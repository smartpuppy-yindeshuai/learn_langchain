"""
Map-Rerank 链详解 (Map-Rerank Chain Explained)
==============================================

核心思想：
  1. Map    — 每个文档片段独立处理，生成答案 + 评分（并行，互不影响）
  2. Rerank — 按评分降序排列，筛选出 Top-K 最优结果
  3. Reduce — 基于最优结果生成最终答案

与 Map-Reduce 的关键区别：
  Map-Reduce: 合并所有片段的结果 → 可能引入噪音
  Map-Rerank: 评分排序后只选最优 → 精准过滤，质量更高

本文件数据流：
  问题 + 5个文档片段
    → Map:     每个片段独立回答 + 打分 (qwen3-max 轻量快速)
    → Rerank:  按分数排序，选 Top 3
    → Reduce:  合并 Top 3 生成最终答案 (qwen3.7-max 高质量)
"""

import os
import re
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ============================================================
# 0. 通用配置
# ============================================================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请配置环境变量 OPENAI_API_KEY")

common_kwargs = {
    "api_key": openai_api_key,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "extra_body": {"enable_thinking": False},
}

parser = StrOutputParser()


# ============================================================
# 1. 初始化两个模型（Map 用轻量，Reduce 用强力）
# ============================================================

# Map 阶段用 qwen3-max：只需要生成短答案 + 打分，不需要深度推理
# 低温度确保评分稳定一致
llm_map = ChatOpenAI(
    model_name="qwen3-max",
    temperature=0.1,    # 低温度 → 评分更确定、更一致
    max_tokens=150,
    **common_kwargs,
)

# Reduce 阶段用 qwen3.7-max：需要综合多个答案生成高质量最终回复
llm_reduce = ChatOpenAI(
    model_name="qwen3.7-max",
    temperature=0.5,
    max_tokens=400,
    **common_kwargs,
)


# ============================================================
# 2. 定义数据类型（让代码结构更清晰）
# ============================================================

class MapResult(TypedDict):
    """单个文档片段的 Map 结果"""
    chunk_index: int       # 片段编号
    answer: str            # LLM 生成的答案
    score: int             # 相关性评分 (0-100)
    source_text: str       # 原始片段文本（用于调试展示）


# ============================================================
# 3. Phase 1 — Map（映射阶段）
# ============================================================
# 核心：每个文档片段 + 原始问题 → 独立调用 LLM
# 要求 LLM 同时输出答案和评分，格式固定便于解析

MAP_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一个文档问答助手。请根据以下文档片段回答问题，并对你的答案与该文档片段的相关性打分。

【重要】只使用给定文档片段中的信息来回答。如果片段中没有相关信息，请如实说明。

## 文档片段
{context}

## 问题
{question}

## 请严格按以下格式回答（不要添加任何额外内容）：
答案：<基于文档片段的回答>
评分：<0到100的整数，表示文档片段与问题的相关程度>"""
)

# Map 子链：prompt → LLM → 纯文本
map_chain = MAP_PROMPT | llm_map | parser


def parse_map_result(raw_text: str, chunk_index: int, source_text: str) -> MapResult:
    """
    从 LLM 的文本输出中提取 answer 和 score。

    LLM 输出格式约定：
        答案：<回答内容>
        评分：<数字>

    使用正则表达式匹配，即使 LLM 输出略有偏差也能容错。
    """
    # 提取答案：匹配 "答案：" 后面到 "评分：" 之前的内容
    answer_match = re.search(r"答案[：:]\s*(.*?)(?=\n*评分[：:]|\Z)", raw_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else raw_text.strip()

    # 提取评分：匹配 "评分：" 后面的数字
    score_match = re.search(r"评分[：:]\s*(\d+)", raw_text)
    if score_match:
        score = int(score_match.group(1))
        # 限制在 0-100 范围内
        score = max(0, min(100, score))
    else:
        # 如果解析失败，给一个默认中等分数
        score = 50

    return MapResult(
        chunk_index=chunk_index,
        answer=answer,
        score=score,
        source_text=source_text,
    )


def map_phase(data: dict) -> dict:
    """
    Map 阶段：对每个文档片段独立调用 map_chain。

    输入: {"question": "...", "documents": ["片段1", "片段2", ...]}
    输出: {"question": "...", "documents": [...], "map_results": [MapResult, ...]}
    """
    question = data["question"]
    documents = data["documents"]

    print(f"\n{'─' * 50}")
    print(f"📍 Phase 1: Map — 处理 {len(documents)} 个文档片段")
    print(f"{'─' * 50}")

    results = []
    for i, doc in enumerate(documents):
        print(f"\n  🔍 处理片段 {i + 1}/{len(documents)}...")

        # 独立调用 LLM 处理当前片段
        raw_output = map_chain.invoke({
            "context": doc,
            "question": question,
        })

        # 解析输出，提取答案和评分
        result = parse_map_result(raw_output, chunk_index=i + 1, source_text=doc)
        results.append(result)

        print(f"     评分: {result['score']:>3}/100  |  答案: {result['answer'][:60]}...")

    return {
        "question": question,
        "documents": documents,
        "map_results": results,
    }


# ============================================================
# 4. Phase 2 — Rerank（重排序阶段）
# ============================================================
# 核心：按评分降序排列，选取 Top-K 结果
# 这一步不调用 LLM，纯逻辑操作

def rerank_phase(data: dict) -> dict:
    """
    Rerank 阶段：按评分排序，筛选 Top-K。

    输入: {"question": "...", "map_results": [...]}
    输出: {"question": "...", "ranked_results": [...], "top_k": int}

    Top-K 可通过 data["top_k"] 指定，默认为 3。
    """
    results = data["map_results"]
    top_k = data.get("top_k", 3)

    # 按 score 降序排列（最高分在前）
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # 选取前 K 个
    top_results = sorted_results[:top_k]

    print(f"\n{'─' * 50}")
    print(f"📊 Phase 2: Rerank — 按评分排序，选取 Top {top_k}")
    print(f"{'─' * 50}")
    print(f"\n  {'排名':<4} {'评分':<8} {'片段':<6} {'答案摘要'}")
    print(f"  {'─' * 4} {'─' * 6} {'─' * 6} {'─' * 40}")

    for rank, r in enumerate(sorted_results, 1):
        # 标记是否入选 Top-K
        marker = "✅" if rank <= top_k else "❌"
        preview = r["answer"][:40].replace("\n", " ")
        print(f"  {marker} #{rank:<3} {r['score']:>3}分   片段{r['chunk_index']}  {preview}")

    return {
        "question": data["question"],
        "ranked_results": top_results,
        "all_results": sorted_results,  # 保留完整排名用于展示
        "top_k": top_k,
        "best_answer": top_results[0]["answer"] if top_results else "",
        "best_score": top_results[0]["score"] if top_results else 0,
    }


# ============================================================
# 5. Phase 3 — Reduce（归约阶段）
# ============================================================
# 两种方案：
#   方案 A: 直接取最高分答案（简单快速）
#   方案 B: 合并 Top-K 答案，再调 LLM 生成综合回复（更高质量）

# ---------- 方案 A: 直取最高分 ----------

def reduce_best_only(data: dict) -> str:
    """方案 A：直接返回评分最高的答案，不额外调用 LLM。"""
    best = data["best_answer"]
    score = data["best_score"]

    print(f"\n{'─' * 50}")
    print(f"🏆 Phase 3: Reduce (方案A) — 直取最高分答案 ({score}分)")
    print(f"{'─' * 50}")

    return best


# ---------- 方案 B: 合并 Top-K + LLM 精炼 ----------

REDUCE_PROMPT = PromptTemplate(
    input_variables=["question", "ranked_answers"],
    template="""你是一个高级问答助手。以下是针对同一个问题的多个候选答案，已按相关性从高到低排序。

请综合这些答案的优点，生成一个最终的、最完整且准确的回答。
- 优先使用高相关性答案中的信息
- 补充其他答案中有价值的细节
- 去除矛盾或不准确的内容
- 用清晰、有条理的方式组织回答

## 问题
{question}

## 候选答案（按相关性排序）
{ranked_answers}

## 请给出最终综合回答："""
)

reduce_chain = REDUCE_PROMPT | llm_reduce | parser


def reduce_merge_topk(data: dict) -> str:
    """方案 B：将 Top-K 答案拼接后，调 LLM 生成综合回复。"""
    question = data["question"]
    ranked = data["ranked_results"]

    # 将 Top-K 答案格式化为文本列表
    ranked_answers_text = "\n\n".join(
        f"### 候选答案 {i + 1}（评分: {r['score']}分，来自片段 {r['chunk_index']}）\n{r['answer']}"
        for i, r in enumerate(ranked)
    )

    scores_preview = " > ".join(str(r["score"]) for r in ranked)

    print(f"\n{'─' * 50}")
    print(f"🏆 Phase 3: Reduce (方案B) — 合并 Top {len(ranked)} 答案 [{scores_preview}]")
    print(f"{'─' * 50}")

    # 调用 Reduce LLM 生成综合答案
    final_answer = reduce_chain.invoke({
        "question": question,
        "ranked_answers": ranked_answers_text,
    })

    return final_answer


# ============================================================
# 6. 组装完整链
# ============================================================

# 方案 A 完整链：Map → Rerank → 直取最高分
full_chain_best = (
    RunnableLambda(map_phase)           # Phase 1: 每个片段独立回答 + 打分
    | RunnableLambda(rerank_phase)      # Phase 2: 排序 + 选 Top-K
    | RunnableLambda(reduce_best_only)  # Phase 3: 直接返回最高分答案
)

# 方案 B 完整链：Map → Rerank → 合并 Top-K 精炼
full_chain_merge = (
    RunnableLambda(map_phase)           # Phase 1: 每个片段独立回答 + 打分
    | RunnableLambda(rerank_phase)      # Phase 2: 排序 + 选 Top-K
    | RunnableLambda(reduce_merge_topk) # Phase 3: 合并 Top-K，LLM 生成综合答案
)


# ============================================================
# 7. 测试数据 — 5 个相关性不同的文档片段
# ============================================================
# 设计意图：
#   片段1: 直接回答过拟合 → 预期高分 (~90+)
#   片段2: 讨论欠拟合（相关概念但不是目标）→ 预期低分 (~30)
#   片段3: 防止过拟合的方法 → 预期中高分 (~80)
#   片段4: 机器学习历史（完全无关）→ 预期最低分 (~10)
#   片段5: 过拟合的实际案例 → 预期中等分 (~70)

QUESTION = "什么是机器学习中的过拟合？如何避免？"

DOCUMENTS = [
    # 片段1: 高相关 — 直接定义过拟合
    (
        "过拟合（Overfitting）是指机器学习模型在训练数据上表现极好，"
        "但在未见过的测试数据上表现很差的现象。这通常是因为模型过度学习"
        "了训练数据中的噪声和细节，而不是学到了真正的规律。过拟合的模型"
        "就像死记硬背的学生，考试遇到新题就不会做了。"
        "常见的判断标准是训练集准确率高但验证集准确率明显下降。"
    ),

    # 片段2: 低相关 — 讨论欠拟合（另一个概念）
    (
        "欠拟合（Underfitting）是指模型过于简单，无法捕捉数据中的基本模式。"
        "欠拟合的模型在训练数据和测试数据上都表现不好。解决欠拟合的方法包括"
        "增加模型复杂度、添加更多特征、减少正则化强度等。"
        "欠拟合与过拟合是机器学习中的两个对立问题。"
    ),

    # 片段3: 中高相关 — 防止过拟合的具体方法
    (
        "防止过拟合的常用技术包括：1) 正则化（L1/L2正则），通过在损失函数中"
        "加入惩罚项限制模型复杂度；2) Dropout，训练时随机关闭部分神经元，"
        "防止模型过度依赖某些特征；3) 交叉验证，用多个数据子集评估模型，"
        "确保泛化能力；4) 早停法（Early Stopping），当验证集性能不再提升时"
        "停止训练；5) 数据增强，通过对训练数据做变换来增加样本多样性。"
    ),

    # 片段4: 无关 — 机器学习历史
    (
        "机器学习的概念可以追溯到1959年，当时Arthur Samuel开发了一个能自我"
        "改进的跳棋程序。1997年，IBM的Deep Blue击败国际象棋世界冠军卡斯帕罗夫。"
        "2012年，AlexNet在ImageNet竞赛中获胜，标志着深度学习时代的到来。"
        "2017年，Google发表了Transformer架构，催生了后来的BERT和GPT系列模型。"
    ),

    # 片段5: 中等相关 — 过拟合的实际案例
    (
        "在一个房价预测项目中，我们使用了一个包含100层的深度神经网络来预测"
        "仅有1000条样本的数据集。模型在训练集上达到了99.8%的准确率，但在"
        "新数据上误差高达40%。这是典型的过拟合案例。后来我们将模型简化为"
        "3层网络，并加入L2正则化和Dropout，测试集误差降低到了8%。"
        "这个案例说明了模型复杂度与数据量匹配的重要性。"
    ),
]


# ============================================================
# 8. 执行演示
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Map-Rerank 链演示")
    print("=" * 60)
    print(f"\n❓ 问题: {QUESTION}")
    print(f"📄 文档片段数: {len(DOCUMENTS)}")

    # ── 展示原始片段 ──
    print(f"\n{'─' * 60}")
    print("📚 原始文档片段：")
    print(f"{'─' * 60}")
    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"\n  [片段 {i}] {doc[:80]}...")

    # ══════════════════════════════════════════════════════
    # 执行方案 A: Map → Rerank → 直取最高分
    # ══════════════════════════════════════════════════════
    print(f"\n{'█' * 60}")
    print("🔗 方案 A：Map → Rerank → 直取最高分答案")
    print(f"{'█' * 60}")

    result_a = full_chain_best.invoke({
        "question": QUESTION,
        "documents": DOCUMENTS,
        "top_k": 3,
    })

    print(f"\n{'─' * 50}")
    print(f"✅ 方案A 最终答案：")
    print(f"{'─' * 50}")
    print(f"\n{result_a}")

    # ══════════════════════════════════════════════════════
    # 执行方案 B: Map → Rerank → 合并 Top-K 精炼
    # ══════════════════════════════════════════════════════
    print(f"\n\n{'█' * 60}")
    print("🔗 方案 B：Map → Rerank → 合并 Top-K 精炼答案")
    print(f"{'█' * 60}")

    result_b = full_chain_merge.invoke({
        "question": QUESTION,
        "documents": DOCUMENTS,
        "top_k": 3,
    })

    print(f"\n{'─' * 50}")
    print(f"✅ 方案B 最终答案（综合 Top-3）：")
    print(f"{'─' * 50}")
    print(f"\n{result_b}")

    # ── 对比总结 ──
    print(f"\n{'=' * 60}")
    print("📋 方案对比")
    print(f"{'=' * 60}")
    print(f"  方案A（直取最高分）: 快速、零额外成本，但只用了一个片段的信息")
    print(f"  方案B（合并Top-K） : 更慢、多一次LLM调用，但综合了多个高相关片段")
    print(f"{'=' * 60}")
    print("✅ 演示完成")
