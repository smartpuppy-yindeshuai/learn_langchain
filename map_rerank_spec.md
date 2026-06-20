# Map-Rerank 链学习规格说明书
# Map-Rerank Chain — Technical Specification

## 1. 概念 / Concept

Map-Rerank 是 Map-Reduce 链的增强版：

```
Map-Reduce:  分片处理 → 简单合并        → 最终输出
Map-Rerank:  分片处理 → 评分 + 排序 → 选最优 → 最终输出
```

核心区别：Map-Rerank 在 Map 之后增加了一个**评分排序（Rerank）** 步骤，
确保最终答案来自**最相关、质量最高**的文档片段，而不是简单拼接所有结果。

Map-Rerank enhances Map-Reduce by adding a **scoring & reranking** step after
the Map phase, ensuring the final answer comes from the **most relevant and
highest quality** chunks rather than naively concatenating everything.

---

## 2. 架构总览 / Architecture

```
原始问题 + 多个文档片段
        │
        ▼
┌───────────────────────────────────────┐
│  Phase 1: Map（映射）                  │
│                                       │
│  每个文档片段 + 原始问题                │
│  → 独立调用 LLM 生成答案 + 评分        │
│  （并行执行，互不影响）                 │
│                                       │
│  片段1 + 问题 → LLM → {答案1, 评分:85} │
│  片段2 + 问题 → LLM → {答案2, 评分:60} │
│  片段3 + 问题 → LLM → {答案3, 评分:92} │
│  片段4 + 问题 → LLM → {答案4, 评分:45} │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  Phase 2: Rerank（重排序）             │
│                                       │
│  收集所有 Map 结果                      │
│  → 按评分降序排列                       │
│  → 选取 Top-K 或最高分结果              │
│                                       │
│  排序后: 答案3(92) > 答案1(85) >       │
│          答案2(60) > 答案4(45)          │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────┐
│  Phase 3: Reduce（归约，可选）          │
│                                       │
│  方案 A: 直接取最高分答案作为最终输出    │
│  方案 B: 将 Top-K 答案合并后再生成最终  │
│          答案（可选再调一次 LLM）        │
└───────────────┬───────────────────────┘
                │
                ▼
           最终输出
```

---

## 3. 应用场景 / Use Cases

| 场景 | 示例 | 为什么用 Map-Rerank |
|---|---|---|
| 长文档问答 | 在 50 页合同中查找违约条款 | 只需最相关的几个段落的答案，不需要全文汇总 |
| 多来源筛选 | 从 5 篇论文摘要中找最佳定义 | 评分机制自动过滤低质量内容 |
| RAG 增强 | 从检索到的 10 个 chunk 中选最优回答 | 避免无关 chunk 污染最终答案 |
| 内容过滤 | 从 20 条评论中提取最有价值的反馈 | 评分排序天然适合"选优"场景 |

---

## 4. 与其他链模式的对比 / Comparison

```
┌─────────────┬──────────────────────────────────────────────────┐
│ 模式         │ 特点                                             │
├─────────────┼──────────────────────────────────────────────────┤
│ Stuff       │ 把所有文档塞进一个 prompt → 一次 LLM 调用          │
│             │ 适合：短文档，token 够用的场景                      │
├─────────────┼──────────────────────────────────────────────────┤
│ Map-Reduce  │ 每个片段独立处理 → 合并所有结果                     │
│             │ 适合：需要汇总全部信息的场景                        │
├─────────────┼──────────────────────────────────────────────────┤
│ Map-Rerank  │ 每个片段独立处理 → 评分排序 → 选最优                │
│             │ 适合：只需要最佳答案，不需要全部信息的场景           │
├─────────────┼──────────────────────────────────────────────────┤
│ Refine      │ 逐片段迭代，每次基于前一次结果精炼                   │
│             │ 适合：需要逐步积累上下文的场景                      │
└─────────────┴──────────────────────────────────────────────────┘
```

---

## 5. LCEL 实现方案 / Implementation Approach

### 5.1 核心组件

```python
# Map 阶段：对每个文档片段并行处理
# 使用 prompt + LLM + 自定义 OutputParser 提取答案和评分

map_prompt = PromptTemplate(
    template="""根据以下文档片段回答问题，并对答案的相关性打分（0-100）。

文档片段：{context}

问题：{question}

请按以下格式回答：
答案：<你的回答>
评分：<0-100的整数>""",
    input_variables=["context", "question"]
)

# 自定义解析器：从 LLM 输出中提取 answer 和 score
class MapResult(TypedDict):
    answer: str
    score: int

map_parser = ...  # 解析 "答案:xxx\n评分:85" 格式

# Map 链 = prompt → LLM → parser
map_chain = map_prompt | llm | map_parser
```

### 5.2 Map 阶段（并行处理所有片段）

```python
from langchain_core.runnables import RunnableLambda

def map_phase(data):
    """对每个文档片段独立调用 map_chain"""
    question = data["question"]
    documents = data["documents"]

    # 并行处理所有片段
    results = []
    for doc in documents:
        result = map_chain.invoke({
            "context": doc,
            "question": question
        })
        results.append(result)

    return {"question": question, "results": results}
```

### 5.3 Rerank 阶段（评分排序）

```python
def rerank_phase(data):
    """按评分降序排列，选取 Top-K"""
    results = data["results"]

    # 按 score 降序排序
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # 选取 Top-K（默认取全部，可选配置）
    top_k = data.get("top_k", len(sorted_results))
    top_results = sorted_results[:top_k]

    return {
        "question": data["question"],
        "ranked_results": top_results,
        "best_answer": top_results[0]["answer"],
        "best_score": top_results[0]["score"],
    }
```

### 5.4 Reduce 阶段（可选，合并 Top-K）

```python
# 方案 A：直接返回最高分答案（最简单）
reduce_chain = RunnableLambda(lambda data: data["best_answer"])

# 方案 B：合并 Top-K 后再生成综合答案（更高质量）
reduce_prompt = PromptTemplate(
    template="""以下是针对问题 "{question}" 的多个候选答案（已按相关性排序）。
请综合这些答案，生成一个最终的、最完整的回答。

{ranked_answers}

最终回答：""",
    input_variables=["question", "ranked_answers"]
)
```

### 5.5 完整链组装

```python
full_chain = (
    RunnableLambda(map_phase)       # Phase 1: Map
    | RunnableLambda(rerank_phase)  # Phase 2: Rerank
    | reduce_chain                  # Phase 3: Reduce
)

# 执行
result = full_chain.invoke({
    "question": "什么是机器学习中的过拟合？",
    "documents": [chunk1, chunk2, chunk3, chunk4, chunk5],
})
```

---

## 6. 测试数据设计 / Test Data Design

### 场景：从 5 个文档片段中回答一个技术问题

| 片段 | 内容摘要 | 预期评分 | 说明 |
|---|---|---|---|
| 片段1 | 直接回答过拟合的定义和原因 | ~90+ | 高相关，应排第一 |
| 片段2 | 讨论欠拟合（非目标概念） | ~30 | 低相关，应排靠后 |
| 片段3 | 提到正则化等防止过拟合的方法 | ~80 | 中等相关，补充信息 |
| 片段4 | 完全不相关的机器学习历史 | ~10 | 无关，应排最后 |
| 片段5 | 过拟合在实际项目中的案例 | ~75 | 有一定相关性 |

预期最终输出：综合片段1 + 片段3 + 片段5 的高质量答案，
排除片段2和片段4的低质量内容。

---

## 7. 文件结构 / File Structure

```
langchain/
├── openai_client.py              # 基础 LLM 调用
├── sequential_chain_example.py   # 顺序链示例
├── feedback_router.py            # 智能路由链
├── feedback_router_spec.md       # 路由链规格
└── map_rerank_example.py         # 新文件：本次实现
```

---

## 8. 实现步骤 / Implementation Steps

1. **定义 Map 提示词**：要求 LLM 同时输出答案和相关性评分
2. **编写自定义解析器**：从 LLM 文本输出中提取 `answer` 和 `score`
3. **实现 Map 阶段**：遍历所有文档片段，并行调用 Map 链
4. **实现 Rerank 阶段**：按评分排序，选取 Top-K
5. **实现 Reduce 阶段**：两种方案（直取最高分 / 合并 Top-K）
6. **组装完整链**：用 RunnableLambda 串联三个阶段
7. **用测试数据验证**：5 个片段，验证评分排序是否正确、最终答案是否来自高相关片段
8. **输出可视化**：打印每个片段的评分排名 + 最终答案，便于理解数据流

---

## 9. 关键学习点 / Key Takeaways

1. **Map 阶段可以并行**：每个片段独立处理，天然支持并发，适合大规模文档
2. **评分是关键**：让 LLM 自评答案相关性，比人工筛选更高效
3. **Top-K 选择**：不需要所有片段的信息，只需最好的几个
4. **成本可控**：Map 阶段可以用轻量模型（快速评分），Reduce 阶段用强力模型（生成最终答案）
5. **对比 Map-Reduce**：Map-Reduce 合并所有结果（可能引入噪音），Map-Rerank 只选最优（更精准）
