# 用户反馈智能路由链 — 技术规格说明书
# Smart Feedback Routing Chain — Technical Specification

## 1. 目标 / Objective

构建一个 LangChain 应用，根据用户反馈的**类型**和**复杂度**，自动选择最合适的处理链和 LLM 模型：

- **类型路由**：识别反馈类型（咨询、投诉、表扬、垃圾信息），分发到对应处理链
- **模型分级**：简单任务用轻量模型（低成本、低延迟），复杂任务用强力模型（高质量）

Build a LangChain app that automatically selects the right processing chain and LLM based on feedback **type** and **complexity**:

- **Type routing**: classify feedback (inquiry, complaint, praise, spam) → dispatch to matching chain
- **Model tiering**: simple tasks → lightweight model (low cost, low latency); complex tasks → powerful model (high quality)

---

## 2. 架构总览 / Architecture Overview

```
用户反馈 (User Feedback)
       │
       ▼
┌─────────────────────┐
│  Step 1: 分类器      │  ← 轻量 LLM (qwen-turbo)
│  Classifier Chain    │     快速、便宜，只做分类
└─────────┬───────────┘
          │
          ▼  {type: "inquiry" | "complaint" | "praise" | "spam"}
┌─────────────────────┐
│  Step 2: 路由器      │  ← RunnableBranch (纯逻辑，不调用 LLM)
│  Router (Branch)     │     根据 type 分发到对应子链
└─────────┬───────────┘
          │
    ┌─────┼─────┬──────────┐
    ▼     ▼     ▼          ▼
┌──────┐┌────────┐┌────────┐┌──────────┐
│Inquiry││Complaint││ Praise ││  Spam    │
│Chain  ││ Chain   ││ Chain  ││  Chain   │
│       ││         ││        ││          │
│qwen-  ││qwen-   ││qwen-   ││ 无 LLM   │
│turbo/ ││3.7-max ││turbo   ││ 规则过滤  │
│plus   ││(强力)  ││(轻量)  ││          │
└───┬──┘└───┬────┘└───┬────┘└────┬─────┘
    │       │         │          │
    └───────┴────┬────┴──────────┘
                 ▼
          最终响应 (Final Response)
```

---

## 3. 模型分级策略 / Model Tiering Strategy

| 模型 / Model | 用途 / Purpose | 选择理由 / Rationale |
|---|---|---|
| `qwen-turbo` | 分类器、表扬回复、简单咨询 | 延迟低 (~1s)、成本低，适合短文本生成 |
| `qwen-plus` | 中等复杂度咨询 | 平衡质量与速度 |
| `qwen3.7-max` | 投诉处理、复杂咨询 | 需要共情能力、深度分析，质量优先 |

---

## 4. 四条处理链设计 / Four Chain Definitions

### 4.1 Inquiry Chain（咨询链）— qwen-plus
```
场景：用户提问、寻求帮助
输入：{"feedback": "...", "type": "inquiry"}
输出：专业、有条理的回答
```
提示词示例：
> 你是客服助手。请针对以下用户咨询，给出专业、清晰的回答。如果问题复杂，建议用户联系人工客服。

### 4.2 Complaint Chain（投诉链）— qwen3.7-max
```
场景：用户表达不满、投诉服务
输入：{"feedback": "...", "type": "complaint"}
输出：共情回应 + 解决方案 + 升级建议
```
提示词示例：
> 你是高级客服专员。用户正在投诉，请先表达真诚的歉意和共情，然后分析问题原因，给出具体解决方案，并提供升级渠道。

### 4.3 Praise Chain（表扬链）— qwen-turbo
```
场景：用户好评、感谢
输入：{"feedback": "...", "type": "praise"}
输出：简短感谢 + 鼓励继续使用
```
提示词示例：
> 你是客服助手。用户给了好评，请用一两句话表示感谢，并欢迎继续使用我们的服务。

### 4.4 Spam Chain（垃圾信息链）— 无 LLM
```
场景：无关内容、广告、恶意信息
输入：{"feedback": "...", "type": "spam"}
输出：固定拒绝回复（不调用 LLM，纯 RunnableLambda）
```

---

## 5. 核心组件说明 / Key Components

### 5.1 分类器链 / Classifier Chain
```python
# 轻量模型做分类，只返回类型标签，不需要高质量文本生成
classifier_llm = ChatOpenAI(model="qwen-turbo", ...)
classifier_prompt = PromptTemplate(
    template="""请将以下用户反馈分类为以下四种类型之一：
- inquiry（咨询）
- complaint（投诉）
- praise（表扬）
- spam（垃圾信息）

只输出类型标签，不要解释。

用户反馈：{feedback}
类型：""",
    input_variables=["feedback"]
)
classifier_chain = classifier_prompt | classifier_llm | StrOutputParser()
```

### 5.2 路由器 / Router (RunnableBranch)
```python
from langchain_core.runnables import RunnableBranch

# RunnableBranch：根据条件分发到不同子链
# 格式：(条件函数, 子链) 对，最后一个为默认链
router = RunnableBranch(
    (lambda x: "inquiry"   in x["type"].lower(), inquiry_chain),
    (lambda x: "complaint" in x["type"].lower(), complaint_chain),
    (lambda x: "praise"    in x["type"].lower(), praise_chain),
    spam_chain  # 默认 fallback
)
```

### 5.3 完整链组装 / Full Chain Assembly
```python
from langchain_core.runnables import RunnablePassthrough

full_chain = (
    RunnablePassthrough.assign(           # 保留原始 feedback
        type=classifier_chain             # 新增 type 字段
    )
    | router                              # 根据 type 路由
)
```

---

## 6. 执行流程示例 / Execution Flow Examples

### 示例 1：简单咨询
```
输入: "你们的退款政策是什么？"
  → 分类器 (qwen-turbo): "inquiry"
  → 路由 → Inquiry Chain (qwen-plus)
  → 输出: "您好！我们的退款政策为购买后7天内可申请全额退款..."
```

### 示例 2：复杂投诉
```
输入: "我已经等了两周还没收到退款，你们的客服也不回复我的邮件，这太差了！"
  → 分类器 (qwen-turbo): "complaint"
  → 路由 → Complaint Chain (qwen3.7-max)
  → 输出: "非常抱歉给您带来了如此糟糕的体验。我完全理解您的不满..."
```

### 示例 3：垃圾信息
```
输入: "点击这个链接免费领取iPhone https://scam.com"
  → 分类器 (qwen-turbo): "spam"
  → 路由 → Spam Chain (无 LLM)
  → 输出: "该内容已被系统过滤。"
```

---

## 7. 文件结构 / File Structure

```
langchain/
├── openai_client.py              # 现有文件（基础 LLM 调用）
├── sequential_chain_example.py   # 现有文件（顺序链示例）
└── feedback_router.py            # 新文件：本次实现的智能路由链
```

---

## 8. 依赖 / Dependencies

```
langchain >= 1.3
langchain-openai >= 1.3
langchain-core >= 1.4
openai  (DashScope 兼容模式)
```

---

## 9. 实现步骤 / Implementation Steps

1. **初始化三个 LLM 实例**（turbo / plus / max）
2. **构建分类器链**（turbo + PromptTemplate + StrOutputParser）
3. **构建四条子链**（各自使用合适的 LLM 和提示词）
4. **用 RunnableBranch 组装路由器**
5. **用 RunnablePassthrough.assign 串联分类 + 路由**
6. **用多个测试用例验证**（覆盖四种类型 + 边界情况）
