"""
LangChain 顺序链（Sequential Chain）示例
======================================
顺序链的核心思想：将多个子链串联，前一步的输出作为后一步的输入。
在现代 LangChain (LCEL) 中，使用 `|` 管道操作符 + RunnableLambda 实现。

本示例实现一个三步顺序链：
  第一步：给定主题 → 生成文章标题
  第二步：根据标题 → 生成短文
  第三步：将短文 → 翻译为英文
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ============================================================
# 0. 初始化 LLM（复用你现有的 DashScope 配置）
# ============================================================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请配置环境变量 OPENAI_API_KEY")

llm = ChatOpenAI(
    model_name="qwen3.7-max",
    temperature=0.7,
    max_tokens=200,
    api_key=openai_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False}
)

# StrOutputParser：从 ChatMessage 对象中提取纯文本字符串
# 这样下一步收到的就是 str，而不是 AIMessage 对象
parser = StrOutputParser()


# ============================================================
# 第一步：主题 → 生成文章标题
# ============================================================
# PromptTemplate 定义提示词模板，input_variables 声明需要的变量
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="请为以下主题生成一个吸引人的中文文章标题，只输出标题，不要任何额外内容：\n主题：{topic}"
)

# 子链1 = 提示词模板 → LLM → 文本解析器
# 输入: {"topic": "xxx"}  输出: 纯文本标题字符串
title_chain = title_prompt | llm | parser


# ============================================================
# 第二步：标题 → 生成短文
# ============================================================
article_prompt = PromptTemplate(
    input_variables=["title"],
    template="请根据以下标题写一篇100字左右的短文，只输出正文，不要任何额外内容：\n标题：{title}"
)

# 子链2：接收上一步输出的标题字符串，生成短文
article_chain = article_prompt | llm | parser


# ============================================================
# 第三步：短文 → 翻译为英文
# ============================================================
translate_prompt = PromptTemplate(
    input_variables=["article"],
    template="请将以下中文短文翻译为英文，只输出翻译结果，不要任何解释：\n{article}"
)

# 子链3：接收上一步输出的短文，翻译为英文
translate_chain = translate_prompt | llm | parser


# ============================================================
# 组装顺序链（核心部分）
# ============================================================
# 关键思路：
#   - RunnableLambda 将普通函数包装为 LangChain 的可运行组件
#   - 每一步接收上一步的 str 输出，包装成下一步 PromptTemplate 需要的 dict
#   - 整条链用 | 管道串联，数据像流水线一样依次流过每个环节
#
# 数据流向：
#   {"topic": "人工智能"}
#       → title_chain      → "AI改变世界的十大方式"
#       → RunnableLambda    → {"title": "AI改变世界的十大方式"}
#       → article_chain    → "人工智能正在深刻改变..."
#       → RunnableLambda    → {"article": "人工智能正在深刻改变..."}
#       → translate_chain  → "Artificial intelligence is profoundly..."

sequential_chain = (
    title_chain                                          # 第一步：主题 → 标题
    | RunnableLambda(lambda title: {"title": title})     # 包装为 dict 传给下一步
    | article_chain                                      # 第二步：标题 → 短文
    | RunnableLambda(lambda article: {"article": article}) # 包装为 dict 传给下一步
    | translate_chain                                    # 第三步：短文 → 英文翻译
)


# ============================================================
# 执行顺序链
# ============================================================
print("=" * 60)
print("🔗 顺序链执行：主题 → 标题 → 短文 → 英文翻译")
print("=" * 60)

final_translation = sequential_chain.invoke({"topic": "人工智能"})

print(f"\n✅ 最终输出（英文翻译）：\n{final_translation}")


# ============================================================
# 进阶：使用 RunnablePassthrough 保留中间结果
# ============================================================
# 如果你想在最终输出中同时看到每一步的结果（而不仅是最后一步），
# 可以使用 RunnablePassthrough.assign() 来"累积"中间变量。

from langchain_core.runnables import RunnablePassthrough

# 这条链在每一步都会保留之前的所有字段，同时添加新字段
chain_with_history = (
    RunnablePassthrough.assign(                           # 保留原始输入 topic
        title=title_chain                                 # 新增 title 字段
    )
    | RunnablePassthrough.assign(                         # 保留 topic + title
        article=article_chain                             # 新增 article 字段
    )
    | RunnablePassthrough.assign(                         # 保留 topic + title + article
        translation=translate_chain                       # 新增 translation 字段
    )
)

print("\n" + "=" * 60)
print("🔗 带中间结果的顺序链")
print("=" * 60)

all_results = chain_with_history.invoke({"topic": "LLM"})

print(f"\n📌 主题:     {all_results['topic']}")
print(f"📌 标题:     {all_results['title']}")
print(f"📌 短文:     {all_results['article']}")
print(f"📌 英文翻译: {all_results['translation']}")
