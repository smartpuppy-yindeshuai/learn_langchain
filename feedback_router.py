"""
用户反馈智能路由链 (Smart Feedback Routing Chain)
=================================================
演示两个核心概念：
  1. 类型路由 (Type Routing)  — 根据反馈类型分发到不同处理链
  2. 模型分级 (Model Tiering) — 简单任务用轻量模型，复杂任务用强力模型

架构：
  用户反馈 → 分类器 (qwen3-max 轻量) → 路由器 (RunnableBranch)
                ↓
    ┌───────────┼───────────┬──────────────┐
    ▼           ▼           ▼              ▼
  Inquiry    Complaint    Praise         Spam
 (qwen3-max  (qwen3.7-max (qwen3-max    (无LLM)
  中等)       强力)        轻量)
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

# ============================================================
# 0. 通用配置
# ============================================================
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请配置环境变量 OPENAI_API_KEY")

# DashScope OpenAI 兼容接口的公共参数
common_kwargs = {
    "api_key": openai_api_key,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "extra_body": {"enable_thinking": False},  # 关闭 Qwen3 思考模式
}

# 文本解析器：从 AIMessage 对象中提取纯文本
parser = StrOutputParser()


# ============================================================
# 1. 初始化两个模型，三种配置级别
# ============================================================
# 当前 API Key 可用模型：qwen3-max / qwen3.7-max
# 通过不同 max_tokens + temperature 模拟轻量/中等/强力三档

# 轻量配置：qwen3-max + 低 token + 低温度 → 快速分类、简单回复
llm_light = ChatOpenAI(
    model_name="qwen3-max",
    temperature=0.1,   # 低温度 → 输出更确定，适合分类
    max_tokens=30,
    **common_kwargs,
)

# 中等配置：qwen3-max + 中等 token → 平衡质量与速度
llm_medium = ChatOpenAI(
    model_name="qwen3-max",
    temperature=0.5,
    max_tokens=200,
    **common_kwargs,
)

# 强力配置：qwen3.7-max + 高 token → 共情、深度分析
llm_heavy = ChatOpenAI(
    model_name="qwen3.7-max",
    temperature=0.7,
    max_tokens=400,
    **common_kwargs,
)


# ============================================================
# 2. 分类器链 (Classifier Chain) — qwen-turbo
# ============================================================
# 只负责判断反馈类型，不需要高质量文本生成，所以用最便宜的模型

classifier_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""你是一个反馈分类器。请将以下用户反馈分为以下四种类型之一：
- inquiry（咨询：用户提问或寻求帮助）
- complaint（投诉：用户表达不满或抱怨）
- praise（表扬：用户表达满意或感谢）
- spam（垃圾信息：广告、无关链接、恶意内容）

只输出一个类型标签（inquiry / complaint / praise / spam），不要任何其他内容。

用户反馈：{feedback}
类型：""",
)

# 分类器链：prompt → light LLM → 提取纯文本
# 输出示例："complaint"
classifier_chain = classifier_prompt | llm_light | parser


# ============================================================
# 3. 四条处理子链 (Sub-Chains)
# ============================================================

# ----- 3.1 Inquiry Chain（咨询链）— qwen-plus -----
# 中等复杂度，需要准确、有条理的回答
inquiry_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""你是专业客服助手。请针对以下用户咨询，给出清晰、有条理的回答。
如果涉及具体操作，请分步骤说明。如果问题超出你的能力范围，建议用户联系人工客服。

用户咨询：{feedback}

回答：""",
)
inquiry_chain = inquiry_prompt | llm_medium | parser


# ----- 3.2 Complaint Chain（投诉链）— qwen3.7-max -----
# 高复杂度，需要共情能力、深度分析和解决方案
complaint_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""你是高级客服专员。一位用户正在投诉，请注意以下几点：
1. 首先表达真诚的歉意，展现共情
2. 分析可能的问题原因
3. 给出具体的解决方案
4. 提供后续跟进或升级渠道

用户投诉：{feedback}

回复：""",
)
complaint_chain = complaint_prompt | llm_heavy | parser


# ----- 3.3 Praise Chain（表扬链）— qwen-turbo -----
# 低复杂度，只需简短感谢
praise_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""你是客服助手。用户给了好评，请用一两句话表示感谢，并欢迎继续使用我们的服务。

用户反馈：{feedback}

回复：""",
)
praise_chain = praise_prompt | llm_light | parser


# ----- 3.4 Spam Chain（垃圾信息链）— 无 LLM -----
# 不需要调用模型，直接返回固定拒绝消息，节省成本
# RunnableLambda 将普通函数包装为 LangChain 可运行组件
spam_chain = RunnableLambda(
    lambda _: "⚠️ 该内容已被系统识别为垃圾信息并已过滤。"
)


# ============================================================
# 4. 路由器 (Router) — RunnableBranch
# ============================================================
# RunnableBranch 接收一组 (条件, 链) 对：
#   - 依次检查每个条件，第一个为 True 的执行对应链
#   - 最后一个参数是无条件的 fallback 链
#
# 注意：条件函数接收的是上一步输出的 dict，包含 "type" 字段

def route_feedback(data):
    """根据分类结果选择处理链"""
    feedback_type = data.get("type", "").strip().lower()

    if "inquiry" in feedback_type:
        return "inquiry"
    elif "complaint" in feedback_type:
        return "complaint"
    elif "praise" in feedback_type:
        return "praise"
    else:
        return "spam"


# 用 RunnableLambda 实现路由分发
# 根据分类结果选择对应的子链执行
router = RunnableLambda(route_feedback) | RunnableBranch(
    (lambda x: x == "inquiry",   RunnableLambda(lambda x: x) | inquiry_chain),
    (lambda x: x == "complaint", RunnableLambda(lambda x: x) | complaint_chain),
    (lambda x: x == "praise",    RunnableLambda(lambda x: x) | praise_chain),
    # fallback：任何未匹配的类型都走 spam 链
    spam_chain,
)


# ============================================================
# 5. 完整链组装 (Full Chain Assembly)
# ============================================================
# 数据流：
#   {"feedback": "..."}
#     → RunnablePassthrough.assign(type=classifier_chain)
#       → {"feedback": "...", "type": "complaint"}
#     → extract_type (提取 type 字符串)
#       → "complaint"
#     → router (RunnableBranch)
#       → complaint_chain 的输出

# 为了同时展示分类结果和最终回复，用 RunnablePassthrough 保留中间数据
def process_and_display(data):
    """接收包含 type 和 feedback 的 dict，执行路由并格式化输出"""
    feedback_type = data.get("type", "unknown").strip().lower()
    feedback_text = data["feedback"]

    # 根据类型选择子链并执行
    if "inquiry" in feedback_type:
        chain = inquiry_chain
        label = "📋 咨询 (Inquiry) → qwen3-max (中等)"
    elif "complaint" in feedback_type:
        chain = complaint_chain
        label = "🚨 投诉 (Complaint) → qwen3.7-max (强力)"
    elif "praise" in feedback_type:
        chain = praise_chain
        label = "👍 表扬 (Praise) → qwen3-max (轻量)"
    else:
        chain = spam_chain
        label = "🗑️ 垃圾信息 (Spam) → 无LLM"

    # 执行子链
    response = chain.invoke({"feedback": feedback_text})

    return {
        "type": feedback_type,
        "label": label,
        "response": response,
    }


full_chain = (
    RunnablePassthrough.assign(
        type=classifier_chain  # 第一步：分类，新增 type 字段
    )
    | RunnableLambda(process_and_display)  # 第二步：路由 + 执行 + 格式化
)


# ============================================================
# 6. 测试用例 (Test Cases)
# ============================================================
test_cases = [
    {
        "name": "简单咨询",
        "feedback": "你们的退款政策是什么？购买后多久可以申请退款？",
    },
    {
        "name": "复杂投诉",
        "feedback": "我已经等了两周还没收到退款，发了三封邮件都没人回复，你们的客服系统太差了！我要求立即处理！",
    },
    {
        "name": "用户表扬",
        "feedback": "你们的产品真的很好用，界面简洁，功能强大，已经推荐给朋友了！",
    },
    {
        "name": "垃圾信息",
        "feedback": "🎉恭喜您中奖100万！点击链接领取：http://scam-site.com/prize 先到先得！",
    },
    {
        "name": "技术深度咨询",
        "feedback": "请问你们的API支持哪些认证方式？OAuth 2.0和API Key有什么区别？在微服务架构下推荐用哪种？",
    },
]


# ============================================================
# 7. 执行测试
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧠 用户反馈智能路由链 — 演示不同类型的反馈走不同的处理链和模型")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"📝 测试 {i}：{case['name']}")
        print(f"   用户反馈：{case['feedback']}")
        print(f"{'─' * 70}")

        result = full_chain.invoke({"feedback": case["feedback"]})

        print(f"   🔀 分类结果：{result['type']}")
        print(f"   🔗 路由选择：{result['label']}")
        print(f"\n   💬 系统回复：\n   {result['response']}")

    print(f"\n{'=' * 70}")
    print("✅ 全部测试完成")
    print("=" * 70)
