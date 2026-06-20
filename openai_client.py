import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 读取环境变量中的API密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请配置环境变量 OPENAI_API_KEY")

# 2. 定义带动态变量的提示词模板（可复用）
prompt_template = PromptTemplate(
    input_variables=["subject"],
    template="请简要描述{subject}的基本概念。"
)

# 3. 初始化对话式GPT模型（使用阿里云DashScope兼容接口）
llm = ChatOpenAI(
    model_name="qwen3.7-max",
    temperature=0.7,
    max_tokens=100,
    api_key=openai_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    extra_body={"enable_thinking": False}
)

# 4. 使用 LCEL 构建执行链（替代已移除的 LLMChain）
chain = prompt_template | llm

# 5. 传入参数执行（动态替换主题）
response = chain.invoke({"subject": "机器学习"})
# Qwen3 系列模型支持"思考模式"，内容可能在 reasoning_content 中
print("生成的回答:", response.content or response.additional_kwargs.get("reasoning_content", "(无内容)"))