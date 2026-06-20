"""
动态消息记忆策略 (Dynamic Message Memory Strategy)
====================================================

本模块实现了一套完整的动态消息记忆策略系统，用于智能对话中的上下文管理。
核心设计理念：根据对话情境灵活调整存储内容，使模型的记忆更符合当前对话需求。

系统包含五大核心模块:
  1. 条件消息过滤器 (ConditionalFilter)    — 存储前多维度评估消息价值
  2. 动态记忆深度调节 (DynamicDepthAdjuster) — 基于对话轮数的指数衰减模型
  3. 优先级消息保留 (PriorityScorer)        — 多维度优先级评分 + 最小堆淘汰
  4. 话题切换检测 (TopicDetector)           — 关键词漂移 + 语义相似度复合检测
  5. 会话缓冲区管理 (SessionBuffer)         — 四区域缓冲 + 动态容量 + 摘要压缩

辅助组件:
  - ArchiveStore: 归档存储，支持关键词检索
  - ContextAssembler: 上下文组装，按 token 预算构建 LLM 输入
  - DynamicMemoryPipeline: 流水线编排器，串联所有模块

参考规格书: dynamic-memory-strategy-spec.md
"""

import os
import re
import math
import uuid
import heapq
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


# ══════════════════════════════════════════════════════════════════════════════
# 0. 全局配置 — LLM 初始化 & DashScope 连接参数
# ══════════════════════════════════════════════════════════════════════════════

# 从环境变量加载 API 密钥（DashScope 兼容 OpenAI 接口）
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量（DashScope API Key）")

# DashScope 公共连接参数 — 所有 LLM 实例共享此配置
common_kwargs = dict(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=openai_api_key,
    extra_body={"enable_thinking": False},  # 关闭 Qwen3 思考模式，直接输出
)

# 轻量模型 — 用于摘要压缩等辅助任务（成本低、速度快）
llm_light = ChatOpenAI(
    model="qwen3-max",
    temperature=0.3,
    **common_kwargs,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 数据模型 — 使用 dataclass 定义核心数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """
    消息对象 — 对话系统中的单条消息

    基础字段在创建时填充，流水线字段（filter_score, priority_score 等）
    在处理过程中由各模块逐步填充。

    Attributes:
        role:           消息角色 — "user" | "assistant" | "system"
        content:        消息文本内容
        id:             唯一标识符，自动生成 UUID
        timestamp:      消息创建时间
        turn_number:    所在对话轮次（由流水线填充）
        topic_id:       所属话题 ID（由话题检测器填充）
        filter_score:   过滤器综合评分 [0, 1]
        priority_score: 优先级评分 [0, 1]
        is_pinned:      是否被钉住（高优先级消息不可淘汰）
        intent_type:    识别的意图类型（requirement/question/decision/info/chat）
        metadata:       扩展元数据字典
    """
    role: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    turn_number: int = 0
    topic_id: str = ""
    filter_score: Optional[float] = None
    priority_score: Optional[float] = None
    is_pinned: bool = False
    intent_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """支持堆排序 — 按 priority_score 比较，None 视为 0"""
        self_score = self.priority_score or 0.0
        other_score = other.priority_score or 0.0
        return self_score < other_score


@dataclass
class TopicContext:
    """
    话题上下文 — 跟踪一个对话话题的生命周期

    当话题检测器识别到新话题时创建，话题切换时标记 is_active=False 并归档。

    Attributes:
        topic_id:   话题唯一 ID
        keywords:   从该话题消息中提取的关键词集合
        start_turn: 话题起始的对话轮次
        end_turn:   话题结束的对话轮次（活跃时为 None）
        summary:    话题摘要（归档后由 LLM 生成）
        is_active:  是否为当前活跃话题
    """
    topic_id: str = field(default_factory=lambda: f"topic_{uuid.uuid4().hex[:6]}")
    keywords: List[str] = field(default_factory=list)
    start_turn: int = 0
    end_turn: Optional[int] = None
    summary: Optional[str] = None
    is_active: bool = True


@dataclass
class FilterResult:
    """
    过滤结果 — 条件过滤器的输出

    Attributes:
        passed:  消息是否通过过滤（True = 允许存储）
        score:   综合评分 [0, 1]
        details: 各维度评分明细
    """
    passed: bool
    score: float
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class TopicResult:
    """
    话题检测结果 — 话题检测器的输出

    Attributes:
        switched:    是否检测到话题切换
        topic_id:    当前话题 ID（新话题或延续的旧话题）
        confidence:  检测置信度 [0, 1]
        details:     检测细节（关键词漂移分、语义相似度分等）
    """
    switched: bool
    topic_id: str
    confidence: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArchiveEntry:
    """
    归档条目 — 一个已完成话题的完整记忆归档

    Attributes:
        topic_id:    话题 ID
        summary:     LLM 生成的话题摘要
        messages:    该话题下的所有消息
        archived_at: 归档时间
        keywords:    从消息中提取的关键词（用于检索）
    """
    topic_id: str
    summary: str
    messages: List[Message]
    archived_at: datetime = field(default_factory=datetime.now)
    keywords: List[str] = field(default_factory=list)


@dataclass
class ProcessResult:
    """
    处理结果 — 流水线对单条消息的完整处理输出

    Attributes:
        stored:          消息是否被存入记忆
        filter_score:    过滤器评分
        priority_score:  优先级评分
        topic_id:        所属话题 ID
        topic_switched:  是否触发了话题切换
        current_depth:   当前记忆深度
        buffer_usage:    缓冲区使用情况 (如 "8/20")
        actions:         执行的操作列表
    """
    stored: bool
    filter_score: float = 0.0
    priority_score: float = 0.0
    topic_id: str = ""
    topic_switched: bool = False
    current_depth: int = 0
    buffer_usage: str = ""
    actions: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 模块一：条件消息过滤器 (ConditionalFilter)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 消息存储前的多维度价值评估
# 维度: 关键词匹配 (40%) + 消息长度 (20%) + 上下文相关性 (40%)
# 输出: FilterResult (通过/拒绝 + 评分 + 明细)
#
# 参考规格书 §3
# ══════════════════════════════════════════════════════════════════════════════

class ConditionalFilter:
    """
    条件消息过滤器

    消息进入记忆系统的第一道关卡。通过关键词、长度、上下文三个维度
    综合评估消息的存储价值，只有评分超过阈值的消息才被允许进入
    后续处理流程。

    评分公式:
        final = w_keyword × keyword_score
              + w_length  × length_score
              + w_context × context_score

    配置参数 (均有合理默认值):
        boost_keywords:    提升保留概率的关键词列表
        suppress_keywords: 降低保留概率的关键词列表
        min_chars:         最小消息长度（低于此值直接拒绝）
        max_chars:         最大消息长度（高于此值需要摘要）
        sweet_spot:        最佳长度区间 (lower, upper)
        weights:           三个维度的权重字典
        threshold:         通过的最低综合评分
    """

    def __init__(
        self,
        boost_keywords: Optional[List[str]] = None,
        suppress_keywords: Optional[List[str]] = None,
        min_chars: int = 5,
        max_chars: int = 500,
        sweet_spot: Tuple[int, int] = (20, 200),
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.35,
        recent_window: int = 3,
        high_similarity: float = 0.6,
        low_similarity: float = 0.3,
    ):
        # --- 关键词规则 ---
        # boost: 命中时 +1 分/个，表示消息含重要信息
        self.boost_keywords = boost_keywords or [
            "需求", "必须", "重要", "确认", "决定",
            "bug", "故障", "安全", "密码", "不允许",
        ]
        # suppress: 命中时 -0.5 分/个，表示消息为低价值寒暄
        self.suppress_keywords = suppress_keywords or [
            "嗯", "好的", "收到", "谢谢",
        ]

        # --- 长度规则 ---
        self.min_chars = min_chars          # 低于此长度 → 直接拒绝
        self.max_chars = max_chars          # 高于此长度 → 标记需摘要
        self.sweet_spot = sweet_spot        # (lower, upper) 最佳区间

        # --- 权重配置 ---
        # 默认: 关键词 40% + 长度 20% + 上下文 40%
        self.weights = weights or {
            "keyword": 0.4,
            "length": 0.2,
            "context": 0.4,
        }

        # --- 阈值 ---
        self.threshold = threshold          # 综合评分 ≥ 此值才通过

        # --- 上下文相关性参数 ---
        self.recent_window = recent_window      # 计算相似度时参考的最近 N 条消息
        self.high_similarity = high_similarity  # 高相关性阈值
        self.low_similarity = low_similarity    # 低相关性阈值

    def _keyword_score(self, content: str) -> float:
        """
        关键词维度评分

        算法:
          1. 遍历 boost_keywords，每命中一个 +1 分
          2. 遍历 suppress_keywords，每命中一个 -0.5 分
          3. 将原始分归一化到 [0, 1] 区间

        归一化公式:
          raw_score = boost_hits × 1.0 + suppress_hits × (-0.5)
          max_possible = len(boost_keywords)  (理论最大分)
          normalized = max(0, min(1, (raw_score + max_possible × 0.5) / (max_possible × 1.5)))

        Args:
            content: 消息文本

        Returns:
            float: [0, 1] 区间的关键词评分
        """
        boost_hits = sum(1 for kw in self.boost_keywords if kw in content)
        suppress_hits = sum(1 for kw in self.suppress_keywords if kw in content)

        # 原始分: 正向命中加分，负向命中扣分
        raw_score = boost_hits * 1.0 + suppress_hits * (-0.5)

        # 归一化到 [0, 1]
        max_possible = max(len(self.boost_keywords), 1)
        normalized = (raw_score + max_possible * 0.5) / (max_possible * 1.5)
        return max(0.0, min(1.0, normalized))

    def _length_score(self, content: str) -> Tuple[float, bool]:
        """
        长度维度评分

        分段评分函数 (参考规格书 §3.2.2):
          - len < min_chars:       score = 0 (直接拒绝)
          - min_chars ≤ len < sweet[0]: score = 线性插值 (0→1)
          - sweet[0] ≤ len ≤ sweet[1]:  score = 1.0 (最佳区间)
          - sweet[1] < len ≤ max_chars: score = 衰减但不低于 0.3
          - len > max_chars:       score = 0 + needs_summary = True

        Args:
            content: 消息文本

        Returns:
            Tuple[float, bool]: (评分, 是否需要摘要化)
        """
        length = len(content)
        sweet_low, sweet_high = self.sweet_spot
        needs_summary = False

        # 过短: 直接拒绝
        if length < self.min_chars:
            return 0.0, False

        # 过长: 需要摘要化处理
        if length > self.max_chars:
            needs_summary = True
            return 0.3, needs_summary  # 给予较低分数，标记需要摘要

        # 最佳区间: 满分
        if sweet_low <= length <= sweet_high:
            return 1.0, False

        # 偏短 (min_chars → sweet_low): 线性增长
        if length < sweet_low:
            score = (length - self.min_chars) / max(sweet_low - self.min_chars, 1)
            return max(0.0, min(1.0, score)), False

        # 偏长 (sweet_high → max_chars): 衰减但不低于 0.3
        decay = (length - sweet_high) / max(self.max_chars - sweet_high, 1)
        score = max(0.3, 1.0 - decay)
        return score, needs_summary

    def _context_score(self, content: str, recent_messages: List[Message]) -> float:
        """
        上下文相关性评分

        通过计算当前消息与最近 N 条消息的词级重叠度来评估相关性。
        使用简单的字符级 bigram 重叠（无需外部嵌入模型）。

        评分规则:
          - 平均重叠度 ≥ high_similarity → 1.0 (高相关)
          - 平均重叠度 ∈ [low, high)      → 0.6 (中相关)
          - 平均重叠度 < low_similarity   → 0.3 (低相关，但不直接拒绝)

        当历史消息不足 recent_window 条时，默认返回 0.8（新对话宽容处理）。

        Args:
            content:         当前消息文本
            recent_messages: 最近的 N 条消息

        Returns:
            float: [0, 1] 区间的上下文相关性评分
        """
        # 历史消息不足时，宽容处理（新对话前期应该多存储）
        if len(recent_messages) < self.recent_window:
            return 0.8

        # 提取当前消息的字符 bigram 集合
        current_bigrams = self._extract_bigrams(content)
        if not current_bigrams:
            return 0.3

        # 计算与最近 N 条消息的平均重叠度
        similarities = []
        for msg in recent_messages[-self.recent_window:]:
            msg_bigrams = self._extract_bigrams(msg.content)
            if not msg_bigrams:
                continue
            # Jaccard 相似度: |交集| / |并集|
            intersection = current_bigrams & msg_bigrams
            union = current_bigrams | msg_bigrams
            sim = len(intersection) / len(union) if union else 0.0
            similarities.append(sim)

        if not similarities:
            return 0.3

        avg_sim = sum(similarities) / len(similarities)

        # 分段评分
        if avg_sim >= self.high_similarity:
            return 1.0
        elif avg_sim >= self.low_similarity:
            return 0.6
        else:
            return 0.3

    @staticmethod
    def _extract_bigrams(text: str) -> set:
        """
        提取文本的字符级 bigram 集合

        用于计算两条消息之间的词级重叠度。
        先移除空白，再提取连续的两个字符组合。

        Args:
            text: 输入文本

        Returns:
            set: bigram 字符串集合
        """
        # 移除空白字符后提取 bigram
        cleaned = re.sub(r'\s+', '', text)
        if len(cleaned) < 2:
            return set()
        return {cleaned[i:i+2] for i in range(len(cleaned) - 1)}

    def evaluate(self, message: Message, recent_messages: List[Message]) -> FilterResult:
        """
        综合评估消息是否应被存储

        执行流程:
          1. 计算关键词得分
          2. 计算长度得分（可能标记需要摘要）
          3. 计算上下文相关性得分
          4. 加权求和得到综合评分
          5. 与阈值比较，决定通过或拒绝

        Args:
            message:         待评估的消息对象
            recent_messages: 最近的对话历史（用于上下文评分）

        Returns:
            FilterResult: 包含通过决策、综合评分和各维度明细
        """
        content = message.content

        # --- 维度 1: 关键词评分 ---
        kw_score = self._keyword_score(content)

        # --- 维度 2: 长度评分 ---
        len_score, needs_summary = self._length_score(content)

        # 长度过短直接拒绝（不走加权）
        if len_score == 0.0 and len(content) < self.min_chars:
            return FilterResult(
                passed=False,
                score=0.0,
                details={
                    "keyword": kw_score,
                    "length": len_score,
                    "context": 0.0,
                    "reason": "消息过短",
                },
            )

        # --- 维度 3: 上下文相关性评分 ---
        ctx_score = self._context_score(content, recent_messages)

        # --- 加权综合评分 ---
        final_score = (
            self.weights["keyword"] * kw_score
            + self.weights["length"] * len_score
            + self.weights["context"] * ctx_score
        )

        # --- 通过/拒绝决策 ---
        passed = final_score >= self.threshold

        # 如果需要摘要，在 metadata 中标记
        if needs_summary:
            message.metadata["needs_summary"] = True

        return FilterResult(
            passed=passed,
            score=final_score,
            details={
                "keyword": round(kw_score, 3),
                "length": round(len_score, 3),
                "context": round(ctx_score, 3),
                "needs_summary": needs_summary,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. 模块二：动态记忆深度调节 (DynamicDepthAdjuster)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 根据对话轮数动态调整系统应保留的消息数量
# 模型: 指数衰减 — 对话初期全量存储，深入后逐步压缩
# 公式: depth = max(min_depth, floor(max_depth × decay_rate ^ effective_turns))
#
# 参考规格书 §4
# ══════════════════════════════════════════════════════════════════════════════

class DynamicDepthAdjuster:
    """
    动态记忆深度调节器

    核心思想：对话初期应尽量多存消息以建立完整上下文；
    对话深入后逐步压缩记忆深度，保留精华而非全量。

    三个阶段:
      - 建立期 (turn ≤ decay_start):  depth = max_depth (全量存储)
      - 衰减期 (decay_start < turn < floor_after): 指数衰减
      - 稳定期 (turn ≥ floor_after):  depth = min_depth (仅保留精华)

    配置参数:
        max_depth:        最大记忆深度（消息条数）
        min_depth:        最小记忆深度
        decay_start_turn: 从第几轮开始衰减
        decay_rate:       每轮衰减系数 (0 < rate < 1)
        floor_after_turn: 多少轮后达到最小深度
        priority_quota_ratio: 高优先级消息的保留配额占比
    """

    def __init__(
        self,
        max_depth: int = 20,
        min_depth: int = 5,
        decay_start_turn: int = 5,
        decay_rate: float = 0.85,
        floor_after_turn: int = 30,
        priority_quota_ratio: float = 0.6,
    ):
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.decay_start_turn = decay_start_turn
        self.decay_rate = decay_rate
        self.floor_after_turn = floor_after_turn
        self.priority_quota_ratio = priority_quota_ratio

    def get_depth(self, current_turn: int) -> int:
        """
        计算当前轮次的记忆深度

        衰减公式:
          if turn ≤ decay_start:       depth = max_depth
          elif turn ≥ floor_after:     depth = min_depth
          else:
            effective = turn - decay_start
            depth = max(min_depth, floor(max_depth × decay_rate ^ effective))

        Args:
            current_turn: 当前对话轮次

        Returns:
            int: 建议的记忆深度（消息条数）
        """
        # 建立期: 全量存储
        if current_turn <= self.decay_start_turn:
            return self.max_depth

        # 稳定期: 最小深度
        if current_turn >= self.floor_after_turn:
            return self.min_depth

        # 衰减期: 指数衰减
        effective_turns = current_turn - self.decay_start_turn
        depth = math.floor(self.max_depth * (self.decay_rate ** effective_turns))
        return max(self.min_depth, depth)

    def get_quotas(self, current_turn: int) -> Tuple[int, int]:
        """
        计算优先级配额和时效性配额

        当记忆深度缩小时，保留策略:
          - priority_quota (60%): 留给高优先级消息
          - recency_quota (40%):  留给最近消息（保证连续性）

        Args:
            current_turn: 当前对话轮次

        Returns:
            Tuple[int, int]: (priority_quota, recency_quota)
        """
        depth = self.get_depth(current_turn)
        priority_quota = math.floor(depth * self.priority_quota_ratio)
        recency_quota = depth - priority_quota
        return priority_quota, recency_quota

    def get_phase(self, current_turn: int) -> str:
        """
        获取当前所处的阶段名称

        Args:
            current_turn: 当前对话轮次

        Returns:
            str: "建立期" | "衰减期" | "稳定期"
        """
        if current_turn <= self.decay_start_turn:
            return "建立期"
        elif current_turn >= self.floor_after_turn:
            return "稳定期"
        else:
            return "衰减期"


# ══════════════════════════════════════════════════════════════════════════════
# 4. 模块三：优先级消息保留 (PriorityScorer + PriorityMessageQueue)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 为消息计算优先级分数，空间不足时优先淘汰低分消息
# 评分维度: 关键词权重 (30%) + 意图信号 (35%) + 时间衰减 (20%) + 角色权重 (15%)
# 淘汰策略: 最小堆 + pinned 保护机制
#
# 参考规格书 §5
# ══════════════════════════════════════════════════════════════════════════════

class PriorityScorer:
    """
    优先级评分器

    为每条消息计算一个 [0, 1] 区间的优先级分数。
    分数由四个维度加权得出:
      - 关键词权重 (keyword_weight): 消息中包含的关键词类别
      - 意图信号 (intent_signal):    消息表达的用户意图类型
      - 时间衰减 (recency_score):    消息距今的时间距离
      - 角色权重 (role_weight):      消息发送者的角色

    评分公式:
      priority = w_kw × keyword + w_intent × intent + w_recency × recency + w_role × role
    """

    # ---- 关键词类别定义 ----
    # critical: 涉及安全、故障等不可丢失的信息
    CRITICAL_KEYWORDS = [
        "需求", "必须", "不允许", "安全", "密码",
        "bug", "故障", "紧急", "严重", "崩溃",
    ]
    # important: 涉及方案决策、计划等重要但非紧急的信息
    IMPORTANT_KEYWORDS = [
        "建议", "方案", "设计", "计划", "截止日期",
        "预算", "架构", "选型", "优化", "性能",
    ]
    # trivial: 寒暄、确认等低价值信息
    TRIVIAL_KEYWORDS = [
        "嗯", "好的", "收到", "谢谢", "了解",
        "哈哈", "呵呵", "哦",
    ]

    # ---- 意图分类模式 ----
    # 使用简单的关键词/标点模式进行意图分类（无需 NLU 模型）
    INTENT_PATTERNS = {
        "requirement": [   # 需求声明 — 信号分 1.0
            "需要", "要求", "必须", "想要", "希望有",
            "功能", "实现", "支持",
        ],
        "decision": [      # 决策确认 — 信号分 0.9
            "决定", "确定", "就用", "采用", "选择",
            "方案A", "方案B", "最终",
        ],
        "question": [      # 问题提出 — 信号分 0.8
            "怎么", "如何", "为什么", "是否", "能不能",
            "？", "?",
        ],
        "info": [          # 信息补充 — 信号分 0.6
            "补充", "另外", "对了", "其实", "实际上",
            "注意", "提示",
        ],
        "chat": [          # 闲聊确认 — 信号分 0.1
            "嗯", "好的", "收到", "谢谢", "明白",
            "了解", "知道了",
        ],
    }

    # 意图类型对应的信号分
    INTENT_SCORES = {
        "requirement": 1.0,
        "decision": 0.9,
        "question": 0.8,
        "info": 0.6,
        "chat": 0.1,
    }

    # 角色对应的权重
    ROLE_WEIGHTS = {
        "user": 0.8,       # 用户消息更重要（包含需求）
        "assistant": 0.5,  # 助手消息为辅助参考
        "system": 1.0,     # 系统消息始终保留
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        recency_decay: float = 0.05,
        pin_threshold: float = 0.85,
        max_pinned: int = 5,
        eviction_threshold: float = 0.25,
    ):
        """
        初始化优先级评分器

        Args:
            weights:            四维权重 (默认: kw=0.30, intent=0.35, recency=0.20, role=0.15)
            recency_decay:      时间衰减系数 α (越大衰减越快)
            pin_threshold:      钉住阈值 — 分数 ≥ 此值的消息自动钉住
            max_pinned:         最大钉住消息数（超出时最低分降级）
            eviction_threshold: 淘汰阈值 — 低于此分的消息优先被淘汰
        """
        self.weights = weights or {
            "keyword": 0.30,
            "intent": 0.35,
            "recency": 0.20,
            "role": 0.15,
        }
        self.recency_decay = recency_decay
        self.pin_threshold = pin_threshold
        self.max_pinned = max_pinned
        self.eviction_threshold = eviction_threshold

    def _keyword_weight(self, content: str) -> float:
        """
        计算关键词维度权重

        根据消息中命中的最高关键词类别返回对应权重:
          - critical 命中 → 1.0
          - important 命中 → 0.7
          - trivial 全命中 → 0.1
          - 其他 → 0.3 (默认正常消息)

        Args:
            content: 消息文本

        Returns:
            float: [0, 1] 区间的关键词权重
        """
        # 优先检查最高级别
        if any(kw in content for kw in self.CRITICAL_KEYWORDS):
            return 1.0
        if any(kw in content for kw in self.IMPORTANT_KEYWORDS):
            return 0.7
        if all(kw in content for kw in content.split() if any(t in content for t in self.TRIVIAL_KEYWORDS)):
            # 如果消息主要由 trivial 词组成
            trivial_ratio = sum(1 for t in self.TRIVIAL_KEYWORDS if t in content) / max(len(self.TRIVIAL_KEYWORDS), 1)
            if trivial_ratio > 0.3:
                return 0.1
        return 0.3  # 默认: 正常消息

    def _intent_signal(self, content: str) -> Tuple[str, float]:
        """
        识别消息意图并返回对应信号分

        通过简单的关键词模式匹配进行意图分类:
          1. 依次检查每种意图的关键词列表
          2. 返回第一个匹配到的意图类型
          3. 如果都未匹配，默认归类为 "info" (信息)

        优先级: requirement > decision > question > info > chat

        Args:
            content: 消息文本

        Returns:
            Tuple[str, float]: (意图类型, 信号分)
        """
        # 按优先级顺序检查意图模式
        for intent_type in ["requirement", "decision", "question", "info", "chat"]:
            patterns = self.INTENT_PATTERNS[intent_type]
            # 只要命中一个模式关键词就判定为该意图
            if any(p in content for p in patterns):
                return intent_type, self.INTENT_SCORES[intent_type]

        # 默认归类为 "info"
        return "info", 0.6

    def _recency_score(self, messages_since: int) -> float:
        """
        计算时间衰减分数

        使用倒数衰减公式:
          score = 1.0 / (1 + α × messages_since)

        特性:
          - 最新消息: score = 1.0
          - 5 条前:   score ≈ 0.80 (α=0.05)
          - 20 条前:  score ≈ 0.50
          - 永远不会降到 0（渐进趋近）

        Args:
            messages_since: 该消息距今的消息条数

        Returns:
            float: (0, 1] 区间的时间衰减分数
        """
        return 1.0 / (1 + self.recency_decay * messages_since)

    def _role_weight(self, role: str) -> float:
        """
        获取角色权重

        Args:
            role: 消息角色 ("user" | "assistant" | "system")

        Returns:
            float: 角色权重
        """
        return self.ROLE_WEIGHTS.get(role, 0.5)

    def score(self, message: Message, total_messages: int) -> float:
        """
        计算消息的综合优先级分数

        执行流程:
          1. 提取四个维度的分数
          2. 加权求和
          3. 将结果写回 message.priority_score
          4. 将识别的意图写回 message.intent_type
          5. 如果分数 ≥ pin_threshold，自动钉住

        Args:
            message:        待评分的消息对象
            total_messages: 当前缓冲区中的总消息数（用于计算 messages_since）

        Returns:
            float: [0, 1] 区间的综合优先级分数
        """
        content = message.content

        # 维度 1: 关键词权重
        kw_score = self._keyword_weight(content)

        # 维度 2: 意图信号
        intent_type, intent_score = self._intent_signal(content)
        message.intent_type = intent_type

        # 维度 3: 时间衰减（消息距今的距离）
        # 假设消息按时间顺序排列，最后一条的 messages_since = 0
        messages_since = max(0, total_messages - 1)
        recency = self._recency_score(messages_since)

        # 维度 4: 角色权重
        role_w = self._role_weight(message.role)

        # 加权求和
        final = (
            self.weights["keyword"] * kw_score
            + self.weights["intent"] * intent_score
            + self.weights["recency"] * recency
            + self.weights["role"] * role_w
        )

        # 写回消息对象
        message.priority_score = round(final, 4)

        # 自动钉住: 分数超过阈值的消息标记为不可淘汰
        if final >= self.pin_threshold:
            message.is_pinned = True

        return final

    def rescore_recency(self, messages: List[Message]) -> None:
        """
        重新计算所有消息的时间衰减分数

        当消息列表发生变化（新增/删除）时调用，确保每条消息的
        recency_score 反映其在列表中的实际位置。

        算法: 从最新消息开始，messages_since 从 0 递增

        Args:
            messages: 按时间顺序排列的消息列表（最新在末尾）
        """
        total = len(messages)
        for i, msg in enumerate(messages):
            messages_since = total - 1 - i  # 最后一条 = 0, 倒数第二条 = 1, ...
            recency = self._recency_score(messages_since)

            # 重新计算综合分数（只更新 recency 维度）
            kw = self._keyword_weight(msg.content)
            _, intent_s = self._intent_signal(msg.content)
            role_w = self._role_weight(msg.role)

            new_score = (
                self.weights["keyword"] * kw
                + self.weights["intent"] * intent_s
                + self.weights["recency"] * recency
                + self.weights["role"] * role_w
            )
            msg.priority_score = round(new_score, 4)


class PriorityMessageQueue:
    """
    优先级消息队列 — 基于最小堆的高效淘汰结构

    数据结构:
      - _heap: 最小堆，按 priority_score 排序，支持 O(log n) 的插入和淘汰
      - _pinned_set: 钉住消息 ID 集合，O(1) 查找
      - _message_map: 消息 ID → Message 的映射，O(1) 查找

    操作复杂度:
      - insert:       O(log n)
      - evict_lowest: O(log n)
      - get_top_k:    O(k log n)
      - pin/unpin:    O(1)

    淘汰规则:
      1. 跳过所有 pinned 消息
      2. 优先淘汰 priority_score < eviction_threshold 的消息
      3. 如果所有消息分数都 ≥ threshold，淘汰最旧的非 pinned 消息
    """

    def __init__(self, eviction_threshold: float = 0.25):
        self._heap: List[Tuple[float, str, Message]] = []  # (score, id, message)
        self._pinned_set: set = set()     # 钉住消息的 ID 集合
        self._message_map: Dict[str, Message] = {}  # ID → Message 快速查找
        self._counter = 0  # 用于堆排序的递增计数器，确保同分时 FIFO
        self.eviction_threshold = eviction_threshold

    def insert(self, message: Message) -> None:
        """
        插入消息到队列

        使用 (priority_score, counter, message) 三元组入堆。
        counter 确保同分时按插入顺序排列（FIFO）。

        Args:
            message: 待插入的消息（需已设置 priority_score）
        """
        score = message.priority_score or 0.0
        self._counter += 1
        heapq.heappush(self._heap, (score, self._counter, message))
        self._message_map[message.id] = message

        # 如果是 pinned 消息，加入 pinned 集合
        if message.is_pinned:
            self._pinned_set.add(message.id)

    def evict_lowest(self) -> Optional[Message]:
        """
        淘汰优先级最低的非 pinned 消息

        策略:
          1. 从堆顶取出分数最低的消息
          2. 如果是 pinned 消息，跳过并尝试下一条
          3. 如果所有消息都是 pinned，返回 None

        Returns:
            Optional[Message]: 被淘汰的消息，如果无法淘汰则返回 None
        """
        # 临时存放跳过的 pinned 消息
        skipped = []
        evicted = None

        while self._heap:
            score, counter, msg = heapq.heappop(self._heap)

            # 检查消息是否仍在 message_map 中（可能已被外部移除）
            if msg.id not in self._message_map:
                continue

            # 跳过 pinned 消息
            if msg.id in self._pinned_set or msg.is_pinned:
                skipped.append((score, counter, msg))
                continue

            # 找到可淘汰的消息
            evicted = msg
            del self._message_map[msg.id]
            break

        # 将跳过的 pinned 消息放回堆中
        for item in skipped:
            heapq.heappush(self._heap, item)

        return evicted

    def get_top_k(self, k: int) -> List[Message]:
        """
        获取 top-k 高优先级消息（不修改堆）

        实现: 将堆中所有非淘汰消息按分数降序排列，取前 k 条。

        Args:
            k: 需要获取的消息数量

        Returns:
            List[Message]: 分数最高的 k 条消息
        """
        # 收集所有有效消息
        valid_messages = [
            msg for _, _, msg in self._heap
            if msg.id in self._message_map
        ]
        # 按 priority_score 降序排列
        valid_messages.sort(key=lambda m: m.priority_score or 0.0, reverse=True)
        return valid_messages[:k]

    def pin(self, message_id: str) -> bool:
        """
        钉住指定消息（使其不可被自动淘汰）

        Args:
            message_id: 消息 ID

        Returns:
            bool: 是否成功（消息不存在时返回 False）
        """
        if message_id in self._message_map:
            self._pinned_set.add(message_id)
            self._message_map[message_id].is_pinned = True
            return True
        return False

    def unpin(self, message_id: str) -> bool:
        """
        取消钉住指定消息

        Args:
            message_id: 消息 ID

        Returns:
            bool: 是否成功
        """
        if message_id in self._pinned_set:
            self._pinned_set.discard(message_id)
            if message_id in self._message_map:
                self._message_map[message_id].is_pinned = False
            return True
        return False

    @property
    def size(self) -> int:
        """当前队列中的有效消息数"""
        return len(self._message_map)

    @property
    def pinned_count(self) -> int:
        """当前钉住的消息数"""
        return len(self._pinned_set)

    def get_all_messages(self) -> List[Message]:
        """获取队列中所有有效消息（按时间顺序）"""
        return [msg for msg in self._message_map.values()]


# ══════════════════════════════════════════════════════════════════════════════
# 5. 模块四：话题切换检测与记忆清理 (TopicDetector)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 检测对话话题是否发生切换，触发旧话题归档和新话题初始化
# 算法: 关键词漂移 (Jaccard) + 语义相似度 (bigram overlap) 复合判定
# 响应: 归档旧话题 → 清理活跃记忆 → 初始化新话题上下文
#
# 参考规格书 §6
# ══════════════════════════════════════════════════════════════════════════════

class TopicDetector:
    """
    话题切换检测器

    使用复合算法检测对话话题是否发生切换:
      1. 关键词漂移检测: 计算当前消息与近期消息的关键词 Jaccard 相似度
      2. 语义相似度检测: 基于 bigram 重叠的轻量语义相似度
      3. 复合判定: 综合两个维度的结果做出最终决策

    配置参数:
        method:                  检测方法 (composite | keyword_only | semantic_only)
        keyword_drift_threshold: 关键词漂移阈值 (J < 此值 → 话题切换)
        semantic_threshold:      语义相似度硬阈值 (sim < 此值 → 话题切换)
        soft_threshold:          语义相似度软阈值 (sim < 此值 → 需二次确认)
        recent_window:           参考的最近消息数
    """

    def __init__(
        self,
        method: str = "composite",
        keyword_drift_threshold: float = 0.15,
        semantic_threshold: float = 0.4,
        soft_threshold: float = 0.55,
        recent_window: int = 5,
    ):
        self.method = method
        self.keyword_drift_threshold = keyword_drift_threshold
        self.semantic_threshold = semantic_threshold
        self.soft_threshold = soft_threshold
        self.recent_window = recent_window

    def _extract_keywords(self, text: str) -> set:
        """
        从文本中提取关键词集合

        使用简单的分词策略:
          1. 提取所有中文字符序列（≥2 字符）
          2. 提取所有英文单词（≥2 字符）
          3. 过滤停用词

        这是一个轻量实现，不依赖 jieba 等外部分词库。
        对于生产环境，建议替换为 jieba + TF-IDF 或 TextRank。

        Args:
            text: 输入文本

        Returns:
            set: 关键词字符串集合
        """
        # 停用词列表 — 高频但无实际语义的词汇
        stopwords = {
            "这个", "那个", "什么", "怎么", "可以", "我们", "他们",
            "一个", "不是", "就是", "还是", "或者", "因为", "所以",
            "但是", "然后", "如果", "虽然", "已经", "应该", "需要",
            "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does",
        }

        keywords = set()

        # 提取中文词组（2-4 字符的连续中文）
        chinese_words = re.findall(r'[一-鿿]{2,4}', text)
        for word in chinese_words:
            if word not in stopwords:
                keywords.add(word)

        # 提取英文单词
        english_words = re.findall(r'[a-zA-Z]{2,}', text)
        for word in english_words:
            lower_word = word.lower()
            if lower_word not in stopwords:
                keywords.add(lower_word)

        return keywords

    def _jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """
        计算两个集合的 Jaccard 相似度

        Jaccard = |A ∩ B| / |A ∪ B|

        值域: [0, 1]
          - 0: 完全不同
          - 1: 完全相同

        Args:
            set_a: 集合 A
            set_b: 集合 B

        Returns:
            float: Jaccard 相似度
        """
        if not set_a and not set_b:
            return 1.0  # 两个空集视为相同
        if not set_a or not set_b:
            return 0.0

        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def _keyword_drift_detect(
        self, current_msg: Message, recent_messages: List[Message]
    ) -> Tuple[float, bool]:
        """
        基于关键词漂移的话题切换检测

        算法步骤:
          1. 提取当前消息的关键词集合 KW_current
          2. 合并最近 N 条消息的关键词集合 KW_recent
          3. 计算 Jaccard 相似度 J
          4. 如果 J < keyword_drift_threshold → 判定为话题切换

        Args:
            current_msg:     当前消息
            recent_messages: 最近的 N 条消息

        Returns:
            Tuple[float, bool]: (Jaccard 相似度, 是否检测到漂移)
        """
        kw_current = self._extract_keywords(current_msg.content)

        # 合并近期消息的关键词
        kw_recent = set()
        for msg in recent_messages[-self.recent_window:]:
            kw_recent |= self._extract_keywords(msg.content)

        jaccard = self._jaccard_similarity(kw_current, kw_recent)
        drifted = jaccard < self.keyword_drift_threshold

        return jaccard, drifted

    def _semantic_similarity(
        self, current_msg: Message, recent_messages: List[Message]
    ) -> Tuple[float, bool]:
        """
        基于语义相似度的话题切换检测

        使用 bigram 重叠作为轻量语义相似度（不依赖嵌入模型）:
          1. 提取当前消息的 bigram 集合
          2. 计算与最近 N 条消息的平均 Jaccard 相似度
          3. 如果 avg_sim < semantic_threshold → 判定为话题切换

        Args:
            current_msg:     当前消息
            recent_messages: 最近的 N 条消息

        Returns:
            Tuple[float, bool]: (平均语义相似度, 是否检测到切换)
        """
        current_bigrams = ConditionalFilter._extract_bigrams(current_msg.content)
        if not current_bigrams:
            return 0.5, False  # 无法计算时返回中间值

        similarities = []
        for msg in recent_messages[-self.recent_window:]:
            msg_bigrams = ConditionalFilter._extract_bigrams(msg.content)
            if not msg_bigrams:
                continue
            sim = self._jaccard_similarity(current_bigrams, msg_bigrams)
            similarities.append(sim)

        if not similarities:
            return 0.5, False

        avg_sim = sum(similarities) / len(similarities)
        switched = avg_sim < self.semantic_threshold

        return avg_sim, switched

    def detect(
        self, current_msg: Message, recent_messages: List[Message], current_topic: TopicContext
    ) -> TopicResult:
        """
        综合话题切换检测

        复合判定逻辑 (参考规格书 §6.2.3):
          1. 并行计算关键词漂移 J 和语义相似度 avg_sim
          2. 根据 method 选择判定策略:
             - composite:    两个维度综合判定
             - keyword_only: 仅使用关键词漂移
             - semantic_only: 仅使用语义相似度
          3. 当消息历史不足 recent_window 条时，默认不触发切换

        Args:
            current_msg:     当前消息
            recent_messages: 最近的对话历史
            current_topic:   当前活跃的话题上下文

        Returns:
            TopicResult: 检测结果（是否切换 + 话题 ID + 置信度）
        """
        # 历史消息不足时，不触发切换（新对话需要积累上下文）
        if len(recent_messages) < self.recent_window:
            return TopicResult(
                switched=False,
                topic_id=current_topic.topic_id,
                confidence=0.0,
                details={"reason": "历史消息不足"},
            )

        # --- 计算两个维度 ---
        jaccard, kw_drifted = self._keyword_drift_detect(current_msg, recent_messages)
        avg_sim, sem_switched = self._semantic_similarity(current_msg, recent_messages)

        # --- 根据检测方法做最终判定 ---
        switched = False
        confidence = 0.0

        if self.method == "keyword_only":
            # 仅依赖关键词漂移
            switched = kw_drifted
            confidence = 1.0 - jaccard  # Jaccard 越低，切换置信度越高

        elif self.method == "semantic_only":
            # 仅依赖语义相似度
            switched = sem_switched
            confidence = 1.0 - avg_sim

        else:  # composite — 复合判定
            # 复合判定规则 (参考规格书 §6.2.3 流程图):
            #   - 关键词漂移 + 语义低于硬阈值 → 确认切换
            #   - 关键词漂移 + 语义高于软阈值 → 不切换
            #   - 关键词未漂移 + 语义低于硬阈值 → 确认切换
            #   - 其他情况 → 不切换
            if kw_drifted and avg_sim < self.soft_threshold:
                switched = True
                confidence = (1.0 - jaccard + 1.0 - avg_sim) / 2
            elif not kw_drifted and sem_switched:
                switched = True
                confidence = 1.0 - avg_sim
            else:
                switched = False
                confidence = (jaccard + avg_sim) / 2

        # --- 生成结果 ---
        if switched:
            # 检测到话题切换 → 创建新话题 ID
            new_topic_id = f"topic_{uuid.uuid4().hex[:6]}"
            return TopicResult(
                switched=True,
                topic_id=new_topic_id,
                confidence=min(1.0, confidence),
                details={
                    "jaccard": round(jaccard, 3),
                    "avg_sim": round(avg_sim, 3),
                    "kw_drifted": kw_drifted,
                    "sem_switched": sem_switched,
                },
            )
        else:
            # 话题延续 → 保持当前话题 ID
            return TopicResult(
                switched=False,
                topic_id=current_topic.topic_id,
                confidence=confidence,
                details={
                    "jaccard": round(jaccard, 3),
                    "avg_sim": round(avg_sim, 3),
                },
            )

    def update_topic_keywords(self, topic: TopicContext, message: Message) -> None:
        """
        更新话题的关键词集合

        每当新消息归属某话题时，将该消息的关键词合并到话题的关键词集合中。
        用于后续的归档检索和话题匹配。

        Args:
            topic:   话题上下文
            message: 新消息
        """
        new_keywords = self._extract_keywords(message.content)
        # 合并关键词（去重）
        existing = set(topic.keywords)
        existing |= new_keywords
        topic.keywords = list(existing)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 模块五：会话缓冲区与缓冲窗口 (SessionBuffer)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 管理消息的存储、淘汰和检索
# 架构: 四区域缓冲 (System / Pinned / Active / Summary)
# 容量: 动态调节 — 前重后轻策略
# 溢出: 优先级淘汰 → LRU 淘汰 → 摘要压缩
#
# 参考规格书 §7
# ══════════════════════════════════════════════════════════════════════════════

class SessionBuffer:
    """
    会话缓冲区 — 对话记忆的核心存储容器

    四个存储区域:
      - system_zone:  系统消息（角色设定、系统提示），永不淘汰
      - pinned_zone:  高优先级消息（pinned），仅手动或降级淘汰
      - active_zone:  普通对话消息，按优先级/LRU 淘汰
      - summary_zone: 压缩后的历史摘要，被新摘要替换

    容量管理:
      - buffer_capacity: 根据对话轮数动态调节的物理上限
      - token_budget:    LLM 上下文窗口的 token 预算

    窗口选择策略:
      1. 始终包含 system + summary + pinned
      2. 按 token 预算从最近消息向前填充 active
    """

    def __init__(
        self,
        max_buffer_size: int = 20,
        min_buffer_size: int = 5,
        warm_up_turns: int = 3,
        step_down: float = 0.5,
        max_pinned: int = 5,
        max_summaries: int = 2,
    ):
        """
        初始化会话缓冲区

        Args:
            max_buffer_size: 最大缓冲区容量（消息条数）
            min_buffer_size: 最小缓冲区容量
            warm_up_turns:   预热期轮数（期间全量分配）
            step_down:       预热后每轮减少的容量
            max_pinned:      最大钉住消息数
            max_summaries:   摘要区最大摘要条数
        """
        self.max_buffer_size = max_buffer_size
        self.min_buffer_size = min_buffer_size
        self.warm_up_turns = warm_up_turns
        self.step_down = step_down
        self.max_pinned = max_pinned
        self.max_summaries = max_summaries

        # --- 四个存储区域 ---
        self.system_zone: List[Message] = []    # 系统消息（不淘汰）
        self.pinned_zone: List[Message] = []    # 钉住消息
        self.active_zone: List[Message] = []    # 活跃消息
        self.summary_zone: List[str] = []       # 历史摘要

    def get_buffer_capacity(self, current_turn: int) -> int:
        """
        计算当前轮次的缓冲区容量

        前重后轻策略:
          - turn ≤ warm_up_turns: capacity = max_buffer_size (全量分配)
          - turn > warm_up_turns: capacity 线性递减
          - capacity 不低于 min_buffer_size

        Args:
            current_turn: 当前对话轮次

        Returns:
            int: 当前缓冲区容量
        """
        if current_turn <= self.warm_up_turns:
            return self.max_buffer_size

        # 线性递减: 每轮减少 step_down 条
        reduction = (current_turn - self.warm_up_turns) * self.step_down
        capacity = self.max_buffer_size - reduction
        return max(self.min_buffer_size, int(capacity))

    def insert(self, message: Message, current_turn: int) -> List[str]:
        """
        将消息插入缓冲区

        插入策略:
          1. system 角色消息 → 直接放入 system_zone
          2. pinned 消息 → 放入 pinned_zone（检查上限）
          3. 普通消息 → 放入 active_zone（检查容量，必要时淘汰）

        Args:
            message:      待插入的消息
            current_turn: 当前对话轮次

        Returns:
            List[str]: 执行的操作列表（如 ["inserted", "evicted_msg_123"]）
        """
        actions = []

        # --- 系统消息: 直接存入，不占 active 容量 ---
        if message.role == "system":
            self.system_zone.append(message)
            actions.append("inserted_to_system_zone")
            return actions

        # --- Pinned 消息: 存入 pinned_zone ---
        if message.is_pinned:
            if len(self.pinned_zone) >= self.max_pinned:
                # pinned 区已满: 将最低分的 pinned 消息降级到 active
                lowest_pinned = min(
                    self.pinned_zone,
                    key=lambda m: m.priority_score or 0.0
                )
                lowest_pinned.is_pinned = False
                self.pinned_zone.remove(lowest_pinned)
                self.active_zone.append(lowest_pinned)
                actions.append(f"demoted_pinned_{lowest_pinned.id}")

            self.pinned_zone.append(message)
            actions.append("inserted_to_pinned_zone")
            return actions

        # --- 普通消息: 存入 active_zone ---
        capacity = self.get_buffer_capacity(current_turn)
        active_capacity = capacity - len(self.pinned_zone)

        # 容量充足: 直接插入
        if len(self.active_zone) < max(1, active_capacity):
            self.active_zone.append(message)
            actions.append("inserted_to_active_zone")
        else:
            # 容量不足: 需要先淘汰一条消息
            evicted = self._evict_from_active()
            if evicted:
                actions.append(f"evicted_{evicted.id}(score={evicted.priority_score:.2f})")
            self.active_zone.append(message)
            actions.append("inserted_to_active_zone")

        return actions

    def _evict_from_active(self) -> Optional[Message]:
        """
        从 active_zone 中淘汰一条消息

        淘汰优先级:
          1. 分数最低的非 pinned 消息
          2. 如果所有消息分数相同，淘汰最旧的

        Returns:
            Optional[Message]: 被淘汰的消息
        """
        if not self.active_zone:
            return None

        # 找到分数最低的消息
        evicted = min(
            self.active_zone,
            key=lambda m: (m.priority_score or 0.0, m.timestamp)
        )
        self.active_zone.remove(evicted)
        return evicted

    def clear_active_zone(self) -> List[Message]:
        """
        清空活跃消息区（话题切换时使用）

        保留 system_zone 和 pinned_zone，仅清空 active_zone。
        返回被清空的消息列表（用于归档）。

        Returns:
            List[Message]: 被清空的消息列表
        """
        cleared = list(self.active_zone)
        self.active_zone.clear()
        return cleared

    def add_summary(self, summary: str) -> None:
        """
        添加摘要到摘要区

        如果摘要区已满（≥ max_summaries），先合并已有摘要再存入。

        Args:
            summary: 摘要文本
        """
        if len(self.summary_zone) >= self.max_summaries:
            # 合并已有摘要: 将所有摘要拼接为一条
            merged = " | ".join(self.summary_zone)
            self.summary_zone.clear()
            self.summary_zone.append(f"[历史摘要] {merged}")

        self.summary_zone.append(summary)

    def get_all_active_messages(self) -> List[Message]:
        """
        获取所有活跃区域的消息（按区域优先级排序）

        返回顺序: system → summary(as message) → pinned → active

        Returns:
            List[Message]: 所有活跃消息
        """
        result = list(self.system_zone)
        result.extend(self.pinned_zone)
        result.extend(self.active_zone)
        return result

    def get_usage(self, current_turn: int) -> str:
        """
        获取缓冲区使用情况字符串

        Args:
            current_turn: 当前对话轮次

        Returns:
            str: 如 "8/20" 格式的使用情况
        """
        total = len(self.active_zone) + len(self.pinned_zone)
        capacity = self.get_buffer_capacity(current_turn)
        return f"{total}/{capacity}"

    def get_state(self, current_turn: int) -> Dict[str, Any]:
        """
        获取缓冲区完整状态信息

        Args:
            current_turn: 当前对话轮次

        Returns:
            Dict: 包含各区域消息数、容量、摘要数等
        """
        return {
            "system_count": len(self.system_zone),
            "pinned_count": len(self.pinned_zone),
            "active_count": len(self.active_zone),
            "summary_count": len(self.summary_zone),
            "capacity": self.get_buffer_capacity(current_turn),
            "usage": self.get_usage(current_turn),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. 归档存储 (ArchiveStore)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 存储已完成话题的记忆归档，支持关键词检索和按话题/时间检索
# 实现: 简单的内存存储 + 关键词匹配检索（生产环境可替换为向量数据库）
#
# 参考规格书 §6.4, §9.1
# ══════════════════════════════════════════════════════════════════════════════

class ArchiveStore:
    """
    归档存储 — 保存已完成话题的记忆

    当话题切换时，旧话题的所有消息被打包为一个 ArchiveEntry 存入此处。
    支持以下检索方式:
      - 关键词检索: search() — 按关键词匹配相关归档
      - 话题检索:   get_by_topic() — 按话题 ID 精确查找
      - 时间检索:   get_recent() — 获取最近的归档

    生产环境建议:
      替换为向量数据库（如 FAISS、Pinecone）以支持语义检索。
    """

    def __init__(self):
        self.entries: List[ArchiveEntry] = []

    def store(self, entry: ArchiveEntry) -> None:
        """
        存储一个归档条目

        Args:
            entry: 归档条目（包含话题摘要、消息列表和关键词）
        """
        self.entries.append(entry)

    def search(self, query: str, top_k: int = 3) -> List[ArchiveEntry]:
        """
        按关键词检索相关归档

        算法:
          1. 从查询文本中提取关键词
          2. 计算查询关键词与每个归档条目的关键词重叠度
          3. 同时检查归档摘要中是否包含查询文本
          4. 按匹配分数降序排列，返回 top_k

        Args:
            query:  查询文本
            top_k:  返回的最大结果数

        Returns:
            List[ArchiveEntry]: 最相关的归档条目列表
        """
        if not self.entries:
            return []

        # 提取查询关键词
        query_keywords = set(re.findall(r'[一-鿿]{2,4}', query))
        query_keywords |= {w.lower() for w in re.findall(r'[a-zA-Z]{2,}', query)}

        scored_entries = []
        for entry in self.entries:
            score = 0.0

            # 关键词匹配
            entry_keywords = set(entry.keywords)
            if query_keywords and entry_keywords:
                overlap = query_keywords & entry_keywords
                score += len(overlap) / max(len(query_keywords), 1)

            # 摘要文本匹配
            if entry.summary:
                for kw in query_keywords:
                    if kw in entry.summary:
                        score += 0.5

            if score > 0:
                scored_entries.append((score, entry))

        # 按分数降序排列
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:top_k]]

    def get_by_topic(self, topic_id: str) -> Optional[ArchiveEntry]:
        """
        按话题 ID 精确查找归档

        Args:
            topic_id: 话题 ID

        Returns:
            Optional[ArchiveEntry]: 匹配的归档条目
        """
        for entry in self.entries:
            if entry.topic_id == topic_id:
                return entry
        return None

    def get_recent(self, limit: int = 3) -> List[ArchiveEntry]:
        """
        获取最近的归档条目

        Args:
            limit: 返回的最大条目数

        Returns:
            List[ArchiveEntry]: 最近的归档条目（按时间降序）
        """
        sorted_entries = sorted(
            self.entries,
            key=lambda e: e.archived_at,
            reverse=True
        )
        return sorted_entries[:limit]

    @property
    def total_archived_messages(self) -> int:
        """归档中保存的总消息数"""
        return sum(len(e.messages) for e in self.entries)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 上下文组装器 (ContextAssembler)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 从缓冲区 + 归档中构建 LLM 可用的上下文消息列表
# 策略: 始终包含 system + summary + pinned → 按 token 预算填充 active
#
# 参考规格书 §7.4
# ══════════════════════════════════════════════════════════════════════════════

class ContextAssembler:
    """
    上下文组装器

    将会话缓冲区中的消息组装成 LLM 可接受的输入格式。
    按优先级填充:
      1. System Zone — 始终包含（系统提示、角色设定）
      2. Summary Zone — 始终包含（历史摘要）
      3. Pinned Zone — 始终包含（高优先级消息）
      4. Active Zone — 按 token 预算从最近消息向前填充
      5. Archive Summaries — 如果仍有剩余预算，补充归档摘要
    """

    def __init__(self, chars_per_token: float = 1.5):
        """
        Args:
            chars_per_token: 每个 token 对应的平均字符数
                             中文约 1.5 字符/token，英文约 4 字符/token
        """
        self.chars_per_token = chars_per_token

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的 token 数

        简单的字符级估算:
          - 中文字符: 约 1.5 字符/token
          - 英文单词: 约 0.75 词/token (即约 4 字符/token)
          - 混合文本: 取加权平均

        Args:
            text: 输入文本

        Returns:
            int: 估算的 token 数
        """
        # 统计中文字符数和英文单词数
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))

        # 中文 token 估算
        chinese_tokens = chinese_chars / self.chars_per_token

        # 英文 token 估算 (约 0.75 词/token)
        english_tokens = english_words / 0.75

        # 标点和其他字符
        other_chars = len(text) - chinese_chars - sum(
            len(w) for w in re.findall(r'[a-zA-Z]+', text)
        )
        other_tokens = other_chars / 2.0  # 标点约 2 字符/token

        return int(chinese_tokens + english_tokens + other_tokens) + 1

    def assemble(
        self,
        buffer: SessionBuffer,
        archive: ArchiveStore,
        token_budget: int = 4000,
    ) -> List[Dict[str, str]]:
        """
        组装 LLM 上下文消息列表

        填充顺序:
          1. system 消息 → 始终全部包含
          2. summary → 包装为 system 消息，始终包含
          3. pinned 消息 → 始终全部包含
          4. active 消息 → 从最新到最旧，逐个填充直到 token 预算耗尽
          5. archive 摘要 → 如果仍有预算，补充最近的归档摘要

        Args:
            buffer:       会话缓冲区
            archive:      归档存储
            token_budget: token 预算上限

        Returns:
            List[Dict[str, str]]: LLM 消息列表 [{"role": "...", "content": "..."}]
        """
        result = []
        used_tokens = 0

        # --- 1. System Zone: 始终全部包含 ---
        for msg in buffer.system_zone:
            tokens = self.estimate_tokens(msg.content)
            result.append({"role": msg.role, "content": msg.content})
            used_tokens += tokens

        # --- 2. Summary Zone: 包装为 system 消息 ---
        if buffer.summary_zone:
            combined_summary = "\n".join(buffer.summary_zone)
            summary_content = f"[对话历史摘要]\n{combined_summary}"
            tokens = self.estimate_tokens(summary_content)
            if used_tokens + tokens <= token_budget:
                result.append({"role": "system", "content": summary_content})
                used_tokens += tokens

        # --- 3. Pinned Zone: 始终全部包含 ---
        for msg in buffer.pinned_zone:
            tokens = self.estimate_tokens(msg.content)
            if used_tokens + tokens <= token_budget:
                result.append({"role": msg.role, "content": msg.content})
                used_tokens += tokens

        # --- 4. Active Zone: 从最新到最旧填充 ---
        # 反转使最新消息排在前面（优先保留）
        remaining_budget = token_budget - used_tokens
        active_messages = list(reversed(buffer.active_zone))

        # 先收集能放下的消息，然后按原始顺序排列
        selected_active = []
        for msg in active_messages:
            tokens = self.estimate_tokens(msg.content)
            if remaining_budget >= tokens:
                selected_active.append(msg)
                remaining_budget -= tokens
            else:
                break  # 预算不足，停止填充

        # 恢复时间顺序（最旧 → 最新）
        selected_active.reverse()
        for msg in selected_active:
            result.append({"role": msg.role, "content": msg.content})

        # --- 5. Archive Summaries: 补充归档摘要 ---
        if remaining_budget > 100:  # 至少需要 100 tokens 的预算才有意义
            recent_archives = archive.get_recent(limit=2)
            for entry in recent_archives:
                if entry.summary:
                    archive_content = f"[历史话题: {entry.topic_id}] {entry.summary}"
                    tokens = self.estimate_tokens(archive_content)
                    if remaining_budget >= tokens:
                        result.append({"role": "system", "content": archive_content})
                        remaining_budget -= tokens

        return result


# ══════════════════════════════════════════════════════════════════════════════
# 9. 动态记忆流水线编排器 (DynamicMemoryPipeline)
# ══════════════════════════════════════════════════════════════════════════════
#
# 功能: 串联所有模块，提供统一的消息处理和上下文获取接口
# 流程: 消息 → 过滤 → 评分 → 话题检测 → 深度调节 → 缓冲存储
#
# 参考规格书 §8
# ══════════════════════════════════════════════════════════════════════════════

class DynamicMemoryPipeline:
    """
    动态记忆流水线编排器

    将五大核心模块串联为完整的消息处理流水线:
      ① 条件过滤器 → ② 优先级评分器 → ③ 话题检测器 → ④ 深度调节器 → ⑤ 缓冲区管理器

    使用方式:
        pipeline = DynamicMemoryPipeline()
        result = pipeline.process_message("user", "我需要一个登录功能")
        context = pipeline.get_context(token_budget=4000)

    所有模块均可通过构造函数替换为自定义实现，也可使用默认配置。
    """

    def __init__(
        self,
        conditional_filter: Optional[ConditionalFilter] = None,
        priority_scorer: Optional[PriorityScorer] = None,
        topic_detector: Optional[TopicDetector] = None,
        depth_adjuster: Optional[DynamicDepthAdjuster] = None,
        session_buffer: Optional[SessionBuffer] = None,
        archive_store: Optional[ArchiveStore] = None,
        context_assembler: Optional[ContextAssembler] = None,
        llm: Optional[ChatOpenAI] = None,
        enable_filter: bool = True,
        enable_priority: bool = True,
        enable_topic_detection: bool = True,
        enable_depth_adjust: bool = True,
    ):
        """
        初始化流水线 — 所有模块可选，均提供合理默认值

        Args:
            conditional_filter:     条件过滤器实例（None 使用默认配置）
            priority_scorer:        优先级评分器实例
            topic_detector:         话题检测器实例
            depth_adjuster:         深度调节器实例
            session_buffer:         会话缓冲区实例
            archive_store:          归档存储实例
            context_assembler:      上下文组装器实例
            llm:                    LLM 实例（用于摘要压缩）
            enable_filter:          是否启用条件过滤
            enable_priority:        是否启用优先级评分
            enable_topic_detection: 是否启用话题检测
            enable_depth_adjust:    是否启用深度调节
        """
        # 初始化各模块（使用默认配置或传入的实例）
        self.filter = conditional_filter or ConditionalFilter()
        self.scorer = priority_scorer or PriorityScorer()
        self.detector = topic_detector or TopicDetector()
        self.adjuster = depth_adjuster or DynamicDepthAdjuster()
        self.buffer = session_buffer or SessionBuffer()
        self.archive = archive_store or ArchiveStore()
        self.assembler = context_assembler or ContextAssembler()
        self.llm = llm or llm_light

        # 模块启用开关
        self.enable_filter = enable_filter
        self.enable_priority = enable_priority
        self.enable_topic_detection = enable_topic_detection
        self.enable_depth_adjust = enable_depth_adjust

        # --- 运行时状态 ---
        self.current_turn = 0                    # 当前对话轮次
        self.current_topic = TopicContext()      # 当前活跃话题
        self.all_messages: List[Message] = []    # 所有消息历史（含被过滤的）
        self.priority_queue = PriorityMessageQueue(
            eviction_threshold=self.scorer.eviction_threshold
        )

        # 摘要压缩的 LCEL 链（用于压缩旧消息为摘要）
        self._summary_chain = (
            ChatPromptTemplate.from_template(
                "请将以下对话内容压缩为一段简洁的摘要（不超过100字），"
                "保留关键信息和决策要点:\n\n{messages}"
            )
            | self.llm
            | StrOutputParser()
        )

    def process_message(self, role: str, content: str) -> ProcessResult:
        """
        处理一条消息 — 流水线的核心入口

        处理流程:
          1. 创建 Message 对象，分配轮次号
          2. [可选] 条件过滤 → 拒绝低价值消息
          3. [可选] 优先级评分 → 计算 priority_score
          4. [可选] 话题检测 → 检测是否话题切换
          5. [可选] 深度调节 → 计算当前记忆深度
          6. 缓冲区存储 → 插入消息，必要时触发淘汰

        Args:
            role:    消息角色 ("user" | "assistant" | "system")
            content: 消息文本内容

        Returns:
            ProcessResult: 处理结果（存储状态、评分、操作列表等）
        """
        # 用户消息计数为一轮新的对话
        if role == "user":
            self.current_turn += 1

        # Step 0: 创建消息对象
        message = Message(
            role=role,
            content=content,
            turn_number=self.current_turn,
            topic_id=self.current_topic.topic_id,
        )
        self.all_messages.append(message)

        # 获取最近的消息列表（用于过滤和话题检测）
        recent = self.buffer.get_all_active_messages()

        actions = []

        # ── Step 1: 条件过滤 ──
        if self.enable_filter:
            filter_result = self.filter.evaluate(message, recent)
            message.filter_score = filter_result.score

            if not filter_result.passed:
                # 消息被过滤拒绝 — 不存入记忆
                return ProcessResult(
                    stored=False,
                    filter_score=filter_result.score,
                    topic_id=self.current_topic.topic_id,
                    current_depth=self.adjuster.get_depth(self.current_turn) if self.enable_depth_adjust else 0,
                    buffer_usage=self.buffer.get_usage(self.current_turn),
                    actions=[f"filtered_out({filter_result.details})"],
                )

            actions.append(f"filter_passed(score={filter_result.score:.2f})")

            # 标记需要摘要的消息
            if filter_result.details.get("needs_summary"):
                message.metadata["needs_summary"] = True

        # ── Step 2: 优先级评分 ──
        if self.enable_priority:
            total_msgs = len(self.buffer.active_zone) + 1
            priority = self.scorer.score(message, total_msgs)
            actions.append(f"priority_scored({priority:.2f}, intent={message.intent_type})")
        else:
            message.priority_score = 0.5  # 默认中等优先级

        # ── Step 3: 话题检测 ──
        topic_switched = False
        if self.enable_topic_detection and role != "system":
            topic_result = self.detector.detect(message, recent, self.current_topic)

            if topic_result.switched:
                topic_switched = True
                actions.append(f"topic_switch(旧={self.current_topic.topic_id}, 新={topic_result.topic_id})")

                # 执行话题切换: 归档旧话题 → 清理缓冲 → 初始化新话题
                self._handle_topic_switch(message, topic_result)

            # 更新话题关键词
            message.topic_id = topic_result.topic_id
            self.detector.update_topic_keywords(self.current_topic, message)

        # ── Step 4: 深度调节 → 触发淘汰 ──
        if self.enable_depth_adjust:
            target_depth = self.adjuster.get_depth(self.current_turn)
            phase = self.adjuster.get_phase(self.current_turn)
            actions.append(f"depth={target_depth}(phase={phase})")

            # 如果当前 active_zone 超过目标深度，触发批量淘汰
            while len(self.buffer.active_zone) > target_depth:
                evicted = self.buffer._evict_from_active()
                if evicted:
                    actions.append(f"depth_evicted_{evicted.id}")
                else:
                    break

        # ── Step 5: 缓冲区存储 ──
        buffer_actions = self.buffer.insert(message, self.current_turn)
        actions.extend(buffer_actions)

        # 将消息加入优先级队列
        if "inserted_to_active_zone" in buffer_actions or "inserted_to_pinned_zone" in buffer_actions:
            self.priority_queue.insert(message)

        # 重新计算所有 active 消息的时间衰减分数
        if self.enable_priority:
            self.scorer.rescore_recency(self.buffer.active_zone)

        return ProcessResult(
            stored=True,
            filter_score=message.filter_score or 0.0,
            priority_score=message.priority_score or 0.0,
            topic_id=message.topic_id,
            topic_switched=topic_switched,
            current_depth=self.adjuster.get_depth(self.current_turn) if self.enable_depth_adjust else len(self.buffer.active_zone),
            buffer_usage=self.buffer.get_usage(self.current_turn),
            actions=actions,
        )

    def _handle_topic_switch(self, new_message: Message, topic_result: TopicResult) -> None:
        """
        处理话题切换事件

        执行步骤:
          1. 归档旧话题的所有消息
          2. 生成旧话题摘要（使用 LLM）
          3. 清空 active_zone
          4. 创建新的 TopicContext
          5. 将旧话题摘要加入 summary_zone

        Args:
            new_message:  触发话题切换的新消息
            topic_result: 话题检测结果
        """
        # --- 1. 收集旧话题消息 ---
        old_messages = self.buffer.clear_active_zone()

        if not old_messages:
            # 没有旧消息需要归档
            self.current_topic = TopicContext(
                topic_id=topic_result.topic_id,
                start_turn=self.current_turn,
            )
            return

        # --- 2. 生成摘要 ---
        summary = self._generate_summary(old_messages)

        # --- 3. 归档旧话题 ---
        archive_entry = ArchiveEntry(
            topic_id=self.current_topic.topic_id,
            summary=summary,
            messages=old_messages,
            keywords=self.current_topic.keywords,
        )
        self.archive.store(archive_entry)

        # --- 4. 将摘要加入 summary_zone ---
        if summary:
            self.buffer.add_summary(f"[话题: {self.current_topic.topic_id}] {summary}")

        # --- 5. 标记旧话题为非活跃 ---
        self.current_topic.end_turn = self.current_turn
        self.current_topic.is_active = False

        # --- 6. 初始化新话题 ---
        self.current_topic = TopicContext(
            topic_id=topic_result.topic_id,
            start_turn=self.current_turn,
        )

    def _generate_summary(self, messages: List[Message]) -> str:
        """
        使用 LLM 生成消息列表的摘要

        通过 LCEL 链调用 LLM，将对话内容压缩为简洁摘要。
        摘要目标: ≤ 100 字，保留关键信息和决策要点。

        Args:
            messages: 待摘要的消息列表

        Returns:
            str: 生成的摘要文本（LLM 调用失败时返回简单拼接）
        """
        if not messages:
            return ""

        # 拼接消息文本
        message_text = "\n".join(
            f"{msg.role}: {msg.content}" for msg in messages
        )

        try:
            # 调用 LCEL 摘要链
            summary = self._summary_chain.invoke({"messages": message_text})
            return summary.strip()
        except Exception:
            # LLM 调用失败时的降级方案: 简单截取
            combined = " ".join(msg.content[:30] for msg in messages[:5])
            return f"[自动摘要] {combined}..."

    def get_context(self, token_budget: int = 4000) -> List[Dict[str, str]]:
        """
        获取 LLM 上下文 — 组装记忆中的消息为 LLM 输入

        Args:
            token_budget: token 预算上限

        Returns:
            List[Dict[str, str]]: LLM 消息列表
        """
        return self.assembler.assemble(
            buffer=self.buffer,
            archive=self.archive,
            token_budget=token_budget,
        )

    def get_state(self) -> Dict[str, Any]:
        """
        获取流水线完整状态

        Returns:
            Dict: 包含缓冲区状态、话题信息、归档统计等
        """
        return {
            "current_turn": self.current_turn,
            "current_topic": self.current_topic.topic_id,
            "topic_is_active": self.current_topic.is_active,
            "buffer": self.buffer.get_state(self.current_turn),
            "depth": self.adjuster.get_depth(self.current_turn) if self.enable_depth_adjust else None,
            "phase": self.adjuster.get_phase(self.current_turn) if self.enable_depth_adjust else None,
            "archive_count": len(self.archive.entries),
            "archived_messages": self.archive.total_archived_messages,
            "pinned_count": self.buffer.pinned_zone.__len__(),
            "total_processed": len(self.all_messages),
        }

    def search_archive(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索归档记忆

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            List[Dict]: 匹配的归档条目
        """
        results = self.archive.search(query, top_k)
        return [
            {
                "topic_id": entry.topic_id,
                "summary": entry.summary,
                "message_count": len(entry.messages),
                "archived_at": entry.archived_at.isoformat(),
                "keywords": entry.keywords[:10],
            }
            for entry in results
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 10. 演示程序
# ══════════════════════════════════════════════════════════════════════════════

def print_separator(title: str = "", char: str = "═", width: int = 70):
    """打印分隔线（带可选标题）"""
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char * side} {title} {char * side}")
    else:
        print(f"\n{char * width}")


def print_result(result: ProcessResult, content: str):
    """打印消息处理结果"""
    status = "✅ 已存储" if result.stored else "❌ 已过滤"
    print(f"  {status} | {content[:40]}{'...' if len(content) > 40 else ''}")
    print(f"    📊 过滤={result.filter_score:.2f} 优先={result.priority_score:.2f} "
          f"深度={result.current_depth} 缓冲={result.buffer_usage}")
    if result.actions:
        print(f"    🔧 {', '.join(result.actions[:3])}")


if __name__ == "__main__":
    print_separator("动态消息记忆策略 — 演示程序", "█")

    # ────────────────────────────────────────────────────────────────
    # 场景 1: 短对话（5 轮以内）— 展示消息过滤
    # ────────────────────────────────────────────────────────────────
    print_separator("场景 1: 短对话 — 消息过滤与存储")

    pipeline = DynamicMemoryPipeline()

    # 模拟 5 轮对话
    scenario1_messages = [
        ("user", "我想设计一个登录页面"),
        ("assistant", "好的，需要什么功能？"),
        ("user", "需要用户名、密码输入框和记住我选项"),
        ("assistant", "是否需要社交账号登录？"),
        ("user", "嗯"),  # ← 预期被过滤（过短 + suppress 关键词）
    ]

    for role, content in scenario1_messages:
        result = pipeline.process_message(role, content)
        print_result(result, content)

    print(f"\n  📍 场景1结果: 处理 {len(pipeline.all_messages)} 条, "
          f"Active={len(pipeline.buffer.active_zone)}, "
          f"Pinned={len(pipeline.buffer.pinned_zone)}")

    # 展示组装的上下文
    context = pipeline.get_context(token_budget=2000)
    print(f"  📋 上下文消息数: {len(context)} 条")

    # ────────────────────────────────────────────────────────────────
    # 场景 2: 话题切换 — 展示归档 + 清理 + 新话题
    # ────────────────────────────────────────────────────────────────
    print_separator("场景 2: 话题切换 — 归档与记忆清理")

    pipeline2 = DynamicMemoryPipeline()

    # 先讨论数据库设计（8 轮）
    db_messages = [
        ("user", "我们来讨论数据库设计方案"),
        ("assistant", "好的，您打算使用什么类型的数据库？"),
        ("user", "我计划用 PostgreSQL 作为主数据库"),
        ("assistant", "PostgreSQL 是个很好的选择，需要设计哪些表？"),
        ("user", "需要 users 表和 orders 表，users 包含用户名和密码"),
        ("assistant", "orders 表需要哪些字段？"),
        ("user", "orders 需要订单号、金额、状态和外键关联 users"),
        ("assistant", "建议给 orders 表加上 created_at 和 updated_at 时间戳"),
    ]

    print("  📌 话题: 数据库设计")
    for role, content in db_messages:
        result = pipeline2.process_message(role, content)
        print_result(result, content)

    # 切换到前端框架话题
    frontend_messages = [
        ("user", "好，数据库就这样。接下来聊聊前端框架选型的问题"),
        ("user", "你觉得 React 和 Vue 哪个更适合我们的项目？"),
        ("assistant", "这取决于团队技术栈和项目规模"),
    ]

    print("\n  📌 话题切换: 前端框架选型")
    for role, content in frontend_messages:
        result = pipeline2.process_message(role, content)
        print_result(result, content)
        if result.topic_switched:
            print(f"    🔄 话题切换触发! 旧话题已归档")

    # 展示归档状态
    state = pipeline2.get_state()
    print(f"\n  📍 场景2结果: 归档数={state['archive_count']}, "
          f"归档消息={state['archived_messages']}, "
          f"Active={state['buffer']['active_count']}")

    # 测试归档检索
    search_results = pipeline2.search_archive("数据库 PostgreSQL 表设计")
    if search_results:
        print(f"  🔍 归档检索 '数据库': 找到 {len(search_results)} 条")
        for r in search_results:
            print(f"    - [{r['topic_id']}] {r['summary'][:50]}...")

    # ────────────────────────────────────────────────────────────────
    # 场景 3: 长对话 — 展示深度衰减和优先级淘汰
    # ────────────────────────────────────────────────────────────────
    print_separator("场景 3: 长对话 — 深度衰减与优先级淘汰")

    pipeline3 = DynamicMemoryPipeline(
        depth_adjuster=DynamicDepthAdjuster(
            max_depth=10,       # 缩小以加速演示
            min_depth=3,
            decay_start_turn=3,
            floor_after_turn=12,
        ),
        session_buffer=SessionBuffer(
            max_buffer_size=10,
            min_buffer_size=3,
        ),
    )

    # 模拟 15 轮对话
    long_messages = [
        # 建立期 (轮次 1-3)
        ("user", "我要开发一个在线教育平台"),
        ("assistant", "很好，请问目标用户群体是什么？"),
        ("user", "面向大学生和职场新人，需要提供视频课程和在线考试功能"),
        # 衰减期开始 (轮次 4+)
        ("assistant", "视频课程需要支持直播还是录播？"),
        ("user", "必须支持录播，直播功能可以作为第二阶段的规划"),
        ("assistant", "考试系统需要什么类型的题目？"),
        ("user", "支持单选题、多选题和编程题三种类型"),
        ("assistant", "编程题的自动评判方案有什么想法吗？"),
        ("user", "可以用 Docker 容器来运行用户提交的代码，确保安全"),
        ("assistant", "这是个好方案，需要考虑并发执行的性能问题"),
        ("user", "嗯"),  # ← 低价值消息
        ("assistant", "建议用消息队列来处理代码执行任务"),
        ("user", "好的，用 RabbitMQ 还是 Kafka？"),
        ("assistant", "对于教育平台来说 RabbitMQ 足够了"),
        ("user", "决定了，就用 RabbitMQ，接下来讨论支付模块的集成方案"),
    ]

    for role, content in long_messages:
        result = pipeline3.process_message(role, content)
        phase = pipeline3.adjuster.get_phase(pipeline3.current_turn)
        depth = pipeline3.adjuster.get_depth(pipeline3.current_turn)
        print(f"  📍 轮次{pipeline3.current_turn:2d} [{phase}] depth={depth} | "
              f"{'✅' if result.stored else '❌'} {content[:30]}{'...' if len(content) > 30 else ''}")

    # 展示最终状态
    state3 = pipeline3.get_state()
    print(f"\n  📊 最终状态:")
    print(f"    对话轮次: {state3['current_turn']}")
    print(f"    记忆深度: {state3['depth']} ({state3['phase']})")
    print(f"    缓冲区:   {state3['buffer']['usage']}")
    print(f"    Active:   {state3['buffer']['active_count']} 条")
    print(f"    Pinned:   {state3['pinned_count']} 条")
    print(f"    归档:     {state3['archive_count']} 个话题, {state3['archived_messages']} 条消息")

    # 展示保留的消息
    print(f"\n  📋 当前活跃记忆:")
    for msg in pipeline3.buffer.active_zone:
        pin_mark = "📌" if msg.is_pinned else "  "
        intent = msg.intent_type or "unknown"
        print(f"    {pin_mark} [{msg.role:9s}] score={msg.priority_score:.2f} "
              f"intent={intent:12s} | {msg.content[:35]}{'...' if len(msg.content) > 35 else ''}")

    print_separator("演示完成", "█")
    print()
