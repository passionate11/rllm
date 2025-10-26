"""
Hierarchical MultiHop Agent: 协调Summary Agent和Reasoning Agent

这个Agent管理两个模型的交互：
1. Reasoning Agent进行推理和搜索
2. 当上下文过长时，触发Summary Agent压缩
3. 将压缩结果注入Reasoning Agent继续推理
"""

import copy
import logging
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.reasoning_agent import ReasoningAgent
from rllm.agents.summary_agent import SummaryAgent
from rllm.tools.tool_base import Tool

logger = logging.getLogger(__name__)


class HierarchicalMultiHopAgent(BaseAgent):
    """
    分层多跳Agent：协调Summary和Reasoning两个模型

    工作流程：
    1. 初始化时只有Reasoning Agent活跃
    2. 每次更新检查是否需要压缩（基于token数或步数）
    3. 如果需要压缩：
       a. 调用Summary Agent压缩历史
       b. 清空Reasoning Agent的历史
       c. 注入压缩摘要
    4. 继续使用Reasoning Agent推理
    """

    def __init__(
        self,
        reasoning_agent: ReasoningAgent | None = None,
        summary_agent: SummaryAgent | None = None,
        # Reasoning Agent配置
        reasoning_system_prompt: str | None = None,
        parser_name: str = "qwen",
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        # Summary触发条件
        compression_trigger: str = "token",  # "token", "step", "both"
        max_tokens_before_compress: int = 3000,  # 超过此token数触发压缩
        max_steps_before_compress: int = 3,     # 超过此步数触发压缩
        # Summary Agent配置
        summary_system_prompt: str | None = None,
        max_summary_length: int = 300,
        # 其他配置
        enable_compression: bool = True,  # 是否启用压缩
    ):
        """
        初始化分层多跳Agent

        Args:
            reasoning_agent: Reasoning Agent实例（可选）
            summary_agent: Summary Agent实例（可选）
            reasoning_system_prompt: Reasoning Agent的系统提示词
            parser_name: 工具解析器名称
            tools: 工具列表
            tool_map: 工具映射
            compression_trigger: 压缩触发条件 ("token", "step", "both")
            max_tokens_before_compress: 触发压缩的最大token数
            max_steps_before_compress: 触发压缩的最大步数
            summary_system_prompt: Summary Agent的系统提示词
            max_summary_length: 摘要最大长度
            enable_compression: 是否启用压缩
        """
        # 创建或使用提供的Reasoning Agent
        if reasoning_agent is None:
            from rllm.agents.reasoning_agent import REASONING_SYSTEM_PROMPT
            reasoning_agent = ReasoningAgent(
                system_prompt=reasoning_system_prompt or REASONING_SYSTEM_PROMPT,
                parser_name=parser_name,
                tools=tools,
                tool_map=tool_map,
            )
        self.reasoning_agent = reasoning_agent

        # 创建或使用提供的Summary Agent
        if summary_agent is None:
            from rllm.agents.summary_agent import SUMMARY_SYSTEM_PROMPT
            summary_agent = SummaryAgent(
                system_prompt=summary_system_prompt or SUMMARY_SYSTEM_PROMPT,
                max_summary_length=max_summary_length,
            )
        self.summary_agent = summary_agent

        # 压缩配置
        self.enable_compression = enable_compression
        self.compression_trigger = compression_trigger
        self.max_tokens_before_compress = max_tokens_before_compress
        self.max_steps_before_compress = max_steps_before_compress

        # 状态跟踪
        self.compression_count = 0  # 压缩次数
        self.steps_since_last_compress = 0  # 自上次压缩以来的步数
        self.compression_history: list[str] = []  # 压缩历史

        # 统一的trajectory
        self._trajectory = Trajectory()

        # 标记当前是否在等待Summary Agent
        self.waiting_for_summary = False
        self.pending_summary_request: str | None = None

    def should_compress(self) -> bool:
        """
        判断是否应该触发压缩

        Returns:
            是否需要压缩
        """
        if not self.enable_compression:
            return False

        # 估算当前token数
        current_tokens = self.reasoning_agent.estimate_token_count()

        token_trigger = current_tokens > self.max_tokens_before_compress
        step_trigger = self.steps_since_last_compress >= self.max_steps_before_compress

        if self.compression_trigger == "token":
            return token_trigger
        elif self.compression_trigger == "step":
            return step_trigger
        elif self.compression_trigger == "both":
            return token_trigger or step_trigger
        else:
            return False

    def trigger_compression(self) -> str:
        """
        触发压缩流程

        Returns:
            压缩请求的prompt
        """
        # 获取Reasoning Agent的当前消息历史
        history_messages = copy.deepcopy(self.reasoning_agent.messages)

        # 准备压缩请求
        compress_request = self.summary_agent.prepare_summarization_request(
            history_messages
        )

        self.waiting_for_summary = True
        self.pending_summary_request = compress_request

        logger.info(
            f"Triggering compression #{self.compression_count + 1} "
            f"(steps since last: {self.steps_since_last_compress})"
        )

        return compress_request

    def apply_compression(self, compressed_summary: str):
        """
        应用压缩后的摘要

        Args:
            compressed_summary: 压缩后的摘要
        """
        # 提取实际摘要
        actual_summary = self.summary_agent.extract_summary(compressed_summary)

        # 保存到历史
        self.compression_history.append(actual_summary)
        self.compression_count += 1
        self.steps_since_last_compress = 0

        # 重置Reasoning Agent并注入压缩上下文
        # 保存当前问题
        current_task = self.reasoning_agent.current_observation

        # 重置
        self.reasoning_agent.reset()

        # 注入压缩的上下文
        self.reasoning_agent.inject_compressed_context(actual_summary)

        # 如果有当前任务，重新加载
        if current_task:
            self.reasoning_agent.update_from_env(
                observation=current_task,
                reward=0.0,
                done=False,
                info={}
            )

        self.waiting_for_summary = False
        self.pending_summary_request = None

        logger.info(
            f"Applied compression #{self.compression_count} "
            f"(summary length: {len(actual_summary)} chars)"
        )

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs
    ):
        """
        更新来自环境的观察

        如果正在等待摘要，则更新Summary Agent；
        否则更新Reasoning Agent。

        Args:
            observation: 环境观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.waiting_for_summary:
            # 正在等待Summary Agent的响应
            self.summary_agent.update_from_env(observation, reward, done, info)
        else:
            # 正常的Reasoning Agent流程
            self.reasoning_agent.update_from_env(observation, reward, done, info)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        更新来自模型的响应

        Args:
            response: 模型响应

        Returns:
            Action对象
        """
        if self.waiting_for_summary:
            # 处理Summary Agent的响应
            summary_action = self.summary_agent.update_from_model(response)

            # 应用压缩
            compressed_summary = summary_action.action
            self.apply_compression(compressed_summary)

            # 记录到统一trajectory
            summary_step = Step(
                chat_completions=[],
                action=f"[COMPRESSION #{self.compression_count}]",
                model_response=compressed_summary,
                observation="Compression triggered",
            )
            self._trajectory.steps.append(summary_step)

            # 返回一个特殊的action，表示需要继续推理
            # 这个action不会被环境执行，而是触发新一轮model调用
            return Action(action="__CONTINUE_REASONING__")

        else:
            # 处理Reasoning Agent的响应
            action = self.reasoning_agent.update_from_model(response)

            # 增加步数计数
            self.steps_since_last_compress += 1

            # 合并step到统一trajectory
            if self.reasoning_agent.trajectory.steps:
                latest_step = self.reasoning_agent.trajectory.steps[-1]
                self._trajectory.steps.append(latest_step)

            # 检查是否需要压缩（在返回action之前）
            if self.should_compress() and action.action != "finish":
                logger.info("Compression needed, will trigger after this step")

            return action

    def reset(self):
        """重置Agent状态"""
        self.reasoning_agent.reset()
        self.summary_agent.reset()
        self._trajectory = Trajectory()
        self.compression_count = 0
        self.steps_since_last_compress = 0
        self.compression_history = []
        self.waiting_for_summary = False
        self.pending_summary_request = None

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        返回当前活跃agent的消息历史

        如果正在等待摘要，返回Summary Agent的消息；
        否则返回Reasoning Agent的消息。
        """
        if self.waiting_for_summary and self.pending_summary_request:
            # 返回Summary Agent的消息（包括压缩请求）
            summary_messages = copy.deepcopy(self.summary_agent.messages)
            # 添加压缩请求
            summary_messages.append({
                "role": "user",
                "content": self.pending_summary_request
            })
            return summary_messages
        else:
            # 检查是否需要压缩
            if self.should_compress():
                # 触发压缩
                compress_request = self.trigger_compression()
                # 返回Summary Agent的消息
                summary_messages = copy.deepcopy(self.summary_agent.messages)
                summary_messages.append({
                    "role": "user",
                    "content": compress_request
                })
                return summary_messages
            else:
                # 返回Reasoning Agent的消息
                return self.reasoning_agent.messages

    @property
    def trajectory(self) -> Trajectory:
        """返回统一的trajectory"""
        return self._trajectory

    def get_stats(self) -> dict:
        """
        获取统计信息

        Returns:
            包含统计信息的字典
        """
        return {
            "compression_count": self.compression_count,
            "steps_since_last_compress": self.steps_since_last_compress,
            "current_tokens": self.reasoning_agent.estimate_token_count(),
            "compression_history_count": len(self.compression_history),
            "waiting_for_summary": self.waiting_for_summary,
        }
