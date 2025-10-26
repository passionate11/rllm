"""
Summary Agent: 负责压缩历史上下文

这个Agent接收一段历史对话（包括推理和搜索结果），
输出一个压缩的摘要表示，用于后续推理。
"""

import copy
import json
import logging
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Summary Agent的系统提示词
SUMMARY_SYSTEM_PROMPT = """You are a context summarization agent. Your task is to compress historical reasoning and search results into a concise, information-rich summary.

Your summary should:
1. Extract key facts and entities discovered so far
2. Identify relationships between entities
3. Track the reasoning progress (what has been tried, what worked, what didn't)
4. Preserve crucial information needed for answering the question
5. Remove redundant or irrelevant details

Format your summary as:
```summary
## Key Facts:
- [Fact 1]
- [Fact 2]
...

## Entity Relationships:
- [Relationship 1]
...

## Reasoning Progress:
- [What has been explored]
- [Current insights]
- [Remaining questions]

## Compressed Context:
[A concise paragraph summarizing everything above]
```

Keep the summary under 300 tokens while preserving all critical information."""


class SummaryAgent(BaseAgent):
    """
    Summary Agent: 压缩历史上下文

    输入: 历史对话（messages list）
    输出: 压缩后的摘要
    """

    def __init__(
        self,
        system_prompt: str = SUMMARY_SYSTEM_PROMPT,
        max_summary_length: int = 300,
    ):
        """
        初始化Summary Agent

        Args:
            system_prompt: 系统提示词
            max_summary_length: 摘要的最大token长度
        """
        self.system_prompt = system_prompt
        self.max_summary_length = max_summary_length

        # 内部状态
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()

    def reset(self):
        """重置Agent状态"""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.current_observation = None

    def prepare_summarization_request(self, history_messages: list[dict]) -> str:
        """
        准备摘要请求

        Args:
            history_messages: 需要压缩的历史消息

        Returns:
            格式化的摘要请求
        """
        # 过滤掉system消息，只保留user/assistant/tool消息
        filtered_messages = [
            msg for msg in history_messages
            if msg.get("role") in ["user", "assistant", "tool"]
        ]

        # 构建摘要请求
        context_str = json.dumps(filtered_messages, indent=2, ensure_ascii=False)

        request = f"""Please summarize the following conversation history:

{context_str}

Provide a concise summary following the format specified in the system prompt."""

        return request

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: dict,
        **kwargs
    ):
        """
        更新来自环境的反馈（Summary Agent通常不需要环境交互）

        Args:
            observation: 环境观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # Summary Agent 通常是单步生成，不需要复杂的环境交互
        if observation:
            self.messages.append({"role": "user", "content": str(observation)})
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        处理模型响应（生成的摘要）

        Args:
            response: 模型生成的摘要

        Returns:
            Action对象，包含生成的摘要
        """
        # 将摘要作为action返回
        assistant_message = {"role": "assistant", "content": response}
        self.messages.append(assistant_message)

        # 记录到trajectory
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=response,  # 摘要就是action
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        return Action(action=response)

    def extract_summary(self, response: str) -> str:
        """
        从模型响应中提取摘要

        Args:
            response: 模型原始响应

        Returns:
            提取的摘要文本
        """
        # 尝试提取```summary```代码块
        import re
        summary_match = re.search(r"```summary\n(.*?)\n```", response, re.DOTALL)
        if summary_match:
            return summary_match.group(1).strip()

        # 如果没有代码块，返回整个响应
        return response.strip()

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """返回当前的消息历史"""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """返回当前的轨迹"""
        return self._trajectory
