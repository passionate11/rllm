"""
Reasoning Agent: 负责推理和搜索

这个Agent接收压缩的上下文摘要，继续进行推理和工具调用。
"""

import copy
import json
import logging
import uuid
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.parser import get_tool_parser
from rllm.parser.tool_parser.tool_parser_base import ToolParser
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool

logger = logging.getLogger(__name__)

# Reasoning Agent的系统提示词
REASONING_SYSTEM_PROMPT = """You are a multi-hop reasoning agent with access to search tools.

Your task is to answer complex questions that may require multiple steps of reasoning and information gathering.

When you receive a compressed context summary, use it as your knowledge base and continue reasoning from there.

Guidelines:
1. Read the summary carefully to understand what has been discovered
2. Identify what information is still missing to answer the question
3. Use the search tool strategically to find missing information
4. Connect facts from different sources to form a complete answer
5. Put your final answer in \\boxed{} format when you're confident

Available actions:
- Call search tools to gather information
- Call finish tool with your final answer when ready

Remember: You may have a compressed summary from previous reasoning steps. Build upon that knowledge."""


class ReasoningAgent(BaseAgent):
    """
    Reasoning Agent: 基于压缩上下文进行推理和搜索

    这个Agent类似于ToolAgent，但增强了对压缩上下文的处理能力。
    """

    def __init__(
        self,
        system_prompt: str = REASONING_SYSTEM_PROMPT,
        parser_name: str = "qwen",
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
        enable_context_summary: bool = True,
    ):
        """
        初始化Reasoning Agent

        Args:
            system_prompt: 系统提示词
            parser_name: 工具调用解析器名称
            tools: 工具名称列表（旧方式）
            tool_map: 工具名称到类的映射（新方式）
            enable_context_summary: 是否启用上下文摘要
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.system_prompt = system_prompt
        self.enable_context_summary = enable_context_summary

        # 初始化工具
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        # 初始化解析器
        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        # 工具提示
        self.tools_prompt = self.tool_parser.get_tool_prompt(
            json.dumps(self.tools.json, indent=2)
        )

        # 状态变量
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.compressed_context: str | None = None  # 存储压缩的上下文
        self.reset()

    def inject_compressed_context(self, compressed_summary: str):
        """
        注入压缩的上下文摘要

        Args:
            compressed_summary: 压缩后的上下文摘要
        """
        self.compressed_context = compressed_summary

        # 将压缩摘要作为系统消息注入
        context_message = {
            "role": "system",
            "content": f"""## Compressed Context from Previous Reasoning:

{compressed_summary}

---

Continue your reasoning based on the above context. Use search tools to gather additional information if needed."""
        }

        # 在系统提示之后、用户消息之前插入
        # 找到第一个非system消息的位置
        insert_pos = 1
        for i, msg in enumerate(self.messages):
            if msg["role"] != "system":
                insert_pos = i
                break

        self.messages.insert(insert_pos, context_message)

        logger.info(f"Injected compressed context ({len(compressed_summary)} chars)")

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """将观察转换为消息格式"""
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                # 格式化工具输出
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append({
                        "role": "tool",
                        "content": tool_output_str,
                        "tool_call_id": tool_call_id,
                    })
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

        return messages

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

        Args:
            observation: 环境观察
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        更新来自模型的响应

        Args:
            response: 模型响应

        Returns:
            Action对象
        """
        tool_calls_dict = []
        assistant_content = response

        # 解析工具调用
        try:
            tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]
        except Exception as e:
            logger.error(f"Failed to parse tool calls: {e}")
            tool_calls_dict = []

        # 添加assistant消息
        assistant_message = {"role": "assistant", "content": assistant_content}
        if len(tool_calls_dict) > 0:
            # 确保arguments是字符串
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(
                        call["function"]["arguments"]
                    )
        else:
            # 没有工具调用，自动添加finish
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {"response": assistant_content},
                    },
                }
            ]

        self.messages.append(assistant_message)

        # 记录到trajectory
        new_step = Step(
            chat_completions=copy.deepcopy(self.chat_completions),
            action=tool_calls_dict,
            model_response=response,
            observation=self.current_observation,
        )
        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)

    def reset(self):
        """重置Agent状态"""
        self._trajectory = Trajectory()
        self.messages = [
            {"role": "system", "content": self.system_prompt + self.tools_prompt}
        ]
        self.compressed_context = None

    def get_message_count(self) -> int:
        """获取当前消息数量"""
        return len(self.messages)

    def estimate_token_count(self) -> int:
        """
        估算当前消息的token数量（粗略估计）

        Returns:
            估算的token数量
        """
        total_chars = sum(
            len(msg.get("content", ""))
            for msg in self.messages
        )
        # 粗略估计：4个字符约等于1个token
        return total_chars // 4

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """返回当前消息历史"""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """返回当前轨迹"""
        return self._trajectory
