"""
快速开始：分层多跳问答Agent

这是一个最简单的示例，演示如何使用HierarchicalMultiHopAgent。
"""

import asyncio
import os

from dotenv import load_dotenv
from transformers import AutoTokenizer

from rllm.agents.hierarchical_multihop_agent import HierarchicalMultiHopAgent
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn

# 导入本地检索工具
import sys
sys.path.insert(0, os.path.dirname(__file__))
from local_retrieval_tool import LocalRetrievalTool


def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()

    # 设置环境
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ.setdefault("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")

    print("\n" + "="*60)
    print("  快速开始：分层多跳问答Agent")
    print("="*60 + "\n")

    # 定义一个测试问题
    test_task = {
        "question": "Who is older, Barack Obama or Michelle Obama?",
        "ground_truth": "Barack Obama",
        "data_source": "demo"
    }

    print(f"测试问题: {test_task['question']}")
    print(f"正确答案: {test_task['ground_truth']}\n")

    # 配置
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"使用模型: {model_name}")

    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 工具配置
    tool_map = {"local_search": LocalRetrievalTool}

    # Agent配置
    agent_args = {
        "tool_map": tool_map,
        "parser_name": "qwen",
        "enable_compression": True,
        "compression_trigger": "step",
        "max_steps_before_compress": 2,  # 每2步压缩一次
    }

    print("\nAgent配置:")
    print(f"  - 压缩启用: {agent_args['enable_compression']}")
    print(f"  - 压缩触发: 每 {agent_args['max_steps_before_compress']} 步")

    # 环境配置
    env_args = {
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
        "max_steps": 10,
    }

    # 创建执行引擎
    print("\n创建执行引擎...")
    engine = AgentExecutionEngine(
        agent_class=HierarchicalMultiHopAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "model": model_name,
        },
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        n_parallel_agents=1,
    )

    print("\n开始推理...")
    print("-"*60)

    # 执行任务
    results = asyncio.run(engine.execute_tasks([test_task]))

    print("-"*60)
    print("\n推理完成!")

    # 显示结果
    if results:
        result = results[0]
        reward = result.get("reward", 0)
        trajectory = result.get("trajectory", {})
        steps = trajectory.get("steps", [])

        print(f"\n奖励得分: {reward:.3f}")
        print(f"总步数: {len(steps)}")

        # 统计压缩次数
        compressions = sum(
            1 for step in steps
            if "[COMPRESSION" in str(step.get("action", ""))
        )
        print(f"压缩次数: {compressions}")

        # 显示最终答案
        if steps:
            last_step = steps[-1]
            final_answer = last_step.get("model_response", "N/A")
            print(f"\n最终答案:\n{final_answer[:200]}...")

        # 判断是否正确
        is_correct = reward > 0.5
        print(f"\n结果: {'✓ 正确' if is_correct else '✗ 错误'}")

    print("\n" + "="*60)
    print("  演示完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
