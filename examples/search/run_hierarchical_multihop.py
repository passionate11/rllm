"""
运行分层多跳问答Agent

这个脚本演示如何使用HierarchicalMultiHopAgent进行推理。
"""

import asyncio
import os

from dotenv import load_dotenv
from transformers import AutoTokenizer

from rllm.agents.hierarchical_multihop_agent import HierarchicalMultiHopAgent
from rllm.agents.reasoning_agent import REASONING_SYSTEM_PROMPT
from rllm.agents.summary_agent import SUMMARY_SYSTEM_PROMPT
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.utils import save_trajectories

from .local_retrieval_tool import LocalRetrievalTool


def load_search_data(train_size=3000, test_size=100):
    """加载搜索数据集"""
    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")
    if test_dataset is None:
        print("Dataset not found, preparing search dataset...")
        from .prepare_hotpotqa_data import prepare_hotpotqa_data
        _, test_dataset = prepare_hotpotqa_data(
            train_size=train_size,
            test_size=test_size
        )

    return test_dataset.get_data()


def print_banner(text: str):
    """打印漂亮的banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # 环境配置
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:8000"

    load_dotenv()

    print_banner("Hierarchical MultiHop Agent Demo")

    # 配置
    n_parallel_agents = 8  # 并行agent数量
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 使用的模型

    print(f"Model: {model_name}")
    print(f"Parallel Agents: {n_parallel_agents}")

    # 加载tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 采样参数
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "model": model_name
    }

    # 工具配置
    tool_map = {"local_search": LocalRetrievalTool}

    # Agent配置
    agent_args = {
        "tool_map": tool_map,
        "parser_name": "qwen",
        "reasoning_system_prompt": REASONING_SYSTEM_PROMPT,
        "summary_system_prompt": SUMMARY_SYSTEM_PROMPT,
        # 压缩配置
        "enable_compression": True,
        "compression_trigger": "both",  # token或step都可以触发
        "max_tokens_before_compress": 3000,  # 3000 tokens触发压缩
        "max_steps_before_compress": 3,      # 3步触发压缩
        "max_summary_length": 300,           # 摘要最大300 tokens
    }

    print("\nAgent Configuration:")
    print(f"  - Compression Enabled: {agent_args['enable_compression']}")
    print(f"  - Compression Trigger: {agent_args['compression_trigger']}")
    print(f"  - Max Tokens: {agent_args['max_tokens_before_compress']}")
    print(f"  - Max Steps: {agent_args['max_steps_before_compress']}")

    # 环境配置
    env_args = {
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
        "max_steps": 20,  # 最多20步
    }

    print_banner("Creating Execution Engine")

    # 创建执行引擎
    engine = AgentExecutionEngine(
        agent_class=HierarchicalMultiHopAgent,
        agent_args=agent_args,
        env_class=ToolEnvironment,
        env_args=env_args,
        rollout_engine=None,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",  # vLLM或SGLang服务地址
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=8192,
        config=None,
        n_parallel_agents=n_parallel_agents,
    )

    print_banner("Loading Test Data")

    # 加载测试数据
    tasks = load_search_data(test_size=10)  # 只测试10个样本
    print(f"Loaded {len(tasks)} test examples")

    # 显示第一个样本
    if tasks:
        print("\nExample Question:")
        print(f"  Q: {tasks[0]['question']}")
        print(f"  A: {tasks[0]['ground_truth']}")

    print_banner("Running Inference")

    # 执行推理
    results = asyncio.run(engine.execute_tasks(tasks))

    print_banner("Inference Completed")

    # 保存结果
    output_file = "hierarchical_multihop_trajectories.pt"
    save_trajectories(results, filename=output_file)
    print(f"✓ Results saved to: {output_file}")

    # 显示统计信息
    print("\n" + "-"*70)
    print("Statistics:")
    print("-"*70)

    total_correct = sum(1 for r in results if r.get("reward", 0) > 0.5)
    total_tasks = len(results)
    accuracy = total_correct / total_tasks if total_tasks > 0 else 0

    print(f"Total Tasks: {total_tasks}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.2%}")

    # 计算平均步数和压缩次数
    total_steps = sum(len(r.get("trajectory", {}).get("steps", [])) for r in results)
    avg_steps = total_steps / total_tasks if total_tasks > 0 else 0

    print(f"Average Steps: {avg_steps:.2f}")

    # 尝试统计压缩次数
    compression_counts = []
    for r in results:
        trajectory = r.get("trajectory", {})
        steps = trajectory.get("steps", [])
        compressions = sum(
            1 for step in steps
            if "[COMPRESSION" in str(step.get("action", ""))
        )
        compression_counts.append(compressions)

    if compression_counts:
        avg_compressions = sum(compression_counts) / len(compression_counts)
        print(f"Average Compressions per Task: {avg_compressions:.2f}")
        print(f"Max Compressions in a Task: {max(compression_counts)}")

    print("-"*70)

    # 显示一些样例结果
    print("\n" + "-"*70)
    print("Sample Results:")
    print("-"*70)

    for i, result in enumerate(results[:3]):  # 显示前3个
        task_info = result.get("task_info", {})
        question = task_info.get("question", "N/A")
        ground_truth = task_info.get("ground_truth", "N/A")
        reward = result.get("reward", 0)

        # 提取模型答案
        trajectory = result.get("trajectory", {})
        steps = trajectory.get("steps", [])
        model_answer = "N/A"
        if steps:
            last_step = steps[-1]
            model_answer = last_step.get("model_response", "N/A")[:100]

        print(f"\n[Example {i+1}]")
        print(f"Question: {question[:80]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"Model Answer: {model_answer}...")
        print(f"Reward: {reward:.3f}")
        print(f"Steps: {len(steps)}")

    print("\n" + "="*70)
    print("Demo Completed!")
    print("="*70 + "\n")
