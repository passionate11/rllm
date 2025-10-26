"""
训练分层多跳问答Agent

支持三种训练模式：
1. joint: 联合训练Summary和Reasoning两个模型
2. summary_only: 只训练Summary模型
3. reasoning_only: 只训练Reasoning模型
"""

import argparse
import os

import hydra
from dotenv import load_dotenv

from rllm.agents.hierarchical_multihop_agent import HierarchicalMultiHopAgent
from rllm.agents.reasoning_agent import REASONING_SYSTEM_PROMPT
from rllm.agents.summary_agent import SUMMARY_SYSTEM_PROMPT
from rllm.data import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import search_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

from .local_retrieval_tool import LocalRetrievalTool


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train Hierarchical MultiHop Agent")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["joint", "summary_only", "reasoning_only"],
        default="joint",
        help="Training mode: joint (both models), summary_only, or reasoning_only"
    )

    parser.add_argument(
        "--enable-compression",
        action="store_true",
        default=True,
        help="Enable context compression during training"
    )

    parser.add_argument(
        "--compression-trigger",
        type=str,
        choices=["token", "step", "both"],
        default="both",
        help="Compression trigger condition"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum tokens before triggering compression"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum steps before triggering compression"
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=3000,
        help="Training dataset size"
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Test dataset size"
    )

    return parser.parse_args()


def load_or_prepare_data(train_size=3000, test_size=100):
    """加载或准备HotpotQA数据"""
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")

    if train_dataset is None or test_dataset is None:
        print("Dataset not found, preparing HotpotQA dataset...")
        from .prepare_hotpotqa_data import prepare_hotpotqa_data
        _, _ = prepare_hotpotqa_data(train_size=train_size, test_size=test_size)

        train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
        test_dataset = DatasetRegistry.load_dataset("hotpotqa", "test")

    return train_dataset, test_dataset


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="ppo_trainer",
    version_base=None
)
def main(config):
    """主训练函数"""
    # 加载环境变量
    load_dotenv()

    # 解析参数
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Training Mode: {args.mode}")
    print(f"Compression Enabled: {args.enable_compression}")
    print(f"Compression Trigger: {args.compression_trigger}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Max Steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if "RETRIEVAL_SERVER_URL" not in os.environ:
        os.environ["RETRIEVAL_SERVER_URL"] = "http://127.0.0.1:8000"

    # 加载数据集
    train_dataset, test_dataset = load_or_prepare_data(
        train_size=args.train_size,
        test_size=args.test_size
    )

    # 工具配置
    tool_map = {"local_search": LocalRetrievalTool}

    # 环境配置
    env_args = {
        "max_steps": 20,  # 环境的最大步数
        "tool_map": tool_map,
        "reward_fn": search_reward_fn,
    }

    # Agent配置
    agent_args = {
        "tool_map": tool_map,
        "parser_name": "qwen",
        "reasoning_system_prompt": REASONING_SYSTEM_PROMPT,
        "summary_system_prompt": SUMMARY_SYSTEM_PROMPT,
        "enable_compression": args.enable_compression,
        "compression_trigger": args.compression_trigger,
        "max_tokens_before_compress": args.max_tokens,
        "max_steps_before_compress": args.max_steps,
    }

    # 根据训练模式调整配置
    if args.mode == "summary_only":
        print("⚠️  Training Summary Agent only")
        print("    Reasoning Agent will use pretrained weights")
        # TODO: 添加加载预训练Reasoning模型的逻辑
        agent_args["freeze_reasoning"] = True

    elif args.mode == "reasoning_only":
        print("⚠️  Training Reasoning Agent only")
        print("    Summary Agent will use pretrained weights")
        # TODO: 添加加载预训练Summary模型的逻辑
        agent_args["freeze_summary"] = True
        # 可以禁用压缩来简化训练
        agent_args["enable_compression"] = False

    else:  # joint mode
        print("✓  Training both models jointly")

    # 创建训练器
    trainer = AgentTrainer(
        agent_class=HierarchicalMultiHopAgent,
        env_class=ToolEnvironment,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    print("\nStarting training...")
    trainer.train()

    print("\n✓ Training completed!")


if __name__ == "__main__":
    main()
