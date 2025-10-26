# 分层多跳问答 Agent (Hierarchical MultiHop QA)

这个实现提供了一个**两模型分层架构**来解决复杂的多跳问答问题。

## 🎯 架构概述

```
┌─────────────────────────────────────────────────────────┐
│         HierarchicalMultiHopAgent (协调器)               │
│                                                          │
│  ┌────────────────┐         ┌────────────────┐         │
│  │ Summary Agent  │         │ Reasoning Agent│         │
│  │   (模型1)      │────────>│    (模型2)     │         │
│  │                │         │                │         │
│  │ 压缩历史上下文  │         │ 推理 + Search  │         │
│  └────────────────┘         └────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

### 组件说明

1. **Summary Agent (模型1)**
   - 职责：压缩历史对话和搜索结果
   - 输入：完整的历史消息（推理 + 搜索结果）
   - 输出：结构化的压缩摘要（~300 tokens）
   - 文件：`rllm/agents/summary_agent.py`

2. **Reasoning Agent (模型2)**
   - 职责：基于当前上下文进行推理和搜索
   - 输入：压缩的上下文摘要 + 当前问题
   - 输出：工具调用或最终答案
   - 文件：`rllm/agents/reasoning_agent.py`

3. **Hierarchical MultiHop Agent (协调器)**
   - 职责：协调两个模型的交互
   - 功能：
     - 监控上下文长度
     - 触发压缩（基于token数或步数）
     - 管理压缩历史
     - 统一轨迹记录
   - 文件：`rllm/agents/hierarchical_multihop_agent.py`

## 🔄 工作流程

```
初始化
  ↓
Reasoning Agent: 推理 + Search (步骤1)
  ↓
Reasoning Agent: 推理 + Search (步骤2)
  ↓
Reasoning Agent: 推理 + Search (步骤3)
  ↓
[触发条件满足：步数 >= 3 或 tokens > 3000]
  ↓
Summary Agent: 压缩历史 → 生成摘要
  ↓
清空 Reasoning Agent 历史
  ↓
注入压缩摘要到 Reasoning Agent
  ↓
Reasoning Agent: 基于摘要继续推理
  ↓
... (循环直到问题解决)
```

## 📦 安装和设置

### 1. 启动检索服务器

```bash
cd examples/search/retrieval
bash launch_server.sh
```

服务器将在 `http://127.0.0.1:8000` 运行。

### 2. 启动模型推理服务

使用 vLLM 或 SGLang 启动模型服务：

```bash
# vLLM 示例
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 30000

# 或使用 SGLang
python -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 30000
```

### 3. 准备数据集

数据集会在首次运行时自动下载和准备（HotpotQA）。

## 🚀 使用方法

### 运行推理

```bash
cd examples/search
python run_hierarchical_multihop.py
```

这会：
- 加载 HotpotQA 测试集
- 使用分层架构进行推理
- 自动触发上下文压缩
- 保存结果到 `hierarchical_multihop_trajectories.pt`
- 显示性能统计

### 训练模型

#### 1. 联合训练（推荐）

同时训练 Summary 和 Reasoning 两个模型：

```bash
cd examples/search
python train_hierarchical_multihop.py \
    --mode joint \
    --enable-compression \
    --compression-trigger both \
    --max-tokens 3000 \
    --max-steps 3
```

#### 2. 只训练 Summary Agent

```bash
python train_hierarchical_multihop.py \
    --mode summary_only \
    --train-size 3000 \
    --test-size 100
```

#### 3. 只训练 Reasoning Agent

```bash
python train_hierarchical_multihop.py \
    --mode reasoning_only \
    --enable-compression false
```

## ⚙️ 配置选项

### Agent 配置

在 `run_hierarchical_multihop.py` 或 `train_hierarchical_multihop.py` 中配置：

```python
agent_args = {
    # 工具配置
    "tool_map": {"local_search": LocalRetrievalTool},
    "parser_name": "qwen",  # 或 "r1"

    # 系统提示词
    "reasoning_system_prompt": REASONING_SYSTEM_PROMPT,
    "summary_system_prompt": SUMMARY_SYSTEM_PROMPT,

    # 压缩配置
    "enable_compression": True,
    "compression_trigger": "both",  # "token", "step", "both"
    "max_tokens_before_compress": 3000,
    "max_steps_before_compress": 3,
    "max_summary_length": 300,
}
```

### 压缩触发条件

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `compression_trigger` | 触发条件 | `"both"` |
| `max_tokens_before_compress` | token阈值 | `3000` |
| `max_steps_before_compress` | 步数阈值 | `3` |

**触发模式：**
- `"token"`: 仅基于token数触发
- `"step"`: 仅基于步数触发
- `"both"`: 满足任一条件即触发

### 环境配置

```python
env_args = {
    "tool_map": tool_map,
    "reward_fn": search_reward_fn,
    "max_steps": 20,  # 最大交互步数
}
```

## 📊 性能监控

### 统计信息

运行后会显示：
- 准确率（Accuracy）
- 平均步数（Average Steps）
- 平均压缩次数（Average Compressions）
- 每个任务的详细结果

### 获取Agent统计

```python
# 在推理过程中
stats = agent.get_stats()
print(stats)
# 输出:
# {
#     "compression_count": 2,
#     "steps_since_last_compress": 1,
#     "current_tokens": 2150,
#     "compression_history_count": 2,
#     "waiting_for_summary": False
# }
```

## 🔍 调试和日志

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.INFO)

# 或者只启用特定模块
logging.getLogger("rllm.agents.hierarchical_multihop_agent").setLevel(logging.DEBUG)
```

## 📈 训练建议

### 联合训练 vs 分开训练

| 模式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **联合训练** | • 端到端优化<br>• 两个模型协同学习 | • 训练复杂<br>• 需要更多资源 | 从头训练新模型 |
| **Summary Only** | • 专注优化压缩质量<br>• 训练简单 | • 需要预训练的Reasoning模型 | 改进摘要质量 |
| **Reasoning Only** | • 专注优化推理能力<br>• 可禁用压缩简化 | • 需要预训练的Summary模型 | 改进推理能力 |

### 超参数建议

```python
# 短上下文任务（简单问题）
max_tokens_before_compress = 2000
max_steps_before_compress = 2

# 长上下文任务（复杂问题）
max_tokens_before_compress = 4000
max_steps_before_compress = 4

# 摘要长度
max_summary_length = 300  # 标准
max_summary_length = 500  # 复杂任务需要更多上下文
```

## 🧪 测试

### 单元测试

```bash
pytest tests/test_hierarchical_multihop.py
```

### 快速验证

```python
from rllm.agents.hierarchical_multihop_agent import HierarchicalMultiHopAgent
from examples.search.local_retrieval_tool import LocalRetrievalTool

# 创建agent
agent = HierarchicalMultiHopAgent(
    tool_map={"local_search": LocalRetrievalTool},
    max_steps_before_compress=2,
    enable_compression=True,
)

# 验证初始化
assert agent.compression_count == 0
assert not agent.waiting_for_summary

print("✓ Agent initialized successfully")
```

## 🐛 常见问题

### Q1: 压缩从不触发

**原因：** token计数或步数未达到阈值

**解决：**
```python
# 降低阈值
agent_args["max_steps_before_compress"] = 1
agent_args["max_tokens_before_compress"] = 500
```

### Q2: 检索服务器连接失败

**解决：**
```bash
# 检查服务器状态
curl http://127.0.0.1:8000/health

# 设置环境变量
export RETRIEVAL_SERVER_URL="http://your-server:8000"
```

### Q3: 模型响应解析失败

**原因：** 模型输出格式不符合预期

**解决：**
- 检查 `parser_name` 是否匹配模型
- 查看模型原始输出：`logger.setLevel(logging.DEBUG)`
- 调整系统提示词

## 📝 扩展

### 添加新的检索工具

```python
from rllm.tools.tool_base import Tool, ToolOutput

class MyCustomRetriever(Tool):
    NAME = "my_retriever"
    DESCRIPTION = "My custom retrieval tool"

    def forward(self, query: str) -> ToolOutput:
        # 实现检索逻辑
        results = self.retrieve(query)
        return ToolOutput(
            name=self.name,
            output=results,
        )

# 使用
tool_map = {
    "local_search": LocalRetrievalTool,
    "my_retriever": MyCustomRetriever,
}
```

### 自定义压缩策略

继承 `HierarchicalMultiHopAgent` 并重写 `should_compress()`:

```python
class CustomHierarchicalAgent(HierarchicalMultiHopAgent):
    def should_compress(self) -> bool:
        # 自定义逻辑
        if self.reasoning_agent.get_message_count() > 10:
            return True
        return super().should_compress()
```

## 📚 参考

- 论文：[Memory Compression for Multi-Hop Reasoning]
- 代码：`rllm/agents/hierarchical_multihop_agent.py`
- 示例：`examples/search/run_hierarchical_multihop.py`

## 📞 支持

遇到问题？
1. 查看日志输出
2. 检查 [Issues](https://github.com/your-repo/issues)
3. 提交新 Issue

---

**更新日期：** 2025-10-26
**版本：** 1.0.0
