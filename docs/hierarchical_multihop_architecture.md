# 分层多跳问答架构详解

## 📐 整体架构

### 系统组件

```
┌─────────────────────────────────────────────────────────────────┐
│                    AgentExecutionEngine                         │
│                   (协调和执行引擎)                               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │        HierarchicalMultiHopAgent (协调器)                 │ │
│  │                                                           │ │
│  │  ┌──────────────────┐         ┌──────────────────┐       │ │
│  │  │  Summary Agent   │         │ Reasoning Agent  │       │ │
│  │  │   (模型1)        │◄────────│    (模型2)       │       │ │
│  │  │                  │         │                  │       │ │
│  │  │ 输入: 历史消息    │         │ 输入: 压缩摘要    │       │ │
│  │  │ 输出: 压缩摘要    │         │ 输出: 工具调用    │       │ │
│  │  │                  │         │                  │       │ │
│  │  │ Prompt:          │         │ Tools:           │       │ │
│  │  │ - 摘要指令       │         │ - local_search   │       │ │
│  │  │ - 格式要求       │         │ - finish         │       │ │
│  │  └──────────────────┘         └──────────────────┘       │ │
│  │           ▲                            │                 │ │
│  │           │                            ▼                 │ │
│  │           │                    ┌──────────────┐          │ │
│  │           └────────────────────│ 触发条件检查  │          │ │
│  │                                │              │          │ │
│  │                                │ • Token > N  │          │ │
│  │                                │ • Steps > M  │          │ │
│  │                                └──────────────┘          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   ToolEnvironment                         │ │
│  │                                                           │ │
│  │  ┌────────────────┐         ┌────────────────┐           │ │
│  │  │ Tool Execution │         │ Reward Function│           │ │
│  │  │                │         │                │           │ │
│  │  │ • 并行执行     │         │ • F1 Score     │           │ │
│  │  │ • 格式化输出   │         │ • Exact Match  │           │ │
│  │  └────────────────┘         └────────────────┘           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 执行流程

### 完整的交互循环

```
[1] 初始化
    ├─ 创建 Reasoning Agent
    ├─ 创建 Summary Agent
    ├─ 设置压缩阈值
    └─ 加载系统提示词

[2] 接收问题
    └─ Environment.reset()
        └─ 返回: {"question": "..."}

[3] Reasoning Agent 推理循环
    │
    ├─ [步骤 1]
    │   ├─ Model生成响应 → "调用local_search('Barack Obama')"
    │   ├─ 解析工具调用
    │   ├─ Environment执行 → 返回搜索结果
    │   ├─ 更新消息历史
    │   └─ steps_since_compress++
    │
    ├─ [步骤 2]
    │   ├─ Model生成响应 → "调用local_search('Michelle Obama')"
    │   ├─ 执行搜索
    │   └─ steps_since_compress++
    │
    ├─ [步骤 3]
    │   ├─ Model生成响应 → "继续推理..."
    │   └─ steps_since_compress++ (现在 = 3)
    │
    └─ [检查压缩条件]
        └─ steps_since_compress >= max_steps_before_compress
            └─ 触发压缩! ✓

[4] 压缩流程
    │
    ├─ [准备压缩]
    │   ├─ 收集 Reasoning Agent 的所有消息
    │   ├─ 构建压缩请求
    │   └─ 设置 waiting_for_summary = True
    │
    ├─ [Summary Agent 生成]
    │   ├─ 输入: 历史消息列表
    │   ├─ Model生成摘要
    │   └─ 输出: 结构化摘要
    │
    └─ [应用压缩]
        ├─ 提取摘要内容
        ├─ 保存到 compression_history
        ├─ 重置 Reasoning Agent
        ├─ 注入压缩摘要为系统消息
        ├─ 重新加载当前问题
        └─ steps_since_compress = 0

[5] 继续推理（基于压缩上下文）
    │
    ├─ [步骤 4]
    │   ├─ Reasoning Agent 现在的上下文:
    │   │   └─ System: 原始提示词 + 工具定义
    │   │   └─ System: 压缩摘要
    │   │   └─ User: 当前问题
    │   ├─ Model基于摘要生成响应
    │   └─ 继续搜索或推理
    │
    └─ [最终答案]
        ├─ Model生成: "finish({'response': '\\boxed{Barack Obama}'})"
        ├─ Environment评估答案
        └─ 返回奖励

[6] 结束
    └─ 保存完整的 Trajectory
        ├─ 所有推理步骤
        ├─ 所有压缩记录
        └─ 最终奖励
```

## 🧩 核心类详解

### 1. SummaryAgent

**职责：** 压缩历史上下文

```python
class SummaryAgent(BaseAgent):
    """
    压缩历史对话的Agent
    """

    # 核心方法
    def prepare_summarization_request(self, history_messages):
        """准备摘要请求"""
        # 1. 过滤消息（只保留user/assistant/tool）
        # 2. 格式化为JSON
        # 3. 构建压缩请求

    def update_from_model(self, response):
        """处理模型生成的摘要"""
        # 1. 接收摘要
        # 2. 记录到trajectory
        # 3. 返回Action

    def extract_summary(self, response):
        """提取结构化摘要"""
        # 1. 尝试提取```summary```代码块
        # 2. 如果没有，返回整个响应
```

**输入示例：**
```json
[
  {"role": "user", "content": "Who is older..."},
  {"role": "assistant", "content": "I'll search for..."},
  {"role": "tool", "content": "Barack Obama born 1961..."},
  {"role": "assistant", "content": "Now searching for Michelle..."},
  {"role": "tool", "content": "Michelle Obama born 1964..."}
]
```

**输出示例：**
```markdown
## Key Facts:
- Barack Obama: born August 4, 1961
- Michelle Obama: born January 17, 1964

## Entity Relationships:
- Both are married
- Barack is 3 years older

## Reasoning Progress:
- Searched for both birthdates
- Found: Barack (1961), Michelle (1964)
- Still need to compare and answer

## Compressed Context:
We searched for birthdates: Barack Obama (1961) and Michelle Obama (1964).
Barack is older by 3 years. Need to formulate final answer.
```

### 2. ReasoningAgent

**职责：** 基于上下文进行推理和搜索

```python
class ReasoningAgent(BaseAgent):
    """
    推理和搜索Agent
    """

    # 核心方法
    def inject_compressed_context(self, compressed_summary):
        """注入压缩的上下文"""
        # 1. 保存摘要
        # 2. 构建context_message
        # 3. 插入到消息历史中

    def update_from_model(self, response):
        """处理模型响应"""
        # 1. 解析工具调用
        # 2. 如果没有工具调用，自动添加finish
        # 3. 更新消息和trajectory

    def estimate_token_count(self):
        """估算当前token数"""
        # 粗略估计：字符数 / 4
```

**消息历史示例（注入摘要后）：**
```python
[
  {
    "role": "system",
    "content": "You are a reasoning agent..."
  },
  {
    "role": "system",
    "content": "## Compressed Context:\n[摘要内容]\n---\nContinue reasoning..."
  },
  {
    "role": "user",
    "content": "Who is older..."
  }
]
```

### 3. HierarchicalMultiHopAgent

**职责：** 协调两个模型的交互

```python
class HierarchicalMultiHopAgent(BaseAgent):
    """
    分层协调器
    """

    # 状态管理
    self.reasoning_agent: ReasoningAgent
    self.summary_agent: SummaryAgent
    self.compression_count: int
    self.steps_since_last_compress: int
    self.waiting_for_summary: bool

    # 核心方法
    def should_compress(self) -> bool:
        """判断是否应该压缩"""
        # 1. 检查enable_compression
        # 2. 计算当前tokens
        # 3. 检查steps
        # 4. 根据trigger模式返回

    def trigger_compression(self) -> str:
        """触发压缩"""
        # 1. 获取历史消息
        # 2. 准备压缩请求
        # 3. 设置waiting_for_summary = True
        # 4. 返回请求

    def apply_compression(self, summary):
        """应用压缩"""
        # 1. 提取摘要
        # 2. 保存历史
        # 3. 重置Reasoning Agent
        # 4. 注入摘要

    @property
    def chat_completions(self):
        """动态返回消息"""
        if self.waiting_for_summary:
            return # Summary Agent消息
        elif self.should_compress():
            # 触发压缩，返回Summary请求
        else:
            return # Reasoning Agent消息
```

## 🎛️ 配置选项详解

### 压缩触发条件

| 配置 | 含义 | 示例场景 |
|------|------|---------|
| `compression_trigger="token"` | 仅当tokens > N时压缩 | 长文档任务 |
| `compression_trigger="step"` | 仅当steps > M时压缩 | 固定步数压缩 |
| `compression_trigger="both"` | 满足任一条件即压缩 | **推荐默认** |

### 参数调优指南

#### 简单问题（2-3跳）
```python
max_tokens_before_compress = 2000
max_steps_before_compress = 2
max_summary_length = 200
```

#### 中等复杂度（4-6跳）
```python
max_tokens_before_compress = 3000
max_steps_before_compress = 3
max_summary_length = 300
```

#### 复杂问题（7+跳）
```python
max_tokens_before_compress = 4000
max_steps_before_compress = 4
max_summary_length = 500
```

## 🔬 训练策略

### 模式1: 联合训练（Joint Training）

**适用场景：** 从头训练新模型

```python
trainer = AgentTrainer(
    agent_class=HierarchicalMultiHopAgent,
    agent_args={
        "enable_compression": True,
        # 两个模型都会被训练
    }
)
```

**优点：**
- 端到端优化
- Summary和Reasoning协同学习
- 摘要更适合后续推理

**缺点：**
- 训练复杂度高
- 需要更多GPU资源
- 收敛可能较慢

### 模式2: 分阶段训练（Staged Training）

#### 阶段1: 训练Summary Agent
```python
# 第一阶段：只训练摘要能力
trainer = AgentTrainer(
    agent_args={
        "mode": "summary_only",
        # 使用固定的Reasoning Agent
    }
)
```

#### 阶段2: 训练Reasoning Agent
```python
# 第二阶段：基于训练好的Summary训练Reasoning
trainer = AgentTrainer(
    agent_args={
        "mode": "reasoning_only",
        # 加载训练好的Summary Agent
        "summary_model_path": "path/to/summary/model",
    }
)
```

**优点：**
- 每个阶段目标明确
- 易于调试和优化
- 可以重复优化某一部分

**缺点：**
- 需要分两次训练
- 可能不是全局最优

### 模式3: 迁移学习（Transfer Learning）

```python
# 使用预训练的通用模型初始化
summary_agent = SummaryAgent(
    pretrained_model="Qwen2.5-7B-Instruct"
)
reasoning_agent = ReasoningAgent(
    pretrained_model="Qwen2.5-7B-Instruct"
)

# 在多跳数据上微调
trainer.train()
```

## 📊 性能优化

### 内存优化

1. **限制历史长度**
```python
max_tokens_before_compress = 2000  # 更激进的压缩
```

2. **梯度检查点**
```python
config.trainer.gradient_checkpointing = True
```

3. **混合精度训练**
```python
config.trainer.fp16 = True
```

### 速度优化

1. **并行Agent**
```python
n_parallel_agents = 64  # 增加并行度
```

2. **批处理**
```python
config.data.train_batch_size = 16
```

3. **异步工具调用**
- 已默认启用（ThreadPoolExecutor）

## 🐛 调试技巧

### 1. 可视化压缩过程

```python
import logging
logging.basicConfig(level=logging.INFO)

# 会看到类似输出：
# INFO: Triggering compression #1 (steps since last: 3)
# INFO: Applied compression #1 (summary length: 245 chars)
```

### 2. 检查摘要质量

```python
# 获取压缩历史
stats = agent.get_stats()
print(stats["compression_history_count"])

# 查看实际摘要
for i, summary in enumerate(agent.compression_history):
    print(f"\n=== Compression {i+1} ===")
    print(summary)
```

### 3. 验证工具调用

```python
# 记录所有工具调用
for step in trajectory.steps:
    action = step.action
    if isinstance(action, list):
        for call in action:
            print(f"Tool: {call['function']['name']}")
            print(f"Args: {call['function']['arguments']}")
```

## 🔮 未来扩展

### 1. 多级压缩

```python
# 压缩的压缩
if compression_count > 5:
    # 压缩所有历史摘要为一个超级摘要
    meta_summary = compress_compression_history()
```

### 2. 选择性压缩

```python
# 只压缩不重要的部分，保留关键信息
def should_compress_message(msg):
    if is_critical_fact(msg):
        return False  # 不压缩
    return True
```

### 3. 动态压缩率

```python
# 根据剩余步数调整压缩强度
remaining_steps = max_steps - current_step
compression_ratio = min(0.3, remaining_steps / max_steps)
```

## 📚 参考文献

1. **Memory Compression in LLMs**
   - 长上下文问题的记忆管理

2. **Multi-Hop QA Datasets**
   - HotpotQA
   - 2WikiMultiHopQA

3. **Tool-Using Agents**
   - ReAct: Reasoning and Acting
   - Toolformer

---

**维护者：** RLLM Team
**最后更新：** 2025-10-26
