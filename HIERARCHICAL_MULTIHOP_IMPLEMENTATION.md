# 分层多跳问答Agent实现总结

## 📦 已实现的功能

本次实现了一个**两模型分层架构**的多跳问答系统，包含以下组件：

### 🎯 核心组件

#### 1. Summary Agent (模型1) - 上下文压缩
- **文件**: `rllm/agents/summary_agent.py`
- **功能**:
  - 接收历史对话（推理 + 搜索结果）
  - 生成结构化的压缩摘要
  - 提取关键事实、实体关系、推理进度
  - 控制摘要长度（默认300 tokens）

#### 2. Reasoning Agent (模型2) - 推理和搜索
- **文件**: `rllm/agents/reasoning_agent.py`
- **功能**:
  - 基于压缩上下文进行推理
  - 调用搜索工具获取信息
  - 支持多轮工具调用
  - 估算token使用量

#### 3. Hierarchical MultiHop Agent - 协调器
- **文件**: `rllm/agents/hierarchical_multihop_agent.py`
- **功能**:
  - 协调两个模型的交互
  - 监控上下文长度和步数
  - 自动触发压缩（可配置）
  - 管理压缩历史
  - 统一轨迹记录

### 🚀 训练和推理脚本

#### 1. 训练脚本
- **文件**: `examples/search/train_hierarchical_multihop.py`
- **支持三种模式**:
  - `joint`: 联合训练两个模型
  - `summary_only`: 只训练Summary Agent
  - `reasoning_only`: 只训练Reasoning Agent
- **可配置参数**:
  - 压缩触发条件（token/step/both）
  - Token阈值
  - 步数阈值
  - 数据集大小

#### 2. 推理脚本
- **文件**: `examples/search/run_hierarchical_multihop.py`
- **功能**:
  - 加载HotpotQA测试集
  - 执行多跳问答推理
  - 自动压缩长上下文
  - 统计性能指标
  - 保存轨迹

#### 3. 快速开始脚本
- **文件**: `examples/search/quick_start_hierarchical.py`
- **功能**:
  - 最简单的使用示例
  - 单个问题演示
  - 适合理解流程

### 📚 文档

#### 1. 使用指南
- **文件**: `examples/search/HIERARCHICAL_MULTIHOP_README.md`
- **内容**:
  - 架构概述
  - 安装和配置
  - 使用方法
  - 训练建议
  - 常见问题

#### 2. 架构详解
- **文件**: `docs/hierarchical_multihop_architecture.md`
- **内容**:
  - 详细的系统架构图
  - 完整的执行流程
  - 核心类详解
  - 配置选项说明
  - 性能优化技巧

## 🔄 工作流程

```
1. 初始化
   └─ 创建 Reasoning Agent 和 Summary Agent

2. 推理循环
   ├─ Reasoning Agent: 推理 + 搜索 (步骤1)
   ├─ Reasoning Agent: 推理 + 搜索 (步骤2)
   ├─ Reasoning Agent: 推理 + 搜索 (步骤3)
   └─ [触发条件: steps >= 3 或 tokens > 3000]

3. 压缩流程
   ├─ Summary Agent: 压缩历史消息
   ├─ 清空 Reasoning Agent 历史
   └─ 注入压缩摘要到 Reasoning Agent

4. 继续推理
   └─ Reasoning Agent: 基于压缩上下文继续

5. 循环直到问题解决
```

## 📁 文件结构

```
rllm/
├── agents/
│   ├── summary_agent.py                    # 新增：Summary Agent
│   ├── reasoning_agent.py                  # 新增：Reasoning Agent
│   └── hierarchical_multihop_agent.py      # 新增：协调器
│
├── docs/
│   └── hierarchical_multihop_architecture.md  # 新增：架构文档
│
└── examples/search/
    ├── train_hierarchical_multihop.py         # 新增：训练脚本
    ├── run_hierarchical_multihop.py           # 新增：推理脚本
    ├── quick_start_hierarchical.py            # 新增：快速开始
    └── HIERARCHICAL_MULTIHOP_README.md        # 新增：使用指南
```

## 🎛️ 关键配置

### Agent配置
```python
agent_args = {
    "tool_map": {"local_search": LocalRetrievalTool},
    "parser_name": "qwen",

    # 压缩配置
    "enable_compression": True,
    "compression_trigger": "both",  # "token", "step", "both"
    "max_tokens_before_compress": 3000,
    "max_steps_before_compress": 3,
    "max_summary_length": 300,
}
```

### 环境配置
```python
env_args = {
    "tool_map": tool_map,
    "reward_fn": search_reward_fn,
    "max_steps": 20,
}
```

## 🔧 使用方法

### 1. 快速测试
```bash
cd examples/search
python quick_start_hierarchical.py
```

### 2. 运行完整推理
```bash
# 确保检索服务器运行
cd examples/search/retrieval
bash launch_server.sh

# 确保模型服务运行
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 30000

# 运行推理
cd examples/search
python run_hierarchical_multihop.py
```

### 3. 训练模型

#### 联合训练（推荐）
```bash
cd examples/search
python train_hierarchical_multihop.py \
    --mode joint \
    --enable-compression \
    --compression-trigger both \
    --max-tokens 3000 \
    --max-steps 3
```

#### 只训练Summary
```bash
python train_hierarchical_multihop.py \
    --mode summary_only
```

#### 只训练Reasoning
```bash
python train_hierarchical_multihop.py \
    --mode reasoning_only
```

## 🎯 核心特性

### 1. 自动上下文压缩
- 监控token数和步数
- 可配置的触发条件
- 结构化摘要格式
- 保留关键信息

### 2. 灵活的训练模式
- **联合训练**: 端到端优化两个模型
- **分开训练**: 独立优化每个模型
- **迁移学习**: 基于预训练模型微调

### 3. 完整的轨迹记录
- 记录所有推理步骤
- 记录所有压缩操作
- 统计压缩次数
- 评估最终性能

### 4. 可扩展架构
- 易于添加新的检索工具
- 可自定义压缩策略
- 支持多种LLM后端

## 📊 性能特点

### 优势
1. **内存效率**: 通过压缩控制上下文长度
2. **可扩展性**: 支持更多跳数的复杂问题
3. **灵活性**: 可配置的压缩策略
4. **可解释性**: 明确的压缩历史记录

### 适用场景
- ✅ 复杂的多跳问答（4跳以上）
- ✅ 长上下文推理任务
- ✅ 需要多次搜索的问题
- ✅ 需要记忆管理的任务

### 不适用场景
- ❌ 简单的单跳问答（开销过大）
- ❌ 极短上下文任务
- ❌ 不需要搜索的任务

## 🔍 下一步

### 建议的优化方向

1. **模型优化**
   - [ ] 实现模型加载和保存
   - [ ] 添加LoRA微调支持
   - [ ] 实现模型并行

2. **压缩策略优化**
   - [ ] 实现选择性压缩（保留关键信息）
   - [ ] 添加多级压缩（压缩的压缩）
   - [ ] 动态调整压缩率

3. **功能增强**
   - [ ] 添加更多检索工具
   - [ ] 支持混合检索（dense + sparse）
   - [ ] 实现压缩质量评估

4. **性能优化**
   - [ ] 批处理压缩请求
   - [ ] 异步压缩
   - [ ] 缓存压缩结果

## 🧪 测试建议

### 单元测试
```bash
pytest tests/test_summary_agent.py
pytest tests/test_reasoning_agent.py
pytest tests/test_hierarchical_multihop_agent.py
```

### 集成测试
```python
# 测试完整流程
from rllm.agents.hierarchical_multihop_agent import HierarchicalMultiHopAgent

agent = HierarchicalMultiHopAgent(
    enable_compression=True,
    max_steps_before_compress=2,
)

# 验证压缩触发
assert agent.should_compress() == False
agent.steps_since_last_compress = 2
assert agent.should_compress() == True
```

## 📞 支持

- **文档**: 查看 `HIERARCHICAL_MULTIHOP_README.md`
- **架构**: 查看 `hierarchical_multihop_architecture.md`
- **示例**: 查看 `quick_start_hierarchical.py`

## 📝 更新日志

### v1.0.0 (2025-10-26)
- ✅ 实现 Summary Agent
- ✅ 实现 Reasoning Agent
- ✅ 实现 Hierarchical MultiHop Agent
- ✅ 创建训练脚本（支持3种模式）
- ✅ 创建推理脚本
- ✅ 创建快速开始示例
- ✅ 编写完整文档

---

**作者**: RLLM Development Team
**日期**: 2025-10-26
**版本**: 1.0.0
