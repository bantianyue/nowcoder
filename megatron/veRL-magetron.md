# veRL 调用 Megatron-LM 接口分析文档

> 本文档详细分析 veRL (Volcano Engine Reinforcement Learning) 框架如何调用 Megatron-LM 的接口，包括接口名称、实现功能以及 veRL 调用该接口的作用。

---

## 目录

1. [概述](#概述)
2. [核心并行模块接口](#核心并行模块接口)
3. [张量并行接口](#张量并行接口)
4. [流水线并行接口](#流水线并行接口)
5. [分布式优化器接口](#分布式优化器接口)
6. [模型检查点接口](#模型检查点接口)
7. [注意力机制接口](#注意力机制接口)
8. [工具函数接口](#工具函数接口)

---

## 概述

**veRL** 是字节跳动开发的强化学习训练框架，用于大语言模型的 PPO（Proximal Policy Optimization）训练。它利用 **Megatron-LM** 提供的分布式训练能力，实现高效的模型并行训练。

### 主要调用文件

| veRL 文件 | 用途 |
|-----------|------|
| `verl/workers/megatron_workers.py` | Megatron 训练工作器主入口 |
| `verl/model_merger/megatron_model_merger.py` | 模型检查点合并工具 |
| `verl/models/qwen2/megatron/modeling_qwen2_megatron.py` | Qwen2 模型的 Megatron 实现 |
| `verl/models/qwen2/megatron/layers/*.py` | Qwen2 模型的并行层实现 |
| `scripts/converter_hf_to_mcore.py` | HuggingFace 到 Megatron 格式转换 |
| `verl/utils/megatron_utils.py` | Megatron 工具函数 |
| `verl/utils/megatron/tensor_parallel.py` | 张量并行工具 |
| `verl/utils/megatron/sequence_parallel.py` | 序列并行工具 |

---

## 核心并行模块接口

### 1. 并行状态初始化接口

#### `mpu.initialize_model_parallel()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**函数签名:**
```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 10,
    tensor_parallel_ecd: bool = False,
    expert_tensor_parallel_size: int = 1,
) -> None:
```

**实现功能:**
- 初始化模型并行环境
- 创建通信进程组（TP、PP、CP、EP）
- 设置并行拓扑结构

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:270-279` (ActorRolloutRefWorker)
- `verl/workers/megatron_workers.py:1034-1043` (CriticWorker)
- `verl/workers/megatron_workers.py:1336-1345` (RewardModelWorker)
- `verl/model_merger/megatron_model_merger.py` (模型合并)
- `scripts/converter_hf_to_mcore.py:469-475`

**veRL 调用作用:**
```python
# 初始化 Actor/Reference 模型的并行环境
mpu.initialize_model_parallel(
    tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
    pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
    use_sharp=False,
    context_parallel_size=self.config.actor.megatron.context_parallel_size,
    expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
    expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
)
```

在 PPO 训练中，veRL 需要同时支持 Actor、Critic、Rollout、Reference 等多个模型，每个模型可能使用不同的并行策略。此接口确保每个模型的并行环境正确初始化。

---

### 2. 并行状态查询接口

#### `mpu.get_tensor_model_parallel_world_size()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**实现功能:** 返回张量并行世界的进程数量

**veRL 调用位置:**
- `verl/models/qwen2/megatron/layers/parallel_mlp.py:47`

**veRL 调用作用:**
```python
tp_size = mpu.get_tensor_model_parallel_world_size()

self.gate_up_proj = MergedColumnParallelLinear(
    input_size=self.hidden_size,
    gate_ouput_size=self.intermediate_size,
    up_output_size=self.intermediate_size,
    ...
)
self.gate_size = self.intermediate_size // tp_size
```

用于计算 MLP 层中每个 TP rank 应处理的中间层大小。

---

#### `mpu.get_tensor_model_parallel_rank()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**实现功能:** 返回当前进程在张量并行组中的 rank

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:283` (判断是否为主 rank)
- `verl/workers/megatron_workers.py:1046` (Critic)
- `verl/workers/megatron_workers.py:1348` (Reward Model)

**veRL 调用作用:**
```python
is_collect = (
    mpu.get_tensor_model_parallel_rank() == 0
    and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    and mpu.get_context_parallel_rank() == 0
)
self._register_dispatch_collect_info(
    mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
)
```

用于判断当前 rank 是否负责收集和分发结果，避免重复操作。

---

#### `mpu.get_pipeline_model_parallel_rank()` / `mpu.get_pipeline_model_parallel_world_size()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**实现功能:** 返回当前进程的流水线并行 rank 和总 stage 数量

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:128-129` (分布式转换)
- `scripts/converter_hf_to_mcore.py:544` (检查点加载)

**veRL 调用作用:**
```python
pp_rank = mpu.get_pipeline_model_parallel_rank()
pp_size = mpu.get_pipeline_model_parallel_world_size()

if pp_rank == 0:
    numel += safe_copy(hf_model.model.embed_tokens.weight, model.embedding.word_embeddings.weight)

if pp_rank == pp_size - 1:
    numel += safe_copy(hf_model.model.norm.weight, model.decoder.final_layernorm.weight)
```

在分布式模型转换时，根据 PP rank 确定当前进程需要处理哪些层的权重转换。

---

#### `mpu.get_expert_model_parallel_rank()` / `mpu.get_expert_model_parallel_world_size()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**实现功能:** 返回专家并行相关状态（用于 MoE 模型）

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:130-131, 344-345`

**veRL 调用作用:**
```python
ep_rank = mpu.get_expert_model_parallel_rank()
ep_size = mpu.get_expert_model_parallel_world_size()

# 在 MoE 模型转换中，根据 EP rank 分配专家权重
for idx, hf_expert in enumerate(hf_layer.mlp.experts):
    num_experts = len(hf_layer.mlp.experts)
    num_local_experts = num_experts // ep_size
    expert_idx_start = ep_rank * num_local_experts
    expert_idx_end = (ep_rank + 1) * num_local_experts
    if idx < expert_idx_start or idx >= expert_idx_end:
        continue
```

用于 MoE 模型的专家权重分片处理。

---

#### `mpu.get_data_parallel_rank()` / `mpu.get_data_parallel_world_size()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**实现功能:** 返回数据并行相关状态

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:288, 1051, 1353` (获取 DP rank)

**veRL 调用作用:**
```python
self._register_dispatch_collect_info(
    mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
)
```

用于数据分发和结果收集时的 rank 识别。

---

#### `mpu.get_context_parallel_rank()`

**Megatron 模块路径:** `megatron.core.parallel_state`

**实现功能:** 返回上下文并行 rank

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:285, 1048, 1350`

**veRL 调用作用:**
```python
is_collect = (
    mpu.get_tensor_model_parallel_rank() == 0
    and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    and mpu.get_context_parallel_rank() == 0
)
```

用于判断是否是 CP 主 rank，负责结果收集。

---

## 张量并行接口

### 1. 并行线性层接口

#### `tensor_parallel.ColumnParallelLinear`

**Megatron 模块路径:** `megatron.core.tensor_parallel`

**类签名:**
```python
class ColumnParallelLinear(torch.autograd.Function):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[ProcessGroup] = None,
    ):
```

**实现功能:**
- 按列分割权重的线性层（输出维度被分割）
- 在前向传播后可选择 gather 输出
- 支持梯度累积和通信重叠

**veRL 调用位置:**
- `verl/models/qwen2/megatron/modeling_qwen2_megatron.py` (LM Head)
- `verl/models/qwen2/megatron/layers/parallel_mlp.py:60-67` (MLP down_proj)
- `verl/models/qwen2/megatron/layers/parallel_linear.py:20-51` (QKV 并行线性层)

**veRL 调用作用:**
```python
# 用于 Qwen2 模型的 LM Head
self.lm_head = tensor_parallel.ColumnParallelLinear(
    input_size=config.hidden_size,
    output_size=config.vocab_size,
    bias=False,
    gather_output=False,  # 保持分片状态用于训练
    **column_kwargs,
)

# 用于 MLP 的 down projection 层
self.down_proj = tensor_parallel.RowParallelLinear(
    input_size=self.intermediate_size,
    output_size=self.hidden_size,
    bias=False,
    input_is_parallel=True,
    skip_bias_add=False,
    **row_kwargs,
)

# 用于 QKV 投影
class QKVParallelLinear(tensor_parallel.ColumnParallelLinear):
    def __init__(self, input_size, num_heads, num_key_value_heads, head_dim, ...):
        output_size = (num_heads + 2 * num_key_value_heads) * head_dim
        super().__init__(input_size=input_size, output_size=output_size, ...)
```

---

#### `tensor_parallel.RowParallelLinear`

**Megatron 模块路径:** `megatron.core.tensor_parallel`

**类签名:**
```python
class RowParallelLinear(torch.autograd.Function):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = False,
        skip_bias_add: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[ProcessGroup] = None,
    ):
```

**实现功能:**
- 按行分割权重的线性层（输入维度被分割）
- 在前向传播前进行 all-gather 操作
- 支持异步 all-reduce 通信重叠

**veRL 调用位置:**
- `verl/models/qwen2/megatron/layers/parallel_mlp.py:60-67`

**veRL 调用作用:**
```python
self.down_proj = tensor_parallel.RowParallelLinear(
    input_size=self.intermediate_size,
    output_size=self.hidden_size,
    bias=False,
    input_is_parallel=True,  # 输入已经是并行状态
    skip_bias_add=False,
    **row_kwargs,
)
```

---

### 2. 并行嵌入层接口

#### `tensor_parallel.VocabParallelEmbedding`

**Megatron 模块路径:** `megatron.core.tensor_parallel`

**类签名:**
```python
class VocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        config: ModelParallelConfig,
        init_method: Callable = init.xavier_normal_,
    ):
```

**实现功能:**
- 按词汇表维度分割的嵌入层
- 自动处理跨 rank 的嵌入查找
- 支持梯度同步

**veRL 调用位置:**
- `verl/models/qwen2/megatron/modeling_qwen2_megatron.py`

**veRL 调用作用:**
```python
self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
    num_embeddings=config.vocab_size,
    embedding_dim=config.hidden_size,
    **embedding_kwargs
)
```

将词嵌入表按 TP rank 分片，减少每个 rank 的显存占用。

---

### 3. 随机数管理接口

#### `tensor_parallel.model_parallel_cuda_manual_seed()`

**Megatron 模块路径:** `megatron.core.tensor_parallel.random`

**函数签名:**
```python
def model_parallel_cuda_manual_seed(seed: int) -> None:
```

**实现功能:**
- 为所有 TP rank 设置相同的 CUDA 随机种子
- 确保跨 rank 的初始化一致性
- 支持 dropout 等操作的确定性

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:100` (set_random_seed 函数)
- `scripts/converter_hf_to_mcore.py:476`

**veRL 调用作用:**
```python
def set_random_seed(seed, only_rollout=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_rollout and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)
```

确保 PPO 训练中所有 TP rank 的随机操作（如 dropout、随机初始化）保持一致，保证训练可复现性。

---

### 4. 张量并行工具接口

#### `tensor_parallel.vocab_parallel_cross_entropy()`

**Megatron 模块路径:** `megatron.core.tensor_parallel`

**函数签名:**
```python
def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
```

**实现功能:**
- 在 vocab 并行状态下计算交叉熵损失
- 无需 gather 完整 logits
- 支持 label smoothing

**veRL 调用位置:**
- `verl/utils/megatron/tensor_parallel.py:156-158`

**veRL 调用作用:**
```python
def vocab_parallel_log_probs_from_logits(logits, labels):
    from megatron.core import tensor_parallel
    return -tensor_parallel.vocab_parallel_cross_entropy(
        vocab_parallel_logits=logits, target=labels
    )
```

在 PPO 训练中计算策略梯度损失时，避免 gather 大规模的 vocab logits，节省显存和通信开销。

---

## 流水线并行接口

### 1. 流水线调度接口

#### `get_dynamic_pipeline_shards()`

**Megatron 模块路径:** `megatron.core.pipeline_parallel`

**实现功能:**
- 计算动态流水线分片策略
- 根据层数和 PP size 均匀分配层到各 stage

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:490`

**veRL 调用作用:**
```python
pipeline_shards = get_dynamic_pipeline_shards(hf_config.num_hidden_layers, pp_size)
print(f"Pipeline shards: {pipeline_shards}")

tfconfig = hf_to_mcore_config(
    hf_config,
    torch.bfloat16,
    num_layers_in_first_pipeline_stage=pipeline_shards[0] if len(pipeline_shards) > 1 else None,
    num_layers_in_last_pipeline_stage=pipeline_shards[-1] if len(pipeline_shards) > 2 else None,
)
```

在模型格式转换时，根据流水线并行配置合理分配 Transformer 层到各个 pipeline stage。

---

## 分布式优化器接口

### 1. 分布式优化器初始化

#### `get_megatron_optimizer()`

**Megatron 模块路径:** `megatron.core.optimizer` (通过 veRL wrapper)

**实现功能:**
- 创建分布式优化器实例
- 支持 ZeRO-1/2/3 分片策略
- 处理参数分片和梯度同步

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:473` (Actor 优化器)
- `verl/workers/megatron_workers.py:1146` (Critic 优化器)

**veRL 调用作用:**
```python
actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
```

为 PPO 训练创建支持分布式训练的优化器，实现参数分片和梯度同步。

---

#### `get_megatron_optimizer_param_scheduler()`

**Megatron 模块路径:** `megatron.core.optimizer`

**实现功能:**
- 创建学习率调度器
- 支持 warmup、衰减等策略
- 与分布式优化器集成

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:474-476` (Actor)
- `verl/workers/megatron_workers.py:1147-1149` (Critic)

**veRL 调用作用:**
```python
actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
    optimizer=actor_optimizer, config=optim_config
)
```

在 PPO 训练中管理学习率变化。

---

#### `get_megatron_last_lr()`

**Megatron 模块路径:** `megatron.core.optimizer`

**实现功能:** 获取优化器当前学习率

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:753` (Actor 更新)
- `verl/workers/megatron_workers.py:1262` (Critic 更新)

**veRL 调用作用:**
```python
metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
self.actor_optimizer_scheduler.step(1)
```

记录训练时的学习率变化。

---

### 2. 优化器状态管理

#### `register_megatron_training_hooks()`

**Megatron 模块路径:** `megatron.core.optimizer`

**实现功能:**
- 注册前向/反向传播钩子
- 处理梯度累积和同步
- 支持 activation checkpointing

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:484` (Actor)
- `verl/workers/megatron_workers.py:1152` (Critic)

**veRL 调用作用:**
```python
register_megatron_training_hooks(actor_module, actor_optimizer)
```

注册训练所需的钩子函数，确保分布式训练的正确执行。

---

## 模型检查点接口

### 1. 分布式检查点保存/加载

#### `dist_checkpointing.save()`

**Megatron 模块路径:** `megatron.core.dist_checkpointing`

**函数签名:**
```python
def save(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    sharded_strategy: Optional[ShardedStrategy] = None,
    async_sharded_save: bool = False,
) -> None:
```

**实现功能:**
- 保存分片模型权重
- 支持异步保存
- 处理 TP/PP/EP 参数分布

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:595`

**veRL 调用作用:**
```python
dist_checkpointing.save(
    megatron_state_dict,
    output_path,
    sharded_strategy=None,
    async_sharded_save=False
)
```

在格式转换后保存 Megatron 格式的分布式检查点。

---

#### `dist_checkpointing.load()`

**Megatron 模块路径:** `megatron.core.dist_checkpointing`

**函数签名:**
```python
def load(
    sharded_state_dict: ShardedStateDict,
    checkpoint_dir: str,
    strict: StrictHandling = StrictHandling.ASSUME_OK_UNEXPECTED,
) -> ShardedStateDict:
```

**实现功能:**
- 加载分片模型权重
- 支持 TP/PP/EP 参数合并
- 处理缺失/额外参数

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:83`

**veRL 调用作用:**
```python
dist_checkpointing.load(
    ref_state_dict,
    output_path,
    strict=StrictHandling.ASSUME_OK_UNEXPECTED
)
```

在测试转换结果时加载保存的检查点进行验证。

---

### 2. 状态字典接口

#### `sharded_state_dict()`

**Megatron 模块路径:** `megatron.core.transformer.module.MegatronModule`

**实现功能:**
- 生成模型的分片状态字典
- 包含分片元数据（offset、fragments）
- 支持跨 TP/PP 的状态重组

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:82, 590`

**veRL 调用作用:**
```python
ref_state_dict = model_test[0].module.sharded_state_dict()
megatron_state_dict = model[0].module.sharded_state_dict()
```

在保存检查点前生成包含分片信息的状态字典。

---

#### `ShardedTensor`

**Megatron 模块路径:** `megatron.core.dist_checkpointing.mapping`

**实现功能:**
- 表示分片张量的元数据类
- 存储分片形状、偏移、副本信息
- 支持跨 rank 的张量重组

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:93-96, 107-110`

**veRL 调用作用:**
```python
if isinstance(ref_data, ShardedTensor):
    ref_data = ref_data.data.view(ref_data.local_shape)
else:
    ref_data = ref_data.data
```

处理分片张量的数据提取和比较。

---

#### `ShardedTensorFactory`

**Megatron 模块路径:** `megatron.core.dist_checkpointing.mapping`

**实现功能:**
- 用于 SwiGLU 等需要特殊分片处理的层
- 在保存时自动 chunk、加载时自动 cat

**veRL 调用位置:**
- `megatron/core/transformer/mlp.py:345-352` (通过 Megatron-LM)

**veRL 调用作用:**
```python
return ShardedTensorFactory(
    original_sh_ten.key,
    original_sh_ten.data,
    sh_ten_build_fn,  # chunk 函数
    sh_ten_merge_fn,  # cat 函数
    original_sh_ten.replica_id,
    flattened_range=original_sh_ten.flattened_range,
)
```

处理 SwiGLU 层的 gate 和 up 投影的分片保存/加载。

---

### 3. 异步保存接口

#### `async_calls.maybe_finalize_async_calls()`

**Megatron 模块路径:** `megatron.core.dist_checkpointing.strategies.base`

**实现功能:**
- 完成异步保存操作
- 确保所有数据写入磁盘
- 支持非阻塞 checkpoint

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:935-937`

**veRL 调用作用:**
```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def async_calls_finalize_fn_exec(self, blocking=False):
    from megatron.core.dist_checkpointing.strategies.base import async_calls
    async_calls.maybe_finalize_async_calls(blocking=blocking)
```

在 PPO 训练中完成异步检查点保存，避免阻塞训练流程。

---

## 注意力机制接口

### 1. 注意力后端枚举

#### `AttnBackend`

**Megatron 模块路径:** `megatron.core.transformer.enums`

**实现功能:**
- 定义注意力后端类型
- 支持 flash、local、unfused 等

**veRL 调用位置:**
- `verl/workers/megatron_workers.py:200-203`

**veRL 调用作用:**
```python
from megatron.core.transformer.enums import AttnBackend

provider.attention_backend = AttnBackend.flash
provider.variable_seq_lengths = True
provider.moe_token_dispatcher_type = "alltoall"
```

配置使用 FlashAttention 后端以提高训练效率。

---

## 工具函数接口

### 1. 模型类型枚举

#### `ModelType`

**Megatron 模块路径:** `megatron.core.models.gpt.gpt_model`

**实现功能:**
- 定义模型类型（encoder_or_decoder、encoder_only、decoder_only）

**veRL 调用位置:**
- `scripts/converter_hf_to_mcore.py:78, 520`

**veRL 调用作用:**
```python
model_test = get_model(
    model_provider_func=megatron_model_provider,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    transformer_config=tfconfig,
)
```

在创建模型时指定模型类型。

---

### 2. 并行配置类

#### `ModelParallelConfig`

**Megatron 模块路径:** `megatron.core.model_parallel_config`

**实现功能:**
- 核心并行配置类
- 包含所有并行策略参数

**veRL 调用位置:**
- `verl/models/qwen2/megatron/layers/parallel_rmsnorm.py:27`
- `verl/models/qwen2/megatron/layers/parallel_mlp.py:31`
- `verl/utils/megatron/tensor_parallel.py:31, 64, 85`

**veRL 调用作用:**
```python
class ParallelQwen2RMSNorm(nn.Module):
    def __init__(self, config: Qwen2Config, megatron_config: ModelParallelConfig):
        ...
        if megatron_config.sequence_parallel:
            sp_utils.mark_parameter_as_sequence_parallel(self.weight)
```

传递并行配置给模型组件，控制并行行为。

---

### 3. 序列并行工具

#### `sequence_parallel.mark_parameter_as_sequence_parallel()`

**Megatron 模块路径:** `megatron.core.tensor_parallel` (通过 veRL wrapper)

**实现功能:**
- 标记参数为序列并行状态
- 用于 LayerNorm 等需要跨 TP 同步的层

**veRL 调用位置:**
- `verl/models/qwen2/megatron/layers/parallel_rmsnorm.py:39`

**veRL 调用作用:**
```python
if megatron_config.sequence_parallel:
    sp_utils.mark_parameter_as_sequence_parallel(self.weight)
```

启用序列并行时，标记 RMSNorm 的权重参数需要特殊的分布式处理。

---

## 完整接口调用汇总表

| 接口名称 | Megatron 模块 | veRL 调用文件 | 主要用途 |
|---------|--------------|-------------|---------|
| `mpu.initialize_model_parallel()` | `parallel_state` | `megatron_workers.py`, `converter_hf_to_mcore.py` | 初始化模型并行环境 |
| `mpu.get_tensor_model_parallel_world_size()` | `parallel_state` | `parallel_mlp.py` | 获取 TP 进程数 |
| `mpu.get_tensor_model_parallel_rank()` | `parallel_state` | `megatron_workers.py` | 获取 TP rank |
| `mpu.get_pipeline_model_parallel_rank()` | `parallel_state` | `converter_hf_to_mcore.py` | 获取 PP rank |
| `mpu.get_pipeline_model_parallel_world_size()` | `parallel_state` | `converter_hf_to_mcore.py` | 获取 PP stage 数 |
| `mpu.get_expert_model_parallel_rank()` | `parallel_state` | `converter_hf_to_mcore.py` | 获取 EP rank |
| `mpu.get_expert_model_parallel_world_size()` | `parallel_state` | `converter_hf_to_mcore.py` | 获取 EP 进程数 |
| `mpu.get_data_parallel_rank()` | `parallel_state` | `megatron_workers.py` | 获取 DP rank |
| `mpu.get_data_parallel_world_size()` | `parallel_state` | `megatron_workers.py` | 获取 DP 进程数 |
| `mpu.get_context_parallel_rank()` | `parallel_state` | `megatron_workers.py` | 获取 CP rank |
| `tensor_parallel.ColumnParallelLinear` | `tensor_parallel` | `modeling_qwen2_megatron.py`, `parallel_mlp.py` | 列并行线性层 |
| `tensor_parallel.RowParallelLinear` | `tensor_parallel` | `parallel_mlp.py` | 行并行线性层 |
| `tensor_parallel.VocabParallelEmbedding` | `tensor_parallel` | `modeling_qwen2_megatron.py` | 词汇表并行嵌入 |
| `tensor_parallel.model_parallel_cuda_manual_seed()` | `tensor_parallel.random` | `megatron_workers.py` | 设置 TP 随机种子 |
| `tensor_parallel.vocab_parallel_cross_entropy()` | `tensor_parallel` | `tensor_parallel.py` | 并行交叉熵计算 |
| `get_megatron_optimizer()` | `optimizer` | `megatron_workers.py` | 创建分布式优化器 |
| `get_megatron_optimizer_param_scheduler()` | `optimizer` | `megatron_workers.py` | 创建学习率调度器 |
| `get_megatron_last_lr()` | `optimizer` | `megatron_workers.py` | 获取当前学习率 |
| `register_megatron_training_hooks()` | `optimizer` | `megatron_workers.py` | 注册训练钩子 |
| `dist_checkpointing.save()` | `dist_checkpointing` | `converter_hf_to_mcore.py` | 保存分布式检查点 |
| `dist_checkpointing.load()` | `dist_checkpointing` | `converter_hf_to_mcore.py` | 加载分布式检查点 |
| `sharded_state_dict()` | `transformer.module` | `converter_hf_to_mcore.py` | 生成分片状态字典 |
| `ShardedTensor` | `dist_checkpointing.mapping` | `converter_hf_to_mcore.py` | 分片张量元数据 |
| `ShardedTensorFactory` | `dist_checkpointing.mapping` | `mlp.py` (Megatron) | 动态分片工厂 |
| `async_calls.maybe_finalize_async_calls()` | `dist_checkpointing.strategies.base` | `megatron_workers.py` | 完成异步保存 |
| `AttnBackend` | `transformer.enums` | `megatron_workers.py` | 注意力后端类型 |
| `ModelType` | `models.gpt.gpt_model` | `converter_hf_to_mcore.py` | 模型类型枚举 |
| `ModelParallelConfig` | `model_parallel_config` | `parallel_rmsnorm.py`, `parallel_mlp.py` | 并行配置类 |
| `get_dynamic_pipeline_shards()` | `pipeline_parallel` | `converter_hf_to_mcore.py` | 计算流水线分片 |

---

## veRL 架构中的 Megatron 集成模式

### 1. PPO 训练流程中的接口调用

```
初始化阶段:
├── mpu.initialize_model_parallel()     # 初始化并行环境
├── tensor_parallel.model_parallel_cuda_manual_seed()  # 设置随机种子
└── get_megatron_optimizer()            # 创建优化器

训练循环:
├── register_megatron_training_hooks()  # 注册训练钩子
├── 前向传播 (使用 ColumnParallelLinear/RowParallelLinear)
├── 损失计算 (使用 vocab_parallel_cross_entropy)
├── 反向传播 (梯度自动分片)
└── 优化器更新 (get_megatron_last_lr 获取学习率)

检查点保存:
├── sharded_state_dict()                # 生成分片状态
├── dist_checkpointing.save()           # 保存检查点
└── async_calls.maybe_finalize_async_calls()  # 完成异步保存
```

### 2. 模型格式转换流程

```
HF -> Megatron 转换:
├── mpu.initialize_model_parallel()     # 初始化并行环境
├── get_dynamic_pipeline_shards()       # 计算流水线分片
├── 权重映射 (根据 TP/PP/EP rank 分配权重)
├── sharded_state_dict()                # 生成分片状态字典
└── dist_checkpointing.save()           # 保存 Megatron 格式

Megatron -> HF 合并:
├── 加载分片检查点
├── TP 参数合并 (torch.cat)
├── PP 参数合并
├── EP 参数合并
└── 保存 HuggingFace 格式
```

---

## 参考资源

- **Megatron-LM 仓库**: https://github.com/NVIDIA/Megatron-LM
- **Megatron Core 文档**: https://docs.nvidia.com/Megatron-Core/
- **veRL 仓库**: https://github.com/volcengine/verl
- **相关论文**:
  - Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2019)
  - Reducing Activation Recomputation in Large Transformer Models (2021)
  - DeepSpeed-Megatron: Megatron-LM + DeepSpeed for Extreme-Scale Language Models (2021)
