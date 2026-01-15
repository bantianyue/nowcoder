# Megatron-LM 架构分析文档

> 本文档深入分析 Megatron-LM 项目的代码架构、技术架构和核心实现技术
>
> 生成日期: 2026-01-12

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术架构](#2-技术架构)
- [3. 代码架构](#3-代码架构)
- [4. 核心实现技术](#4-核心实现技术)
- [5. 分布式训练模块深度分析](#5-分布式训练模块深度分析)
- [6. 流水线并行模块深度分析](#6-流水线并行模块深度分析)
- [7. 张量并行模块深度分析](#7-张量并行模块深度分析)

---

## 1. 项目概述

### 1.1 项目简介

Megatron-LM 是 NVIDIA 开发的大规模 Transformer 模型训练框架，专门用于训练从数十亿到万亿参数级别的语言模型。该项目提供了全面的 GPU 优化、分布式训练支持和高效的内存管理。

### 1.2 核心特性

- **多维并行策略**: 支持数据并行、张量并行、流水线并行、上下文并行、专家并行
- **混合精度训练**: FP16、BF16、FP8、FP4 支持
- **内存优化**: 激活检查点、梯度累积、内存高效的注意力机制
- **模型支持**: GPT、BERT、T5、Mamba、多模态模型
- **MoE 支持**: 完整的混合专家模型训练支持
- **生产就绪**: 支持大规模部署（数千 GPU）

---

## 2. 技术架构

### 2.1 整体架构设计

Megatron-LM 采用分层模块化架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                 │
│    预训练脚本 (pretrain_*.py) | 微调脚本 | 推理服务           │
├─────────────────────────────────────────────────────────────┤
│                  模型层 (Model Layer)                         │
│    GPT | BERT | T5 | Mamba | Multimodal | Retro              │
├─────────────────────────────────────────────────────────────┤
│                核心引擎层 (Megatron Core)                     │
│    Transformer Engine | 模型构建块 | 注意力机制               │
├─────────────────────────────────────────────────────────────┤
│              并行计算层 (Parallel Computing Layer)            │
│    Distributed | Pipeline Parallel | Tensor Parallel         │
├─────────────────────────────────────────────────────────────┤
│              基础设施层 (Infrastructure Layer)                │
│    数据加载 | 检查点管理 | 日志 | 优化器                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 配置系统架构

#### TransformerConfig 核心配置

Megatron-LM 使用 `TransformerConfig` 数据类作为中心配置对象，包含超过 200 个配置参数：

```python
@dataclass
class TransformerConfig(ModelParallelConfig):
    # 模型架构参数
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    ffn_hidden_size: Optional[int] = None

    # 并行配置
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1

    # 混合精度配置
    fp16: bool = False
    bf16: bool = False
    fp8: Optional[str] = None

    # MoE 配置
    num_moe_experts: Optional[int] = None
    moe_router_load_balancing_type: str = "aux_loss"

    # 激活重计算配置
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
```

#### 配置验证机制

配置类实现了严格的参数验证和兼容性检查：

- 并行维度验证（head 数必须能被 TP 大小整除）
- 精度配置验证（不能同时启用 FP16 和 BF16）
- MoE 参数验证
- 流水线并行布局验证
- CUDA Graph 兼容性验证

### 2.3 训练流程架构

```
初始化 Megatron
    │
    ├─→ 设置全局变量
    ├─→ 初始化分布式环境
    │   └─→ 创建进程组（TP/PP/DP/EP/CP）
    │
    ├─→ 构建模型
    │   └─→ 使用 LayerSpec 定义层结构
    │
    ├─→ 设置数据集
    │   └─→ BlendedMegatronDatasetBuilder
    │
    └─→ 训练循环
        ├─→ Warmup 阶段（流水线并行）
        ├─→ 稳态训练
        │   ├─→ 前向传播
        │   ├─→ 损失计算
        │   ├─→ 反向传播
        │   └─→ 参数更新
        ├─→ Cooldown 阶段（流水线并行）
        └─→ 检查点保存
```

---

## 3. 代码架构

### 3.1 目录结构分析

```
Megatron-LM/
├── megatron/
│   ├── core/                           # 核心引擎
│   │   ├── models/                     # 模型实现
│   │   │   ├── gpt/                    # GPT 模型
│   │   │   ├── bert/                   # BERT 模型
│   │   │   ├── T5/                     # T5 模型
│   │   │   ├── mamba/                  # Mamba 模型
│   │   │   ├── multimodal/             # 多模态模型
│   │   │   └── common/                 # 通用组件
│   │   ├── transformer/                # Transformer 构建块
│   │   │   ├── attention/              # 注意力机制
│   │   │   ├── mlp/                    # 前馈网络
│   │   │   └── module/                 # 基础模块
│   │   ├── distributed/                # 分布式训练 ★
│   │   ├── pipeline_parallel/          # 流水线并行 ★
│   │   ├── tensor_parallel/            # 张量并行 ★
│   │   ├── datasets/                   # 数据加载
│   │   ├── optimizer/                  # 优化器
│   │   ├── inference/                  # 推理引擎
│   │   └── export/                     # 模型导出
│   ├── training/                       # 训练工具
│   ├── inference/                      # 推理服务器
│   ├── legacy/                         # 向后兼容
│   └── post_training/                  # 训练后处理
├── examples/                           # 示例脚本
├── tools/                              # 工具脚本
├── tests/                              # 测试套件
└── docs/                               # 文档
```

> 标记 ★ 的模块将在后续章节深入分析

### 3.2 模型构建架构

#### 3.2.1 LayerSpec 系统

Megatron-LM 引入了 `LayerSpec` 系统来灵活定义层结构：

```python
# GPT Layer Spec 示例
 transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
    num_experts=num_moe_experts,
    moe_grouped_gemm=moe_grouped_gemm,
    qk_layernorm=qk_layernorm,
    ...
)
```

**LayerSpec 组成部分：**

- `self_attention`: 自注意力模块规格
- `self_attention_attn_mask`: 注意力掩码
- `pre_mlp_layernorm`: MLP 前归一化
- `mlp`: MLP 模块规格（可以是 Dense 或 MoE）
- `pre_cross_attention_layernorm`: 交叉注意力前归一化
- `cross_attention`: 交叉注意力模块
- `post_self_attention_layernorm`: 自注意力后归一化

#### 3.2.2 构建器模式

模型使用构建器模式创建，支持多种配置：

```python
def gpt_builder(args, pre_process, post_process):
    if args.use_legacy_models:
        # 旧版模型
        model = legacy.model.GPTModel(...)
    else:
        # 新版核心模型
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            ...
        )
    return model
```

### 3.3 数据架构

#### 3.3.1 数据集层次结构

```
BlendedMegatronDatasetBuilder
    │
    ├─→ LowLevelDataset (基础数据集)
    │   └─→ 内存映射数据集 / 分词数据集
    │
    ├─→ MegatronDataset (中间层)
    │   ├─→ 并行采样支持
    │   ├─→ 分片管理
    │   └─→ 数据预处理
    │
    ├─→ BlendedDataset (混合层)
    │   ├─→ 多数据集加权混合
    │   ├─→ 动态比例调整
    │   └─→ 数据集分割
    │
    └─→ TopLevelDataset (最终数据集)
        ├─→ 数据增强
        ├─→ 批处理
        └─→ 动态批大小
```

#### 3.3.2 数据加载优化

**关键优化技术：**

1. **分布式采样**: 每个进程只加载需要的数据分片
2. **预取机制**: 异步加载下一批数据
3. **内存映射**: 避免全部加载到内存
4. **动态批处理**: 根据序列长度动态组合样本

---

## 4. 核心实现技术

### 4.1 混合精度训练

#### 4.1.1 精度类型

| 精度类型 | 存储位数 | 用途 | 优势 |
|---------|---------|------|------|
| **FP32** | 32 bits | 权重更新、梯度累积 | 数值稳定性高 |
| **FP16** | 16 bits | 前向/反向传播 | 内存占用小，速度快 |
| **BF16** | 16 bits | 前向/反向传播 | 数值范围大，训练稳定 |
| **FP8** | 8 bits | 权重、激活、梯度 | 极致内存优化 |
| **FP4** | 4 bits | 权重存储 | Blackwell 架构专用 |

#### 4.1.2 FP8 延迟缩放 (Delayed Scaling)

```python
# FP8 配置示例
config.fp8 = "hybrid"  # e4m3 for weights/activations, e5m2 for gradients
config.fp8_recipe = "delayed"
config.fp8_amax_history_len = 1
config.fp8_amax_compute_algo = "most_recent"
```

**缩放因子计算：**

```python
# 伪代码：延迟缩放算法
if step == 0:
    scale = 1.0
else:
    # 根据历史 amax 计算缩放因子
    amax = max(amax_history)
    scale = fp8_max / amax
```

### 4.2 激活重计算 (Activation Recomputation)

#### 4.2.1 选择性重计算 (Selective Recomputation)

Megatron-LM 默认使用选择性重计算，只重计算内存密集部分：

```python
config.recompute_granularity = "selective"
config.recompute_modules = ["core_attn"]  # 默认
```

**支持的重计算模块：**

- `core_attn`: 核心注意力部分
- `moe_act`: MoE 激活函数
- `layernorm`: 层归一化
- `mla_up_proj`: MLA 上投影
- `mlp`: MLP 模块
- `moe`: MoE 层
- `shared_experts`: MoE 共享专家

#### 4.2.2 输出丢弃检查点

```python
class CheckpointWithoutOutput:
    def discard_output_and_register_recompute(self, hook_tensor):
        # 释放输出存储
        for output in self.outputs:
            output.untyped_storage().resize_(0)
        # 注册重新计算钩子
        hook_tensor.register_hook(self._recompute)
```

### 4.3 Flash Attention 集成

#### 4.3.1 注意力后端选择

```python
config.attention_backend = AttnBackend.auto  # 自动选择
# 可选: auto, flash, local, unfused
```

#### 4.3.2 Flash Attention v3

```python
if HAVE_FA3:
    output = flash_attn3_with_kvcache(
        q, k, v, kv_cache,
        softmax_scale,
        causal=True,
        ...
    )
```

**优势：**
- 内存复杂度 O(N) 而非 O(N²)
- 减少内存访问次数
- 支持长序列训练

### 4.4 CUDA Graph 优化

#### 4.4.1 CUDA Graph 作用域

```python
config.cuda_graph_impl = "transformer_engine"
config.cuda_graph_scope = [
    CudaGraphScope.attn,      # 注意力层
    CudaGraphScope.mlp,       # MLP 层
    CudaGraphScope.moe,       # MoE 层
    CudaGraphScope.moe_router # MoE 路由器
]
```

#### 4.4.2 CUDA Graph 优势

- 减少内核启动开销
- 优化内存分配
- 提升推理吞吐量 20-30%

---

## 5. 分布式训练模块深度分析

> 目录: `megatron/core/distributed/`

### 5.1 模块概述

分布式训练模块是 Megatron-LM 的核心，实现了高效的数据并行和分布式优化器支持。

### 5.2 目录结构与文件功能

```
distributed/
├── __init__.py                          # 模块导出
├── data_parallel_base.py               # DDP 基类
├── distributed_data_parallel.py         # 核心 DDP 实现
├── distributed_data_parallel_config.py  # DDP 配置
├── param_and_grad_buffer.py            # 参数梯度缓冲区
├── finalize_model_grads.py              # 模型梯度同步
└── fsdp/                                # FSDP 实现
    ├── mcore_fsdp_adapter.py
    └── src/megatron_fsdp/
```

### 5.3 并行状态管理

虽然 `parallel_state.py` 不在 distributed 目录中，但它是分布式训练的核心：

```python
# 全局进程组变量
_TENSOR_MODEL_PARALLEL_GROUP = None        # 张量并行组
_PIPELINE_MODEL_PARALLEL_GROUP = None      # 流水线并行组
_DATA_PARALLEL_GROUP = None               # 数据并行组
_EXPERT_MODEL_PARALLEL_GROUP = None       # 专家并行组
_EXPERT_DATA_PARALLEL_GROUP = None        # 专家数据并行组
_CONTEXT_PARALLEL_GROUP = None             # 上下文并行组
```

#### 5.3.1 初始化函数

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    order: str = "tp-cp-ep-dp-pp",  # 并行顺序
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    ...
):
```

**关键特性：**

1. **灵活的并行顺序**: 通过 `order` 参数支持不同的并行化策略组合
2. **层次化进程组**: 支持嵌套的进程组结构
3. **NCCL 优化**: 支持自定义 NCCL 通信配置

#### 5.3.2 Rank 生成器

```python
class RankGenerator:
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str):
        self.name_to_size = {
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "ep": ep,
            "cp": cp,
        }
        self.order = order

    def get_rank(self, global_rank: int, parallel_name: str) -> int:
        """根据全局 rank 和并行类型计算特定并行域的 rank"""
```

### 5.4 梯度同步机制

#### 5.4.1 Bucketing 策略

```python
class _ParamAndGradBucket:
    """梯度桶实现"""
    def __init__(self, params, param_data, grad_data, offset,
                 numel_unpadded, gradient_scaling_factor, bucket_id):
        self.params_list = params
        self.param_data = param_data      # 参数数据
        self.grad_data = grad_data        # 梯度数据
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
```

**Bucket 分组策略：**

```python
def partition_buffers(buffers, force_single_bucket_group=False):
    if force_single_bucket_group:
        # 强制单组
        return [_ParamAndGradBucketGroup(all_buckets)]
    elif no_fp8:
        # 每个 bucket 独立
        return [bucket_group for bucket in buffers]
    else:
        # FP8 优化：合并非 FP8 bucket
        return merge_fp8_buckets(buffers)
```

#### 5.4.2 梯度同步流程

```python
class _ParamAndGradBucketGroup:
    def start_grad_sync(self):
        """启动梯度同步"""
        # 1. 梯度缩放
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # 2. 选择归约操作
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # 3. 执行通信
        async_op = self.ddp_config.overlap_grad_reduce
        with _coalescing_manager(communication_group, async_ops=async_op):
            for bucket in self.buckets:
                if self.ddp_config.use_distributed_optimizer:
                    # 分布式优化器使用 reduce-scatter
                    dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op
                    )
                else:
                    # 传统 DDP 使用 all-reduce
                    torch.distributed.all_reduce(
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op
                    )
```

### 5.5 参数同步机制

#### 5.5.1 All-Gather 实现

```python
def start_param_sync(self, force_sync=False):
    """启动参数同步（all-gather）"""
    async_op = (
        self.ddp_config.overlap_param_gather and
        not force_sync
    )

    with _coalescing_manager(
        self.intra_distributed_optimizer_instance_group,
        async_ops=async_op
    ) as cm:
        for bucket in self.buckets:
            # 参数 all-gather
            dist_all_gather_func(
                bucket.param_data,
                local_data_view,
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op
            )
```

#### 5.5.2 梯度累积融合

```python
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion, ...):
        ctx.save_for_backward(input, weight)
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion

        # 计算前向输出
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # CUDA 内核融合的梯度累积
        if weight.main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input, grad_output, weight.main_grad
            )
        elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                total_input, grad_output, weight.main_grad
            )
```

### 5.6 通信优化技术

#### 5.6.1 通信计算重叠

```python
# 支持异步通信
async_op = self.ddp_config.overlap_grad_reduce
with _coalescing_manager(communication_group, async_ops=async_op) as cm:
    # 异步通信操作
    handle = torch.distributed.all_reduce(
        data,
        group=communication_group,
        async_op=True
    )
    # 在等待通信完成的同时进行计算
    result = compute_something()
    handle.wait()  # 等待通信完成
```

#### 5.6.2 通信融合

```python
@contextlib.contextmanager
def _coalescing_manager(group, async_ops=False):
    """融合多个通信操作"""
    if async_ops:
        handles = []
        yield _CoalescedHandles(handles)

        # 批量执行所有通信操作
        reqs = torch.distributed.batch_isend_irecv(handles)
        for req in reqs:
            req.wait()
    else:
        yield None
```

#### 5.6.3 NCCL 优化

```python
def get_nccl_options(pg_name, nccl_comm_cfgs):
    nccl_options = torch.distributed.ProcessGroupNCCL.Options(
        is_high_priority_stream=nccl_comm_cfgs[pg_name].get(
            "is_high_priority_stream", False
        )
    )
    if "cga_cluster_size" in nccl_comm_cfgs[pg_name]:
        nccl_options.config.cga_cluster_size = (
            nccl_comm_cfgs[pg_name]["cga_cluster_size"]
        )
    return nccl_options
```

### 5.7 专家并行支持

#### 5.7.1 专家模型并行

```python
# 专家在专家之间分配
_EXPERT_MODEL_PARALLEL_GROUP = None

def get_expert_model_parallel_group():
    """获取专家模型并行组"""
    return _EXPERT_MODEL_PARALLEL_GROUP
```

#### 5.7.2 专家梯度缩放

```python
# 专家梯度特殊缩放
if self.ddp_config.average_in_collective:
    expert_gradient_scaling_factor = (
        self.expt_dp_group.size() / self.dp_cp_group.size()
    )
else:
    expert_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
```

### 5.8 设计模式分析

#### 5.8.1 策略模式

- **通信策略**: 根据配置选择 all-reduce 或 reduce-scatter
- **并行策略**: 通过不同的并行组合策略

#### 5.8.2 观察者模式

- **Backward Hook**: 注册梯度计算完成后的回调
- **Forward Hook**: 注册前向传播前的回调

#### 5.8.3 工厂模式

- **进程组创建**: `create_group` 函数创建各种进程组
- **Rank 生成器**: 根据配置生成不同的 rank 分配

---

## 6. 流水线并行模块深度分析

> 目录: `megatron/core/pipeline_parallel/`

### 6.1 模块概述

流水线并行模块实现了高效的模型层切分和调度，支持非交错和交错两种流水线模式。

### 6.2 目录结构与文件功能

```
pipeline_parallel/
├── __init__.py                    # 接口导出
├── schedules.py                   # 流水线调度核心实现 (2307行)
├── p2p_communication.py           # 点对点通信原语 (646行)
├── bridge_communicator.py         # 桥接通信器 (923行)
├── combined_1f1b.py               # 组合 1F1B 调度 (445行)
└── utils.py                       # 工具函数 (316行)
```

### 6.3 核心调度策略

#### 6.3.1 非交错流水线调度

**特点：**
- 模型被连续分割到不同 stage
- 实现 1F1B (One Forward, One Backward) 调度
- 每个 stage 处理连续的层块

```python
def forward_backward_pipelining_without_interleaving(
    forward_step_func,
    backward_step_func,
    num_microbatches,
    p2p_communicator,
    ...
):
    # 计算 warmup 微批次数量
    num_warmup_microbatches = (
        p2p_communicator.pp_group.size() -
        p2p_communicator.pp_group.rank() - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

    # Warmup 阶段
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communicator.recv_forward(
            timout=cfg.pipeline_model_parallel_forward_timeout
        )
        output_tensor = forward_step_func(
            forward_step_func, data_iterator, model, ...
        )
        p2p_communicator.send_forward(output_tensor)

    # 1F1B 稳态阶段
    num_microbatches_remaining = (
        num_microbatches - num_warmup_microbatches
    )

    for i in range(num_microbatches_remaining):
        # 前向传播
        output_tensor = forward_step_func(...)

        if forward_only:
            p2p_communicator.send_forward(output_tensor)
        else:
            # 发送输出并接收梯度
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor,
                timout=cfg.pipeline_model_parallel_forward_timeout
            )

            # 后向传播
            input_tensor_grad = backward_step_func(...)

            # 发送梯度并接收下一个输入
            input_tensor = p2p_communicator.send_backward_recv_forward(
                input_tensor_grad,
                timout=cfg.pipeline_model_parallel_backward_timeout
            )

    # Cooldown 阶段
    if not forward_only:
        for i in range(num_warmup_microbatches - 1):
            input_tensor_grad = p2p_communicator.recv_backward(...)
            backward_step_func(...)
            p2p_communicator.send_backward(input_tensor_grad)
```

#### 6.3.2 交错流水线调度

**特点：**
- 模型被分割成多个块（model chunks）
- 支持虚拟流水线并行
- 实现更复杂的交错调度策略

```python
def forward_backward_pipelining_with_interleaving(
    forward_step_func,
    backward_step_func,
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    ...
):
    # 调度表生成 - 核心创新
    schedule_table = get_schedule_table(
        num_microbatches,
        num_model_chunks,
        microbatch_group_size_per_vp_stage
    )

    # 遍历调度表执行计算
    for microbatch_id, model_chunk_id in schedule_table:
        # 前向传播
        if model_chunk_id < num_model_chunks:
            output_tensor = forward_step_func(
                forward_step_func,
                data_iterator,
                model[model_chunk_id],  # 使用特定的 model chunk
                num_microbatches,
                microbatch_id,
                ...
            )

        # 后向传播
        if not forward_only:
            input_tensor_grad = backward_step_func(...)

    return forward_model_parallel_output
```

#### 6.3.3 调度表生成算法

```python
def get_schedule_table(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage
):
    """生成交错流水线调度表"""
    schedule_table = []

    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if (min_microbatch_id_in_group +
            microbatch_group_size_per_vp_stage >= num_microbatches):
            # 最后一个微批次组
            schedule_table.extend([
                (microbatch_id, model_chunk_id)
                for model_chunk_id in range(num_model_chunks)
                for microbatch_id in range(
                    min_microbatch_id_in_group, num_microbatches
                )
            ])
        else:
            # 其他微批次组
            schedule_table.extend([
                (microbatch_id, model_chunk_id)
                for model_chunk_id in range(num_model_chunks)
                for microbatch_id in range(
                    min_microbatch_id_in_group,
                    min_microbatch_id_in_group + microbatch_group_size_per_vp_stage
                )
            ])

    return schedule_table
```

### 6.4 点对点通信原语

#### 6.4.1 P2P 通信架构

```python
class P2PCommunicator:
    def __init__(self, pp_group, config):
        self.pp_group = pp_group
        self.config = config
        self.curr_rank = pp_group.rank()
        self.world_size = pp_group.size()

        # 计算相邻 rank
        self.next_rank = (self.curr_rank + 1) % self.world_size
        self.prev_rank = (self.curr_rank - 1) % self.world_size

        # 通信模式
        self.use_ring_exchange = config.use_ring_exchange_p2p
```

#### 6.4.2 通信模式

| 模式 | 实现方式 | 优点 | 缺点 | 适用场景 |
|------|----------|------|------|----------|
| **Ring Exchange** | `torch.distributed.ring_exchange` | 最低延迟 | 需要 UCCL 后端 | 小规模 PP |
| **Batched** | `batch_isend_irecv` | 批量处理效率高 | 需要同步 | 大规模 PP |
| **Unbatched** | 分离的 `isend/irecv` | 灵活性高 | 开销大 | 异步调度 |

#### 6.4.3 变长序列处理

```python
def _communicate_shapes(
    self,
    tensor_send_next,
    tensor_send_prev,
    recv_prev,
    recv_next
):
    """通信张量形状，支持变长序列"""
    # 创建形状张量
    send_prev_shape_tensor = None
    send_next_shape_tensor = None

    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            [tensor_send_prev.shape[0],
             tensor_send_prev.shape[1],
             tensor_send_prev.shape[2]],
            device="cuda",
            dtype=torch.int64
        )

    # 创建接收缓冲
    recv_prev_shape_tensor = torch.empty((3,), device="cuda", dtype=torch.int64)
    recv_next_shape_tensor = torch.empty((3,), device="cuda", dtype=torch.int64)

    # 通信形状信息
    if self.use_ring_exchange:
        torch.distributed.ring_exchange(
            [send_prev_shape_tensor, send_next_shape_tensor],
            [recv_prev_shape_tensor, recv_next_shape_tensor],
            group=self.pp_group
        )
    else:
        # 使用 P2POp 进行批量通信
        ops = []
        if recv_prev and tensor_send_next is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                self.next_rank
            ))
        # ... 添加更多操作

        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
```

#### 6.4.4 奇偶 Rank 优化

```python
def _p2p_ops(
    self,
    tensor_send_next,
    tensor_recv_prev,
    tensor_send_prev,
    tensor_recv_next
):
    """优化点对点通信，减少通信冲突"""
    reqs = {}

    if self.pp_group.rank() % 2 == 0:
        # 偶数 rank：先发送后接收
        if tensor_send_next is not None:
            reqs["send_next"] = torch.distributed.isend(
                tensor_send_next, self.next_rank, group=self.pp_group
            )
        if tensor_recv_prev is not None:
            reqs["recv_prev"] = torch.distributed.irecv(
                tensor_recv_prev, self.prev_rank, group=self.pp_group
            )
    else:
        # 奇数 rank：先接收后发送
        if tensor_recv_prev is not None:
            reqs["recv_prev"] = torch.distributed.irecv(
                tensor_recv_prev, self.prev_rank, group=self.pp_group
            )
        if tensor_send_next is not None:
            reqs["send_next"] = torch.distributed.isend(
                tensor_send_next, self.next_rank, group=self.pp_group
            )

    return reqs
```

### 6.5 桥接通信器

#### 6.5.1 桥接通信架构

`BridgeCommunicator` 处理不同并行配置间的智能通信：

```python
class CommRole(Enum):
    SENDER = "SENDER"       # 源网格的 leader TP-CP rank
    RECEIVER = "RECEIVER"   # 目标网格的 leader TP-CP rank
    MEMBER = "MEMBER"       # DP 副本中的非 leader rank

class BridgeCommunicator:
    def build_comm_map(self, src_tp_leaders, dest_tp_leaders):
        """构建通信映射，支持风扇-in/fan-out模式"""
        src_count = len(src_tp_leaders)
        dest_count = len(dest_tp_leaders)

        if src_count % dest_count != 0 and dest_count % src_count != 0:
            raise ValueError(
                "源和目标 leader 数量必须能整除"
            )

        scale_factor = int(src_count / dest_count)
        if scale_factor > 1:
            # Fan-in: 多个源发送到较少目标
            for i, dest_rank in enumerate(dest_tp_leaders):
                src_ranks = src_tp_leaders[
                    i * scale_factor : (i + 1) * scale_factor
                ]
                for src_rank in src_ranks:
                    self.comm_map[src_rank] = CommInfo(
                        role=CommRole.SENDER,
                        send_to_ranks=[dest_rank]
                    )
```

### 6.6 组合 1F1B 调度

#### 6.6.1 与 MoE 通信重叠

```python
def combined_forward_backward_step(
    f_schedule_plan,
    b_schedule_plan,
    ...
):
    """合并前向和后向计算，隐藏 MoE 通信"""

    # 合并预处理
    if f_model is not None:
        f_schedule_plan, loss_func = forward_step_func(...)

    # 合并计算
    with context_manager and outer_fp8_context:
        # 调度计划执行，隐藏通信
        output_tensor = type(f_schedule_plan or b_schedule_plan).run(
            f_schedule_plan,
            b_schedule_plan,
            b_grad=b_grad,
            pre_forward=pre_forward,
            pre_backward=pre_backward,
            post_forward=post_forward,
            post_backward=post_backward,
        )

    return output_tensor
```

### 6.7 内存优化技术

#### 6.7.1 激活值伪释放

```python
def deallocate_output_tensor(
    out,
    deallocate_pipeline_outputs=False
):
    """伪释放输出张量的 .data 字段"""
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    assert isinstance(out, torch.Tensor)
    assert out._base is None, "不能释放另一个张量的视图"

    # 将数据替换为空张量
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

#### 6.7.2 延迟嵌入梯度计算

```python
# 嵌入层延迟梯度计算
if config.defer_embedding_wgrad_compute:
    # 保存激活值在缓冲区中
    self.embedding_activation_buffer = []
    self.grad_output_buffer = []

    # 在 pipeline 刷新阶段计算梯度
    finish_embedding_wgrad_compute(
        config,
        embedding_module,
        is_pp_last_stage(p2p_communicator.pp_group),
        tp_group
    )
```

### 6.8 气泡优化

#### 6.8.1 气泡检测

```python
# 检查最终微批次组大小是否会产生依赖气泡
final_microbatch_group_size = (
    num_microbatches % config.microbatch_group_size_per_vp_stage
)

if 0 < final_microbatch_group_size < pipeline_parallel_size:
    raise RuntimeError(
        f'余数 {final_microbatch_group_size} 应等于 0 或 ≥ PP 大小'
        f' {pipeline_parallel_size}，否则会产生依赖气泡'
    )
```

#### 6.8.2 通信隐藏

```python
# 在 warmup 阶段预取通信
if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(pp_group):
    # 预取下一个迭代的接收
    fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
        p2p_communicator.send_forward_recv_forward(
            tensor_prev=None,
            tensor_next=None,
            prev_rank=prev_rank,
            next_rank=next_rank,
        )
    )
```

---

## 7. 张量并行模块深度分析

> 目录: `megatron/core/tensor_parallel/`

### 7.1 模块概述

张量并行模块实现了模型权重在多 GPU 间的切分，是训练超大模型的核心技术。

### 7.2 目录结构与文件功能

```
tensor_parallel/
├── __init__.py                    # 模块导出
├── layers.py                      # 核心实现 (1316行)
├── mappings.py                    # 通信操作 (597行)
├── cross_entropy.py               # 并行交叉熵 (233行)
├── random.py                      # 并行 RNG (689行)
├── data.py                        # 数据广播 (102行)
├── inference_layers.py            # 推理优化层 (295行)
└── utils.py                       # 工具函数 (122行)
```

### 7.3 列并行线性层

#### 7.3.1 数学原理

对于线性变换 Y = XA，其中 X ∈ ℝ^(s×b×h)，A ∈ ℝ^(h×d)

- 将权重矩阵 A 沿列方向分割：A = [A₁, A₂, ..., A_p]
- 每个 A_i ∈ ℝ^(h×d/p)
- 每个 GPU 只计算 Y_i = XA_i，输出维度为 ℝ^(s×b×d/p)
- 如果需要完整输出，通过 All-Gather 操作拼接各 GPU 的结果

```
完整计算: Y = XA, 其中 A ∈ ℝ^(h×d)
并行化后: Y_i = XA_i, 其中 A_i ∈ ℝ^(h×d/p)
最终输出: Y = concat([Y_1, Y_2, ..., Y_p])
```

#### 7.3.2 核心实现

```python
class ColumnParallelLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input_,
        weight,
        bias,
        gradient_accumulation_fusion,
        ...
    ):
        # 前向传播
        output = torch.matmul(input_, weight.t())
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(input_, weight)
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # 反向传播
        input_, weight = ctx.saved_tensors

        # 梯度计算
        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), input_)

        # 如果启用序列并行，进行 reduce-scatter
        if ctx.sequence_parallel:
            handle = dist_reduce_scatter_func(
                sub_grad_input,
                grad_input,
                group=ctx.tp_group,
                async_op=True
            )
        else:
            # 否则进行 all-reduce
            handle = torch.distributed.all_reduce(
                grad_input,
                group=ctx.tp_group,
                async_op=True
            )

        return grad_input, grad_weight, grad_bias, ...
```

#### 7.3.3 权重初始化

```python
# 权重初始化（CPU 端）
master_weight = torch.empty(
    output_size, input_size,
    dtype=torch.float
)
init_method(master_weight)  # 在所有 GPU 上初始化完整权重

# 分割权重
weight_list = torch.split(
    master_weight,
    per_partition_size,
    dim=0
)

# 每个 GPU 获取对应分片
my_weight_list = weight_list[rank::world_size]
```

### 7.4 行并行线性层

#### 7.4.1 数学原理

对于线性变换 Y = XA，其中 X ∈ ℝ^(s×b×h)，A ∈ ℝ^(h×d)

- 将输入 X 沿行方向分割：X = [X₁; X₂; ...; X_p]
- 每个 X_i ∈ ℝ^(s×b×h/p)
- 将权重矩阵 A 沿行方向分割：A = [A₁; A₂; ...; A_p]^T
- 每个 A_i ∈ ℝ^(h/p×d)
- 每个 GPU 计算 Y_i = X_iA_i，输出维度为 ℝ^(s×b×d)
- 通过 All-Reduce 操作合并各 GPU 的结果

```
完整计算: Y = XA, 其中 X ∈ ℝ^(s×b×h), A ∈ ℝ^(h×d)
并行化后: Y_i = X_iA_i, 其中 X_i ∈ ℝ^(s×b×h/p), A_i ∈ ℝ^(h/p×d)
最终输出: Y = Y_1 + Y_2 + ... + Y_p
```

#### 7.4.2 核心实现

```python
class RowParallelLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input_,
        weight,
        bias,
        input_is_parallel,
        gradient_accumulation_fusion,
        ...
    ):
        # 如果输入不是并行化的，进行 scatter
        if not input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_,
                group=ctx.tp_group
            )
        else:
            input_parallel = input_

        # 矩阵乘法
        output_parallel = torch.matmul(input_parallel, weight.t())

        # All-Reduce 合并结果
        if ctx.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(
                output_parallel,
                group=ctx.tp_group
            )
        else:
            output = copy_to_tensor_model_parallel_region(
                output_parallel,
                group=ctx.tp_group
            )

        if bias is not None:
            output = output + bias

        return output
```

### 7.5 并行注意力机制

#### 7.5.1 QKV 矩阵并行化

```
Q = XW_q, K = XW_k, V = XW_v

并行化策略:
- Q: 列并行，每个 GPU 处理完整的 Q 输出
- K 和 V: 列并行，每个 GPU 处理 K、V 的一部分
- 输出投影: 行并行，输入被分割
```

```python
# 在 attention.py 中的实现
self.linear_qkv = tensor_parallel.ColumnParallelLinear(
    hidden_size,
    3 * hidden_size,  # QKV 合并处理
    config=config,
    init_method=init_method,
    bias=add_bias,
    skip_weight_param_allocation=only_query_position,
    ...
)

self.linear_proj = tensor_parallel.RowParallelLinear(
    hidden_size,
    hidden_size,
    config=config,
    init_method=output_layer_init_method,
    bias=add_bias,
    ...
)
```

#### 7.5.2 Flash Attention 集成

```python
# Flash Attention v3 集成
if HAVE_FA3:
    output = flash_attn3_with_kvcache(
        q, k, v, kv_cache,
        softmax_scale,
        causal=True,
        ...
    )
```

### 7.6 词汇表并行嵌入

#### 7.6.1 实现原理

- 将词汇表均分到各个 GPU：vocab_size = vocab_per_partition × world_size
- 每个 GPU 只存储和处理自己的词汇表分片
- 嵌入查找后通过 Reduce-Scatter 或 All-Reduce 合并结果

```python
class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ...):
        super().__init__()

        # 计算词汇表范围
        self.vocab_start_index, self.vocab_end_index = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_pg_rank(self.tp_group),
                get_pg_size(self.tp_group)
            )
        )

        # 只存储自己的词汇表分片
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.weight = torch.nn.Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                embedding_dim,
                dtype=torch.float
            )
        )

    def forward(self, input_):
        # 创建掩码
        input_mask = (
            (input_ < self.vocab_start_index) |
            (input_ >= self.vocab_end_index)
        )

        # 调整输入索引
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        # 嵌入查找
        output_parallel = torch.nn.functional.embedding(
            masked_input,
            self.weight,
            ...
        )

        # 掩码无效结果
        output_parallel[input_mask, :] = 0.0

        # All-Reduce 合并
        if self.sequence_parallel:
            output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output = copy_to_tensor_model_parallel_region(output_parallel)

        return output
```

#### 7.6.2 并行交叉熵

```python
def vocab_parallel_cross_entropy(vocab_parallel_logits, target):

    # 最大值同步
    logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=get_tensor_model_parallel_group()
    )

    # 按目标索引获取预测值
    predicted_logits = vocab_parallel_logits[target]

    # 计算指数和
    sum_exp_logits = torch.exp(vocab_parallel_logits).sum(dim=-1)

    # All-Reduce 归约
    torch.distributed.all_reduce(
        predicted_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group()
    )
    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group()
    )

    # 计算损失
    loss = -predicted_logits + torch.log(sum_exp_logits)
    return loss
```

### 7.7 通信操作实现

#### 7.7.1 All-Reduce

```python
def _reduce(input_, group):
    if group.size() == 1:
        return input_

    torch.distributed.all_reduce(
        input_.contiguous(),
        group=group
    )
    return input_
```

#### 7.7.2 All-Gather

```python
def _gather_along_last_dim(input_, group):
    world_size = group.size()

    # 创建输出张量
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size
    output = torch.empty(
        dim_size,
        dtype=input_.dtype,
        device='cuda'
    )

    # All-Gather
    dist_all_gather_func(
        output,
        input_.contiguous(),
        group=group
    )

    # 重塑输出
    return output.view(-1, input_.shape[-1])
```

#### 7.7.3 Reduce-Scatter

```python
def _reduce_scatter_along_first_dim(
    input_,
    group,
    input_split_sizes=None,
    use_global_buffer=False
):
    world_size = group.size()

    # 计算输出大小
    if input_split_sizes is None:
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] // world_size
    else:
        dim_size = [input_split_sizes[group.rank()]]

    # 使用全局内存缓冲区减少分配开销
    if use_global_buffer:
        output = get_global_memory_buffer().get_tensor(
            dim_size,
            input_.dtype,
            "mpu"
        )
    else:
        output = torch.empty(
            dim_size,
            dtype=input_.dtype,
            device='cuda'
        )

    # Reduce-Scatter
    dist_reduce_scatter_func(
        output,
        input_.contiguous(),
        group=group
    )

    return output
```

#### 7.7.4 All-to-All

```python
def all_to_all_sp2hp(input_, group=None):
    """从 [num_tokens/TP, H] 转换到 [num_tokens, H/TP]"""
    world_size = group.size()
    input_ = input_.reshape(-1, input_.shape[-1])

    # 分割张量
    split_tensors = torch.split(
        input_,
        input_.shape[-1] // world_size,
        dim=1
    )

    # 拼接张量
    concat_tensor = torch.cat(split_tensors, dim=0)

    # All-to-All 通信
    output = all_to_all(group, concat_tensor)
    return output
```

### 7.8 激活重计算和梯度同步

#### 7.8.1 梯度累积融合

```python
class LinearWithGradAccumulationAndAsyncCommunication(
    torch.autograd.Function
):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        ...
    ):
        ctx.save_for_backward(input, weight)
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion

        # 计算前向输出
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # 使用自定义 CUDA 内核进行梯度累积
        if weight.main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input,
                grad_output,
                weight.main_grad
            )
        elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                total_input,
                grad_output,
                weight.main_grad
            )
```

#### 7.8.2 异步梯度同步

```python
# 异步 All-Reduce
if ctx.allreduce_dgrad:
    handle = torch.distributed.all_reduce(
        grad_input,
        group=tp_group,
        async_op=True
    )
    # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 确保调度顺序

# 序列并行处理
if ctx.sequence_parallel:
    handle = dist_reduce_scatter_func(
        sub_grad_input,
        grad_input,
        group=tp_group,
        async_op=True
    )
```

### 7.9 性能优化技术

#### 7.9.1 推理优化层

**融合内核：**

```python
# reduce-scatter + add + rms-norm + all-gather 融合
fused_multimem_rs_add_norm_ag(
    residual,                          # 残差
    symm_mem_buffer["tensor"],         # 输出
    symm_mem_buffer["handle"],         # 句柄
    residual,                          # 输出残差
    next_layer_norm_weights,          # 下一层权重
    eps,                              # epsilon
)
```

**对称内存使用：**

```python
def _maybe_allocate_symmetric_buffer(self, x: torch.Tensor):
    symm_mem_buffer_dims = list(x.size())
    symm_mem_buffer_dims[0] *= self.tp_size

    symm_mem_buffer = (
        get_global_symmetric_memory_buffer().maybe_get_tensor(
            symm_mem_buffer_dims,
            dtype=x.dtype
        )
    )
    return symm_mem_buffer
```

#### 7.9.2 并行随机数生成

```python
class CudaRNGStatesTracker:
    def __init__(self, use_cudagraphable_rng=False):
        self.states_ = {}     # 存储不同名称的 RNG 状态
        self.seeds_ = set()   # 防止重复种子

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        # 保存当前状态
        orig_cuda_rng_state = _get_cuda_rng_state()

        # 切换到指定状态
        _set_cuda_rng_state(self.states_[name])

        try:
            yield
        finally:
            # 恢复原始状态
            _set_cuda_rng_state(orig_cuda_rng_state)
```

---

## 8. 总结

### 8.1 架构优势

1. **模块化设计**: 各组件职责清晰，易于扩展和维护
2. **多维并行**: 支持灵活的并行策略组合
3. **性能优化**: 全面的 GPU 优化，包括内核、内存和通信
4. **向后兼容**: 保留 legacy 支持平滑迁移
5. **生产就绪**: 支持大规模生产环境部署

### 8.2 技术创新

1. **选择性激活重计算**: 减少内存占用同时保持训练效率
2. **交错流水线并行**: 减少流水线气泡
3. **组合 1F1B 调度**: 隐藏 MoE 通信开销
4. **梯度累积融合**: 减少内存访问和提升计算效率
5. **桥接通信**: 支持不同并行配置间的灵活通信

### 8.3 适用场景

- **超大规模模型训练**: 数十亿到万亿参数的模型
- **多模态模型**: 视觉-语言模型、音频-语言模型
- **MoE 模型**: 支持数千专家的混合专家模型
- **长序列训练**: 支持百万级 token 序列
- **生产部署**: 高吞吐量推理服务

---

## 9. 训练流程详细图解 ⭐新增

### 9.1 完整训练流程图

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          Megatron-LM 训练流程                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                               阶段 1: 初始化                                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ├───┬──────────────────────────────────────────────────────────────┐
    │   │ 1.1 参数解析 (parse_args)                                   │
    │   │     • 命令行参数                                             │
    │   │     • YAML 配置文件                                           │
    │   │     • 从 checkpoint 恢复参数                                   │
    │   └──────────────────────────────────────────────────────────────┘
    │
    ├───┬──────────────────────────────────────────────────────────────┐
    │   │ 1.2 初始化分布式环境 (initialize_megatron)                   │
    │   │     • torch.distributed 初始化                                │
    │   │     • 设置随机种子                                             │
    │   │     • 创建并行通信组                                           │
    │   │       ├─ TP: 张量并行组                                        │
    │   │       ├─ PP: 流水线并行组                                      │
    │   │       ├─ DP: 数据并行组                                        │
    │   │       ├─ CP: 上下文并行组                                      │
    │   │       └─ EP: 专家并行组                                        │
    │   └──────────────────────────────────────────────────────────────┘
    │
    ├───┬──────────────────────────────────────────────────────────────┐
    │   │ 1.3 模型和优化器设置 (setup_model_and_optimizer)             │
    │   │     • 模型构建 (get_model)                                    │
    │   │       ├─ 使用 LayerSpec 定义层结构                             │
    │   │       ├─ 包装 DDP/FSDP                                        │
    │   │       └─ 多模型块 (虚拟流水线)                                 │
    │   │     • 优化器创建 (get_megatron_optimizer)                     │
    │   │       ├─ 分布式优化器                                         │
    │   │       ├─ 学习率调度器                                         │
    │   │       └─ 梯度缩放器                                           │
    │   └──────────────────────────────────────────────────────────────┘
    │
    └───┬──────────────────────────────────────────────────────────────┐
        │ 1.4 数据迭代器构建                                           │
        │     • 数据集配置 (BlendedMegatronDatasetBuilder)              │
        │     • 混合数据集支持                                          │
        │     • 虚拟流水线数据分片                                       │
        └──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         阶段 2: 主训练循环 (train)                            │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    └─── while iteration < train_iters:
        │
        ├───┬──────────────────────────────────────────────────────┐
        │   │ 步骤 1: 数据获取                                       │
        │   │   data = next(data_iterator)                          │
        │   │   • 支持 FIM (Fill-in-the-Middle)                     │
        │   │   • 支持混合数据集                                     │
        │   │   • 支持 RL 数据                                       │
        │   └──────────────────────────────────────────────────────┘
        │
        ├───┬──────────────────────────────────────────────────────┐
        │   │ 步骤 2: 前向传播 (forward_backward_func)              │
        │   │                                                        │
        │   │   ┌────────────────────────────────────────────────┐ │
        │   │   │ 根据并行配置选择调度策略                        │ │
        │   │   │                                                │ │
        │   │   │   PP > 1 ? ──Yes──> 交错/非交错流水线          │ │
        │   │   │       │                                        │ │
        │   │   │       No                                       │ │
        │   │   │       │                                        │ │
        │   │   │       ↓                                        │ │
        │   │   │   无流水线: 直接前向+反向                        │ │
        │   │   └────────────────────────────────────────────────┘ │
        │   │                                                        │
        │   │   ┌────────────────────────────────────────────────┐ │
        │   │   │ 每个 microbatch 的处理:                        │ │
        │   │   │                                                │ │
        │   │   │   [输入] → [嵌入层] → [Transformer 层 N]      │ │
        │   │   │                ↓                               │ │
        │   │   │            [输出投影] → [损失计算]            │ │
        │   │   │                                                │ │
        │   │   │   流水线并行时:                                  │ │
        │   │   │   • Warmup: 填充流水线                           │ │
        │   │   │   • 1F1B: 前向+反向交替                         │ │
        │   │   │   • Cooldown: 清空流水线                        │ │
        │   │   └────────────────────────────────────────────────┘ │
        │   └──────────────────────────────────────────────────────┘
        │
        ├───┬──────────────────────────────────────────────────────┐
        │   │ 步骤 3: 梯度处理                                       │
        │   │   • finalize_model_grads(model)                      │
        │   │   • 梯度同步 (AllReduce/ReduceScatter)                │
        │   │   • 梯度裁剪 (clip_grad_norm)                        │
        │   │   • NaN/Inf 检查                                       │
        │   └──────────────────────────────────────────────────────┘
        │
        ├───┬──────────────────────────────────────────────────────┐
        │   │ 步骤 4: 参数更新 (optimizer.step)                    │
        │   │   • 梯度缩放更新                                       │
        │   │   • 优化器步进                                       │
        │   │   • 参数同步 (AllGather)                             │
        │   │   • 学习率更新                                        │
        │   └──────────────────────────────────────────────────────┘
        │
        ├───┬──────────────────────────────────────────────────────┐
        │   │ 步骤 5: 检查点保存                                     │
        │   │   if should_checkpoint:                              │
        │   │       save_checkpoint()                              │
        │   │       • 保存模型权重                                  │
        │   │       • 保存优化器状态                                │
        │   │       • 保存训练状态                                  │
        │   └──────────────────────────────────────────────────────┘
        │
        └───┬──────────────────────────────────────────────────────┐
            │ 步骤 6: 日志和评估                                     │
            │   • training_log(loss, lr, grad_norm, ...)           │
            │   • if iteration % eval_interval == 0:               │
            │   │     evaluate_and_print_results()                 │
            └──────────────────────────────────────────────────────┘
```

### 9.2 单步训练详细流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    train_step() - 单步训练详解                               │
└─────────────────────────────────────────────────────────────────────────────┘

输入:
    • forward_step_func: 前向传播函数
    • data_iterator: 数据迭代器
    • model: 模型 (可能是多个 model chunks)
    • optimizer: 优化器
    • config: 配置对象
    • num_microbatches: microbatch 数量

流程:

    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. 梯度缓冲区清零                                               │
    │    for model_chunk in model:                                   │
    │        model_chunk.zero_grad_buffer()                          │
    │    optimizer.zero_grad()                                       │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ 2. 前向+反向传播 (forward_backward_func)                         │
    │                                                                 │
    │    ┌─────────────────────────────────────────────────────────┐  │
    │    │ 无流水线模式 (PP=1):                                    │  │
    │    │                                                         │  │
    │    │   for microbatch_id in range(num_microbatches):        │  │
    │    │       # 前向传播                                         │  │
    │    │       data = next(data_iterator)                        │  │
    │    │       output = model(data)                              │  │
    │    │       loss = loss_func(output)                          │  │
    │    │                                                         │  │
    │    │       # 反向传播                                         │  │
    │    │       loss.backward()                                   │  │
    │    └─────────────────────────────────────────────────────────┘  │
    │                                                                 │
    │    ┌─────────────────────────────────────────────────────────┐  │
    │    │ 流水线模式 (PP>1):                                       │  │
    │    │                                                         │  │
    │    │   非交错流水线:                                          │  │
    │    │   ┌───────────────────────────────────────────────────┐│  │
    │    │   │ Warmup: F0, F1, ..., F(PP-2)                      ││  │
    │    │   │ 1F1B:  (F(PP-1)+B0), (FPP+B1), ..., (Fn+B(n-PP+1))││  │
    │    │   │ Cooldown: B(n-PP+2), ..., Bn                       ││  │
    │    │   └───────────────────────────────────────────────────┘│  │
    │    │                                                         │  │
    │    │   交错流水线:                                            │  │
    │    │   ┌───────────────────────────────────────────────────┐│  │
    │    │   │ 生成调度表 (schedule_table)                       ││  │
    │    │   │ 执行顺序: [+1,+1,+2,+2,...,-N,-N,...]            ││  │
    │    │   │   (+N = 前向 Chunk N)                             ││  │
    │    │   │   (-N = 反向 Chunk N)                             ││  │
    │    │   └───────────────────────────────────────────────────┘│  │
    │    └─────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ 3. 梯度最终化                                                   │
    │    finalize_model_grads(model)                                │
    │    • 收集 DDP 梯度缓冲区的已减少梯度                            │
    │    • 处理 FP8 参数                                             │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ 4. 梯度裁剪                                                     │
    │    grad_norm = clip_grad_norm(model.parameters())             │
    │    • 支持 L2 范数裁剪                                           │
    │    • 支持 INF 范数裁剪                                          │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ 5. 梯度缩放检查                                                 │
    │    found_inf = check_for_nan_inf(grad_norm)                    │
    │    if found_inf:                                               │
    │        grad_scaler.update(found_inf=True)                     │
    │        return False  # 跳过本次更新                            │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ 6. 优化器步进                                                   │
    │    update_successful, grad_norm, num_zeros = optimizer.step() │
    │                                                                 │
    │    ┌─────────────────────────────────────────────────────────┐ │
    │    │ DistributedOptimizer.step():                           │ │
    │    │                                                         │ │
    │    │ 1. prepare_grads():                                    │ │
    │    │    - 检查 NaN/Inf                                      │ │
    │    │    - 转换梯度精度                                      │ │
    │    │                                                         │ │
    │    │ 2. 梯度裁剪                                             │ │
    │    │    - 计算全局梯度范数                                   │ │
    │    │    - 应用裁剪系数                                       │ │
    │    │                                                         │ │
    │    │ 3. step_with_ready_grads():                            │ │
    │    │    - 调用底层优化器 (Adam)                              │ │
    │    │    - 更新分片参数                                       │ │
    │    │                                                         │ │
    │    │ 4. start_param_sync():                                 │ │
    │    │    - AllGather 更新后的参数                              │ │
    │    └─────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │ 7. 学习率和梯度缩放更新                                         │
    │    if update_successful:                                       │
    │        opt_param_scheduler.step(increment=1)                  │
    │        grad_scaler.update(found_inf=False)                    │
    └─────────────────────────────────────────────────────────────────┘

输出:
    • loss_dict: 损失字典
    • grad_norm: 梯度范数
    • num_zeros: 零值梯度数量
    • learning_rate: 当前学习率
```

### 9.3 流水线并行调度可视化

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    流水线并行 1F1B 调度可视化                               │
└─────────────────────────────────────────────────────────────────────────────┘

示例配置:
    • PP (流水线并行) = 4
    • Microbatches = 8
    • 非交错流水线

时间线:
    GPU 0 (Stage 0):  F0→F1→F2→F3───────────────B0→B1→B2→B3
    GPU 1 (Stage 1):  ────F0→F1→F2→F3───────────B0→B1→B2→B3
    GPU 2 (Stage 2):  ───────F0→F1→F2→F3─────────B0→B1→B2→B3
    GPU 3 (Stage 3):  ─────────F0→F1→F2→F3→B0→B1→B2→B3

详细步骤:
    Step  │ GPU0         │ GPU1         │ GPU2         │ GPU3
    ──────┼──────────────┼──────────────┼──────────────┼──────────────
      0  │ F0           │              │              │
      1  │ F1           │ F0           │              │
      2  │ F2           │ F1           │ F0           │
      3  │ F3           │ F2           │ F1           │ F0  ← Warmup结束
      4  │ B0           │ F3           │ F2           │ F1  ← 1F1B开始
      5  │ B1           │ B0           │ F3           │ F2
      6  │ B2           │ B1           │ B0           │ F3
      7  │ B3           │ B2           │ B1           │ B0
      8  │              │ B3           │ B2           │ B1
      9  │              │              │ B3           │ B2  ← Cooldown结束
     10  │              │              │              │ B3

流水线气泡:
    • GPU 0: 3 steps (Step 7-9 等待)
    • GPU 1: 2 steps (Step 8-9 等待)
    • GPU 2: 1 step (Step 9 等待)
    • GPU 3: 无等待

GPU 利用率 = 32 / 44 ≈ 72.7%
```

### 9.4 数据流向图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         单个 Transformer 层数据流向                          │
└─────────────────────────────────────────────────────────────────────────────┘

输入: hidden_states [batch, seq_len, hidden_size]

    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LayerNorm (可选，取决于配置)                               │
│  output = LayerNorm(hidden_states)                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Self-Attention                                             │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ QKV 投影 (ColumnParallelLinear)                       │  │
│  │ qkv = Linear_qkv(hidden_states)  [TP 列并行]          │  │
│  │ Q, K, V = split(qkv, 3, dim=-1)                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 注意力计算                                             │  │
│  │ attn_output = Attention(Q, K, V)                      │  │
│  │   • 如果启用 Flash Attention: 使用 FA 内核            │  │
│  │   • 否则: 标准注意力实现                               │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 输出投影 (RowParallelLinear)                          │  │
│  │ attn_output = Linear_proj(attn_output)  [TP 行并行]   │  │
│  │   • All-Reduce 合并 TP 结果                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  残差连接 + Dropout                                          │
│  hidden_states = hidden_states + attn_output               │
│  hidden_states = dropout(hidden_states)                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  LayerNorm (可选，取决于配置)                               │
│  output = LayerNorm(hidden_states)                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  MLP                                                        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ FC1 (ColumnParallelLinear)                             │  │
│  │ intermediate = Linear_fc1(hidden_states)  [TP 列并行]  │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 激活函数                                                │  │
│  │ intermediate = activation_func(intermediate)           │  │
│  │   • SwiGLU: if config.gated_linear_unit               │  │
│  │   • GeGLU: if config.add_bias_linear                  │  │
│  │   • GELU: otherwise                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ FC2 (RowParallelLinear)                               │  │
│  │ mlp_output = Linear_fc2(intermediate)  [TP 行并行]    │  │
│  │   • All-Reduce 合并 TP 结果                            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  残差连接 + Dropout                                          │
│  hidden_states = hidden_states + mlp_output               │
│  hidden_states = dropout(hidden_states)                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
输出: hidden_states [batch, seq_len, hidden_size]
```

---

## 10. Transformer 模块深度分析 ⭐新增

> 目录: `megatron/core/transformer/`

### 10.1 模块概述

Transformer 模块是 Megatron-LM 的核心构建块，包含注意力机制、MLP、层归一化等组件。

### 10.2 目录结构与文件功能

```
transformer/
├── __init__.py                          # 模块导出
├── attention.py                         # 注意力机制实现 ⭐
├── mlp.py                               # MLP 层实现 ⭐
├── transformer_layer.py                 # Transformer 层
├── transformer_block.py                 # Transformer 块
├── transformer_config.py                # 配置类
├── module.py                            # 基础模块类
├── enums.py                             # 枚举定义
├── utils.py                             # 工具函数
├── identity_op.py                       # 身份操作
├── custom_layers/                       # 自定义层
│   ├── transformer_engine.py            # TE 集成
│   └── batch_invariant_kernels.py       # 批处理不变内核
└── moe/                                 # MoE 实现
    ├── moe_layer.py                     # MoE 层
    ├── router.py                        # 路由器
    ├── experts.py                       # 专家实现
    └── token_dispatcher.py              # Token 分发器
```

### 10.3 注意力机制深度分析 (attention.py)

#### 10.3.1 MultiHeadAttention 类结构

```python
class MultiHeadAttention(MegatronModule):
    """
    多头注意力实现

    关键特性:
    - 支持分组查询注意力 (GQA)
    - 集成 Flash Attention
    - 支持张量并行
    - 支持 RoPE 位置编码
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
    ):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        # 计算每个注意力头的大小
        self.hidden_size_per_attention_head = divide(
            config.hidden_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            config.num_attention_heads,
            config.tensor_model_parallel_size
        )

        # QKV 投影 (列并行)
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            config.hidden_size,
            3 * config.kv_channels * config.num_query_groups,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            skip_weight_param_allocation=config.only_query_position,
            ...
        )

        # 核心注意力
        self.core_attention = build_module(
            submodules.core_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )

        # 输出投影 (行并行)
        self.linear_proj = build_module(
            submodules.linear_proj,
            config.kv_channels * config.num_query_groups,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            ...
        )

        # QK LayerNorm (可选，用于稳定训练)
        if config.qk_layernorm:
            self.q_layernorm = build_module(submodules.q_layernorm, ...)
            self.k_layernorm = build_module(submodules.k_layernorm, ...)
```

#### 10.3.2 前向传播流程

```python
def forward(
    self,
    hidden_states,
    attention_mask=None,
    key_value_states=None,
    inference_params=None,
    rotary_pos_emb=None,
):
    """
    前向传播流程:

    1. QKV 投影
    2. QK LayerNorm (可选)
    3. 应用 RoPE (可选)
    4. 核心注意力计算
    5. 输出投影
    """

    # 1. QKV 投影 (列并行，每个 GPU 处理部分头)
    mixed_qkv, _ = self.linear_qkv(hidden_states)

    # 2. 分离 Q, K, V
    if self.config.only_query_position:
        # 只计算 Q，K/V 来自缓存 (推理时)
        query, key, value = mixed_qkv, key_value_states[0], key_value_states[1]
    else:
        # 分离 QKV
        query, key, value = mixed_qkv.chunk(3, dim=-1)

    # 3. QK LayerNorm (可选，提高稳定性)
    if self.config.qk_layernorm:
        query = self.q_layernorm(query)
        key = self.k_layernorm(key)

    # 4. 重塑为多头格式
    # [batch, seq_len, num_heads * head_dim] → [batch, seq_len, num_heads, head_dim]
    query = query.view(batch_size, seq_len, -1, self.hidden_size_per_head)
    key = key.view(batch_size, seq_len, -1, self.hidden_size_per_head)
    value = value.view(batch_size, seq_len, -1, self.hidden_size_per_head)

    # 5. 应用 RoPE (旋转位置编码)
    if rotary_pos_emb is not None:
        if self.config.apply_rope_fusion:
            # 融合 RoPE (更快)
            query, key = apply_fused_qkv_rotary_pos_emb(
                query, key, rotary_pos_emb
            )
        else:
            # 标准 RoPE
            query = apply_rotary_pos_emb(query, rotary_pos_emb)
            key = apply_rotary_pos_emb(key, rotary_pos_emb)

    # 6. 核心注意力计算
    # 支持 Flash Attention v3 / v2 / 标准
    context_layer = self.core_attention(
        query, key, value,
        attention_mask=attention_mask,
        inference_params=inference_params,
    )

    # 7. 输出投影 (行并行，需要 All-Reduce)
    output, bias = self.linear_proj(context_layer)

    return output, bias
```

#### 10.3.3 Flash Attention 集成

```python
# Flash Attention v3 集成 (最快，需要 Hopper GPU)
if HAVE_FA3:
    output = flash_attn3_with_kvcache(
        q=query,           # [batch, seq_len, num_heads, head_dim]
        k=key,
        v=value,
        kvcache=kv_cache,  # KV 缓存 (推理时)
        softmax_scale=softmax_scale,
        causal=True,       # 因果掩码 (decoder-only)
        rotary_cos_sin=rotary_pos_emb,  # RoPE
    )

# Flash Attention v2 (Hopper/Ampere)
elif HAVE_FLASH_ATTN:
    output = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens,  # 累积序列长度 (可变长度)
        max_seqlen_q=max_seqlen,
        dropout_p=self.config.attention_dropout,
        causal=True,
    )

# 标准注意力 (回退)
else:
    # 标准的 softmax(QK^T/√d)V
    attn_scores = torch.matmul(query, key.transpose(-2, -1))
    attn_scores = attn_scores / math.sqrt(self.hidden_size_per_head)
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, value)
```

### 10.4 MLP 层深度分析 (mlp.py)

#### 10.4.1 MLP 类结构

```python
class MLP(MegatronModule):
    """
    MLP 前馈网络

    标准配置: hidden_size → 4*hidden_size → hidden_size

    支持:
    - Gated Linear Unit (GLU/SwiGLU)
    - 张量并行
    - 激活函数融合
    - MoE 专家
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.input_size = input_size if input_size is not None else config.hidden_size

        # 计算 FFN 隐藏层大小
        if ffn_hidden_size is None:
            ffn_hidden_size = config.ffn_hidden_size

        # 如果使用 GLU，需要双倍输出宽度
        if config.gated_linear_unit:
            ffn_hidden_size *= 2
            fc1_stride = 2  # TP 时交错存储 [gate, up] 部分
        else:
            fc1_stride = 1

        # FC1: hidden_size → 4*hidden_size (列并行)
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,  # 不收集，保持分片
            bias=config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
            stride=fc1_stride,  # GLU 特殊处理
        )

        # 激活函数
        if config.use_te_activation_func:
            self.activation_func = build_module(
                submodules.activation_func, config=config
            )
        else:
            self.activation_func = config.activation_func

        # FC2: 4*hidden_size → hidden_size (行并行)
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            config.ffn_hidden_size,  # 注意: 使用原始大小
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,  # 输入已分片
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )
```

#### 10.4.2 前向传播

```python
def forward(self, hidden_states):
    """
    MLP 前向传播:

    输入: [batch, seq_len, hidden_size]
    输出: [batch, seq_len, hidden_size]
    """

    # 1. FC1 投影 (列并行)
    # 输出: [batch, seq_len, 4*hidden_size/TP] 或 [batch, seq_len, 8*hidden_size/TP] (GLU)
    intermediate, bias = self.linear_fc1(hidden_states)

    # 2. 激活函数
    if self.config.gated_linear_unit:
        # SwiGLU: activation(x) = Swish(x) * x
        # 分割为 gate 和 up 两部分
        gate, up = intermediate.chunk(2, dim=-1)

        if self.config.activation_func == F.silu:
            # SwiGLU = SiLU(gate) * up
            intermediate = self.config.activation_func(gate) * up
        elif self.config.activation_func == F.gelu:
            # GeGLU = GELU(gate) * up
            intermediate = self.config.activation_func(gate) * up
        else:
            # 其他 GLU 变体
            intermediate = self.config.activation_func(gate) * up
    else:
        # 标准激活函数
        intermediate = self.activation_func(intermediate)

    # 3. FC2 投影 (行并行，需要 All-Reduce)
    # 输出: [batch, seq_len, hidden_size]
    output, bias = self.linear_fc2(intermediate)

    return output, bias
```

#### 10.4.3 激活函数实现

```python
# 标准 GELU
def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

# 快速 GELU (融合版本)
@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

# SiLU (Swish)
def silu(x):
    return x * torch.sigmoid(x)

# SwiGLU (融合版本)
@torch.jit.script
def bias_swiglu(bias, y):
    x = bias + y
    return x * torch.sigmoid(x) * (bias + y * 3)  # SiLU(x) * 3x

# 加权 SwiGLU (用于某些特殊配置)
@torch.jit.script
def weighted_bias_swiglu(gate_bias, gate, up_bias, up):
    gate = gate + gate_bias
    up = up + up_bias
    return torch.sigmoid(gate) * gate * up
```

### 10.5 Transformer Layer 组合

```python
class TransformerLayer(MegatronModule):
    """
    完整的 Transformer 层

    包含:
    - Self-Attention
    - MLP
    - 残差连接
    - Layer Normalization
    """

    def __init__(self, config, submodules, layer_number):
        super().__init__(config=config)
        self.layer_number = layer_number

        # Self-Attention
        self.self_attention = build_module(
            submodules.self_attention,
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
        )

        # Post-Attention LayerNorm (可选)
        if self.config.post_layer_norm:
            self.self_attention_layernorm = LayerNorm(...)

        # MLP
        self.mlp = build_module(
            submodules.mlp,
            config=config,
        )

        # Post-MLP LayerNorm (可选)
        if self.config.post_layer_norm:
            self.mlp_layernorm = LayerNorm(...)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        rotary_pos_emb=None,
        ...
    ):
        """
        Transformer Layer 前向传播
        """

        # Self-Attention 分支
        residual = hidden_states

        # Pre-LN 或 Post-LN
        if not self.config.post_layer_norm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        # Self-Attention
        attn_output, attn_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )

        # 残差连接
        hidden_states = residual + attn_output
        if attn_bias is not None:
            hidden_states = hidden_states + attn_bias

        # Post-LN
        if self.config.post_layer_norm:
            hidden_states = self.self_attention_layernorm(hidden_states)

        # MLP 分支
        residual = hidden_states

        # Pre-LN 或 Post-LN
        if not self.config.post_layer_norm:
            hidden_states = self.mlp_layernorm(hidden_states)

        # MLP
        mlp_output, mlp_bias = self.mlp(hidden_states)

        # 残差连接
        hidden_states = residual + mlp_output
        if mlp_bias is not None:
            hidden_states = hidden_states + mlp_bias

        # Post-LN
        if self.config.post_layer_norm:
            hidden_states = self.mlp_layernorm(hidden_states)

        return hidden_states
```

---

## 11. 分布式训练模块增强分析 ⭐新增

### 11.1 DDP 实现深度剖析

#### 11.1.1 DistributedDataParallel 完整流程

```python
class DistributedDataParallel(_BaseDataParallel):
    """
    分布式数据并行实现

    核心特性:
    1. 梯度分桶: 将参数梯度组织到连续缓冲区
    2. 通信重叠: 支持梯度 reduce 与反向计算重叠
    3. 桶组管理: 使通信可以聚合在一起
    4. FP8 支持: 支持 FP8 参数的特殊处理
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        ...):
        super().__init__(config, module)
        self.ddp_config = ddp_config

        # 1. 设置通信组
        self.dp_group = parallel_state.get_data_parallel_group()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()

        # 2. 构建通信域 (CP × EP 可选)
        self.intra_distributed_optimizer_instance_group = self._build_communication_domain(
            self.dp_group,
            ddp_config.context_parallel_size,
            ddp_config.expert_model_parallel_size
        )

        # 3. 收集参数
        dense_params = []
        fp8_params = []
        for param in module.parameters():
            if param.requires_grad:
                if is_float8tensor(param):
                    fp8_params.append(param)
                else:
                    dense_params.append(param)

        # 4. 分配缓冲区
        self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
            dense_params,
            self.dp_group,
            ddp_config.gradient_scaling_factor,
            ddp_config.bucket_size,
            ddp_config.force_single_bucket_group
        )

        # 5. 处理 FP8 参数
        if fp8_params:
            self.fp8_bucket_groups = _allocate_fp8_buffers(
                fp8_params,
                self.dp_group,
                ddp_config.gradient_scaling_factor
            )

        # 6. 注册反向 hook
        self._register_backward_hooks(module)

        # 7. 设置通信重叠
        if ddp_config.overlap_grad_reduce:
            self.no_sync_func = self._get_no_sync_context()
        else:
            self.no_sync_func = None

    def _register_backward_hooks(self, module: torch.nn.Module):
        """为参数注册反向传播钩子"""

        for param in module.parameters():
            if param.requires_grad:
                # 获取梯度累加函数
                param_tmp = param.detach()
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                # 创建 hook 函数
                hook = self._make_backward_post_hook(param)

                # 注册 hook
                grad_acc.register_hook(hook)

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """创建反向传播后钩子"""

        def hook(*unused):
            # 1. 累积梯度到 main_grad
            if param.grad is not None:
                param.main_grad.add_(param.grad.data)
                param.grad = None

            # 2. 如果启用重叠，通知 bucket 组
            if self.ddp_config.overlap_grad_reduce:
                bucket_group = self.param_to_bucket_group[param]
                bucket_group.register_grad_ready(param)

        return hook
```

#### 11.1.2 参数到梯度缓冲区映射

```python
def _allocate_buffers_for_parameters(
    params,
    dp_group,
    gradient_scaling_factor,
    bucket_size,
    force_single_bucket_group
):
    """
    为参数分配梯度缓冲区并分桶

    映射策略:
    1. 参数按 size 排序
    2. 按 bucket_size 分桶
    3. 每个 bucket 分配连续内存
    4. 创建参数到 buffer 的映射
    """

    # 1. 收集参数信息
    param_info = []
    for param in params:
        param_info.append({
            'param': param,
            'size': param.numel(),
            'dtype': param.dtype,
            'requires_grad': True
        })

    # 2. 按 size 排序 (减少碎片)
    param_info.sort(key=lambda x: x['size'], reverse=True)

    # 3. 分桶
    buckets = []
    current_bucket_params = []
    current_bucket_size = 0

    for info in param_info:
        param_size = info['size']

        # 如果 bucket 已满，创建新 bucket
        if current_bucket_size + param_size > bucket_size and current_bucket_size > 0:
            buckets.append(current_bucket_params)
            current_bucket_params = []
            current_bucket_size = 0

        current_bucket_params.append(info)
        current_bucket_size += param_size

    # 添加最后一个 bucket
    if current_bucket_params:
        buckets.append(current_bucket_params)

    # 4. 分配缓冲区
    buffers = []
    for bucket_params in buckets:
        bucket_size = sum(p['size'] for p in bucket_params)

        # 分配连续内存 (FP32 累加)
        buffer = torch.empty(
            bucket_size,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
            requires_grad=False
        )
        buffers.append(buffer)

    # 5. 创建 bucket 组
    bucket_groups = []
    for bucket_id, (bucket_params, buffer) in enumerate(zip(buckets, buffers)):
        bucket_group = _ParamAndGradBucketGroup(
            buffer=buffer,
            params=[p['param'] for p in bucket_params],
            dp_group=dp_group,
            gradient_scaling_factor=gradient_scaling_factor,
            bucket_id=bucket_id,
            overlap_grad_reduce=ddp_config.overlap_grad_reduce
        )
        bucket_groups.append(bucket_group)

    return buffers, bucket_groups
```

### 11.2 桶组通信机制详解

```python
class _ParamAndGradBucketGroup:
    """
    将多个 bucket 分组，使通信可以聚合在一起

    工作原理:
    1. 收集 bucket 内所有参数
    2. 等待所有参数梯度就绪
    3. 启动聚合通信
    """

    def __init__(
        self,
        buffer,
        params,
        dp_group,
        gradient_scaling_factor=1.0,
        bucket_id=0,
        overlap_grad_reduce=False
    ):
        self.buffer = buffer
        self.params = set(params)
        self.params_with_grad = set()
        self.dp_group = dp_group
        self.gradient_scaling_factor = gradient_scaling_factor
        self.bucket_id = bucket_id
        self.overlap_grad_reduce = overlap_grad_reduce

        # 参数到 buffer 视图的映射
        self.param_to_buffer_view = {}
        offset = 0
        for param in params:
            size = param.numel()
            self.param_to_buffer_view[param] = {
                'offset': offset,
                'size': size,
                'view': buffer[offset:offset + size].view_as(param)
            }
            offset += size

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        注册参数梯度准备就绪

        当所有参数梯度就绪后，自动启动通信
        """
        if param not in self.params:
            return

        self.params_with_grad.add(param)

        # 检查是否所有参数梯度都就绪
        if len(self.params_with_grad) == len(self.params):
            if self.overlap_grad_reduce:
                # 异步启动通信
                self.start_grad_sync(wait=False)
            else:
                # 同步启动通信
                self.start_grad_sync(wait=True)

    def start_grad_sync(self, wait: bool = True):
        """
        启动梯度同步通信

        支持:
        1. All-Reduce (标准 DDP)
        2. Reduce-Scatter (分布式优化器)
        3. 通信融合 (多个操作合并)
        """
        # 1. 梯度缩放
        if self.gradient_scaling_factor != 1.0:
            self.buffer.mul_(self.gradient_scaling_factor)

        # 2. 选择归约操作
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # 3. 执行通信
        if self.use_distributed_optimizer:
            # Reduce-Scatter (分布式优化器)
            self._reduce_scatter(reduce_op, wait)
        else:
            # All-Reduce (标准 DDP)
            self._all_reduce(reduce_op, wait)

    def _all_reduce(self, reduce_op, wait):
        """All-Reduce 实现"""

        if self.overlap_grad_reduce:
            # 异步 All-Reduce
            self._handle = torch.distributed.all_reduce(
                self.buffer,
                op=reduce_op,
                group=self.dp_group,
                async_op=True
            )
            if wait:
                self._handle.wait()
        else:
            # 同步 All-Reduce
            torch.distributed.all_reduce(
                self.buffer,
                op=reduce_op,
                group=self.dp_group
            )

    def _reduce_scatter(self, reduce_op, wait):
        """Reduce-Scatter 实现 (分布式优化器)"""

        # 计算输出大小
        world_size = self.dp_group.size()
        output_size = self.buffer.numel() // world_size

        # 分配输出缓冲区
        output = torch.empty(
            output_size,
            dtype=self.buffer.dtype,
            device=self.buffer.device
        )

        # Reduce-Scatter
        if self.overlap_grad_reduce:
            # 异步 Reduce-Scatter
            self._handle = torch.distributed.reduce_scatter_tensor(
                output,
                self.buffer,
                op=reduce_op,
                group=self.dp_group,
                async_op=True
            )
            if wait:
                self._handle.wait()
        else:
            # 同步 Reduce-Scatter
            torch.distributed.reduce_scatter_tensor(
                output,
                self.buffer,
                op=reduce_op,
                group=self.dp_group
            )

        # 将结果复制回 buffer
        self.buffer.copy_(output)
```

### 11.3 参数同步详解

```python
def start_param_sync(self, force_sync: bool = False):
    """
    启动参数同步 (All-Gather)

    用于:
    1. 优化器更新后同步参数
    2. 检查点恢复
    3. 模型保存
    """

    async_op = (
        self.ddp_config.overlap_param_gather and
        not force_sync
    )

    # 使用融合管理器进行批量通信
    with _coalescing_manager(
        self.intra_distributed_optimizer_instance_group,
        async_ops=async_op
    ) as cm:
        for bucket in self.buffers:
            # 1. 获取参数视图
            param_data_views = []
            for param in bucket.params:
                view_info = bucket.param_to_buffer_view[param]
                param_data_views.append(view_info['view'])

            # 2. All-Gather
            dist_all_gather_func(
                bucket.param_data,  # 完整参数
                local_data_view,    # 本地分片
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op
            )

            # 3. 处理 FP8 参数
            if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                # 复制 All-Gather 后的参数数据到 param.data
                for param, view in zip(bucket.params, param_data_views):
                    param.data.copy_(view)

    # 等待通信完成
    if async_op:
        cm.wait()
```

### 11.4 分布式优化器详解

```python
class DistributedOptimizer(MixedPrecisionOptimizer):
    """
    分布式优化器，支持参数分片

    特性:
    1. 参数和梯度分片到不同 DP rank
    2. 每个 rank 只更新自己的参数分片
    3. All-Gather 同步更新后的参数
    4. 支持多种检查点格式
    """

    def __init__(
        self,
        optimizer,
        config,
        ddp_config,
        ...):
        super().__init__(optimizer, config, ...)

        # 1. 收集参数
        self.params = []
        for model_chunk in self.model_chunks:
            for param in model_chunk.parameters():
                if param.requires_grad:
                    self.params.append(param)

        # 2. 构建参数到梯度缓冲区的映射
        self.param_range_map = self._build_model_gbuf_param_range_map(
            self.params,
            self.dp_group,
            ddp_config.data_parallel_sharding_strategy
        )

        # 3. 设置梯度分片
        self.gbuf_all_shards = []
        for r in range(self.dp_world_size):
            shard = self._build_gbuf_shard(
                self.param_and_grad_buffer,
                r,
                self.param_range_map,
                self.dp_world_size
            )
            self.gbuf_all_shards.append(shard)

        # 4. 重构优化器以使用分片参数
        self._build_optimizer_for_sharded_model(optimizer)

    def _build_model_gbuf_param_range_map(
        cls,
        param_world_index_map: Dict,
        gbuf_world_range: Range,
        bucket_offset: int
    ):
        """
        构建参数到梯度缓冲区分片范围的映射

        为每个参数创建四个范围:
        1. world_range: 参数在整个梯度缓冲区中的范围
        2. bucket_range: 参数在 bucket 中的范围
        3. local_range: 参数在当前 DP rank 本地视图中的范围
        4. shard_range: 参数本身的分片范围
        """

        param_range_map = {}

        for param, param_world_indexes in param_world_index_map.items():
            # 参数的世界范围
            param_world_start, param_world_end, _ = param_world_indexes

            # 参数的本地范围
            param_local_start = max(
                0,
                param_world_start - gbuf_world_range.start
            )
            param_local_end = min(
                gbuf_world_range.size,
                param_world_end - gbuf_world_range.start
            )

            # 只有在本地范围内的参数才添加
            if param_local_end > param_local_start:
                param_range_map[param] = {
                    'gbuf_world': Range(
                        param_world_start + bucket_offset,
                        param_world_end + bucket_offset
                    ),
                    'gbuf_world_in_bucket': Range(
                        param_world_start,
                        param_world_end
                    ),
                    'gbuf_local': Range(
                        param_local_start,
                        param_local_end
                    ),
                    'param': Range(0, param.numel())
                }

        return param_range_map

    def step_with_ready_grads(self) -> bool:
        """
        执行优化器步骤并启动参数通信

        流程:
        1. 调用父类的优化器步骤 (使用分片参数)
        2. 启动参数 All-Gather 通信
        """

        # 1. 调用父类步骤
        update_successful = super().step_with_ready_grads()

        # 2. 启动参数 All-Gather
        if not self.ddp_config.overlap_param_gather:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()

        return update_successful
```

---

## 12. 流水线并行模块增强分析 ⭐新增

### 12.1 调度算法实现详解

#### 12.1.1 非交错流水线完整实现

```python
def forward_backward_pipelining_without_interleaving(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    p2p_communicator,
    config,
    ...
):
    """
    非交错流水线并行实现

    核心思想:
    1. Warmup: 填充流水线 (num_warmup = PP - rank - 1)
    2. 1F1B: 稳态执行 (前向+反向交替)
    3. Cooldown: 清空流水线

    时间复杂度: O(PP + num_microbatches)
    """

    # 1. 获取配置
    pipeline_model_parallel_size = p2p_communicator.pp_group.size()
    pipeline_model_parallel_rank = p2p_communicator.pp_group.rank()

    # 2. 计算 warmup microbatch 数量
    if forward_only:
        num_warmup_microbatches = num_microbatches - 1
    else:
        num_warmup_microbatches = (
            pipeline_model_parallel_size - pipeline_model_parallel_rank - 1
        )
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)

    # 3. 初始化缓冲区
    input_tensors = [[] for _ in range(num_microbatches)]
    output_tensors = [[] for _ in range(num_microbatches)]

    # ============== Warmup 阶段 ==============
    for k in range(num_warmup_microbatches):
        # 1. 接收前向输入 (除第一 stage)
        if not p2p_communicator.is_pp_first_stage():
            input_tensors[k] = p2p_communicator.recv_forward(
                timeout=config.pipeline_model_parallel_forward_timeout
            )

        # 2. 前向传播
        output_tensors[k] = forward_step_func(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            k,
            input_tensors[k],
            config
        )

        # 3. 发送前向输出 (除最后 stage)
        if not p2p_communicator.is_pp_last_stage():
            p2p_communicator.send_forward(output_tensors[k])

    # ============== 1F1B 稳态阶段 ==============
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    for k in range(num_microbatches_remaining):
        # 1. 前向传播
        microbatch_id = k + num_warmup_microbatches

        # 接收输入
        if not p2p_communicator.is_pp_first_stage():
            input_tensors[microbatch_id] = p2p_communicator.recv_forward(
                timeout=config.pipeline_model_parallel_forward_timeout
            )

        # 前向计算
        output_tensors[microbatch_id] = forward_step_func(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            microbatch_id,
            input_tensors[microbatch_id],
            config
        )

        # 2. 反向传播
        if not forward_only:
            # 计算需要反向的 microbatch
            k_bwd = microbatch_id - num_warmup_microbatches

            # 发送前向输出并接收反向梯度
            if p2p_communicator.is_pp_last_stage():
                # 最后 stage: 发送输出，接收梯度
                output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                    output_tensors[microbatch_id],
                    timeout=config.pipeline_model_parallel_backward_timeout
                )
            else:
                # 中间 stage: 转发输出
                p2p_communicator.send_forward(output_tensors[microbatch_id])
                output_tensor_grad = None

            # 反向计算
            input_tensor_grad = backward_step_func(
                backward_step_func,
                input_tensors[k_bwd],
                output_tensors[k_bwd],
                output_tensor_grad,
                model,
                num_microbatches,
                k_bwd,
                config
            )

            # 发送反向梯度并接收下一个前向输入
            if not p2p_communicator.is_pp_first_stage():
                input_tensors[microbatch_id] = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad,
                    timeout=config.pipeline_model_parallel_forward_timeout
                )

            # 释放输出张量 (内存优化)
            deallocate_output_tensor(output_tensors[k_bwd])

    # ============== Cooldown 阶段 ==============
    if not forward_only:
        for k in range(num_warmup_microbatches - 1, -1, -1):
            # 反向计算
            if p2p_communicator.is_pp_last_stage():
                output_tensor_grad = p2p_communicator.recv_backward(
                    timeout=config.pipeline_model_parallel_backward_timeout
                )
            else:
                output_tensor_grad = None

            input_tensor_grad = backward_step_func(
                backward_step_func,
                input_tensors[k],
                output_tensors[k],
                output_tensor_grad,
                model,
                num_microbatches,
                k,
                config
            )

            # 发送梯度
            if not p2p_communicator.is_pp_first_stage():
                p2p_communicator.send_backward(input_tensor_grad)

            # 释放输出张量
            deallocate_output_tensor(output_tensors[k])

    return forward_model_parallel_output
```

#### 12.1.2 交错流水线完整实现

```python
def forward_backward_pipelining_with_interleaving(
    forward_step_func,
    data_iterators,  # 注意: 复数，每个 chunk 一个迭代器
    model,           # 注意: list，每个 chunk 一个模型
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    p2p_communicator,
    config,
    ...
):
    """
    交错流水线并行实现

    核心思想:
    1. 模型分成多个 chunks (虚拟流水线)
    2. 每个 stage 处理多个不连续的 chunks
    3. 减少流水线气泡，提高 GPU 利用率

    优势:
    - GPU 利用率提升到 80-90%
    - 更好的负载均衡
    - 支持更灵活的并行配置
    """

    # 1. 生成调度表
    schedule_table = get_schedule_table(
        num_microbatches,
        num_model_chunks,
        microbatch_group_size_per_vp_stage
    )

    # 2. 计算执行顺序
    # 转换为: [+1,+1,+2,+2,...,-N,-N,...]
    # +N = 前向 Chunk N
    # -N = 反向 Chunk N
    order = convert_schedule_table_to_order(
        num_warmup_microbatches,
        num_model_chunks,
        schedule_table
    )

    # 3. 初始化
    input_tensors = [[] for _ in range(num_microbatches)]
    output_tensors = [[] for _ in range(num_microbatches)]
    forward_data_store = [None for _ in range(num_model_chunks)

    # 4. 遍历执行顺序
    for i, order_id in enumerate(order):
        model_chunk_id = None
        microbatch_id = None

        if order_id > 0:
            # 前向传播
            model_chunk_id = order_id - 1
            microbatch_id = get_microbatch_id_from_schedule_index(
                i,
                schedule_table
            )

            # 接收输入
            if not p2p_communicator.is_pp_first_stage():
                input_tensors[microbatch_id] = p2p_communicator.recv_forward(
                    timeout=config.pipeline_model_parallel_forward_timeout
                )

            # 前向计算
            output_tensors[microbatch_id], forward_data_store[model_chunk_id] = forward_step_func(
                forward_step_func,
                data_iterators[model_chunk_id],
                model[model_chunk_id],
                num_microbatches,
                microbatch_id,
                input_tensors[microbatch_id],
                config
            )

            # 发送输出
            if not p2p_communicator.is_pp_last_stage():
                p2p_communicator.send_forward(output_tensors[microbatch_id])

        elif order_id < 0:
            # 反向传播
            model_chunk_id = -order_id - num_model_chunks - 1
            microbatch_id = get_microbatch_id_from_schedule_index(
                i - num_warmup_microbatches,
                schedule_table
            )

            # 接收梯度
            if p2p_communicator.is_pp_last_stage():
                output_tensor_grad = p2p_communicator.recv_backward(
                    timeout=config.pipeline_model_parallel_backward_timeout
                )
            else:
                output_tensor_grad = None

            # 反向计算
            input_tensor_grad = backward_step_func(
                backward_step_func,
                input_tensors[microbatch_id],
                output_tensors[microbatch_id],
                output_tensor_grad,
                model[model_chunk_id],
                num_microbatches,
                microbatch_id,
                forward_data_store[model_chunk_id],
                config
            )

            # 发送梯度
            if not p2p_communicator.is_pp_first_stage():
                p2p_communicator.send_backward(input_tensor_grad)

            # 释放输出张量
            deallocate_output_tensor(output_tensors[microbatch_id])

    return forward_model_parallel_output
```

### 12.2 调度表生成算法

```python
def get_schedule_table(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage
):
    """
    生成交错流水线调度表

    输入:
    - num_microbatches: 总 microbatch 数量
    - num_model_chunks: 模型 chunk 数量
    - microbatch_group_size_per_vp_stage: 每个 VP stage 的 group 大小

    输出:
    - schedule_table: [(microbatch_id, model_chunk_id), ...]
    """

    schedule_table = []

    # 按 group 处理 microbatches
    for min_microbatch_id_in_group in range(
        0,
        num_microbatches,
        microbatch_group_size_per_vp_stage
    ):
        # 计算当前 group 的结束位置
        max_microbatch_id_in_group = min(
            min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
            num_microbatches
        )

        # 检查是否是最后一个 group
        if max_microbatch_id_in_group == num_microbatches:
            # 最后一个 group: 添加所有组合
            schedule_table.extend([
                (microbatch_id, model_chunk_id)
                for model_chunk_id in range(num_model_chunks)
                for microbatch_id in range(
                    min_microbatch_id_in_group,
                    max_microbatch_id_in_group
                )
            ])
        else:
            # 其他 groups: 添加完整 group 的组合
            schedule_table.extend([
                (microbatch_id, model_chunk_id)
                for model_chunk_id in range(num_model_chunks)
                for microbatch_id in range(
                    min_microbatch_id_in_group,
                    max_microbatch_id_in_group
                )
            ])

    return schedule_table

def convert_schedule_table_to_order(
    num_warmup_microbatches,
    num_model_chunks,
    schedule_table
):
    """
    将调度表转换为执行顺序

    示例:
    schedule_table = [(0,0), (0,1), (1,0), (1,1)]
    num_warmup_microbatches = 2
    num_model_chunks = 2

    输出: [+1, +1, +2, +2, -1, -2, -1, -2]
    """

    # 提取 model_chunk_id
    _, model_chunk_id_table = zip(*schedule_table)

    # 前向顺序: +1, +1, +2, +2, ...
    forward_order = [
        chunk_id + 1
        for chunk_id in model_chunk_id_table
    ]

    # 反向顺序: -N, -N, ... (N = num_model_chunks + chunk_id)
    backward_order = [
        chunk_id - num_model_chunks
        for chunk_id in model_chunk_id_table
    ]

    # 合并顺序
    order = forward_order[:num_warmup_microbatches]

    for i in range(num_warmup_microbatches, len(forward_order)):
        order.append(forward_order[i])
        order.append(backward_order[i - num_warmup_microbatches])

    # 添加剩余的反向
    if num_warmup_microbatches > 0:
        order.extend(backward_order[-num_warmup_microbatches:])

    return order
```

### 12.3 点对点通信优化

```python
class P2PCommunicator:
    """
    点对点通信器

    特性:
    1. 支持变长序列
    2. 批量通信优化
    3. 奇偶 rank 优化
    4. 超时管理
    """

    def __init__(self, pp_group, config):
        self.pp_group = pp_group
        self.config = config
        self.curr_rank = pp_group.rank()
        self.world_size = pp_group.size()

        # 计算相邻 rank
        self.next_rank = (self.curr_rank + 1) % self.world_size
        self.prev_rank = (self.curr_rank - 1) % self.world_size

        # 全局 rank
        self.next_global_rank = torch.distributed.get_global_rank(
            pp_group, self.next_rank
        )
        self.prev_global_rank = torch.distributed.get_global_rank(
            pp_group, self.prev_rank
        )

        # 通信模式
        self.use_ring_exchange = config.use_ring_exchange_p2p

    def send_forward_recv_forward(
        self,
        tensor_prev=None,
        tensor_next=None,
        prev_rank=None,
        next_rank=None,
        timeout=timedelta(seconds=1000)
    ):
        """
        发送前向输出并接收下一个前向输入

        用于流水线 warmup 阶段的预取
        """

        if prev_rank is None:
            prev_rank = self.prev_global_rank
        if next_rank is None:
            next_rank = self.next_global_rank

        # 通信形状
        send_prev_shape = self._get_shape(tensor_prev)
        send_next_shape = self._get_shape(tensor_next)

        # 分配接收缓冲区
        recv_prev_shape = torch.empty((3,), device="cuda", dtype=torch.int64)
        recv_next_shape = torch.empty((3,), device="cuda", dtype=torch.int64)

        # 批量 P2P 操作
        ops = []
        reqs = {}

        # 发送/接收形状
        if send_prev_shape is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape,
                prev_rank,
                self.pp_group
            ))
        if send_next_shape is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape,
                next_rank,
                self.pp_group
            ))
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_prev_shape,
            prev_rank,
            self.pp_group
        ))
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_next_shape,
            next_rank,
            self.pp_group
        ))

        # 执行批量通信
        if len(ops) > 0:
            p2p_reqs = torch.distributed.batch_isend_irecv(ops)

        # 等待形状通信完成
        for req in p2p_reqs:
            req.wait()

        # 分配数据缓冲区
        recv_prev_tensor = torch.empty(
            recv_prev_shape.tolist(),
            device="cuda",
            dtype=self.config.pipeline_dtype
        )
        recv_next_tensor = torch.empty(
            recv_next_shape.tolist(),
            device="cuda",
            dtype=self.config.pipeline_dtype
        )

        # 批量数据通信
        ops = []
        if tensor_prev is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_prev,
                prev_rank,
                self.pp_group
            ))
        if tensor_next is not None:
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend,
                tensor_next,
                next_rank,
                self.pp_group
            ))
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_prev_tensor,
            prev_rank,
            self.pp_group
        ))
        ops.append(torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_next_tensor,
            next_rank,
            self.pp_group
        ))

        # 执行批量通信
        if len(ops) > 0:
            p2p_reqs = torch.distributed.batch_isend_irecv(ops)

        # 等待通信完成
        for req in p2p_reqs:
            req.wait()

        return recv_prev_tensor, recv_next_tensor
```

---

## 13. 张量并行模块增强分析 ⭐新增

### 13.1 列并行线性层完整实现

```python
class ColumnParallelLinear(torch.autograd.Function):
    """
    列并行线性层

    数学原理:
    Y = XA，其中 A 按列分割
    A = [A1, A2, ..., Ap]^T
    Yi = XAi (每个 GPU)

    特性:
    1. 权重按列分割
    2. 输入复制到所有 GPU
    3. 输出通过 All-Gather 合并
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input_,
        weight,
        bias,
        gradient_accumulation_fusion,
        ...):
        """
        前向传播
        """
        # 1. 保存上下文
        ctx.save_for_backward(input_, weight)
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.use_bias = bias is not None
        ctx.activation_func = activation_func
        ctx.sequence_parallel = sequence_parallel
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.tp_group = tp_group

        # 2. 矩阵乘法
        # input_: [batch, seq_len, hidden_size]
        # weight:  [output_size/TP, hidden_size]
        # output:  [batch, seq_len, output_size/TP]
        output = torch.matmul(input_, weight.t())

        # 3. 添加 bias
        if bias is not None:
            output = output + bias

        # 4. 激活函数
        if activation_func is not None:
            output = activation_func(output)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        反向传播

        梯度计算:
        1. grad_input = grad_output @ weight
        2. grad_weight = grad_output^T @ input
        3. grad_bias = sum(grad_output)

        通信:
        - 如果 sequence_parallel: Reduce-Scatter
        - 否则: All-Reduce
        """
        input_, weight = ctx.saved_tensors

        # 1. 计算 grad_input
        # grad_output: [batch, seq_len, output_size/TP]
        # weight:        [output_size/TP, hidden_size]
        # grad_input:    [batch, seq_len, hidden_size]
        grad_input = torch.matmul(grad_output, weight)

        # 2. 梯度同步
        if ctx.sequence_parallel:
            # Reduce-Scatter: 每个 GPU 得到部分序列
            grad_input = reduce_scatter_to_sequence_parallel_region(
                grad_input,
                group=ctx.tp_group
            )
        elif ctx.tensor_parallel_output_grad:
            # All-Reduce: 每个 GPU 得到完整梯度
            handle = torch.distributed.all_reduce(
                grad_input,
                group=ctx.tp_group,
                async_op=True
            )
            if ctx.sequence_parallel is not None:
                handle.wait()

        # 3. 计算 grad_weight
        # grad_output: [batch, seq_len, output_size/TP]
        # input_:       [batch, seq_len, hidden_size]
        # grad_weight:  [output_size/TP, hidden_size]
        grad_weight = torch.matmul(grad_output.t(), input_)

        # 4. 计算 grad_bias
        if ctx.use_bias:
            grad_bias = grad_output.sum(dim=(0, 1))
        else:
            grad_bias = None

        # 5. 梯度累积融合
        if ctx.gradient_accumulation_fusion:
            # 使用融合内核累积梯度
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    input_,
                    grad_output,
                    weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    input_,
                    grad_output,
                    weight.main_grad
                )
        else:
            # 标准梯度累积
            if weight.main_grad is not None:
                weight.main_grad.add_(grad_weight)

        return grad_input, grad_weight, grad_bias, ...
```

### 13.2 行并行线性层完整实现

```python
class RowParallelLinear(torch.autograd.Function):
    """
    行并行线性层

    数学原理:
    Y = XA，其中 X 和 A 按行分割
    X = [X1, X2, ..., Xp]^T
    A = [A1, A2, ..., Ap]^T
    Yi = XiAi (每个 GPU)
    Y = Y1 + Y2 + ... + Yp (All-Reduce)

    特性:
    1. 输入按序列维度分割
    2. 权重按行分割
    3. 输出通过 All-Reduce 合并
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input_,
        weight,
        bias,
        input_is_parallel,
        gradient_accumulation_fusion,
        ...):
        """
        前向传播
        """
        # 1. 处理输入
        if input_is_parallel:
            input_parallel = input_
        else:
            # Scatter 输入
            if sequence_parallel:
                input_parallel = scatter_to_sequence_parallel_region(
                    input_,
                    group=tp_group
                )
            else:
                input_parallel = scatter_to_tensor_model_parallel_region(
                    input_,
                    group=tp_group
                )

        # 2. 保存上下文
        ctx.save_for_backward(input_parallel, weight)
        ctx.input_is_parallel = input_is_parallel
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.use_bias = bias is not None
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group

        # 3. 矩阵乘法
        # input_parallel: [batch, seq_len/TP, hidden_size]
        # weight:         [hidden_size/TP, output_size]
        # output_parallel: [batch, seq_len/TP, output_size]
        output_parallel = torch.matmul(input_parallel, weight.t())

        # 4. All-Reduce 合并输出
        if sequence_parallel:
            # Reduce-Scatter: 每个 GPU 得到部分序列
            output = reduce_scatter_to_sequence_parallel_region(
                output_parallel,
                group=tp_group
            )
        else:
            # All-Reduce: 每个 GPU 得到完整输出
            output = all_reduce = torch.distributed.all_reduce(
                output_parallel,
                group=tp_group,
                async_op=True
            )
            output = all_reduce.wait() if async_op else all_reduce

        # 5. 添加 bias (只在 rank 0 添加)
        if bias is not None and not ctx.skip_bias_add:
            output = output + bias

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        反向传播

        梯度计算:
        1. grad_input_parallel = grad_output @ weight
        2. grad_weight = grad_output^T @ input_parallel
        3. grad_bias = sum(grad_output)

        通信:
        - 前向: Scatter 输入，All-Reduce 输出
        - 反向: All-Gather grad_input，All-Reduce grad_weight
        """
        input_parallel, weight = ctx.saved_tensors

        # 1. 计算 grad_input_parallel
        # grad_output:      [batch, seq_len/TP, output_size]
        # weight:           [hidden_size/TP, output_size]
        # grad_input_parallel: [batch, seq_len/TP, hidden_size/TP]
        grad_input_parallel = torch.matmul(grad_output, weight)

        # 2. All-Gather grad_input
        if ctx.input_is_parallel:
            grad_input = grad_input_parallel
        else:
            if ctx.sequence_parallel:
                grad_input = gather_from_sequence_parallel_region(
                    grad_input_parallel,
                    group=ctx.tp_group
                )
            else:
                grad_input = all_gather_from_tensor_model_parallel_region(
                    grad_input_parallel,
                    group=ctx.tp_group
                )

        # 3. 计算 grad_weight
        # grad_output: [batch, seq_len/TP, output_size]
        # input_parallel: [batch, seq_len/TP, hidden_size/TP]
        # grad_weight: [hidden_size/TP, output_size]
        grad_weight = torch.matmul(grad_output.t(), input_parallel)

        # 4. All-Reduce grad_weight
        handle = torch.distributed.all_reduce(
            grad_weight,
            group=ctx.tp_group,
            async_op=True
        )
        handle.wait()

        # 5. 计算 grad_bias
        if ctx.use_bias:
            grad_bias = grad_output.sum(dim=(0, 1))
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, ...
```

### 13.3 词汇并行交叉熵完整实现

```python
class VocabParallelCrossEntropy(torch.autograd.Function):
    """
    词汇并行交叉熵损失

    核心思想:
    1. 词汇表分割到不同 GPU
    2. 每个GPU计算部分词表的logits
    3. 通过通信聚合结果

    关键优化:
    - 最大值同步 (MAX)
    - 预测值同步 (SUM)
    - 指数和同步 (SUM)
    """

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits,
        target,
        label_smoothing=0.0
    ):
        """
        前向传播

        vocab_parallel_logits: [batch*seq_len, vocab_size/TP]
        target: [batch*seq_len]
        """
        # 1. 获取词表范围
        vocab_start_index = ...
        vocab_end_index = ...

        # 2. 计算最大值 (数值稳定)
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]

        # All-Reduce (MAX) 同步最大值
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group()
        )

        # 3. 减去最大值
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(-1)

        # 4. 处理目标词
        target_mask = (
            (target < vocab_start_index) |
            (target >= vocab_end_index)
        )
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # 5. 获取预测值
        arange_1d = torch.arange(
            vocab_parallel_logits.size(0),
            device=vocab_parallel_logits.device
        )
        predicted_logits = vocab_parallel_logits[
            arange_1d, masked_target
        ]

        # All-Reduce (SUM) 同步预测值
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group()
        )

        # 6. 计算指数和
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        # All-Reduce (SUM) 同步指数和
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group()
        )

        # 7. 计算损失
        loss = -predicted_logits + torch.log(sum_exp_logits)

        # 8. 计算梯度
        exp_logits.div_(sum_exp_logits.unsqueeze(-1))
        grad_input = exp_logits

        # 设置目标词的梯度
        grad_input[arange_1d, masked_target] -= 1.0

        # 掩码无效梯度
        grad_input[target_mask, :] = 0.0

        # 9. 标签平滑
        if label_smoothing > 0:
            grad_input += label_smoothing / (vocab_end_index - vocab_start_index)

        # 10. 保存上下文
        ctx.save_for_backward(grad_input)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播

        直接返回保存的梯度，乘以 grad_output
        """
        grad_input, = ctx.saved_tensors
        return grad_input * grad_output.unsqueeze(-1), None, None
```

### 13.4 序列并行详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          序列并行 (Sequence Parallelism)                     │
└─────────────────────────────────────────────────────────────────────────────┘

概念:
    在序列维度上进行分割，减少激活内存占用

标准张量并行:
    输入: [batch, seq_len, hidden] → 复制到所有 GPU
    列并行: GPU_i 计算 [batch, seq_len, hidden/TP]
    行并行: GPU_i 计算 [batch, seq_len, hidden] → All-Reduce

序列并行:
    输入: [batch, seq_len, hidden] → Scatter → [batch, seq_len/TP, hidden]
    列并行: GPU_i 计算 [batch, seq_len/TP, hidden/TP]
    行并行: GPU_i 计算 [batch, seq_len/TP, hidden] → Reduce-Scatter

优势:
    1. 减少激活内存: 从 O(seq_len * hidden) 到 O(seq_len/TP * hidden)
    2. 减少通信: All-Reduce → Reduce-Scatter
    3. 更好的扩展性

实现:

    列并行 + 序列并行:
        输入: [batch, seq_len, hidden]
        ├─ Scatter → [batch, seq_len/TP, hidden] (每个 GPU)
        ├─ Linear: [batch, seq_len/TP, hidden] @ [hidden, hidden/TP]
        └─ 输出: [batch, seq_len/TP, hidden/TP]

    行并行 + 序列并行:
        输入: [batch, seq_len/TP, hidden/TP]
        ├─ Linear: [batch, seq_len/TP, hidden/TP] @ [hidden/TP, hidden]
        ├─ 输出: [batch, seq_len/TP, hidden]
        └─ Reduce-Scatter → [batch, seq_len/TP, hidden]

内存对比:
    标准 TP:
        激活内存: 2 * batch * seq_len * hidden * sizeof(dtype)
        通信: All-Reduce (batch * seq_len * hidden)

    序列并行:
        激活内存: 2 * batch * seq_len/TP * hidden * sizeof(dtype)
        通信: Reduce-Scatter (batch * seq_len/TP * hidden)

    节省: TP 倍的激活内存
```

---

## 参考资源

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core 文档](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)

---

*本文档基于 Megatron-LM 代码库的深入分析生成，涵盖了项目的核心架构和实现细节。*
