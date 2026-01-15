# Megatron-LM 张量并行层实现注释详解

> 文件：`megatron/core/tensor_parallel/layers.py`
> 功能：实现张量并行的核心层（Embedding、Linear）

---

## 模块导入与配置

```python
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
```
**说明**：NVIDIA 版权声明

```python
# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
```
**说明**：部分代码改编自 PyTorch

```python
import os
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
```
**说明**：标准库和 PyTorch 导入
- `os`: 环境变量访问
- `warnings`: 警告信息
- `partial`: 函数偏应用（用于固定某些参数）
- `F`: torch.nn.functional，包含各种函数式 API
- `Parameter`: 神经网络参数类

---

## Megatron Core 导入

```python
from megatron.core.model_parallel_config import ModelParallelConfig
```
**说明**：模型并行配置类，包含所有并行相关的配置参数

```python
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
```
**说明**：
- `get_global_memory_buffer`: 获取全局内存缓冲区（通信优化）
- `get_tensor_model_parallel_rank`: 获取当前 tensor parallel rank
- `get_tensor_model_parallel_world_size`: 获取 tensor parallel world size

```python
from megatron.core.utils import (
    divide,
    get_pg_rank,
    get_pg_size,
    get_tensor_model_parallel_group_if_none,
    is_torch_min_version,
    make_tp_sharded_tensor_for_checkpoint,
    prepare_input_tensors_for_wgrad_compute,
)
```
**说明**：
- `divide`: 安全除法（带检查）
- `get_pg_rank`: 获取进程组 rank
- `get_pg_size`: 获取进程组大小
- `get_tensor_model_parallel_group_if_none`: 获取 TP 进程组
- `is_torch_min_version`: 检查 PyTorch 版本
- `make_tp_sharded_tensor_for_checkpoint`: 创建分片张量用于 checkpoint
- `prepare_input_tensors_for_wgrad_compute`: 准备输入张量用于权重梯度计算

```python
from ..dist_checkpointing.mapping import ShardedStateDict
from ..transformer.utils import make_sharded_tensors_for_checkpoint
```
**说明**：
- `ShardedStateDict`: 分片状态字典类型
- `make_sharded_tensors_for_checkpoint`: 为 checkpoint 创建分片张量

```python
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
```
**说明**：导入通信原语（见 mappings.py 注释）

```python
from .random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
```
**说明**：
- `get_cuda_rng_tracker`: 获取 CUDA RNG 追踪器（确保初始化一致性）
- `get_expert_parallel_rng_tracker_name`: 获取专家并行 RNG 追踪器名称

```python
from .utils import VocabUtility
```
**说明**：词汇表工具类（用于计算词汇表分割范围）

---

## 可选依赖检查

```python
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False
```
**说明**：检查梯度累积融合 CUDA 扩展是否可用
- `fused_weight_gradient_mlp_cuda`: 自定义 CUDA kernel，用于优化权重梯度计算

```python
try:
    import transformer_engine  # pylint: disable=unused-import
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    HAVE_TE = True
except ImportError:
    HAVE_TE = False
```
**说明**：检查 Transformer Engine 是否可用
- Transformer Engine: NVIDIA 提供的 FP8 训练库
- `get_dummy_wgrad`: 获取虚拟权重梯度（用于内存优化）

---

## 模型并行属性默认值

```python
_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,  # 是否是张量并行的
    "partition_dim": -1,              # 分割维度
    "partition_stride": 1,            # 分割步长
}
```
**说明**：张量并行属性的默认值

---

## 混合精度（AMP）配置

```python
try:
    if is_torch_min_version("2.4.0a0"):
        # PyTorch 2.4+ 使用新的 API
        custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
        custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
    else:
        # PyTorch 2.4 之前使用旧 API
        custom_fwd = torch.cuda.amp.custom_fwd
        custom_bwd = torch.cuda.amp.custom_bwd
except:
    # 降级到旧 API
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd
```
**说明**：根据 PyTorch 版本选择混合精度装饰器
- `custom_fwd`: 自定义前向传播装饰器（支持自动混合精度）
- `custom_bwd`: 自定义反向传播装饰器

---

## 分布式通信函数选择

```python
try:
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
```
**说明**：根据 PyTorch 版本选择通信 API

---

## 工具函数

### 参数去重检查

```python
def param_is_not_tensor_parallel_duplicate(param):
    """
    检查参数是否不是 tensor parallel 的重复参数

    在 tensor parallel 中，某些参数在多个 rank 上重复。
    只需要在 rank 0 上保存这些参数的 checkpoint。

    Args:
        param: 要检查的参数

    Returns:
        如果参数不是重复的，返回 True
    """
    return (hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel) or (
        get_tensor_model_parallel_rank() == 0
    )
```

---

### 设置张量并行属性

```python
def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    """
    为张量设置张量并行属性

    Args:
        tensor: 要设置属性的张量
        is_parallel: 是否是并行张量
        dim: 分割维度
        stride: 分割步长
    """
    # 确保属性尚未设置
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)

    # 设置属性
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)
```

---

### 设置默认属性

```python
def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    """
    如果尚未显式设置，则设置默认的模型并行属性
    """
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])
```

---

### 复制张量并行属性

```python
def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    """
    从一个张量复制模型并行属性到另一个张量
    """
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)
```

---

## 权重初始化函数

### GPU 权重初始化

```python
def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1, is_expert=False):
    """
    在 GPU 上初始化仿射层权重（模型并行）

    Args:
        weight: 权重张量
        init_method: 初始化方法（如 Xavier、Kaiming）
        partition_dim: 分割维度
        stride: 分割步长
        is_expert: 是否是专家权重
    """
    # 设置张量并行属性
    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    # 使用 RNG 追踪器确保各 rank 初始化一致
    if not is_expert:
        with get_cuda_rng_tracker().fork():
            init_method(weight)
    else:
        # 专家权重使用独立的 RNG 追踪器
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
            init_method(weight)
```
**关键点**：
- RNG 追踪器确保所有 TP rank 的权重初始化一致
- 专家权重使用独立的 RNG 状态

---

### CPU 权重初始化

```python
def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    rank=None,
    world_size=None,
    skip_set_tensor_parallel_attributes=False,
):
    """
    在 CPU 上初始化仿射层权重（模型并行）

    策略：在所有进程上构建完整的主权重，然后分割相关部分

    Args:
        weight: 权重张量（待初始化）
        output_size: 输出维度
        input_size: 输入维度
        per_partition_size: 每个 partition 的大小
        partition_dim: 分割维度
        init_method: 初始化方法
        stride: 分割步长
        return_master_weight: 是否返回主权重
        params_dtype: 参数数据类型
        rank: 当前 rank
        world_size: 总 rank 数
        skip_set_tensor_parallel_attributes: 是否跳过设置 TP 属性

    Returns:
        主权重（如果 return_master_weight=True）
    """
    # 设置张量并行属性
    if not skip_set_tensor_parallel_attributes:
        set_tensor_model_parallel_attributes(
            tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
        )

    # 初始化主权重（在 CPU 上）
    master_weight = torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # 分割并复制
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)

    if rank is None:
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()

    # 获取当前 rank 对应的权重片段
    # 例如：world_size=4, rank=1，取 [1::4] 即第 1、5、9... 个片段
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        # 所有张量必须在同一设备上
        cpu_weight = torch.cat(my_weight_list, dim=partition_dim).to_dense()
        weight.data.copy_(cpu_weight)

    if return_master_weight:
        return master_weight
    return None
```
**设计思路**：
1. 在 CPU 上创建完整的主权重
2. 使用 torch.split 分割权重
3. 根据 rank 选择对应的片段
4. 复制到目标权重张量

---

## 核心层实现

### VocabParallelEmbedding

```python
class VocabParallelEmbedding(torch.nn.Module):
    """
    沿词汇表维度并行的 Embedding 层

    词汇表被分割到多个 GPU，每个 GPU 只存储一部分词向量

    Args:
        num_embeddings: 词汇表大小
        embedding_dim: 嵌入维度
        init_method: 初始化方法
        reduce_scatter_embeddings: 是否在 embedding lookup 后执行 Reduce-Scatter
        config: ModelParallelConfig 对象
        tp_group: Tensor Parallel 进程组
    """
```

---

#### VocabParallelEmbedding.__init__

```python
def __init__(
    self,
    num_embeddings: int,
    embedding_dim: int,
    *,
    init_method: Callable,
    reduce_scatter_embeddings: bool = False,
    config: ModelParallelConfig,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    super(VocabParallelEmbedding, self).__init__()

    # 保存输入参数
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.reduce_scatter_embeddings = reduce_scatter_embeddings
    self.tp_group = tp_group

    # 获取张量并行进程组
    self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)

    # 计算当前 rank 负责的词汇表范围
    (self.vocab_start_index, self.vocab_end_index) = (
        VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, get_pg_rank(self.tp_group), get_pg_size(self.tp_group)
        )
    )
    # 当前 rank 负责的词表大小
    self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
    self.deterministic_mode = config.deterministic_mode

    # 分配权重并初始化
    if config.use_cpu_initialization:
        # CPU 初始化
        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
            )
        )
        if config.perform_initialization:
            _initialize_affine_weight_cpu(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,  # partition_dim=0，沿第一维（词汇表维度）分割
                init_method,
                params_dtype=config.params_dtype,
                rank=get_pg_rank(self.tp_group),
                world_size=get_pg_size(self.tp_group),
            )
    else:
        # GPU 初始化
        self.weight = Parameter(
            torch.empty(
                self.num_embeddings_per_partition,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)
```
**词汇表分割示例**（TP=2, vocab_size=10000）：
```
Rank 0: 负责词汇 0-4999
Rank 1: 负责词汇 5000-9999
```

---

#### VocabParallelEmbedding.forward

```python
def forward(self, input_):
    """
    前向传播

    Args:
        input_: 输入 token IDs，形状 [seq_len, batch_size]

    Returns:
        嵌入向量
    """
    if self.tp_group.size() > 1:
        # 构建掩码：标记哪些 token 属于当前 rank
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # 将输入映射到局部索引
        masked_input = input_.clone() - self.vocab_start_index
        # 不属于当前 rank 的 token 映射到 0（稍后会被 mask 掉）
        masked_input[input_mask] = 0
    else:
        masked_input = input_

    # 获取嵌入向量
    if self.deterministic_mode:
        # 确定性模式（用于测试）
        output_parallel = self.weight[masked_input]
    else:
        # 使用 F.embedding（更快但非确定性）
        output_parallel = F.embedding(masked_input, self.weight)

    # 掩码掉不属于当前 rank 的嵌入
    if self.tp_group.size() > 1:
        output_parallel[input_mask, :] = 0.0

    if self.reduce_scatter_embeddings:
        # 序列并行模式：转置并 Reduce-Scatter
        # 数据格式转换: [batch, seq, hidden] -> [seq, batch, hidden]
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(
            output_parallel, group=self.tp_group
        )
    else:
        # 标准模式：All-Reduce
        output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

    return output
```
**前向传播流程**：
1. **输入掩码**：标记哪些 token 属于当前 rank
2. **局部 lookup**：只在当前 rank 的词表部分查找
3. **输出掩码**：将不属于当前 rank 的输出置零
4. **All-Reduce**：汇总所有 rank 的嵌入结果

---

### LinearWithFrozenWeight

```python
class LinearWithFrozenWeight(torch.autograd.Function):
    """
    权重冻结的 Linear 操作（不计算权重梯度）

    与 `LinearWithGradAccumulationAndAsyncCommunication` 在数学上等价

    概念上等同于 `weight.requires_grad=False` 的 torch.nn.functional.linear，
    但实验证明两者在数学上不完全相同
    """
```

---

#### LinearWithFrozenWeight.forward

```python
@staticmethod
@custom_fwd
def forward(ctx, input, weight, bias, allreduce_dgrad, tp_group):
    """
    前向传播（权重冻结）

    Args:
        ctx: 上下文对象
        input: 输入张量
        weight: 权重张量（不需要梯度）
        bias: 偏置
        allreduce_dgrad: 是否对输入梯度执行 All-Reduce
        tp_group: Tensor Parallel 进程组

    Returns:
        输出张量
    """
    ctx.save_for_backward(weight)  # 只保存权重，不保存输入（节省内存）
    ctx.allreduce_dgrad = allreduce_dgrad
    ctx.tp_group = tp_group

    # Y = XA^T
    output = torch.matmul(input, weight.t())
    if bias is not None:
        output = output + bias
    return output
```

---

#### LinearWithFrozenWeight.backward

```python
@staticmethod
@custom_bwd
def backward(ctx, grad_output):
    """
    反向传播（权重冻结）

    只计算输入梯度，不计算权重梯度
    """
    (weight,) = ctx.saved_tensors

    # dX = dY * A
    grad_input = grad_output.matmul(weight)

    if ctx.allreduce_dgrad:
        # All-Reduce 输入梯度（同步和异步效果相同）
        torch.distributed.all_reduce(grad_input, group=ctx.tp_group)

    return grad_input, None, None, None, None
    # None 表示对 weight, bias, allreduce_dgrad, tp_group 没有梯度
```
**应用场景**：
- 微调时冻结部分层
- 推理模式
- 参数高效微调（如 LoRA）

---

### linear_with_frozen_weight

```python
def linear_with_frozen_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    tp_group: Optional[torch.distributed.ProcessGroup],
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    wgrad_deferral_limit: None = None,
    async_grad_allreduce: Optional[bool] = None,
) -> torch.Tensor:
    """
    权重冻结的 Linear 层执行函数

    参数说明（略，见函数文档）
    """
    if async_grad_allreduce is not None:
        warnings.warn(
            "async_grad_allreduce is deprecated, not in use anymore and will"
            " be fully removed with 0.11.0. Please use allreduce_dgrad instead."
        )

    # 这些参数只用于保持 API 一致性，实际不使用
    assert grad_output_buffer is None, (
        "grad_output_buffer kwarg is only supported with "
        "linear_with_grad_accumulation_and_async_allreduce"
    )
    assert wgrad_deferral_limit is None, (
        "This arg is only supported with " "linear_with_grad_accumulation_and_async_allreduce"
    )

    tp_group = get_tensor_model_parallel_group_if_none(tp_group)

    # 序列并行：All-Gather 输入
    if sequence_parallel:
        input = gather_from_sequence_parallel_region(
            input, tensor_parallel_output_grad=True, group=tp_group
        )

    args = [input, weight, bias, allreduce_dgrad, tp_group]

    return LinearWithFrozenWeight.apply(*args)
```

---

### LinearWithGradAccumulationAndAsyncCommunication

```python
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    支持梯度累积融合和异步通信的 Linear 操作

    关键优化：
    1. 梯度累积融合：直接累积到主梯度缓冲区
    2. 异步通信：通信与计算重叠
    3. 权重梯度延迟：支持延迟计算权重梯度
    """
```

---

#### forward 方法

```python
@staticmethod
@custom_fwd
def forward(
    ctx,
    input,
    weight,
    bias,
    gradient_accumulation_fusion,
    allreduce_dgrad,
    sequence_parallel,
    grad_output_buffer,
    wgrad_deferral_limit,
    tp_group,
):
    """前向传播"""
    # 获取主梯度缓冲区
    if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
        main_grad = weight.main_grad
    else:
        main_grad = None

    # 保存前向传播所需的信息
    ctx.save_for_backward(input, weight)
    ctx.main_grad = main_grad  # 不能用 save_for_backward，因为会被跨层复用
    ctx.use_bias = bias is not None
    ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
    ctx.allreduce_dgrad = allreduce_dgrad
    ctx.sequence_parallel = sequence_parallel
    ctx.wgrad_deferral_limit = wgrad_deferral_limit
    ctx.grad_output_buffer = grad_output_buffer
    ctx.tp_group = tp_group

    # 序列并行：All-Gather 输入
    if sequence_parallel:
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * tp_group.size()

        # 使用全局内存缓冲区（减少分配开销）
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        dist_all_gather_func(all_gather_buffer, input, group=tp_group)
        total_input = all_gather_buffer
    else:
        total_input = input

    # Y = XA^T + b
    output = torch.matmul(total_input, weight.t())
    if bias is not None:
        output = output + bias
    return output
```

---

#### backward 方法

```python
@staticmethod
@custom_bwd
def backward(ctx, grad_output):
    """反向传播"""
    input, weight = ctx.saved_tensors
    main_grad = ctx.main_grad
    use_bias = ctx.use_bias
    grad_output_buffer = ctx.grad_output_buffer
    wgrad_deferral_limit = ctx.wgrad_deferral_limit
    handle = None  # 通信句柄
    tp_group = ctx.tp_group

    # 恢复主梯度指针
    if ctx.gradient_accumulation_fusion:
        weight.main_grad = main_grad

    # 权重梯度延迟检查
    wgrad_compute = True
    if grad_output_buffer is not None:
        if wgrad_deferral_limit == 0 or len(grad_output_buffer) < wgrad_deferral_limit:
            grad_output_buffer.append(grad_output)
            wgrad_compute = False

    # 计算输入梯度
    if wgrad_compute:
        if ctx.sequence_parallel:
            # 异步 All-Gather 输入（用于计算权重梯度）
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            handle = dist_all_gather_func(
                all_gather_buffer, input, group=tp_group, async_op=True
            )
            # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 确保 gather 在计算前调度
            total_input = all_gather_buffer
        else:
            total_input = input

    # dX = dY * A
    grad_input = grad_output.matmul(weight)

    # 等待 All-Gather 完成（如果启用）
    if ctx.sequence_parallel and wgrad_compute:
        handle.wait()

    # 准备权重梯度计算的输入
    if wgrad_compute:
        grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
            grad_output, total_input
        )

    # 异步 All-Reduce 输入梯度
    if ctx.allreduce_dgrad:
        # 异步启动 All-Reduce（与权重梯度计算重叠）
        handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
        # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1

    # 异步 Reduce-Scatter（序列并行模式）
    if ctx.sequence_parallel:
        assert not ctx.allreduce_dgrad  # 序列并行不需要 All-Reduce
        dim_size = list(input.size())
        sub_grad_input = torch.empty(
            dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
        )
        # 异步 Reduce-Scatter
        handle = dist_reduce_scatter_func(
            sub_grad_input, grad_input, group=tp_group, async_op=True
        )
        # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1

    # 计算权重梯度
    if ctx.gradient_accumulation_fusion:
        if wgrad_compute:
            # Megatron-FSDP: 需要就地创建主梯度缓冲区
            if hasattr(weight, "__fsdp_param__"):
                weight.main_grad = weight.get_main_grad()
                torch.matmul(grad_output.t(), total_input, out=weight.main_grad)
            else:
                # 使用融合的 CUDA kernel
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError(
                        "Unsupported gradient type for gradient accumulation fusion"
                    )

        # 设置虚拟梯度（避免向后 hooks 在后台线程运行）
        if hasattr(weight, "grad_added_to_main_grad"):
            if getattr(weight, "zero_out_wgrad", False):
                if HAVE_TE:
                    grad_weight = get_dummy_wgrad(
                        list(weight.main_grad.shape), input.dtype, zero=True
                    )
                else:
                    grad_weight = torch.zeros(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            else:
                if HAVE_TE:
                    grad_weight = get_dummy_wgrad(list(weight.main_grad.shape), input.dtype)
                else:
                    grad_weight = torch.empty(
                        weight.main_grad.shape,
                        dtype=input.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            weight.grad_added_to_main_grad = True
        else:
            grad_weight = None
    else:
        # 标准权重梯度计算
        grad_weight = grad_output.t().matmul(total_input)

    # 偏置梯度
    grad_bias = grad_output.sum(dim=0) if use_bias else None

    # 等待 Reduce-Scatter 完成（序列并行）
    if ctx.sequence_parallel:
        handle.wait()
        return (sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None)

    # 等待 All-Reduce 完成
    if ctx.allreduce_dgrad:
        handle.wait()

    return grad_input, grad_weight, grad_bias, None, None, None, None, None, None
```
**通信与计算重叠**：
```
时间线：
t0: 启动 All-Reduce (dX)
t1: 开始计算 dW（All-Reduce 在后台进行）
t2: All-Reduce 完成
```

---

### ColumnParallelLinear

```python
class ColumnParallelLinear(torch.nn.Module):
    """
    列并行 Linear 层

    线性层定义为 Y = XA + b
    权重矩阵 A 沿第二维（列/输出维度）分割：A = [A_1, ..., A_p]

    Args:
        input_size: 矩阵 A 的第一维
        output_size: 矩阵 A 的第二维
        bias: 是否添加偏置
        gather_output: 是否 All-Gather 输出
        init_method: 权重初始化方法
        stride: 用于跨步 Linear 层
        keep_master_weight_for_test: 是否保留主权重（测试用）
        skip_bias_add: 是否跳过偏置添加（返回给调用者添加）
        skip_weight_param_allocation: 是否跳过权重参数分配
        embedding_activation_buffer: Embedding 激活缓冲区
        grad_output_buffer: 梯度输出缓冲区
        is_expert: 是否是 MoE 专家层
        config: ModelParallelConfig 对象
        tp_comm_buffer_name: 通信缓冲区名称
        disable_grad_reduce: 是否禁用梯度归约
        tp_group: Tensor Parallel 进程组
    """
```

---

#### ColumnParallelLinear.__init__

```python
def __init__(
    self,
    input_size,
    output_size,
    *,
    config: ModelParallelConfig,
    init_method: Callable,
    bias=True,
    gather_output=False,
    stride=1,
    keep_master_weight_for_test=False,
    skip_bias_add=False,
    skip_weight_param_allocation: bool = False,
    embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    is_expert: bool = False,
    tp_comm_buffer_name: str = None,
    disable_grad_reduce: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    super(ColumnParallelLinear, self).__init__()

    # 保存输入参数
    self.input_size = input_size
    self.output_size = output_size
    self.gather_output = gather_output
    self.skip_bias_add = skip_bias_add
    self.is_expert = is_expert
    self.expert_parallel = config.expert_model_parallel_size > 1
    self.embedding_activation_buffer = embedding_activation_buffer
    self.grad_output_buffer = grad_output_buffer
    self.config = config
    self.disable_grad_reduce = disable_grad_reduce
    self.tp_group = tp_group

    # 获取 Tensor Parallel 进程组
    self.tp_group = get_tensor_model_parallel_group_if_none(
        self.tp_group, is_expert=self.is_expert
    )
    world_size = get_pg_size(self.tp_group)
    rank = get_pg_rank(self.tp_group)
    self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

    # 每个partition 的输出大小
    self.output_size_per_partition = divide(output_size, world_size)

    # 权重参数分配
    if not skip_weight_param_allocation:
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition, input_size, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,  # partition_dim=0，沿第一维分割
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    stride=stride,
                    is_expert=self.is_expert,
                )

        setattr(self.weight, "allreduce", not (self.is_expert and self.expert_parallel))
    else:
        self.weight = None

    # 偏置参数分配
    if bias:
        if config.use_cpu_initialization:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
            )
        else:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
        set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
        if config.perform_initialization:
            # 偏置始终初始化为零
            with torch.no_grad():
                self.bias.zero_()
        setattr(self.bias, "allreduce", not (self.is_expert and self.expert_parallel))
    else:
        self.register_parameter("bias", None)

    # 序列并行配置
    self.sequence_parallel = config.sequence_parallel
    if self.sequence_parallel and world_size <= 1:
        warnings.warn(
            "`sequence_parallel` is set to `True`, but tensor model parallel size "
            f"is {world_size}. Disabling sequence parallel."
        )
        self.sequence_parallel = False

    # All-Reduce 输入梯度配置
    self.allreduce_dgrad = (
        world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
    )

    # 梯度累积融合检查
    if config.gradient_accumulation_fusion and not _grad_accum_fusion_available:
        raise RuntimeError(
            "ColumnParallelLinear was called with gradient_accumulation_fusion set "
            "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
            "module is not found. To use gradient_accumulation_fusion you must "
            "install APEX with --cpp_ext and --cuda_ext. For example: "
            'pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." '
            "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
            "gradient accumulation fusion."
        )
    self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

    # 互斥检查
    if self.allreduce_dgrad and self.sequence_parallel:
        raise RuntimeError(
            "`allreduce_dgrad` and `sequence_parallel` cannot be enabled at the same time."
        )

    # Hook: 为 state dict 添加默认的空 _extra_state
    self._register_load_state_dict_pre_hook(
        lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
            f"{prefix}_extra_state"
        )
    )
```

---

#### ColumnParallelLinear.forward

```python
def forward(
    self,
    input_: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    runtime_gather_output: Optional[bool] = None,
):
    """
    前向传播

    Args:
        input_: 输入张量，形状 [sequence, batch, hidden]
        weight: 可选的权重张量（skip_weight_param_allocation=True 时必需）
        runtime_gather_output: 运行时决定是否 gather 输出

    Returns:
        - output: 输出张量
        - bias: 偏置（如果 skip_bias_add=True）
    """
    # 权重处理
    if weight is None:
        if self.weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True."
            )
        weight = self.weight
    else:
        # 检查权重形状
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected"
            )

    bias = self.bias if not self.skip_bias_add else None

    # 输入处理
    if (
        self.allreduce_dgrad
        or self.sequence_parallel
        or self.explicit_expert_comm
        or self.disable_grad_reduce
    ):
        input_parallel = input_
    else:
        # 标准模式：复制到 TP 区域（反向时 All-Reduce）
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

    # Embedding 权重梯度延迟
    if self.config.defer_embedding_wgrad_compute:
        if (
            self.config.wgrad_deferral_limit == 0
            or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
        ):
            self.embedding_activation_buffer.append(input_parallel)

    # 矩阵乘法
    allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad

    # CPU Offloading 支持
    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context is True:
            if not HAVE_TE:
                assert (
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while TE is not present"
            else:
                input_parallel.activation_offloading = self.config.cpu_offloading_activations

    # 执行矩阵乘法
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
        grad_output_buffer=(
            self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None
        ),
        wgrad_deferral_limit=(
            self.config.wgrad_deferral_limit
            if self.config.defer_embedding_wgrad_compute
            else None
        ),
        tp_group=self.tp_group,
    )

    # 输出处理
    gather_output = self.gather_output
    if runtime_gather_output is not None:
        gather_output = runtime_gather_output

    if gather_output:
        # All-Gather 输出
        output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
    else:
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```
**列并行示例**（TP=2）：
```
输入: X = [seq, batch, hidden]
权重:
  Rank 0: A_0, 形状 [hidden/2, hidden]
  Rank 1: A_1, 形状 [hidden/2, hidden]
输出:
  Rank 0: Y_0 = X * A_0^T, 形状 [seq, batch, hidden/2]
  Rank 1: Y_1 = X * A_1^T, 形状 [seq, batch, hidden/2]
All-Gather 后: Y = [Y_0, Y_1], 形状 [seq, batch, hidden]
```

---

### RowParallelLinear

```python
class RowParallelLinear(torch.nn.Module):
    """
    行并行 Linear 层

    线性层定义为 Y = XA + b
    权重矩阵 A 沿第一维（行/输入维度）分割：A = transpose([A_1, ..., A_p])
    输入 X 沿第二维分割：X = [X_1, ..., X_p]

    Args:
        input_size: 矩阵 A 的第一维
        output_size: 矩阵 A 的第二维
        bias: 是否添加偏置（偏置不被分割）
        input_is_parallel: 输入是否已经分割
        init_method: 权重初始化方法
        stride: 用于跨步 Linear 层
        keep_master_weight_for_test: 是否保留主权重
        skip_bias_add: 是否跳过偏置添加
        is_expert: 是否是 MoE 专家层
        tp_comm_buffer_name: 通信缓冲区名称
        config: ModelParallelConfig 对象
    """
```

---

#### RowParallelLinear.__init__

```python
def __init__(
    self,
    input_size: int,
    output_size: int,
    *,
    config: ModelParallelConfig,
    init_method: Callable,
    bias: bool,
    input_is_parallel: bool,
    skip_bias_add: bool,
    stride: int = 1,
    keep_master_weight_for_test: bool = False,
    is_expert: bool = False,
    tp_comm_buffer_name: str = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    super(RowParallelLinear, self).__init__()

    # 保存输入参数
    self.input_size = input_size
    self.output_size = output_size
    self.input_is_parallel = input_is_parallel
    self.skip_bias_add = skip_bias_add
    self.config = config
    self.is_expert = is_expert
    self.expert_parallel = config.expert_model_parallel_size > 1
    self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
    self.sequence_parallel = config.sequence_parallel
    self.tp_group = tp_group

    # 序列并行检查
    if self.sequence_parallel and not self.input_is_parallel:
        raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

    # 获取 Tensor Parallel 进程组
    self.tp_group = get_tensor_model_parallel_group_if_none(
        self.tp_group, is_expert=self.is_expert
    )

    world_size = get_pg_size(self.tp_group)
    rank = get_pg_rank(self.tp_group)
    self.explicit_expert_comm = self.is_expert and (world_size > 1 or self.expert_parallel)

    # 每个 partition 的输入大小
    self.input_size_per_partition = divide(input_size, world_size)

    # 权重参数分配
    if config.use_cpu_initialization:
        self.weight = Parameter(
            torch.empty(
                self.output_size, self.input_size_per_partition, dtype=config.params_dtype
            )
        )
        if config.perform_initialization:
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,  # partition_dim=1，沿第二维分割
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
                params_dtype=config.params_dtype,
                rank=rank,
                world_size=world_size,
            )
    else:
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            _initialize_affine_weight_gpu(
                self.weight,
                init_method,
                partition_dim=1,
                stride=stride,
                is_expert=self.is_expert,
            )
    setattr(self.weight, "allreduce", not (self.is_expert and self.expert_parallel))

    # 偏置参数分配（偏置不被分割）
    if bias:
        if config.use_cpu_initialization:
            self.bias = Parameter(torch.empty(self.output_size, dtype=config.params_dtype))
        else:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )

        if config.perform_initialization:
            with torch.no_grad():
                self.bias.zero_()
        setattr(self.bias, "allreduce", not (self.is_expert and self.expert_parallel))
        setattr(self.bias, "sequence_parallel", self.sequence_parallel)
    else:
        self.register_parameter("bias", None)

    # Hook: 为 state dict 添加默认的空 _extra_state
    self._register_load_state_dict_pre_hook(
        lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
            f"{prefix}_extra_state"
        )
    )
```

---

#### RowParallelLinear.forward

```python
def forward(self, input_):
    """
    前向传播

    Args:
        input_: 输入张量，形状 [sequence, batch, hidden]

    Returns:
        - output: 输出张量
        - bias: 偏置（如果 skip_bias_add=True）
    """
    # 输入处理
    if self.input_is_parallel:
        input_parallel = input_
    else:
        assert not self.sequence_parallel
        # 沿最后一维分割输入
        input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)

    # 矩阵乘法（不需要 All-Reduce 输入梯度）
    allreduce_dgrad = False

    # CPU Offloading 支持
    if self.config._cpu_offloading_context is not None:
        if self.config._cpu_offloading_context.inside_context is True:
            if not HAVE_TE:
                assert (
                    self.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while TE is not present"
            else:
                input_parallel.activation_offloading = self.config.cpu_offloading_activations

    # 执行矩阵乘法
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=self.weight,
        bias=None,  # 偏置稍后添加
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=False,
        tp_group=None,
        grad_output_buffer=None,
    )

    # All-Reduce 或 Reduce-Scatter 输出
    if self.explicit_expert_comm:
        assert self.skip_bias_add
        output_ = output_parallel
    elif self.sequence_parallel:
        # 序列并行：Reduce-Scatter
        output_ = reduce_scatter_to_sequence_parallel_region(
            output_parallel, group=self.tp_group
        )
    else:
        # 标准模式：All-Reduce
        output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)

    # 添加偏置
    if not self.skip_bias_add:
        output = (output_ + self.bias) if self.bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = self.bias

    return output, output_bias
```
**行并行示例**（TP=2）：
```
输入:
  Rank 0: X_0, 形状 [seq, batch, hidden/2]
  Rank 1: X_1, 形状 [seq, batch, hidden/2]
权重:
  Rank 0: A_0, 形状 [output, hidden/2]
  Rank 1: A_1, 形状 [output, hidden/2]
局部输出:
  Rank 0: Y_0 = X_0 * A_0^T, 形状 [seq, batch, output]
  Rank 1: Y_1 = X_1 * A_1^T, 形状 [seq, batch, output]
All-Reduce 后: Y = Y_0 + Y_1, 形状 [seq, batch, output]
```

---

## 总结

### 层类型对比

| 层类型 | 分割维度 | 输入处理 | 输出处理 | 应用场景 |
|--------|---------|---------|---------|---------|
| VocabParallelEmbedding | 词汇表维度 | Mask | All-Reduce | Embedding 层 |
| ColumnParallelLinear | 输出维度 | Copy | All-Gather | QKV 投影、第一层 MLP |
| RowParallelLinear | 输入维度 | Scatter | All-Reduce | 第二层 MLP、输出投影 |

### 通信模式

| 模式 | 前向 | 反向 | 使用场景 |
|------|------|------|---------|
| 标准 TP | Copy → Split | All-Reduce → All-Gather | 大多数情况 |
| 序列并行 | All-Gather → Scatter | Reduce-Scatter → All-Gather | 长序列训练 |

### 性能优化

1. **梯度累积融合**：直接累积到主梯度，避免额外加法
2. **异步通信**：通信与计算重叠
3. **全局内存缓冲区**：减少内存分配
4. **权重梯度延迟**：延迟计算 Embedding 权重梯度

### 关键参数

- `sequence_parallel`: 启用序列并行
- `gradient_accumulation_fusion`: 启用梯度累积融合
- `skip_bias_add`: 跳过偏置添加（用于融合）
- `gather_output`: 是否 All-Gather 输出
- `input_is_parallel`: 输入是否已分割

### 内存优化

1. **frozen weight**: 不保存输入激活
2. **main_grad**: 直接累积到主梯度缓冲区
3. **dummy wgrad**: 使用虚拟权重梯度避免 hooks
4. **global buffer**: 复用通信缓冲区
