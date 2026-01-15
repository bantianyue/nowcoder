# Megatron-LM 张量并行通信映射模块注释详解

> 文件：`megatron/core/tensor_parallel/mappings.py`
> 功能：实现张量并行的通信原语和自动微分函数

---

## 模块导入与配置

```python
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
```
**说明**：NVIDIA 版权声明，Megatron-LM 是 NVIDIA 的开源项目

```python
import torch
```
**说明**：导入 PyTorch 核心库，用于张量计算和分布式通信

```python
from megatron.core.parallel_state import get_global_memory_buffer
```
**说明**：导入全局内存缓冲区，用于在通信过程中复用内存，减少分配开销

```python
from megatron.core.utils import get_tensor_model_parallel_group_if_none, is_torch_min_version
```
**说明**：
- `get_tensor_model_parallel_group_if_none`: 获取张量并行进程组，如果传入的 group 为 None 则使用默认组
- `is_torch_min_version`: 检查 PyTorch 版本是否满足最低要求

```python
from .utils import split_tensor_along_last_dim
```
**说明**：导入工具函数，沿最后一维分割张量

---

## 分布式通信函数选择

```python
try:
    if is_torch_min_version("1.13.0"):
        # PyTorch 1.13.0+ 使用新的 API
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        # PyTorch 1.13.0 之前使用旧 API（带下划线的私有函数）
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
except:
    # 如果版本检测失败，默认使用旧 API
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base
```
**说明**：根据 PyTorch 版本选择合适的分布式通信 API
- **all_gather**: 将所有 rank 的数据收集到每个 rank
- **reduce_scatter**: 先进行归约操作，然后将结果分散到各个 rank

---

## 核心通信原语

### 1. All-Reduce 操作

```python
def _reduce(input_, group):
    """
    在张量并行组内对输入张量执行 All-Reduce 操作

    All-Reduce: 将所有 rank 的张量进行求和（或其他归约操作），
                最终每个 rank 都得到相同的归约结果

    Args:
        input_: 输入张量
        group: 进程组（张量并行组）

    Returns:
        归约后的张量
    """
    assert group is not None, "group should not be None"

    # 如果只有一个 GPU，跳过通信操作
    if group.size() == 1:
        return input_

    # 执行 All-Reduce 操作（默认使用 SUM 作为归约操作）
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_
```
**关键点**：
- All-Reduce 是行并行（Row Parallel）前向传播后的关键操作
- `contiguous()` 确保张量在内存中是连续存储的，避免通信错误

---

### 2. 沿最后一维分割

```python
def _split_along_last_dim(input_, group):
    """
    沿张量的最后一维进行分割，保留当前 rank 对应的分片

    这是列并行（Column Parallel）前向传播的关键操作

    Args:
        input_: 输入张量，形状 [..., hidden_dim]
        group: 进程组

    Returns:
        分割后的张量，形状 [..., hidden_dim/tp_size]
    """
    assert group is not None, "group should not be None"

    world_size = group.size()  # 张量并行大小

    # 单 GPU 情况下直接返回
    if world_size == 1:
        return input_

    # 沿最后一维分割张量
    input_list = split_tensor_along_last_dim(input_, world_size)

    # torch.split 不会自动创建连续张量，需要手动调用
    rank = group.rank()  # 获取当前 rank
    output = input_list[rank].contiguous()  # 取出当前 rank 对应的分片

    return output
```
**应用场景**：
- 列并行 Linear 层：将权重矩阵沿列（输出维度）分割
- 每个 rank 计算输出的一部分

---

### 3. 沿第一维分割

```python
def _split_along_first_dim(input_, group):
    """
    沿张量的第一维（通常是序列/批次维度）进行分割

    这是序列并行（Sequence Parallel）的关键操作

    Args:
        input_: 输入张量，形状 [seq_len, ...] 或 [batch, ...]
        group: 进程组

    Returns:
        分割后的张量，形状 [seq_len/tp_size, ...]
    """
    assert group is not None, "group should not be None"

    world_size = group.size()

    # 单 GPU 情况
    if world_size == 1:
        return input_

    # 检查第一维是否能被 world_size 整除
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    # 计算本地维度大小和偏移量
    local_dim_size = dim_size // world_size
    rank = group.rank()
    dim_offset = rank * local_dim_size

    # 使用切片操作获取对应分片（比 torch.split 更高效）
    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output
```
**应用场景**：
- 序列并行：将长序列分割到多个 GPU
- Ring Attention：实现环形注意力机制

---

### 4. 沿最后一维收集

```python
def _gather_along_last_dim(input_, group):
    """
    沿最后一维收集所有 rank 的张量并拼接

    Args:
        input_: 本地输入张量
        group: 进程组

    Returns:
        收集并拼接后的完整张量
    """
    world_size = group.size()

    # 单 GPU 情况
    if world_size == 1:
        return input_

    # 准备输出张量（注意：先按第一维扩展，后面会重新排列）
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

    # 执行 All-Gather 操作（沿第一维收集）
    dist_all_gather_func(output, input_.contiguous(), group=group)

    # 将收集的数据重新分块，然后沿最后一维拼接
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-1).contiguous()

    return output
```
**技术细节**：
- 由于 PyTorch 的 `all_gather_into_tensor` 只支持沿第一维收集
- 这里使用了技巧：先沿第一维收集，然后通过 chunk + cat 重新排列到最后一维

---

### 5. 沿最后一维 Reduce-Scatter

```python
def _reduce_scatter_along_last_dim(input_, group):
    """
    沿最后一维进行 Reduce-Scatter 操作

    Reduce-Scatter: 先求和，然后将结果分散到各个 rank

    Args:
        input_: 输入张量
        group: 进程组

    Returns:
        Reduce-Scatter 后的张量
    """
    world_size = group.size()
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size  # 最后一维缩小

    # 重塑为 2D 张量
    input_ = input_.reshape(-1, input_.shape[-1])

    # 沿最后一维（列）分割
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )

    # 沿第一维拼接（转置效果）
    concat_tensor = torch.cat(split_tensors, dim=0)

    # 对第一维执行 reduce-scatter（利用现有的第一维操作）
    output = _reduce_scatter_along_first_dim(concat_tensor, group=group).reshape(target_shape)
    return output
```
**技术技巧**：
- 通过 reshape + cat 将最后一维的 reduce-scatter 转换为第一维的操作
- 避免了直接实现沿最后一维 reduce-scatter 的复杂性

---

### 6. 沿第一维收集

```python
def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """
    沿第一维收集张量并拼接

    Args:
        input_: 要收集的张量
        group: 进程组
        output_split_sizes: 可选，指定每个 rank 输出的分割大小（用于不均匀分割）
        use_global_buffer: 是否使用全局内存缓冲区（性能优化）

    Returns:
        收集后的张量
    """
    assert group is not None, "group should not be None"
    world_size = group.size()

    # 单 GPU 情况
    if world_size == 1:
        return input_

    dim_size = list(input_.size())

    if output_split_sizes is None:
        # 均匀分割情况
        dim_size[0] = dim_size[0] * world_size

        # 选择输出缓冲区
        if use_global_buffer:
            # 使用全局缓冲区，避免重复分配
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            # 创建新的输出张量
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

        # 执行 All-Gather
        dist_all_gather_func(output, input_.contiguous(), group=group)
    else:
        # 不均匀分割情况（用于动态批处理等场景）
        dim_size[0] = sum(output_split_sizes)

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

        # 分割输出张量
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))

        # 使用旧版 API 支持不均匀分割
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output
```
**关键特性**：
- 支持不均匀分割（`output_split_sizes`）
- 全局内存缓冲区优化：复用预分配的内存，减少开销

---

### 7. 沿第一维 Reduce-Scatter

```python
def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
    """
    沿第一维进行 Reduce-Scatter 操作

    Args:
        input_: 输入张量
        group: 进程组
        input_split_sizes: 可选，指定每个 rank 输入的分割大小
        use_global_buffer: 是否使用全局内存缓冲区

    Returns:
        Reduce-Scatter 后的张量
    """
    assert group is not None, "group should not be None"
    world_size = group.size()

    # 单 GPU 情况
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        # 均匀分割情况
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())

        # 执行 Reduce-Scatter
        dist_reduce_scatter_func(output, input_.contiguous(), group=group)
    else:
        # 不均匀分割情况
        rank = group.rank()
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(
                input_tensor_list[rank].shape, input_.dtype, "mpu"
            )
        else:
            output = torch.empty_like(input_tensor_list[rank])

        # 使用旧版 API
        torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
    return output
```

---

## 自动微分函数类

这些类继承自 `torch.autograd.Function`，实现自定义的前向和反向传播逻辑，支持通信操作的自动微分。

### 8. Copy To Model Parallel Region

```python
class _CopyToModelParallelRegion(torch.autograd.Function):
    """
    将输入复制到模型并行区域

    前向：直接传递输入（无操作）
    反向：对梯度执行 All-Reduce

    应用场景：在进入张量并行区域之前使用，确保梯度在反向传播时被同步
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """用于 ONNX 导出的符号函数"""
        return input_

    @staticmethod
    def forward(ctx, input_, group):
        """
        前向传播：直接返回输入

        Args:
            ctx: 上下文对象，用于保存反向传播所需的信息
            input_: 输入张量
            group: 进程组

        Returns:
            输入张量（无变化）
        """
        ctx.group = group  # 保存进程组供反向使用
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：对梯度执行 All-Reduce

        由于前向没有操作（输入直接输出），所有 rank 的梯度需要累加

        Args:
            ctx: 上下文对象
            grad_output: 上游梯度

        Returns:
            All-Reduce 后的梯度
        """
        return _reduce(grad_output, ctx.group), None  # None 表示对 group 参数没有梯度
```
**使用场景**：
- 在行并行（Row Parallel）之前使用
- 确保输入的梯度在所有 rank 上一致

---

### 9. Reduce From Model Parallel Region

```python
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """
    从模型并行区域 All-Reduce 输出

    前向：执行 All-Reduce
    反向：直接传递梯度（无操作）

    与 _CopyToModelParallelRegion 相反的操作
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """符号函数"""
        return _reduce(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """
        前向传播：执行 All-Reduce

        Args:
            input_: 张量并行区域的输出
            group: 进程组

        Returns:
            All-Reduce 后的张量
        """
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：直接传递梯度

        由于前向已经 All-Reduce，反向不需要额外操作

        Args:
            grad_output: 上游梯度

        Returns:
            梯度（无变化）
        """
        return grad_output, None
```
**使用场景**：
- 在行并行 Linear 层之后使用
- 将各 rank 计算的部分结果求和

---

### 10. Scatter To Model Parallel Region

```python
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """
    将输入沿最后一维分割并分发到各个 rank

    前向：沿最后一维分割
    反向：沿最后一维收集（All-Gather）

    应用场景：列并行 Linear 层的输入处理
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """符号函数"""
        return _split_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """
        前向传播：沿最后一维分割

        Args:
            input_: 输入张量
            group: 进程组

        Returns:
            当前 rank 对应的分片
        """
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：沿最后一维收集

        由于前向分割了输入，反向需要收集所有 rank 的梯度

        Args:
            grad_output: 当前 rank 的梯度分片

        Returns:
            完整的梯度（All-Gather 结果）
        """
        return _gather_along_last_dim(grad_output, ctx.group), None
```
**使用场景**：
- 列并行 Linear 的输入
- 每个 rank 只处理输入的一部分特征

---

### 11. Gather From Model Parallel Region

```python
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """
    从模型并行区域收集张量并沿最后一维拼接

    前向：沿最后一维收集（All-Gather）
    反向：沿最后一维分割

    与 _ScatterToModelParallelRegion 相反的操作
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """符号函数"""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """
        前向传播：沿最后一维收集

        Args:
            input_: 当前 rank 的张量分片
            group: 进程组

        Returns:
            收集并拼接后的完整张量
        """
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：沿最后一维分割

        由于前向收集了张量，反向需要分割梯度

        Args:
            grad_output: 完整的梯度

        Returns:
            当前 rank 对应的梯度分片
        """
        return _split_along_last_dim(grad_output, ctx.group), None
```
**使用场景**：
- 列并行 Linear 的输出
- 需要收集各 rank 计算的部分结果

---

### 12. Scatter To Sequence Parallel Region

```python
class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    将输入沿第一维分割并分发到各个 rank（序列并行）

    前向：沿第一维分割
    反向：沿第一维收集

    应用场景：序列并行的输入处理
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """符号函数"""
        return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """
        前向传播：沿第一维分割

        Args:
            input_: 输入张量，形状 [seq_len, batch, hidden]
            group: 进程组

        Returns:
            分割后的张量，形状 [seq_len/tp_size, batch, hidden]
        """
        ctx.group = group
        return _split_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：沿第一维收集

        Args:
            grad_output: 当前 rank 的梯度分片

        Returns:
            完整的梯度
        """
        return _gather_along_first_dim(grad_output, ctx.group), None
```
**序列并行 vs 张量并行**：
- 张量并行：沿特征维度（最后一维）分割
- 序列并行：沿序列维度（第一维）分割

---

### 13. Gather From Sequence Parallel Region

```python
class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    从序列并行区域收集张量并沿第一维拼接

    前向：沿第一维收集
    反向：根据后续操作类型决定（Reduce-Scatter 或 Split）

    Args:
        tensor_parallel_output_grad: 如果为 True，后续计算在张量并行模式下，需要 Reduce-Scatter
                                    如果为 False，后续计算是数据并行的，只需要 Split
    """

    @staticmethod
    def symbolic(
        graph,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """符号函数"""
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def forward(
        ctx,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """
        前向传播：沿第一维收集

        Args:
            input_: 序列并行的张量分片
            group: 进程组
            tensor_parallel_output_grad: 控制反向传播行为
            output_split_sizes: 不均匀分割的大小
            use_global_buffer: 是否使用全局缓冲区

        Returns:
            收集后的完整张量
        """
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：根据后续操作类型选择

        如果后续是张量并行（需要各 rank 协作）：
            反向需要 Reduce-Scatter（梯度求和并分散）

        如果后续是数据并行（各 rank 独立）：
            反向只需要 Split（梯度分割）
        """
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        if tensor_parallel_output_grad:
            # 后续计算在张量并行模式，需要 Reduce-Scatter
            return (
                _reduce_scatter_along_first_dim(
                    grad_output, ctx.group, ctx.output_split_sizes, ctx.use_global_buffer
                ),
                None,
                None,
                None,
                None,
            )
        else:
            # 后续计算是数据并行的，只需要 Split
            assert ctx.output_split_sizes is None
            return (_split_along_first_dim(grad_output, ctx.group), None, None, None, None)
```
**关键设计**：
- `tensor_parallel_output_grad` 参数控制反向传播行为
- 这是序列并行和张量并行混合使用时的关键

---

### 14. Reduce-Scatter To Sequence Parallel Region

```python
class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    对序列并行区域的输入执行 Reduce-Scatter

    前向：Reduce-Scatter
    反向：All-Gather

    与 _GatherFromSequenceParallelRegion 相反的操作（当 tensor_parallel_output_grad=True）
    """

    @staticmethod
    def symbolic(graph, input_, group, input_split_sizes=None, use_global_buffer=False):
        """符号函数"""
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, group, input_split_sizes=None, use_global_buffer=False):
        """
        前向传播：Reduce-Scatter

        Args:
            input_: 输入张量
            group: 进程组
            input_split_sizes: 不均匀分割的大小
            use_global_buffer: 是否使用全局缓冲区

        Returns:
            Reduce-Scatter 后的张量
        """
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：All-Gather

        由于前向是 Reduce-Scatter，反向需要 All-Gather
        """
        input_split_sizes = ctx.input_split_sizes
        use_global_buffer = ctx.use_global_buffer
        return (
            _gather_along_first_dim(grad_output, ctx.group, input_split_sizes, use_global_buffer),
            None,
            None,
            None,
        )
```
**应用场景**：
- 在序列并行中，需要在某些点进行 Reduce-Scatter
- 例如：Attention 之后、LayerNorm 之前

---

### 15. All-Gather From Tensor Parallel Region

```python
class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """
    从张量并行区域收集张量（沿最后一维）

    前向：All-Gather
    反向：Reduce-Scatter

    类似于 _GatherFromModelParallelRegion，但反向使用 Reduce-Scatter
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """符号函数"""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """前向传播：All-Gather"""
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：Reduce-Scatter"""
        return _reduce_scatter_along_last_dim(grad_output, ctx.group), None
```
**使用区别**：
- `_GatherFromModelParallelRegion`: 反向用 Split
- `_AllGatherFromTensorParallelRegion`: 反向用 Reduce-Scatter

---

### 16. Reduce-Scatter To Tensor Parallel Region

```python
class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """
    对张量并行区域的输入执行 Reduce-Scatter（沿最后一维）

    前向：Reduce-Scatter
    反向：All-Gather

    与 _AllGatherFromTensorParallelRegion 相反的操作
    """

    @staticmethod
    def symbolic(graph, input_, group):
        """符号函数"""
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """前向传播：Reduce-Scatter"""
        ctx.group = group
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：All-Gather"""
        return _gather_along_last_dim(grad_output, ctx.group), None
```

---

### 17. All-To-All 操作

```python
class _AllToAll(torch.autograd.Function):
    """
    All-To-All 通信原语

    每个 rank 向所有其他 rank 发送数据，同时从所有其他 rank 接收数据

    应用场景：
    - 序列并行与张量并行的转换
    - Flash Attention 的实现
    """

    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """
        前向传播：All-To-All

        Args:
            group: 进程组
            input: 输入张量
            output_split_sizes: 每个 rank 接收的数据大小（None 表示均匀）
            input_split_sizes: 每个 rank 发送的数据大小（None 表示均匀）

        Returns:
            All-To-All 后的张量
        """
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()

        # 单 GPU 情况
        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            # 均匀分割
            output = torch.empty_like(input)
        else:
            # 不均匀分割
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )

        # 执行 All-To-All
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """
        反向传播：All-To-All（交换 input_split_sizes 和 output_split_sizes）

        反向时，发送和接收的大小互换
        """
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )
```
**All-To-All 的作用**：
- 实现数据在不同维度间的重新分配
- 例如：从 [num_tokens/TP, H] 转换为 [num_tokens, H/TP]

---

## 高级通信函数

### 18. 包装函数：张量并行区域操作

```python
def copy_to_tensor_model_parallel_region(input_, group=None):
    """
    包装函数：复制到张量并行区域

    前向：无操作
    反向：All-Reduce
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_, group=None):
    """
    包装函数：从张量并行区域 All-Reduce

    前向：All-Reduce
    反向：无操作
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region(input_, group=None):
    """
    包装函数：沿最后一维分割（张量并行）

    前向：Split（最后一维）
    反向：All-Gather
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region(input_, group=None):
    """
    包装函数：沿最后一维收集（张量并行）

    前向：All-Gather
    反向：Split
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromModelParallelRegion.apply(input_, group)
```

---

### 19. 包装函数：序列并行区域操作

```python
def scatter_to_sequence_parallel_region(input_, group=None):
    """
    包装函数：沿第一维分割（序列并行）

    前向：Split（第一维）
    反向：All-Gather
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToSequenceParallelRegion.apply(input_, group)


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_global_buffer=False,
):
    """
    包装函数：沿第一维收集（序列并行）

    前向：All-Gather
    反向：Reduce-Scatter 或 Split（取决于 tensor_parallel_output_grad）
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallel_output_grad, output_split_sizes, use_global_buffer
    )


def reduce_scatter_to_sequence_parallel_region(
    input_, group=None, input_split_sizes=None, use_global_buffer=False
):
    """
    包装函数：沿第一维 Reduce-Scatter（序列并行）

    前向：Reduce-Scatter
    反向：All-Gather
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, group, input_split_sizes, use_global_buffer
    )
```

---

### 20. All-Gather / Reduce-Scatter（最后一维）

```python
def all_gather_last_dim_from_tensor_parallel_region(input_, group=None):
    """
    包装函数：沿最后一维 All-Gather

    前向：All-Gather
    反向：Reduce-Scatter
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _AllGatherFromTensorParallelRegion.apply(input_, group)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_, group=None):
    """
    包装函数：沿最后一维 Reduce-Scatter

    前向：Reduce-Scatter
    反向：All-Gather
    """
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToTensorParallelRegion.apply(input_, group)
```

---

### 21. All-To-All 包装函数

```python
def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None):
    """
    All-To-All 包装函数
    """
    assert group is not None, "group should not be None"
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes)
```

---

### 22. SP 转 HP（序列并行到隐藏并行）

```python
def all_to_all_sp2hp(input_, group=None):
    """
    将输入从序列并行分布转换为隐藏维度并行分布

    输入形状：[num_tokens/TP, H]
    输出形状：[num_tokens, H/TP]

    应用场景：
    - 从序列并行切换到张量并行
    - Flash Attention 实现

    Args:
        input_: 已沿序列维度分割的张量
        group: 进程组

    Returns:
        沿隐藏维度分割的张量
    """
    group = get_tensor_model_parallel_group_if_none(group)

    world_size = group.size()

    # 重塑为 2D
    input_ = input_.reshape(-1, input_.shape[-1])

    # 沿隐藏维度（列）分割
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )

    # 沿序列维度（行）拼接
    concat_tensor = torch.cat(split_tensors, dim=0)

    # 执行 All-To-All
    output = all_to_all(group, concat_tensor)
    return output
```
**数据转换示例**（TP=2）：
```
输入（SP）:
Rank 0: [seq/2, H]  (序列的前半部分)
Rank 1: [seq/2, H]  (序列的后半部分)

输出（HP）:
Rank 0: [seq, H/2]  (隐藏维度的前半部分)
Rank 1: [seq, H/2]  (隐藏维度的后半部分)
```

---

### 23. HP 转 SP（隐藏并行到序列并行）

```python
def all_to_all_hp2sp(input_, group=None):
    """
    将输入从隐藏维度并行分布转换为序列并行分布

    输入形状：[num_tokens, H/TP]
    输出形状：[num_tokens/TP, H]

    与 all_to_all_sp2hp 相反的操作

    Args:
        input_: 已沿隐藏维度分割的张量
        group: 进程组

    Returns:
        沿序列维度分割的张量
    """
    group = get_tensor_model_parallel_group_if_none(group)

    world_size = group.size()

    # 重塑为 2D
    input_ = input_.reshape(-1, input_.shape[-1])

    # 执行 All-To-All
    input_exchanged = all_to_all(group, input_)

    # 重塑并沿序列维度分割
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
    )

    # 沿隐藏维度拼接
    output = torch.cat(split_tensors, dim=-1)
    return output
```
**数据转换示例**（TP=2）：
```
输入（HP）:
Rank 0: [seq, H/2]  (隐藏维度的前半部分)
Rank 1: [seq, H/2]  (隐藏维度的后半部分)

输出（SP）:
Rank 0: [seq/2, H]  (序列的前半部分)
Rank 1: [seq/2, H]  (序列的后半部分)
```

---

## 总结

### 通信模式对比

| 函数 | 前向操作 | 反向操作 | 应用场景 |
|------|---------|---------|---------|
| `copy_to_tensor_model_parallel_region` | 无操作 | All-Reduce | 进入张量并行区域前 |
| `reduce_from_tensor_model_parallel_region` | All-Reduce | 无操作 | 行并行 Linear 后 |
| `scatter_to_tensor_model_parallel_region` | Split（最后一维） | All-Gather | 列并行 Linear 输入 |
| `gather_from_tensor_model_parallel_region` | All-Gather | Split | 列并行 Linear 输出 |
| `scatter_to_sequence_parallel_region` | Split（第一维） | All-Gather | 序列并行输入 |
| `gather_from_sequence_parallel_region` | All-Gather | Reduce-Scatter/Split | 序列并行输出 |
| `reduce_scatter_to_sequence_parallel_region` | Reduce-Scatter | All-Gather | 序列并行中间点 |
| `all_gather_last_dim_from_tensor_parallel_region` | All-Gather | Reduce-Scatter | 特定场景收集 |
| `reduce_scatter_last_dim_to_tensor_parallel_region` | Reduce-Scatter | All-Gather | 特定场景分散 |
| `all_to_all_sp2hp` | All-To-All | All-To-All | SP→HP 转换 |
| `all_to_all_hp2sp` | All-To-All | All-To-All | HP→SP 转换 |

### 设计原则

1. **对称性**：前向和反向操作互为逆操作
2. **梯度正确性**：确保梯度在各 rank 正确传播
3. **通信效率**：使用全局缓冲区复用内存
4. **灵活性**：支持均匀和不均匀分割

### 关键优化

1. **全局内存缓冲区**：减少频繁的内存分配/释放
2. **连续性保证**：`contiguous()` 确保通信正确性
3. **单 GPU 优化**：单 GPU 时跳过通信操作
4. **版本兼容**：支持多个 PyTorch 版本的 API
