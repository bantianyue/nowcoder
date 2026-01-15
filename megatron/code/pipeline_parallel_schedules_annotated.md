# Megatron-LM 流水线并行调度模块注释详解

> 文件：`megatron/core/pipeline_parallel/schedules.py`
> 功能：实现流水线并行的调度算法（1F1B、交错 1F1B）

---

## 模块导入与配置

```python
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
```
**说明**：NVIDIA 版权声明

```python
import contextlib
from functools import partial
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable
```
**说明**：
- `contextlib`: 上下文管理器
- `partial`: 函数偏应用（固定部分参数）
- `Variable`: PyTorch 自动微分变量

```python
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    nvtx_range_pop,
    nvtx_range_push,
)
```
**说明**：
- `P2PCommunicator`: 点对点通信管理器
- `is_pp_first_stage/last_stage`: 判断是否是 PP 的首/末阶段
- `is_vp_first_stage/last_stage`: 判断是否是虚拟 PP 的首/末阶段
- `ProcessGroupCollection`: 进程组集合
- `create_cudagraphs`: 创建 CUDA Graph
- `MoEAuxLossAutoScaler`: MoE 辅助损失缩放器

```python
from .combined_1f1b import (
    combined_1f1b_schedule_for_interleaved_pipelining,
    combined_1f1b_schedule_for_no_pipelining,
)
```
**说明**：导入组合 1F1B 调度（用于 MoE 专家并行通信重叠）

```python
# Types
Shape = Union[List[int], torch.Size]
```
**说明**：类型别名

---

## 核心调度函数

### get_forward_backward_func

```python
def get_forward_backward_func():
    """
    获取适合当前配置的前向-后向传播函数

    根据 parallel_state 中的配置（PP size、VP size）返回相应的函数

    Returns:
        函数，执行模型的所有前向和后向传播

    函数签名：
        forward_backward_func(
            forward_step_func (required): 用户定义的前向步骤函数
                - 参数: data_iterator, model
                - 返回: (output, loss_func)

            data_iterator (required): 数据迭代器
                - 交错 PP 时是迭代器列表

            model (required): 模型
                - 交错 PP 时是模型块列表

            num_microbatches (int, required): 微批次数量

            seq_length (int, required): 序列长度

            micro_batch_size (int, required): 微批次大小

            decoder_seq_length (int, optional): 解码器序列长度

            forward_only (bool, default=False): 是否只执行前向

            collect_non_loss_data (bool, default=False): 是否收集非损失数据

            first_val_step (bool, optional): 是否是验证的第一步

            adjust_tensor_shapes_fn (Callable, optional): 调整张量形状的函数
        )
    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    if pipeline_model_parallel_size > 1:
        # 流水线并行
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            # 交错流水线并行
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            # 标准流水线并行
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        # 无流水线并行
        forward_backward_func = forward_backward_no_pipelining

    return forward_backward_func
```
**调度选择逻辑**：
```
if PP size > 1:
    if VP size > 1:
        -> 交错 1F1B
    else:
        -> 标准 1F1B
else:
    -> 无流水线（标准训练）
```

---

## 内存优化函数

### deallocate_output_tensor

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """
    伪释放输出张量的 '.data' 字段（内存优化）

    在输出张量发送到下一流水线阶段后立即调用
    此时张量只需要 '.grad_fn' 字段用于反向传播

    Args:
        out: 输出张量
        deallocate_pipeline_outputs: 是否启用释放

    原理：
        将张量的 data 字段替换为一个 1 元素的张量
        保留 grad_fn 用于反向传播
    """
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."

    # 将 .data 字段替换为 1 元素张量
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```
**内存节省**：
- 激活值在发送后立即释放
- 仅保留计算图（grad_fn）
- 显著降低峰值内存使用

---

### custom_backward

```python
def custom_backward(output, grad_output):
    """
    直接调用 C++ 自动微分引擎

    配合 deallocate_output_tensor 使用
    PyTorch 的 backward() 检查输出和梯度形状相同
    C++ 的 backward() 不检查（允许伪释放的张量）

    Args:
        output: 输出张量（已被伪释放）
        grad_output: 输出梯度
    """
    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # 处理标量输出
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # 调用 C++ 引擎 [见 torch/csrc/autograd/python_engine.cpp]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )
```
**为什么需要 custom_backward**：
- `deallocate_output_tensor` 将输出张量缩到 1 元素
- PyTorch 的 `torch.autograd.backward` 检查形状匹配
- 直接调用 C++ 引擎跳过形状检查

---

## 前向传播步骤

### forward_step

```python
def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
):
    """
    前向传播步骤

    Args:
        forward_step_func (callable): 用户定义的前向函数
            参数: data_iterator, model
            返回: (output_object, loss_func)

            loss_func 的返回类型：
                a. (loss, reduced_loss_dict) - 损失除以 num_microbatches
                b. (loss, num_tokens, reduced_loss_dict) - 按词数平均
                c. (non_loss_data) - 收集非损失数据（需要 collect_non_loss_data=True）

        data_iterator: 数据迭代器
        model: 模型（或模型块）
        num_microbatches: 微批次数量
        input_tensor: 输入张量（首阶段从 data_iterator 获取）
        forward_data_store: 存储前向数据
        config: 配置对象
        cp_group_size: 上下文并行组大小
        collect_non_loss_data: 是否收集非损失数据
        checkpoint_activations_microbatch: checkpoint 的微批次 ID
        is_first_microbatch: 是否是第一个微批次
        current_microbatch: 当前微批次 ID
        vp_stage: 虚拟流水线阶段 ID
        is_last_stage: 是否是最后阶段

    Returns:
        output_tensor: 输出张量（或标量损失）
        num_tokens: token 数量
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    # 设置第一个微批次标志
    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()

    # 设置当前微批次 ID
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    # 包装输入张量为列表
    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    # 设置输入张量
    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    # 自动混合精度
    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    # 计算损失
    output_tensor, num_tokens = forward_step_calc_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_last_stage,
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens
```
**前向传播流程**：
1. 设置微批次标志
2. 设置输入张量
3. 执行前向计算（可能用 autocast）
4. 计算损失
5. 设置 MoE/MTP 损失缩放

---

### forward_step_calc_loss

```python
def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
    cp_group_size=None,
    is_last_stage=None,
):
    """
    计算损失和 token 数量

    Args:
        model: 模型
        output_tensor: 前向输出
        loss_func: 损失函数
        config: 配置
        vp_stage: 虚拟流水线阶段
        collect_non_loss_data: 是否收集非损失数据
        num_microbatches: 微批次数量
        forward_data_store: 存储前向数据
        cp_group_size: 上下文并行组大小
        is_last_stage: 是否是最后阶段

    Returns:
        output_tensor: 输出张量（可能是标量损失）
        num_tokens: token 数量
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    # 验证 vp_stage
    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
            vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"

    # 获取参数
    if cp_group_size is None and is_last_stage is None:
        cp_group_size = parallel_state.get_context_parallel_world_size()
        is_last_stage = parallel_state.is_pipeline_last_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
    else:
        assert (
            cp_group_size is not None and is_last_stage is not None
        ), "cp_group_size and is_last_stage must be provided"

    num_tokens = torch.tensor(0, dtype=torch.int)

    # 只在最后阶段计算损失
    if is_last_stage:
        if loss_func is None:
            # 只收集输出
            forward_data_store.append(output_tensor)
        elif not collect_non_loss_data:
            # 计算损失
            outputs = loss_func(output_tensor)

            if len(outputs) == 3:
                # (loss, num_tokens, loss_reduced) - 按词数平均
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    # 防止除以零（所有 token 被掩码）
                    output_tensor /= torch.clamp(num_tokens, min=1)
                    output_tensor /= num_microbatches
            else:
                # (loss, loss_reduced) - 传统平均
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            # 收集非损失数据（用于推理）
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # 设置 MoE 辅助损失缩放
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale * cp_group_size / num_microbatches)

    # 设置 MTP 损失缩放
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens
```

---

## 反向传播步骤

### backward_step

```python
def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """
    反向传播步骤

    Args:
        input_tensor: 输入张量（首阶段为 None）
        output_tensor: 输出张量
        output_tensor_grad: 输出梯度（末阶段为 None）
        model_type: 模型类型
        config: 配置

    Returns:
        input_tensor_grad: 输入梯度（首阶段为 None）

    注意：
        当前代码最多支持一个跳过连接
        需要修改以支持任意数量的跳过连接
    """
    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # 保留输入张量的梯度
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    # 包装输出张量和梯度
    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # 反向传播
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # 多模态模型：某些批次可能没有图像
    # 视觉编码器可能不参与计算
    # 这种情况下跳过反向传播，保持零梯度
    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            # 使用 custom_backward 优化内存
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            # 标准 PyTorch 反向传播
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # 收集输入张量的梯度
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad
```

---

## 无流水线并行

### forward_backward_no_pipelining

```python
def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """
    无流水线并行的训练

    标准的微批次训练：依次处理每个微批次
    最后一个微批次执行梯度同步
    """
    # 设置进程组
    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

    # 验证参数
    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for non-pipeline-parallel schedule"

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # 获取 no_sync 函数（延迟梯度同步）
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)
    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    # MoE 专家并行通信重叠
    if config.overlap_moe_expert_parallel_comm and not forward_only:
        forward_data_store, total_num_tokens = combined_1f1b_schedule_for_no_pipelining(...)
    else:
        # 标准训练循环
        with no_sync_func():
            # 前 N-1 个微批次（延迟梯度同步）
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    pg_collection.cp.size(),
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                )
                total_num_tokens += num_tokens
                if not forward_only:
                    backward_step(
                        input_tensor, output_tensor, output_tensor_grad, model_type, config
                    )

        # 最后一个微批次（执行梯度同步）
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            pg_collection.cp.size(),
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, num_microbatches == 1
            ),
            current_microbatch=num_microbatches - 1,
        )
        total_num_tokens += num_tokens

        if not forward_only:
            backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # 完成梯度同步
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    # CUDA Graph 捕获
    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and CudaGraphScope.full_iteration not in config.cuda_graph_scope
    ):
        create_cudagraphs()

    return forward_data_store
```
**训练流程**：
```
for microbatch 0 to N-2:
    forward(microbatch)
    backward(microbatch)
    # 梯度累积，不同步

forward(microbatch N-1)
backward(microbatch N-1)
# 梯度同步 + 全局归约
```

---

## 流水线并行辅助函数

### get_pp_rank_microbatches

```python
def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
):
    """
    计算 PP 调度的微批次数量

    Args:
        num_microbatches: 每个 PP 阶段的微批次数
        num_model_chunks: 模型块数（虚拟流水线大小）
        microbatch_group_size_per_vp_stage: 每个 VP 阶段的微批次组大小
        forward_only: 是否只执行前向
        overlap_moe_expert_parallel_comm: 是否重叠 MoE 通信
        p2p_communicator: P2P 通信器

    Returns:
        total_num_microbatches: 总微批次数
        are_all_microbatches_in_warmup: 是否所有微批次都在预热阶段
        num_warmup_microbatches: 预热微批次数
        num_microbatches_remaining: 剩余微批次数
    """
    # 获取参数
    if p2p_communicator is not None:
        pipeline_parallel_size = p2p_communicator.pp_group.size()
        pipeline_parallel_rank = p2p_communicator.pp_group.rank()
        virtual_pipeline_parallel_size = p2p_communicator.virtual_pipeline_model_parallel_size
    else:
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        virtual_pipeline_parallel_size = (
            parallel_state.get_virtual_pipeline_model_parallel_world_size()
        )

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        # 只前向：所有批次都是预热
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # 标准 1F1B
            # 预热数量 = PP size - rank - 1
            # Rank 0: PP-1, Rank 1: PP-2, ..., Rank PP-1: 0
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # 交错 1F1B
            # 预热数量 = (PP size - rank - 1) * 2 + (num_model_chunks - 1) * group_size
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage

            # MoE 专家并行通信重叠
            if overlap_moe_expert_parallel_comm:
                num_warmup_microbatches = num_warmup_microbatches + 1
    else:
        # 无流水线（仅用于 CUDA Graph）
        num_warmup_microbatches = 0

    # 边界检查
    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )
```
**预热数量计算示例**（PP=4, VP=2）：
```
Rank 0: warmup = (4-0-1) * 2 + (2-1) * group_size = 6 + group_size
Rank 1: warmup = (4-1-1) * 2 + (2-1) * group_size = 4 + group_size
Rank 2: warmup = (4-2-1) * 2 + (2-1) * group_size = 2 + group_size
Rank 3: warmup = (4-3-1) * 2 + (2-1) * group_size = 0 + group_size
```

---

### get_schedule_table

```python
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """
    构建 PP 调度查找表

    返回一个列表，每个元素是 (microbatch_id, model_chunk_id) 元组

    示例（PP=2, M=5, VP=2）：
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1

    Args:
        num_microbatches: 微批次数量
        num_model_chunks: 模型块数
        microbatch_group_size_per_vp_stage: 每个 VP 阶段的微批次组大小

    Returns:
        schedule_table: 调度表
    """
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # 最后一个微批次组（可能不完整）
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # 其他微批次组（完整）
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table
```

---

## 交错流水线并行（Interleaved 1F1B）

### forward_backward_pipelining_with_interleaving

```python
def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """
    交错 1F1B 调度

    模型被分成多个块（model chunks），每个块是一个虚拟流水线阶段

    约定：
    - num_microbatches: 每个 PP 阶段的微批次数
    - num_model_chunks: 虚拟流水线大小
    - total_num_microbatches = num_microbatches * num_model_chunks
    - microbatch_id: [0, num_microbatches)
    - model_chunk_id: [0, num_model_chunks)
    - virtual_microbatch_id: [0, total_num_microbatches)

    Returns:
        forward_data_store: 前向数据存储（最后阶段）
    """
    # 设置进程组...
    # （省略详细设置代码）

    # 计算张量形状
    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # 计算微批次数量
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # 构建调度表
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    # 辅助函数
    def get_model_chunk_id(virtual_microbatch_id, forward):
        """获取虚拟微批次对应的模型块 ID"""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """获取模型块内的微批次 ID"""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    # ... (更多辅助函数)

    # ==================== 主逻辑 ====================

    # 1. 预热阶段（只有前向）
    nvtx_range_push(suffix="warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)

        # 决定是否从上一阶段接收
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # 执行前向
        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # 发送到下一阶段
        if not _is_vp_last_stage(vp_stage=cur_model_chunk_id) or not is_pp_last_stage(pp_group):
            input_tensor = p2p_communicator.send_forward_recv_forward(
                output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
            )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="warmup")

    # 2. 稳态阶段（1F1B）
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        forward_k = k + num_warmup_microbatches
        backward_k = k

        # 前向 + 后向
        output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=forward_k,
            b_virtual_microbatch_id=backward_k,
            pre_forward=pp_pre_forward,
            pre_backward=pp_pre_backward,
            post_forward=pp_post_forward,
            post_backward=pp_post_backward,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # 通信...

    nvtx_range_pop(suffix="steady")

    # 3. 冷却阶段（只有后向）
    nvtx_range_push(suffix="cooldown")
    if not forward_only:
        for k in range(num_microbatches_remaining, total_num_microbatches):
            # 执行后向
            input_tensor_grad = backward_step_helper(k)

            # 发送梯度
            if not (_is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)):
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
    nvtx_range_pop(suffix="cooldown")

    # 完成梯度同步
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    return forward_data_store
```
**交错 1F1B 调度示意**（PP=2, VP=2, M=4）：
```
时间 →
Rank 0: F0(0) F0(1) F1(0) B0(0) F1(1) B0(1) B1(0) B1(1)
Rank 1: F0(1) F1(0) F1(1) B0(1) B0(0) B1(1) B1(0)

F = Forward, B = Backward
数字 = (model_chunk_id, microbatch_id)
```

---

## 标准流水线并行（Non-Interleaved 1F1B）

### forward_backward_pipelining_without_interleaving

```python
def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """
    标准 1F1B 调度（无交错）

    模型不分割成块，每个 rank 处理一个完整的流水线阶段

    Returns:
        forward_data_store: 前向数据存储（最后阶段）
    """
    # ... (设置代码)

    # 计算预热微批次数
    num_warmup_microbatches = (
        p2p_communicator.pp_group.size() - p2p_communicator.pp_group.rank() - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # 获取张量形状
    recv_tensor_shapes = get_tensor_shapes(...)
    send_tensor_shapes = get_tensor_shapes(...)

    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # 1. 预热阶段（只有前向）
    for i in range(num_warmup_microbatches):
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
        )
        output_tensor, num_tokens = forward_step(...)
        p2p_communicator.send_forward(output_tensor, is_pp_last_stage(p2p_communicator.pp_group))

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # 接收第一个稳态前向张量
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
        )

    # 2. 稳态阶段（1F1B）
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # 前向
        output_tensor, num_tokens = forward_step(...)

        if forward_only:
            p2p_communicator.send_forward(
                output_tensor, is_pp_last_stage(p2p_communicator.pp_group)
            )
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, is_pp_first_stage(p2p_communicator.pp_group)
                )
        else:
            # 前向发送，后向接收
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

            # 保存张量用于反向
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # 弹出最早的张量
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # 后向
            if num_warmup_microbatches == 0 and last_iteration:
                enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            # 后向发送，前向接收
            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group)
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad,
                    recv_tensor_shapes,
                    is_pp_first_stage(p2p_communicator.pp_group),
                )

    # 3. 冷却阶段（只有后向）
    if not forward_only:
        for i in range(num_warmup_microbatches):
            if i == num_warmup_microbatches - 1:
                enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, is_pp_last_stage(p2p_communicator.pp_group)
            )

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            p2p_communicator.send_backward(
                input_tensor_grad, is_pp_first_stage(p2p_communicator.pp_group)
            )

    # 完成梯度同步
    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
        )

    return forward_data_store
```
**标准 1F1B 调度示意**（PP=4, M=8）：
```
时间 →
Rank 0: F0 F1 F2 F3    F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
Rank 1:     F0 F1 F2 F3 F4 F5 F6 F7    B0 B1 B2 B3 B4 B5 B6 B7
Rank 2:          F0 F1 F2 F3 F4 F5 F6 F7    B0 B1 B2 B3 B4 B5 B6 B7
Rank 3:               F0 F1 F2 F3 F4 F5 F6 F7    B0 B1 B2 B3 B4 B5 B6 B7
```

---

## 总结

### 调度模式对比

| 模式 | 模型分割 | 预热长度 | 稳态模式 | 适用场景 |
|------|---------|---------|---------|---------|
| 无 PP | 否 | 0 | F-B | 小模型、单 GPU |
| 标准 1F1B | 否 | PP-rank-1 | 1F1B | 中等模型、中等 GPU 数量 |
| 交错 1F1B | 是 | (PP-rank-1)*2+(VP-1)*group | 1F1B | 大模型、大量 GPU |

### 关键优化

1. **内存优化**：
   - `deallocate_output_tensor`: 立即释放激活值
   - `custom_backward`: 跳过形状检查

2. **通信重叠**：
   - P2P 通信与计算重叠
   - 异步发送/接收

3. **调度优化**：
   - 预热阶段填充流水线
   - 稳态阶段保持高利用率
   - 冷却阶段清空流水线

### 通信模式

| 阶段 | 发送方向 | 接收方向 |
|------|---------|---------|
| 前向 | send_forward | recv_forward |
| 后向 | send_backward | recv_backward |
| 稳态 | send_forward_recv_backward | send_backward_recv_forward |

### 辅助函数

| 函数 | 功能 |
|------|------|
| `get_pp_rank_microbatches` | 计算微批次数量 |
| `get_schedule_table` | 构建调度表 |
| `get_tensor_shapes` | 计算张量形状 |
| `recv_tensor_from_previous_stage` | 判断是否接收 |
