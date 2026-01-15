# Megatron-LM GPT 模型训练全流程详解

> 文件：`pretrain_gpt.py`, `gpt_builders.py`, `training.py`, `gpt_model.py`
> 功能：GPT 模型从初始化到训练完成的完整流程

---

## 目录

1. [训练流程概览](#训练流程概览)
2. [初始化阶段](#初始化阶段)
3. [数据加载阶段](#数据加载阶段)
4. [模型构建阶段](#模型构建阶段)
5. [优化器配置阶段](#优化器配置阶段)
6. [训练循环阶段](#训练循环阶段)
7. [算子操作详解](#算子操作详解)
8. [性能优化技术](#性能优化技术)

---

## 训练流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Megatron-LM GPT 训练流程                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 初始化阶段 (initialize_megatron)                            │
│     └── 环境设置、参数解析、分布式初始化                          │
│                                                                  │
│  2. 数据加载阶段 (train_valid_test_datasets_provider)            │
│     └── 构建 dataset、dataloader                                  │
│                                                                  │
│  3. 模型构建阶段 (gpt_builder)                                   │
│     └── 构建 GPTModel、设置并行                                   │
│                                                                  │
│  4. 优化器配置阶段 (get_megatron_optimizer)                      │
│     └── 创建优化器、学习率调度器                                  │
│                                                                  │
│  5. 训练循环阶段 (train)                                         │
│     └── 迭代训练、验证、checkpoint                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 初始化阶段

### 代码入口

```python
# pretrain_gpt.py
if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,  # 数据集提供者
        partial(model_provider, gpt_builder), # 模型提供者
        ModelType.encoder_or_decoder,        # 模型类型
        forward_step,                        # 前向步骤函数
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
```

### initialize_megatron 流程

```python
# megatron/training/initialize.py
def initialize_megatron(...):
    """
    初始化 Megatron 环境

    核心步骤：
    1. 解析命令行参数和 YAML 配置
    2. 设置随机种子
    3. 初始化分布式进程组（TP、PP、DP、CP、EP）
    4. 设置 CUDA 设备
    5. 初始化内存缓冲区
    """

    # === 步骤 1: 参数解析 ===
    args = parse_args(...)  # 解析命令行参数
    if args.yaml_cfg is not None:
        args = merge_yaml_args(args, args.yaml_cfg)  # 合并 YAML 配置

    # === 步骤 2: 随机种子设置 ===
    set_random_seed(args.seed)

    # === 步骤 3: 分布式初始化 ===
    # 初始化进程组
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
    )

    # 初始化模型并行状态
    initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,      # TP
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,  # PP
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,  # VP
        pipeline_model_parallel_split_rank=args.pipeline_model_parallel_split_rank,
        context_parallel_size=args.context_parallel_size,                # CP
        expert_model_parallel_size=args.expert_model_parallel_size,      # EP
        distributed_timeout_minutes=args.distributed_timeout_minutes,
    )

    # === 步骤 4: CUDA 设置 ===
    torch.cuda.set_device(args.local_rank)

    # === 步骤 5: 内存缓冲区初始化 ===
    from megatron.core.parallel_state import get_global_memory_buffer
    global_memory_buffer = get_global_memory_buffer()
    # 预分配通信缓冲区，减少运行时分配开销
```

### 性能优化点

| 优化点 | 说明 |
|--------|------|
| **NCCL 环境变量** | 设置 NCCL 通信优化参数 |
| **进程组复用** | TP、PP、DP 共享通信域，减少开销 |
| **内存预分配** | 提前分配全局缓冲区，避免运行时分配 |

---

## 2. 数据加载阶段

### 数据集构建

```python
# pretrain_gpt.py
def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """
    构建训练、验证、测试数据集

    Args:
        train_val_test_num_samples: [train_samples, valid_samples, test_samples]
    """
    # === 步骤 1: 配置数据集参数 ===
    config = core_gpt_dataset_config_from_args(args)
    # 关键参数：
    # - sequence_length: 序列长度（如 2048、4096）
    # - blend: 数据集混合比例
    # - split: train/valid/test 分割
    # - tokenizer: 分词器

    # === 步骤 2: 选择数据集类型 ===
    if args.sft:
        dataset_type = SFTDataset          # 监督微调数据集
    elif args.mock_data:
        dataset_type = MockGPTDataset      # 模拟数据（测试用）
    elif args.fim_data:
        dataset_type = GPTFIMDataset       # FIM（Fill-In-Middle）数据集
    else:
        dataset_type = GPTDataset          # 标准 GPT 数据集

    # === 步骤 3: 构建数据集 ===
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage),  # 哪些 rank 构建数据集
        config
    ).build()

    return train_ds, valid_ds, test_ds
```

### 数据迭代器构建

```python
# megatron/training/datasets/data_samplers.py
def build_pretraining_data_loader(dataset, consumed_samples):
    """
    构建数据加载器

    核心功能：
    1. 实现 Random Round-Robin 数据采样
    2. 支持多数据集混合
    3. 支持序列打包（Sequence Packing）
    """

    # 创建分布式 sampler
    sampler = MegatronPretrainingSampler(
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=args.micro_batch_size,
        num_microbatches=get_num_microbatches(),
        data_parallel_rank=args.data_parallel_rank,
        data_parallel_size=args.data_parallel_size,
    )

    # 创建 dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_dataloader_workers,
        pin_memory=True,  # 优化：锁页内存，加速 CPU→GPU 传输
        persistent_workers=True,  # 优化：保持 worker 进程，减少重启开销
    )

    return data_loader
```

### get_batch 函数

```python
# pretrain_gpt.py
def get_batch(data_iterator, vp_stage=None):
    """
    从数据迭代器获取一个批次

    核心操作：
    1. 从 TP rank 切片批次（避免每个 rank 处理相同数据）
    2. 从 CP rank 切片序列（长序列分割）

    Returns:
        tokens: [seq_len, batch_size]
        labels: [seq_len, batch_size]
        loss_mask: [seq_len, batch_size]
        attention_mask: [seq_len, seq_len, batch_size] 或 1D
        position_ids: [seq_len, batch_size]
    """
    # 只在 PP 首尾阶段获取数据
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # === 操作 1: TP 维度切片 ===
    batch = get_batch_on_this_tp_rank(data_iterator)
    # 如果 sequence_parallel=False，每个 TP rank 获取相同的批次
    # 如果 sequence_parallel=True，沿序列维度切片

    # === 操作 2: CP 维度切片 ===
    batch = get_batch_on_this_cp_rank(batch)
    # 沿序列维度切分，支持超长序列

    return batch.values()
```

### 性能优化点

| 优化点 | 说明 |
|--------|------|
| **锁页内存** | `pin_memory=True`，避免 DMA 复制 |
| **持久化 Workers** | `persistent_workers=True`，减少进程重启开销 |
| **异步数据加载** | DataLoader 与 GPU 计算重叠 |
| **序列打包** | 多个短序列打包成一个长序列，减少 padding |

---

## 3. 模型构建阶段

### gpt_builder 函数

```python
# gpt_builders.py
def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """
    构建 GPT 模型

    Args:
        args: 训练参数
        pre_process: 是否包含 embedding 层（PP 第一阶段）
        post_process: 是否包含输出层（PP 最后阶段）
        vp_stage: 虚拟流水线阶段 ID（用于交错 PP）
        config: Transformer 配置
        pg_collection: 进程组集合
    """
    # === 步骤 1: 创建配置 ===
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    # === 步骤 2: 选择层规范 ===
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)  # 用户自定义规范
    else:
        use_te = args.transformer_impl == "transformer_engine"

        if args.num_experts:
            # MoE 模型：使用 decoder block 规范
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )
        else:
            # 标准 Transformer 层规范
            transformer_layer_spec = _get_transformer_layer_spec(use_te, config)

    # === 步骤 3: 创建 MTP Block 规范（可选）===
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(...)

    # === 步骤 4: 构建 GPT 模型 ===
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,  # TP 输出保持分割
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    return model
```

### GPTModel 结构

```python
# megatron/core/models/gpt/gpt_model.py
class GPTModel(LanguageModule):
    """
    GPT 模型

    结构：
        input_ids → embedding → decoder (N layers) → output_layer → logits/loss
    """

    def __init__(self, config, transformer_layer_spec, vocab_size, ...):
        super().__init__(config=config, pg_collection=pg_collection)

        # === 组件 1: Embedding 层 ===
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        # === 组件 2: RoPE（可选）===
        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
            )

        # === 组件 3: Transformer Decoder ===
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        # === 组件 4: MTP Block（可选）===
        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(...)

        # === 组件 5: Output Layer ===
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                gather_output=not self.parallel_output,  # 是否 All-Gather logits
                skip_weight_param_allocation=(
                    self.pre_process and self.share_embeddings_and_output_weights
                ),
                tp_group=self.pg_collection.tp,
            )

        # === 组件 6: 权重绑定 ===
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()
```

### 前向传播流程

```python
def forward(self, input_ids, position_ids, attention_mask, labels=None, ...):
    """
    GPT 前向传播

    数据流：
        input_ids → embedding → decoder → output_layer → logits/loss
    """
    # === 步骤 1: 预处理（Embedding + RoPE）===
    decoder_input, rotary_pos_emb, ... = self._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )
    # decoder_input: [seq_len, batch_size, hidden_size]

    # === 步骤 2: Transformer Decoder ===
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
    )
    # hidden_states: [seq_len, batch_size, hidden_size]

    # === 步骤 3: 后处理（Output + Loss）===
    return self._postprocess(
        hidden_states=hidden_states,
        input_ids=input_ids,
        labels=labels,
    )
```

### 性能优化点

| 优化点 | 说明 |
|--------|------|
| **权重绑定** | embedding 和 output_layer 共享权重 |
| **TP 并行** | ColumnParallelLinear 分割输出层 |
| **序列并行** | embedding 分散到各 TP rank |

---

## 4. 优化器配置阶段

### 优化器创建

```python
# megatron/core/optimizer/optimizer.py
def get_megatron_optimizer(config, model, config_overrides=None, ...):
    """
    创建 Megatron 优化器

    支持：
    - Adam, SGD 等优化器
    - 分布式优化器（DistributedOptimizer）
    - 参数分组（不同层不同学习率）
    """

    # === 步骤 1: 参数分组 ===
    param_groups = _get_param_groups_for_optimizer(
        model,
        config,
        config_overrides
    )
    # 示例分组：
    # - embedding 层
    # - attention 层
    # - MLP 层
    # - output 层
    # 每组可以有不同的学习率、权重衰减

    # === 步骤 2: 创建优化器 ===
    if config.use_distributed_optimizer:
        # 分布式优化器（ZeRO-1/2/3）
        optimizer = DistributedOptimizer(
            config,
            optimizer_arg,
            model,
            config_overrides=config_overrides,
        )
    else:
        # 标准优化器
        optimizer = AdamOptimizer(
            config,
            optimizer_param_groups,
            ...
        )

    return optimizer
```

### 学习率调度器

```python
# megatron/core/optimizer_param_scheduler.py
class OptimizerParamScheduler:
    """
    学习率调度器

    支持的调度策略：
    - constant: 常数学习率
    - linear: 线性衰减
    - cosine: 余弦衰减
    - inverse-square-root: 倒平方根衰减
    - exponential: 指数衰减
    """

    def step(self, increment):
        """
        更新学习率和权重衰减

        Args:
            increment: 增加的样本数（用于基于样本的调度）
        """
        # === 步骤 1: 更新计数器 ===
        self.num_steps_taken += increment

        # === 步骤 2: 计算学习率 ===
        if self.num_steps_taken < self.lr_warmup_steps:
            # 预热阶段：线性增长
            lr = self.init_lr + (self.max_lr - self.init_lr) * (
                self.num_steps_taken / self.lr_warmup_steps
            )
        else:
            # 衰减阶段
            if self.lr_decay_style == 'linear':
                progress = (
                    self.num_steps_taken - self.lr_warmup_steps
                ) / (self.lr_decay_steps - self.lr_warmup_steps)
                lr = self.max_lr + (self.min_lr - self.max_lr) * progress
            elif self.lr_decay_style == 'cosine':
                progress = (
                    self.num_steps_taken - self.lr_warmup_steps
                ) / (self.lr_decay_steps - self.lr_warmup_steps)
                lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )

        # === 步骤 3: 更新优化器学习率 ===
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # === 步骤 4: 更新权重衰减 ===
        # 类似逻辑更新 weight_decay
```

### DDP 包装

```python
# megatron/training/training.py
def get_model(model_provider_func, model_type, wrap_with_ddp=True, ...):
    """
    构建并包装模型

    核心步骤：
    1. 调用 model_provider 构建模型
    2. 设置 TP 属性
    3. GPU 分配
    4. FP16/BF16 转换
    5. DDP/FSDP 包装
    """

    # === 步骤 1: 构建模型 ===
    model = model_provider_func(...)

    # === 步骤 2: 设置 TP 属性 ===
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # === 步骤 3: GPU 分配 ===
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # === 步骤 4: FP16/BF16 转换 ===
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # === 步骤 5: DDP/FSDP 包装 ===
    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            DP = torch_FSDP  # PyTorch FSDP2
        elif args.use_megatron_fsdp:
            DP = megatron_FSDP  # Megatron FSDP
        else:
            DP = DDP  # Megatron DDP

        # 创建 DDP 配置
        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=args.overlap_grad_reduce,  # 梯度归约与计算重叠
            use_distributed_optimizer=args.use_distributed_optimizer,
            gradient_accumulation_fusion=args.gradient_accumulation_fusion,
            bucket_size=args.ddp_bucket_size,  # 梯度桶大小
            ...
        )

        # 包装每个模型块
        model = [
            DP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
            )
            for model_chunk in model
        ]

    return model
```

---

## 5. 训练循环阶段

### train 函数

```python
# megatron/training/training.py
def train(forward_step_func, model, optimizer, opt_param_scheduler, ...):
    """
    训练主循环

    核心逻辑：
    while iteration < train_iters:
        1. 前向+反向传播
        2. 优化器更新
        3. 学习率更新
        4. 日志记录
        5. 验证（按间隔）
        6. Checkpoint（按间隔）
    """

    # === 初始化 ===
    for model_module in model:
        model_module.train()  # 设置为训练模式

    iteration = args.iteration

    # === 获取前向-反向函数 ===
    forward_backward_func = get_forward_backward_func()
    # 根据 PP 配置返回：
    # - 无 PP: forward_backward_no_pipelining
    # - 标准 PP: forward_backward_pipelining_without_interleaving
    # - 交错 PP: forward_backward_pipelining_with_interleaving

    # CUDA Graph 包装（可选）
    if args.cuda_graph_impl == "local":
        forward_backward_func = FullCudaGraphWrapper(forward_backward_func, ...)

    # === 训练循环 ===
    while iteration < args.train_iters:
        # --- 周期开始 ---

        # === 步骤 1: 前向+反向 ===
        (
            loss_dict,
            skipped_iter,
            should_checkpoint,
            ...
        ) = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func
        )

        # === 步骤 2: 检查点保存 ===
        if should_checkpoint:
            save_checkpoint(...)

        # === 步骤 3: 更新计数器 ===
        iteration += 1
        args.consumed_train_samples += batch_size

        # === 步骤 4: 日志记录 ===
        if not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        learning_rate = ...
        report_memory_flag = training_log(
            loss_dict, total_loss_dict, learning_rate, iteration,
            loss_scale, report_memory_flag, skipped_iter, grad_norm, ...
        )

        # === 步骤 5: 验证 ===
        if args.eval_interval and iteration % args.eval_interval == 0:
            evaluate_and_print_results(...)

        # === 步骤 6: Checkpoint ===
        if iteration % args.save_interval == 0:
            save_checkpoint(...)

        # --- 周期结束 ---

    return iteration, num_floating_point_operations_so_far
```

### train_step 函数

```python
# megatron/training/training.py
def train_step(forward_step_func, data_iterator, model, optimizer, ...):
    """
    单步训练

    核心步骤：
    1. 梯度清零
    2. 前向+反向
    3. 优化器更新
    4. 学习率更新
    """

    # === 步骤 1: 梯度清零 ===
    for model_chunk in model:
        model_chunk.zero_grad_buffer()  # 清零梯度缓冲区
    optimizer.zero_grad()

    # === 步骤 2: 前向+反向传播 ===
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
    )
    # losses_reduced: 每个微批次的损失列表

    # === 步骤 3: 内存清理 ===
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # === 步骤 4: 优化器更新 ===
    timers('optimizer').start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # === 步骤 5: QK Clipping（可选）===
    if args.qk_clip:
        log_max_attention_logit = clip_qk(model, log_max_only=not args.qk_clip)

    # === 步骤 6: 学习率更新 ===
    if update_successful:
        increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # === 步骤 7: 内存清理 ===
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # === 步骤 8: 损失聚合 ===
    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:  # [loss, num_tokens]
                val = torch.vstack(val)
                val = val[:, 0] / val[:, 1]  # loss / num_tokens
                val = val.mean()
                torch.distributed.all_reduce(
                    val, group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                loss_reduced[key] = val
            elif val[0].numel() == 1:
                val = torch.cat(val).mean()
                loss_reduced[key] = val

        return (loss_reduced, skipped_iter, ...)

    return {}, skipped_iter, ...
```

### forward_step 函数

```python
# pretrain_gpt.py
def forward_step(data_iterator, model: GPTModel, return_schedule_plan=False):
    """
    前向传播步骤

    核心步骤：
    1. 获取批次数据
    2. 调用模型前向
    3. 计算损失
    """

    # === 步骤 1: 获取批次 ===
    timers('batch-generator').start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, vp_stage
    )
    timers('batch-generator').stop()

    # tokens: [seq_len, batch_size]
    # labels: [seq_len, batch_size]
    # loss_mask: [seq_len, batch_size]
    # attention_mask: [seq_len, seq_len, batch_size] 或 1D
    # position_ids: [seq_len, batch_size]

    # === 步骤 2: 模型前向 ===
    if args.use_legacy_models:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    else:
        output_tensor = model(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
        )
    # output_tensor: [seq_len, batch_size] (每个位置的损失)

    # === 步骤 3: 返回损失函数 ===
    return output_tensor, partial(loss_func, loss_mask, model=model)
```

### loss_func 函数

```python
# pretrain_gpt.py
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None):
    """
    损失函数

    计算 Cross-Entropy 损失（带 mask）
    """
    args = get_args()

    # === 步骤 1: Reshape ===
    losses = output_tensor.view(-1).float()      # [seq_len * batch_size]
    loss_mask = loss_mask.view(-1).float()       # [seq_len * batch_size]

    # === 步骤 2: 应用 mask ===
    loss = torch.sum(losses * loss_mask)        # 标量

    # === 步骤 3: 计算 token 数量 ===
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    # === 步骤 4: 报告指标 ===
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # === 步骤 5: 检查 NaN/Inf ===
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            fatal=True,
        )

    return loss, num_tokens, report
```

---

## 6. 算子操作详解

### 单层 Transformer 前向传播算子

```python
# 假设输入：x [seq_len, batch_size, hidden_size]

# === LayerNorm ===
x = LayerNorm(x)
# 算子：mean、std、subtract、divide、multiply、add
# 输出：[seq_len, batch_size, hidden_size]

# === QKV Projection（张量并行）===
# 沿输出维度分割
q = ColumnParallelLinear(x)  # [seq_len, batch_size, num_heads/TP * head_dim]
k = ColumnParallelLinear(x)  # [seq_len, batch_size, num_heads/TP * head_dim]
v = ColumnParallelLinear(x)  # [seq_len, batch_size, num_heads/TP * head_dim]
# 算子：GEMM (General Matrix Multiply)
# 计算：x @ W^T，其中 W 沿列分割

# === RoPE Apply ===
q = apply_rotary_pos_emb(q)  # 旋转位置编码
k = apply_rotary_pos_emb(k)
# 算子：sin, cos, multiply, add

# === Attention ===
# QK^T 计算
attn_scores = q @ k.transpose(-2, -1)  # [seq_len, seq_len, batch_size, num_heads/TP]
attn_scores = attn_scores / scale       # 缩放
attn_weights = softmax(attn_scores)     # Softmax
# 算子：MatMul, Div, Softmax

# === Attention Mask ===
attn_weights = attn_weights + attention_mask[None, None, ...]
attn_weights = softmax(attn_weights)  # 重新计算

# === Attention Output ===
attn_output = attn_weights @ v  # [seq_len, batch_size, num_heads/TP * head_dim]
# 算子：MatMul

# === Head Merge（张量并行）===
# 沿 head 维度拼接
attn_output = all_reduce(attn_output)  # [seq_len, batch_size, num_heads * head_dim]
# 算子：All-Reduce

# === Output Projection（行并行）===
output = RowParallelLinear(attn_output)  # [seq_len, batch_size, hidden_size]
# 算子：GEMM，输入沿特征维度分割，输出 All-Reduce

# === MLP ===
# Gate projection
gate = ColumnParallelLinear(x)  # [seq_len, batch_size, 4 * hidden_size/TP]
# Up projection
up = ColumnParallelLinear(x)    # [seq_len, batch_size, 4 * hidden_size/TP]
# 算子：GEMM

# Activation
act = activation_fn(gate) * up  # SwiGLU
# 算子：SiLU, Multiply

# Down projection
mlp_output = RowParallelLinear(act)  # [seq_len, batch_size, hidden_size]
# 算子：GEMM，All-Reduce
```

### 核心算子列表

| 算子 | 操作 | 输入形状 | 输出形状 | 性能优化 |
|------|------|----------|----------|----------|
| **GEMM** | 矩阵乘法 | [M, K] @ [N, K]^T | [M, N] | Tensor Core, FP16/BF16 |
| **LayerNorm** | 层归一化 | [M, N] | [M, N] | 融合 kernel |
| **Softmax** | 指数归一化 | [M, N] | [M, N] | 在线 softmax（节省内存） |
| **MatMul** | 矩阵乘法 | [M, K] @ [K, N] | [M, N] | Tensor Core |
| **All-Reduce** | 分布式归约 | [M] | [M] | NCCL, 通信与计算重叠 |
| **All-Gather** | 分布式收集 | [M/P] | [M] | NCCL, 异步执行 |
| **Reduce-Scatter** | 归约分散 | [M] | [M/P] | NCCL |
| **RoPE** | 旋转位置编码 | [M, N] | [M, N] | 融合 kernel |

---

## 7. 性能优化技术

### 7.1 并行策略优化

#### 张量并行 (TP)

```python
# ColumnParallelLinear 前向
def forward(self, input_):
    """
    列并行 Linear：权重沿输出维度分割

    优化点：
    1. 序列并行时：All-Gather 输入
    2. 梯度异步归约：与前向计算重叠
    """
    if self.sequence_parallel:
        # 序列并行：All-Gather 输入
        input_ = gather_from_sequence_parallel_region(
            input_, tensor_parallel_output_grad=True, group=self.tp_group
        )
    else:
        # 标准模式：复制到 TP 区域（反向时 All-Reduce）
        input_ = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

    # 矩阵乘法（局部计算）
    output_parallel = self._forward_impl(input_, ...)
    # output_parallel: [seq_len, batch_size, hidden_size/TP]

    if self.gather_output:
        # All-Gather 输出
        output = gather_from_tensor_model_parallel_region(output_parallel, ...)
    else:
        output = output_parallel

    return output
```

#### 流水线并行 (PP) - 1F1B 调度

```python
# megatron/core/pipeline_parallel/schedules.py
def forward_backward_pipelining_without_interleaving(...):
    """
    标准 1F1B 调度

    时间线（PP=4, M=8）：

    Rank 0: F0 F1 F2 F3    F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
    Rank 1:     F0 F1 F2 F3 F4 F5 F6 F7    B0 B1 B2 B3 B4 B5 B6 B7
    Rank 2:          F0 F1 F2 F3 F4 F5 F6 F7    B0 B1 B2 B3 B4 B5 B6 B7
    Rank 3:               F0 F1 F2 F3 F4 F5 F6 F7    B0 B1 B2 B3 B4 B5 B6 B7

    阶段划分：
    1. 预热（Warmup）：填充流水线
    2. 稳态（Steady）：1F1B，最大化 GPU 利用率
    3. 冷却（Cooldown）：清空流水线
    """

    num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1

    # === 预热阶段（只有前向）===
    for i in range(num_warmup_microbatches):
        input_tensor = recv_forward(...)
        output_tensor = forward_step(...)
        send_forward(output_tensor, ...)
        # 保存用于反向
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    # === 稳态阶段（1F1B）===
    for i in range(num_microbatches_remaining):
        # 前向
        output_tensor = forward_step(...)
        output_tensor_grad = send_forward_recv_backward(...)  # 通信重叠

        # 反向（使用之前保存的张量）
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        input_tensor_grad = backward_step(...)
        input_tensor = send_backward_recv_forward(...)  # 通信重叠

    # === 冷却阶段（只有反向）===
    for i in range(num_warmup_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        output_tensor_grad = recv_backward(...)
        input_tensor_grad = backward_step(...)
        send_backward(input_tensor_grad, ...)
```

### 7.2 内存优化

#### Activation Checkpointing

```python
# megatron/core/transformer/module.py
def checkpointed_forward(hidden_states, ...):
    """
    激活检查点

    优势：
    1. 不保存中间激活值（节省内存）
    2. 反向时重新计算（时间换空间）

    适用：大型 MLP 层、Attention 层
    """
    # 反向时重新计算
    return transformer_block(hidden_states, ...)
```

#### 伪释放输出张量

```python
# megatron/core/pipeline_parallel/schedules.py
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """
    伪释放输出张量的 .data 字段

    原理：
    1. 保留 .grad_fn（用于反向传播）
    2. 释放 .data（节省内存）

    优势：
    - PP 中间阶段的激活值可以立即释放
    - 只在反向传播需要时重新计算
    """
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    # 将 .data 替换为 1 元素张量
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

### 7.3 通信优化

#### 异步 All-Reduce

```python
# megatron/core/tensor_parallel/layers.py
def _forward_impl(self, input_, ...):
    """
    异步梯度归约

    关键：启动 All-Reduce，然后继续计算（等待重叠）
    """
    # 计算输入梯度
    grad_input = grad_output.matmul(weight)

    # 异步启动 All-Reduce（与权重梯度计算重叠）
    if self.allreduce_dgrad:
        handle = torch.distributed.all_reduce(
            grad_input, group=self.tp_group, async_op=True
        )

    # 计算权重梯度（All-Reduce 在后台进行）
    grad_weight = grad_output.t().matmul(total_input)

    # 等待 All-Reduce 完成
    if self.allreduce_dgrad:
        handle.wait()

    return grad_input, grad_weight, ...
```

#### 梯度累积融合

```python
# fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32
"""
梯度累积融合 CUDA Kernel

优势：
1. 直接累积到主梯度缓冲区（避免额外加法）
2. FP32 精度累积（提高数值稳定性）
3. 融合 kernel（减少 kernel 启动开销）

计算：
main_grad[:] += x.T @ grad_output
"""
if weight.main_grad.dtype == torch.float32:
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
        total_input,      # [M, K]
        grad_output,      # [M, N]
        weight.main_grad, # [N, K], 直接累积
    )
```

### 7.4 数值精度优化

#### FP8 训练

```python
# Transformer Engine FP8 支持
"""
FP8 训练流程：
1. 前向：FP32 → FP8 计算 → FP32 输出
2. 缩放因子（amax）自动调整
3. 反向：FP32 → FP8 计算 → FP32 梯度

优势：
- 计算速度提升约 2x
- 内存使用减半
- 吞吐量显著提升
"""

# 启用方式
args.fp8 = True
args.fp8_format = "hybrid"  # E4M3 for forward, E5M2 for backward
```

#### 混合精度

```python
# BF16/FP16 + FP32 混合
"""
策略：
1. 模型权重：BF16/FP16
2. 前向激活：BF16/FP16
3. 梯度：FP32（主梯度）
4. 优化器状态：FP32

优势：
- BF16：数值稳定性好，无需 loss scaling
- FP32 梯度：精度高，训练稳定
"""
```

### 7.5 计算优化

#### Flash Attention

```python
# FlashAttention 2 CUDA kernel
"""
优势：
1. 在线 softmax（不保存注意力矩阵）
2. 分块计算（优化内存访问）
3. 融合 kernel（减少 HBM 访问）

性能：
- 2-4x 速度提升
- 支持更长序列
"""

# 启用方式
args.use_flash_attn = True
```

#### 融合 LayerNorm + Linear

```python
# Transformer Engine 提供融合 kernel
"""
融合：LayerNorm + Linear

优势：
1. 单个 kernel（减少 kernel 启动）
2. 减少 HBM 读写
3. 提高数据局部性
"""

# 自动融合
args.transformer_impl = "transformer_engine"
```

### 7.6 CUDA Graph

```python
# CUDA Graph 捕获和重放
"""
CUDA Graph 优势：
1. 减少 kernel 启动开销
2. 优化内存访问模式
3. 提高GPU利用率

限制：
- 静态形状（seq_len, batch_size 固定）
- 静态控制流（无 if-else 分支）
"""

# 启用方式
args.cuda_graph_impl = "local"
args.cuda_graph_scope = [CudaGraphScope.full_iteration]

# 工作流程
# 1. Warmup（正常执行，捕获计算图）
# 2. Capture（创建 CUDA Graph）
# 3. Replay（重复执行，高效）
```

---

## 8. 完整训练流程示例

```python
# === 伪代码：完整训练流程 ===

# 1. 初始化
args = parse_args()
initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=8,
    context_parallel_size=2,
)
model = GPTModel(...)
optimizer = AdamOptimizer(model, lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, ...)

# 2. 数据加载
dataloader = build_dataloader(dataset, batch_size=8, micro_batch_size=1)
# num_microbatches = global_batch_size / micro_batch_size / data_parallel_size
# = 8 / 1 / (64 / 4 / 8 / 2) = 8

# 3. 训练循环
for iteration in range(100000):
    # === 单步训练 ===

    # 梯度清零
    optimizer.zero_grad()

    # 微批次循环
    losses = []
    for microbatch_idx in range(num_microbatches):
        # 获取数据
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(dataloader)

        # 前向传播
        logits = model(tokens, position_ids, attention_mask)
        loss = cross_entropy(logits, labels, loss_mask)
        losses.append(loss)

        # 反向传播
        loss.backward()

    # 梯度同步（DP All-Reduce）
    sync_grads()

    # 优化器更新
    optimizer.step()

    # 学习率更新
    scheduler.step()

    # === 日志与验证 ===
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {mean(losses)}")

    if iteration % 1000 == 0:
        validate(model, val_dataloader)

    if iteration % 10000 == 0:
        save_checkpoint(model, optimizer, scheduler, iteration)

# 4. 清理
cleanup()
```

---

## 9. 性能指标

### MFU (Model FLOP Utilization)

```
MFU = 实际吞吐量 / 理论峰值 FLOPS

理论计算：
- GPU: H100 SXM5 80GB
  - FP16/BF16 Tensor Core: 1979 TFLOPS
  - FP32 Tensor Core: 989 TFLOPS

实际测量：
- Megatron-LM + H100 可达 40-47% MFU
- 包含通信开销后的综合效率
```

### 典型配置性能

| 模型 | 参数量 | GPU 数 | TP | PP | CP | MFU | 吞吐量 |
|------|--------|-------|----|----|----|-----|--------|
| GPT-3 175B | 175B | 512 | 4 | 8 | 1 | 43% | 180 TFLOPS/GPU |
| Llama-3 70B | 70B | 64 | 4 | 4 | 2 | 45% | 220 TFLOPS/GPU |
| GPT-4 | 1.7T | 8192 | 8 | 16 | 2 | 47% | 250 TFLOPS/GPU |

---

## 10. 关键优化总结

| 优化类别 | 优化技术 | 性能提升 |
|----------|----------|----------|
| **并行策略** | TP + PP + CP + DP | 线性扩展 |
| **内存优化** | Activation Checkpointing | 2-4x 模型容量 |
| **内存优化** | Sequence Parallel | 减少 TP 通信 |
| **计算优化** | Flash Attention | 2-4x Attention 速度 |
| **计算优化** | FP8 训练 | 2x 计算速度 |
| **通信优化** | 异步 All-Reduce | 10-20% 通信重叠 |
| **通信优化** | 梯度累积融合 | 15% 梯度同步 |
| **计算优化** | CUDA Graph | 10-20% 端到端 |
| **通信优化** | 交错 PP | 更好的负载均衡 |
