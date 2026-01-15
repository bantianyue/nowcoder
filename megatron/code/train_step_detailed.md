# Megatron-LM 单步训练详细流程

> 补充文档：深入分析单个训练步骤的完整执行流程，包含详细算子操作说明

---

## 概述

单步训练（train_step）是 Megatron-LM 训练的核心，涉及前向传播、反向传播、优化器更新等关键步骤。本文档详细解析每个步骤的操作和优化技术。

---

## 涉及的主要算子分类

### 1. 计算算子 (CUDA Kernels)
| 算子类型 | 具体操作 | 数学公式 | 说明 |
|---------|---------|----------|------|
| **GEMM** | `torch.matmul` / F.linear | $C_{ij} = \sum_k A_{ik} \cdot B_{kj}$ | 矩阵乘法，计算密集型 |
| **LayerNorm** | `F.layer_norm` | $\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | 层归一化，小矩阵操作 |
| **Softmax** | `F.softmax` | $\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | 注意力权重计算 |
| **Dropout** | `F.dropout` | $y = \frac{x \cdot \text{Bernoulli}(p)}{1-p}$ | 随机失活 |
| **Embedding** | `F.embedding` | $h_i = E[token_i]$ | 词嵌入查表 |
| **SiLU** | `F.silu` | $\text{SiLU}(x) = \frac{x}{1 + e^{-x}} = x \cdot \sigma(x)$ | 激活函数 |
| **GeLU** | `F.gelu` | $\text{GeLU}(x) = x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$ | 激活函数 |
| **SwiGLU** | gate * SiLU(up) | $\text{SwiGLU}(x) = \text{SiLU}(xW_g) \odot (xW_{up})$ | MLP 激活函数 |
| **Transpose** | `tensor.transpose` / `permute` | $y_{ji} = x_{ij}$ | 张量转置 |
| **View/Reshape** | `tensor.view` / `reshape` | - | 形状变换 |
| **Concat** | `torch.cat` | - | 张量拼接 |
| **Element-wise** | add, mul, div, sqrt | $y_i = x_1 \circ x_2$ | 逐元素操作 |

### 2. 通信算子 (NCCL Collectives)
| 算子类型 | 具体操作 | 数学公式 | 说明 | 通信量 |
|---------|---------|----------|------|--------|
| **All-Reduce** | `dist.all_reduce` | $y_i = \sum_{k=1}^{N} x_k^{(i)}$ | 所有 rank 梯度聚合 | 2 × param_size |
| **Reduce-Scatter** | `dist.reduce_scatter` | $y_i^{(k)} = \sum_{j=1}^{N} x_j^{(k \cdot N + i)}$ | 聚合+分散，减少内存 | param_size |
| **All-Gather** | `dist.all_gather` | $y^{(k)} = [x_1^{(k)}, x_2^{(k)}, ..., x_N^{(k)}]$ | 收集所有 rank 的张量 | TP_size × param_size |
| **Broadcast** | `dist.broadcast` | $y^{(k)} = x^{(0)}, \forall k$ | 从一个 rank 广播 | param_size |
| **Send/Recv** | `dist.send/recv` | $y^{(recv)} = x^{(send)}$ | 点对点通信 | 张量大小 |

### 3. 内存操作算子
| 算子类型 | 具体操作 | 说明 |
|---------|---------|------|
| **Copy** | `tensor.cuda()` / `to()` | 设备间数据拷贝 |
| **Fill** | `tensor.fill_()` / `zero_()` | 内存填充 |
| **Slice/Index** | `tensor[]` / `index_select` | 张量切片 |

---

## 完整数据流图（含算子标注）

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        单步训练数据流（含算子）                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 1. 梯度清零 (zero_grad)                                                 │  │
│  │    算子: zero_() / fill_(0.0)                                            │  │
│  │    操作: GPU 内存填充零（约 0.1ms）                                       │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 2. 前向+反向 (forward_backward_func)                                     │  │
│  │    ┌──────────────────────────────────────────────────────────────────┐│  │
│  │    │ 2.1 获取批次 (get_batch)                                         ││  │
│  │    │     算子: slice, copy (CPU→GPU)                                   ││  │
│  │    │     操作: 从 dataloader 加载，TP/CP 切片                          ││  │
│  │    └──────────────────────────────────────────────────────────────────┘│  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │    ┌──────────────────────────────────────────────────────────────────┐│  │
│  │    │ 2.2 微批次循环 - 每个微批次包含以下算子：                        ││  │
│  │    │                                                                  ││  │
│  │    │   【前向传播算子】                                                ││  │
│  │    │   ┌──────────────────────────────────────────────────────────┐  ││  │
│  │    │   │ Embedding: embedding_lookup (查表)                       │  ││  │
│  │    │   │   - tokens [S,B] → hidden [S,B,H]                       │  ││  │
│  │    │   │   - 算子: F.embedding                                     │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ LayerNorm: mean, std, normalize (小矩阵)                 │  ││  │
│  │    │   │   - 算子: F.layer_norm (约 0.5ms)                         │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ QKV Projection: GEMM (大矩阵乘法)                         │  ││  │
│  │    │   │   - hidden [S,B,H] → qkv [S,B,3H]                         │  ││  │
│  │    │   │   - 算子: cublasGemmEx / F.linear (约 2ms)                │  ││  │
│  │    │   │   - TP: 每个 rank 计算 1/TP 的头                           │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ RoPE: apply_rotary (逐元素操作)                           │  ││  │
│  │    │   │   - 算子: cos, sin, mul, add (约 0.2ms)                    │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ Attention: QK^T, softmax, dropout, V×attn                │  ││  │
│  │    │   │   - QK^T: GEMM (约 1ms)                                   │  ││  │
│  │    │   │   - softmax:逐元素 (约 0.5ms)                              │  ││  │
│  │    │   │   - dropout: 随机mask (约 0.1ms)                           │  ││  │
│  │    │   │   - V×attn: GEMM (约 1ms)                                 │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ All-Reduce: 聚合 TP 梯度                                  │  ││  │
│  │    │   │   - 算子: ncclAllReduce (约 0.5ms)                        │  ││  │
│  │    │   │   - 通信量: 2 × seq_len × batch × hidden_size            │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ Output Projection: GEMM                                   │  ││  │
│  │    │   │   - 算子: cublasGemmEx (约 1ms)                            │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ MLP: fc1(GEMM) → SiLU → fc2(GEMM)                         │  ││  │
│  │    │   │   - fc1: GEMM (约 2ms)                                    │  ││  │
│  │    │   │   - SiLU: 逐元素 (约 0.3ms)                               │  ││  │
│  │    │   │   - fc2: GEMM (约 2ms)                                    │  ││  │
│  │    │   │   - All-Reduce: 通信算子 (约 0.5ms)                       │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ Output Layer: GEMM → vocab logits                        │  ││  │
│  │    │   │   - 算子: cublasGemmEx (约 3ms, vocab_size 大)            │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ CrossEntropy: log_softmax, nll_loss                      │  ││  │
│  │    │   │   - 算子: log_softmax (逐元素, 约 0.5ms)                   │  ││  │
│  │    │   │   - gather (查表, 约 0.2ms)                               │  ││  │
│  │    │   └──────────────────────────────────────────────────────────┘  ││  │
│  │    │                                                                  ││  │
│  │    │   【反向传播算子】                                                ││  │
│  │    │   ┌──────────────────────────────────────────────────────────┐  ││  │
│  │    │   │ Output Layer 反向:                                       │  ││  │
│  │    │   │   - d_logits: 扩散算子 (约 0.1ms)                         │  ││  │
│  │    │   │   - d_weight: GEMM (约 3ms)                               │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ MLP 反向:                                                 │  ││  │
│  │    │   │   - d_fc2: GEMM (约 2ms)                                  │  ││  │
│  │    │   │   - d_fc1: GEMM (约 2ms)                                  │  ││  │
│  │    │   │   - 梯度累积: add_ (逐元素, 约 0.1ms)                      │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ Attention 反向:                                          │  ││  │
│  │    │   │   - d_attn: GEMM (约 1ms)                                 │  ││  │
│  │    │   │   - d_V: GEMM (约 1ms)                                    │  ││  │
│  │    │   │   - d_QK: 复杂反向 (约 2ms)                               │  ││  │
│  │    │   │   - d_qkv_weight: GEMM (约 2ms)                           │  ││  │
│  │    │   │   - All-Reduce: 通信算子 (约 0.5ms)                       │  ││  │
│  │    │   │                                                           │  ││  │
│  │    │   │ Embedding 反向:                                          │  ││  │
│  │    │   │   - index_add: 累积梯度到embedding (约 0.5ms)             │  ││  │
│  │    │   └──────────────────────────────────────────────────────────┘  ││  │
│  │    │                                                                  ││  │
│  │    └──────────────────────────────────────────────────────────────────┘│  │
│  │    losses_reduced = [loss_0, loss_1, ..., loss_{N-1}]                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 3. 梯度同步 (DP All-Reduce)                                            │  │
│  │    算子: ncclAllReduce / ReduceScatter                                  │  │
│  │    操作: 聚合所有 DP rank 的梯度                                        │  │
│  │    通信量: 2 × model_size (All-Reduce)                                  │  │
│  │           或 model_size (ReduceScatter + DP)                            │  │
│  │    时间: 约 5-10ms (取决于网络和模型大小)                               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 4. 优化器更新 (optimizer.step)                                          │  │
│  │    算子: norm, sqrt, div, mul, add (逐元素操作)                         │  │
│  │    操作:                                                                │  │
│  │      - grad_norm = sqrt(sum(grad^2))  约 1ms                           │  │
│  │      - m = beta1 * m + (1-beta1) * grad  约 0.5ms                      │  │
│  │      - v = beta2 * v + (1-beta2) * grad^2  约 0.5ms                    │  │
│  │      - param = param - lr * m / (sqrt(v) + eps)  约 0.5ms              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                │
│                              ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ 5. 学习率更新 (opt_param_scheduler.step)                               │  │
│  │    算子: 基础算术运算（CPU，约 0.01ms）                                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  返回: loss_dict, skipped_iter, grad_norm, ...                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 步骤 1: 梯度清零 (Zero Gradient)

### 1.1 代码实现（含详细算子说明）

```python
# megatron/training/training.py
def train_step(forward_step_func, data_iterator, model, optimizer, ...):
    """
    单步训练主函数

    关键算子总览：
    - zero_/fill_: 内存填充算子
    - all_reduce/reduce_scatter: 通信算子
    - gemm: 矩阵乘法算子
    - layer_norm: 归一化算子
    - softmax: 注意力算子
    - embedding: 查表算子
    """

    # ============================================================
    # 步骤 1: 梯度清零
    # ============================================================
    # 【算子类型: 内存操作】
    # - zero_(): GPU kernel，将内存清零
    # - 执行时间: 约 0.1ms (取决于缓冲区大小)
    # - 内存访问: 顺序写，效率高

    # 1.1 清零梯度缓冲区（DistributedOptimizer 使用）
    # 【作用】复用预分配的内存，避免重复 malloc/free
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
        # 内部调用: self.grad_buffer.zero_()
        # CUDA kernel: memset(buffer, 0, size)

    # 1.2 清零优化器梯度
    optimizer.zero_grad()
    # 标准优化器: param.grad.zero_()
    # 或: param.grad.detach_() + param.grad.zero_()
```

### 1.2 zero_grad_buffer 实现（算子级别）

```python
# megatron/core/distributed/distributed_data_parallel.py
def zero_grad_buffer(self):
    """
    清零梯度缓冲区（DistributedOptimizer）

    ====== 算子分析 ======
    算子: cudaMemset / cudaMemsetAsync
    功能: 将 GPU 内存块填充为指定值
    时间复杂度: O(n)，n = buffer_size
    空间复杂度: O(1)，原地操作
    并行度: 高，内存操作可以充分并行

    优化点：
    1. 使用 zero_() 而非 fill_(0.0) - 零值有特殊优化
    2. 异步执行，不阻塞 CPU
    3. 复用内存，减少分配开销
    """
    if self.grad_buffer is not None:
        # 【CUDA 算子: cudaMemsetAsync】
        # GPU kernel: 将连续内存块清零
        # 优势: 硬件级优化，比普通 fill 快
        self.grad_buffer.zero_()

        # 等价于 (但 zero_ 更优):
        # self.grad_buffer.fill_(0.0)
        # CUDA kernel: for i in range(n): buffer[i] = 0
```

### 1.3 optimizer.zero_grad 实现

```python
# megatron/core/optimizer/optimizer.py
def zero_grad(self):
    """
    优化器梯度清零

    ====== 算子分析 ======
    标准优化器:
    - 算子: detach_ + zero_
    - 时间: O(n_params × param_size)

    DistributedOptimizer:
    - 算子: zero_ (仅主梯度)
    - 时间: O(n_params × param_size / DP_size)
    - 优势: 梯度分片，每个 rank 只清零一部分
    """
    if self.config.use_distributed_optimizer:
        # DistributedOptimizer: 梯度分片
        for group in self.param_groups:
            for param in group['params']:
                if hasattr(param, 'main_grad'):
                    # 【CUDA 算子: cudaMemsetAsync】
                    # 只清零主梯度缓冲区（已分片）
                    param.main_grad.zero_()
    else:
        # 标准优化器: 清零所有参数梯度
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # 【算子 1: detach_】
                    # 从计算图中分离，避免梯度传播到清零操作
                    param.grad.detach_()

                    # 【算子 2: zero_】
                    # 清零梯度张量
                    param.grad.zero_()
```

---

## 步骤 2: 前向+反向传播

### 2.1 获取批次数据

```python
# pretrain_gpt.py
def get_batch(data_iterator, vp_stage=None):
    """
    从数据迭代器获取批次数据

    ====== 算子分析 ======
    涉及算子:
    1. slice / index_select: 数据切片
    2. copy / to_: CPU→GPU 数据传输
    3. all_gather (SP): 序列并行数据收集
    4. recv/send (PP): 流水线并行数据传递

    数据流:
    dataloader (CPU) → slice (TP) → slice (CP) → GPU tensor
    """
    # === 检查 PP 阶段 ===
    if not is_first_or_last_pipeline_stage(vp_stage):
        # PP 中间阶段：从上一 stage 接收数据
        # 【通信算子: recv】
        # 点对点接收，阻塞操作
        return None, None, None, None, None

    # === TP 维度切片 ===
    batch = get_batch_on_this_tp_rank(data_iterator)
    """
    ====== 算子分析 ======

    如果 sequence_parallel=True:
    - 【算子: slice】沿序列维度切分
    - tokens [seq_len, batch] → tokens[rank_offset:rank_offset+seq_len/TP, batch]
    - 每个 TP rank 获取不同的序列片段
    - 优势: 减少 LayerNorm 的通信开销

    如果 sequence_parallel=False:
    - 【算子: broadcast 或 copy】
    - 所有 TP rank 获取相同的数据副本
    - 需要在后续 LayerNorm 做 All-Reduce
    """

    # === CP 维度切片 ===
    batch = get_batch_on_this_cp_rank(batch)
    """
    ====== 算子分析 ======
    - 【算子: slice】沿序列维度切分
    - 用于超长序列训练（seq_len > context_window）
    - 每个 CP rank 处理序列的不同段
    - 需要 Ring Attention 或 Ulysses Attention
    """

    return batch.values()
    # 返回：tokens, labels, loss_mask, attention_mask, position_ids
```

### 2.2 数据形状说明

```python
# 数据形状示例（70B 模型）
tokens:        [2048, 2]        # [seq_len, micro_batch_size]
labels:        [2048, 2]        # 目标 token IDs
loss_mask:     [2048, 2]        # 1=有效位置, 0=padding
attention_mask: [2048, 1]       # 因果 mask (下三角矩阵)
position_ids:  [2048, 2]        # RoPE 位置编码

# 内存占用 (FP16)
# tokens: 2048 × 2 × 2 bytes = 8 KB
# labels: 2048 × 2 × 2 bytes = 8 KB
# loss_mask: 2048 × 2 × 2 bytes = 8 KB
# attention_mask: 2048 × 1 × 2 bytes = 4 KB
# position_ids: 2048 × 2 × 2 bytes = 8 KB
# 总计: ~36 KB/微批次
```

### 2.3 单个微批次处理流程（算子级别详解）

```python
# megatron/training/training.py
# forward_backward_func 内部循环
for microbatch_idx in range(num_microbatches):
    # === 获取微批次数据 ===
    # 【算子: copy (如果需要)】
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    # === 前向传播 ===
    # 调用 model.forward，内部包含大量 GEMM 算子
    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask
    )
    # output_tensor: [seq_len, batch_size] - 每个位置的损失值

    # === 反向传播 ===
    # 【算子: 自动微分，计算梯度】
    # 内部调用大量反向 GEMM 算子
    loss.backward()
```

### 2.4 模型前向传播详细流程（算子级别）

```python
# GPTModel.forward()
# ====== 算子统计 ======
# 总 GEMM 数量: 2 × num_layers × 3 (QKV + MLP + output)
# 总 LayerNorm 数量: 2 × num_layers
# 总 Softmax 数量: num_layers
# 总 Embedding 数量: 1

# === 步骤 1: 预处理（Embedding + RoPE）===
# 【算子 1: embedding_lookup】
# - 操作: 根据 token_id 查表获取 embedding
# - 复杂度: O(seq_len × batch_size × hidden_size)
# - 实现: F.embedding = index_select + gather
# - 时间: 约 0.5ms (2048 × 2 × 8192)
decoder_input = model.embedding(tokens)
# decoder_input: [2048, 2, 8192]

# 【算子 2: dropout】(可选)
# - 操作: 随机将部分元素置零
# - 实现: 生成随机 mask × 原张量
# - 时间: 约 0.1ms
decoder_input = F.dropout(decoder_input, p=config.dropout, training=self.training)

# 【算子 3: LayerNorm】
# - 操作: 计算均值、方差，归一化
# - 公式: $\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
#   其中: $\mu = \frac{1}{n}\sum_{i=1}^{n}x_i$, $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2$
# - 实现: 多次 pass (mean → var → normalize)
# - 时间: 约 0.5ms
decoder_input = F.layer_norm(
    decoder_input,
    (config.hidden_size,),
    weight=model.embedding_layer_norm.weight,
    bias=model.embedding_layer_norm.bias,
)

# 【算子 4: RoPE 位置编码生成】
# - 操作: 生成 cos, sin 位置编码
# - 公式: 对于位置 $m$ 和维度 $i$:
#   $\theta_i = 10000^{-2i/d}$, $\text{pos}(m) = (m\theta_0, m\theta_1, ..., m\theta_{d/2-1})$
#   $\text{RoPE}(x, m) = \begin{bmatrix} x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i) \\ x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i) \end{bmatrix}$
# - 实现: arange → unsqueeze → sin/cos
# - 时间: 约 0.1ms
rotary_pos_emb = get_rotary_pos_emb(position_ids, config)

# === 步骤 2: Transformer Decoder ===
# ====== 循环所有层，每层算子：======
hidden_states = decoder_input
for layer_idx, layer in enumerate(model.decoder.layers):
    # ====== Layer 1: Pre-LayerNorm ======
    residual = hidden_states  # 【算子: copy/alias】

    # 【算子: LayerNorm】
    # - 时间: 约 0.5ms
    # - SP 模式: 需要 All-Reduce (后续处理)
    hidden_states = layer.layer_norm(hidden_states)

    # ====== Layer 2: Self-Attention ======

    # 【算子 1: QKV Projection - 3个并行 GEMM】
    # - 输入: hidden_states [2048, 2, 8192]
    # - 权重: qkv_weight [8192, 3×8192/TP] = [8192, 6144] (TP=4)
    # - 输出: qkv [2048, 2, 3×8192/TP] = [2048, 2, 6144]
    # - 【CUDA 算子: cublasGemmEx】
    # - 时间: 约 2ms (2048×2×8192 × 8192×6144)
    # - FLOPs: 2048 × 2 × 8192 × 6144 × 2 = 411B FLOPs
    qkv = layer.qkv_proj(hidden_states)

    # 【算子 2: Reshape/View】
    # - 操作: 重排张量维度，不改变数据
    # - 时间: ~0 (元数据操作)
    q, k, v = qkv.chunk(3, dim=-1)  # 【算子: chunk/slice】
    q = q.view(2048, 2, num_heads_per_partition, qk_head_dim)
    k = k.view(2048, 2, num_heads_per_partition, qk_head_dim)
    v = v.view(2048, 2, num_heads_per_partition, v_head_dim)

    # 【算子 3: Transpose/Permute】
    # - 操作: 交换维度以供 attention 计算
    # - 时间: ~0 (元数据操作)
    q = q.permute(1, 2, 0, 3)  # [batch, heads, seq_len, head_dim]
    k = k.permute(1, 2, 0, 3)
    v = v.permute(1, 2, 0, 3)

    # 【算子 4: RoPE 应用】
    # - 操作: 旋转向量位置编码
    # - 公式: [q0, q1] × [cos, -sin; sin, cos]
    # - 实现: 逐元素 mul + add
    # - 时间: 约 0.2ms
    q = apply_rotary_pos_emb(q, rotary_pos_emb)
    k = apply_rotary_pos_emb(k, rotary_pos_emb)

    # ====== Core Attention ======
    # 【算子 5: QK^T - Attention Score 计算】
    # - 输入: q [2, 32, 2048, 128], k [2, 32, 2048, 128]
    # - 输出: attn_scores [2, 32, 2048, 2048]
    # - 公式: $\text{Scores}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$
    # - 【CUDA 算子: cublasGemmBatched】
    # - 时间: 约 1ms (batch GEMM)
    # - FLOPs: 2 × 32 × 2048 × 2048 × 128 × 2 = 68B FLOPs
    attn_scores = torch.matmul(q, k.transpose(-2, -1))

    # 【算子 6: Scale】
    # - 操作: 除以 sqrt(head_dim)
    # - 公式: $\text{ScaledScores} = \frac{\text{Scores}}{\sqrt{d_k}}$
    # - 实现: 逐元素 mul (预计算 scale)
    # - 时间: ~0.1ms
    attn_scores = attn_scores / math.sqrt(qk_head_dim)

    # 【算子 7: Add attention mask】
    # - 操作: 应用因果 mask (将未来位置设为 -inf)
    # - 公式: $\text{MaskedScores}_{ij} = \text{Scores}_{ij} + M_{ij}$, $M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$
    # - 实现: 逐元素 add
    # - 时间: 约 0.1ms
    attn_scores = attn_scores + attention_mask  # 广播加法

    # 【算子 8: Softmax】
    # - 操作: exp(x) / sum(exp(x))
    # - 公式: $\text{AttnWeight}_{ij} = \frac{\exp(\text{MaskedScores}_{ij})}{\sum_k \exp(\text{MaskedScores}_{ik})}$
    # - 实现: max → exp → sum → div (4-pass 算法)
    # - 时间: 约 0.5ms
    # - 【CUDA 算子: 优化 softmax kernel】
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 【算子 9: Dropout】(可选)
    # - 公式: $y = \frac{x \cdot \text{Bernoulli}(p)}{1-p}$
    # - 时间: 约 0.1ms
    attn_weights = F.dropout(attn_weights, p=config.dropout, training=self.training)

    # 【算子 10: Attention × V - 加权求和】
    # - 输入: attn_weights [2, 32, 2048, 2048], v [2, 32, 2048, 128]
    # - 输出: attn_output [2, 32, 2048, 128]
    # - 公式: $\text{Output}_i = \sum_j \text{AttnWeight}_{ij} \cdot V_j$
    # - 【CUDA 算子: cublasGemmBatched】
    # - 时间: 约 1ms
    # - FLOPs: 2 × 32 × 2048 × 2048 × 128 × 2 = 68B FLOPs
    attn_output = torch.matmul(attn_weights, v)

    # 【算子 11: Transpose】
    # - 操作: 还原维度
    # - 时间: ~0
    attn_output = attn_output.permute(2, 0, 1, 3)  # [seq_len, batch, heads, head_dim]

    # 【算子 12: View/Reshape】
    # - 操作: 合并 head 维度
    # - 时间: ~0
    attn_output = attn_output.contiguous().view(2048, 2, -1)

    # ====== 张量并行通信 ======
    # 【通信算子: All-Reduce】
    # - 操作: 聚合所有 TP rank 的 attn_output
    # - 通信量: 2 × seq_len × batch × hidden_size
    #           = 2 × 2048 × 2 × 8192 × 2 bytes = 128 MB
    # - 【NCCL 算子: ncclAllReduce】
    # - 时间: 约 0.5ms (取决于网络)
    # - 算法: Ring All-Reduce (2× 节点数 步)
    attn_output = reduce_from_tensor_model_parallel_region(attn_output)

    # 【算子 13: Output Projection - GEMM】
    # - 输入: attn_output [2048, 2, 8192]
    # - 权重: proj_weight [8192/TP, 8192] = [2048, 8192] (TP=4)
    # - 输出: [2048, 2, 8192]
    # - 【CUDA 算子: cublasGemmEx】
    # - 时间: 约 1ms
    # - FLOPs: 2048 × 2 × 8192 × 8192 × 2 = 549B FLOPs
    attn_output = layer.linear_proj(attn_output)

    # ====== Residual Connection + Post-LayerNorm ======
    # 【算子 14: Element-wise Add】
    # - 时间: ~0.1ms
    hidden_states = residual + attn_output

    # 【算子 15: LayerNorm】
    # - 时间: 约 0.5ms
    hidden_states = layer.post_layer_norm(hidden_states)

    # ====== Layer 3: MLP ======
    residual = hidden_states

    # 【算子 16: LayerNorm】
    hidden_states = layer.mlp_layer_norm(hidden_states)

    # 【算子 17: Gate Projection - GEMM】
    # - 输入: [2048, 2, 8192]
    # - 权重: [8192, 4×8192/TP] = [8192, 8192] (TP=4, 中间层)
    # - 输出: [2048, 2, 8192]
    # - 【CUDA 算子: cublasGemmEx】
    # - 时间: 约 2ms
    gate = layer.linear_fc1(hidden_states)

    # 【算子 18: Up Projection - GEMM】
    # - 同上
    up = layer.linear_fc2(hidden_states)

    # 【算子 19: Activation (SwiGLU)】
    # - 公式: $\text{SwiGLU}(x) = \text{SiLU}(xW_g) \odot (xW_{up})$
    #   其中 $\text{SiLU}(x) = \frac{x}{1 + e^{-x}} = x \cdot \sigma(x)$
    # - 实现: sigmoid → mul (逐元素)
    # - 时间: 约 0.3ms
    act = F.silu(gate) * up

    # 【算子 20: Down Projection - GEMM】
    # - 输入: [2048, 2, 8192]
    # - 权重: [8192, 8192]
    # - 输出: [2048, 2, 8192]
    # - 时间: 约 2ms
    mlp_output = layer.linear_proj(act)

    # 【通信算子: All-Reduce】
    # - 时间: 约 0.5ms
    mlp_output = reduce_from_tensor_model_parallel_region(mlp_output)

    # 【算子 21: Residual Add】
    hidden_states = residual + mlp_output

# === 步骤 3: 后处理（Output Layer + Loss）===
# 【算子 22: Final LayerNorm】
hidden_states = F.layer_norm(hidden_states, ...)

# 【算子 23: Output Layer - GEMM (最大)】
# - 输入: [2048, 2, 8192]
# - 权重: [8192, vocab_size/TP] = [8192, 24000] (TP=4, vocab=96000)
# - 输出: logits [2048, 2, 24000]
# - 【CUDA 算子: cublasGemmEx】
# - 时间: 约 3ms (vocab_size 大)
# - FLOPs: 2048 × 2 × 8192 × 24000 × 2 = 1.6T FLOPs
logits = model.output_layer(hidden_states)

# 【通信算子: All-Gather】(如果 parallel_output=False)
# - 操作: 收集所有 TP rank 的 logits
# - 通信量: TP_size × seq_len × batch × vocab_size
# - 时间: 约 1ms
if not model.parallel_output:
    logits = gather_from_tensor_model_parallel_region(logits)

# 【算子 24: Cross Entropy Loss】
# - 操作: log_softmax + nll_loss
# - 公式: $\text{CE}(y, \hat{y}) = -\sum_{i} y_i \log(\text{Softmax}(\hat{y}_i))$
#   其中 $\text{Softmax}(\hat{y}_i) = \frac{e^{\hat{y}_i}}{\sum_j e^{\hat{y}_j}}$
#   简化: $\text{CE} = -\log(\text{Softmax}(\hat{y}_{\text{target}})) = \log(\sum_j e^{\hat{y}_j}) - \hat{y}_{\text{target}}$
# - 【CUDA 算子: 融合 log_softmax + nll_loss kernel】
# - 时间: 约 0.5ms
# - 实现: 通常融合为单个 kernel，避免中间结果存储
loss = cross_entropy(logits, labels, loss_mask)

# ====== 前向传播算子总结 ======
# 单层 Transformer (70B 模型, TP=4):
# - GEMM: 6个 (QKV×1, attn_out×1, MLP×3)
# - LayerNorm: 2个
# - Softmax: 1个
# - All-Reduce: 2个 (attn, MLP)
# - 逐元素操作: 约10个
#
# 80层总计:
# - GEMM: 480个
# - LayerNorm: 160个
# - Softmax: 80个
# - All-Reduce: 160个
#
# 预估时间:
# - GEMM: 480 × 2ms = 960ms
# - Attention (softmax + QK^T + V@attn): 80 × 2.5ms = 200ms
# - LayerNorm: 160 × 0.5ms = 80ms
# - 通信: 160 × 0.5ms = 80ms
# - 其他: 约50ms
# 总计: ~1370ms (1.37秒/步，前向)
```

### 2.5 反向传播详细流程（算子级别）

```python
# ====== 反向传播算子详解 ======
# loss.backward() 触发自动微分，计算所有梯度

# === Output Layer 反向 ===
# 【算子 1: CrossEntropy 反向】
# - 操作: 计算 d_logits
# - 公式: $\frac{\partial \text{CE}}{\partial \hat{y}_i} = \text{Softmax}(\hat{y})_i - \mathbb{1}[i = \text{target}]$
#   其中 $\mathbb{1}[\cdot]$ 是指示函数
# - 【CUDA 算子: 融合 kernel】
# - 时间: 约 0.5ms
# grad_logits: [2048, 2, 24000/TP] = [2048, 2, 6000]

# 【算子 2: Output Layer 权重梯度 - GEMM】
# - 操作: d_weight = grad_logits.T @ grad_hidden
# - 公式: $\frac{\partial \text{CE}}{\partial W} = \frac{\partial \text{CE}}{\partial y} \cdot x^T$
# - 【CUDA 算子: cublasGemmEx】
# - 时间: 约 3ms (大矩阵)
# d_output_weight = grad_logits.transpose(-2, -1) @ hidden_states
# shape: [6000, 8192]

# 【算子 3: Output Layer 输入梯度 - GEMM】
# - 操作: d_hidden = grad_logits @ weight
# - 公式: $\frac{\partial \text{CE}}{\partial x} = \frac{\partial \text{CE}}{\partial y} \cdot W^T$
# - 【CUDA 算子: cublasGemmEx】
# - 时间: 约 1ms
grad_hidden = grad_logits @ output_layer.weight
# shape: [2048, 2, 8192]

# === Transformer Layer 反向 (从后往前) ===
for layer in reversed(model.decoder.layers):
    # ====== MLP 反向 ======

    # 【算子 4: MLP 输入梯度 - GEMM】
    # - 操作: d_act = grad_hidden @ proj_weight.T
    # - 时间: 约 2ms
    grad_act = grad_hidden @ layer.linear_proj.weight.T

    # 【算子 5: Down Projection 权重梯度 - GEMM】
    # - 操作: d_proj_weight = act.T @ grad_hidden
    # - 时间: 约 2ms
    layer.linear_proj.weight.grad += act.T @ grad_hidden

    # 【算子 6: SwiGLU 反向】
    # - SiLU(gate) * up 的反向
    # - 公式: $\frac{\partial \text{SwiGLU}}{\partial \text{gate}} = \text{up} \cdot \sigma(\text{gate}) \cdot (1 - \sigma(\text{gate}))$
    #         $\frac{\partial \text{SwiGLU}}{\partial \text{up}} = \text{SiLU}(\text{gate})$
    # - 涉及: sigmoid, mul 的链式法则
    # - 时间: 约 0.3ms
    grad_up = grad_act * F.silu(gate)
    grad_gate = grad_act * up * F.sigmoid(gate) * (1 - F.sigmoid(gate))

    # 【算子 7: Up Projection 权重梯度 - GEMM】
    # - 时间: 约 2ms
    layer.linear_fc2.weight.grad += grad_up.T @ hidden_states

    # 【算子 8: Up Projection 输入梯度 - GEMM】
    # - 时间: 约 2ms
    grad_hidden_from_up = grad_up @ layer.linear_fc2.weight.T

    # 【算子 9: Gate Projection 权重梯度 - GEMM】
    # - 时间: 约 2ms
    layer.linear_fc1.weight.grad += grad_gate.T @ hidden_states

    # 【算子 10: Gate Projection 输入梯度 - GEMM】
    # - 时间: 约 2ms
    grad_hidden_from_gate = grad_gate @ layer.linear_fc1.weight.T

    # 【算子 11: 梯度累加】
    # - 操作: 两个梯度路径相加
    # - 时间: ~0.1ms
    grad_mlp = grad_hidden_from_up + grad_hidden_from_gate

    # ====== Attention 反向 ======

    # 【算子 12: Attention 输出梯度 - GEMM】
    # - 操作: d_attn_output = grad_hidden @ proj_weight.T
    # - 时间: 约 1ms
    grad_attn_output = grad_hidden @ layer.linear_proj.weight.T

    # 【算子 13: Output Projection 权重梯度 - GEMM】
    # - 时间: 约 1ms
    layer.linear_proj.weight.grad += grad_attn_output.transpose(-2, -1) @ attn_output

    # ====== 张量并行通信 ======
    # 【通信算子: All-Reduce】
    # - 操作: 聚合所有 TP rank 的梯度
    # - 注意: 这是反向的 All-Reduce，与前向对称
    # - 时间: 约 0.5ms
    grad_attn_output = reduce_from_tensor_model_parallel_region(grad_attn_output)

    # 【算子 14: Attention Weight 梯度】
    # - 操作: d_attn_weights = grad_attn_output @ v.T
    # - 【CUDA 算子: cublasGemmBatched】
    # - 时间: 约 1ms
    grad_attn_weights = grad_attn_output @ v.transpose(-2, -1)

    # 【算子 15: Value 梯度 - GEMM】
    # - 操作: d_v = attn_weights.T @ grad_attn_output
    # - 时间: 约 1ms
    grad_v = attn_weights.transpose(-2, -1) @ grad_attn_output

    # 【算子 16: Value 权重梯度 - GEMM】
    # - 时间: 约 1ms
    layer.v_weight.grad += grad_v.transpose(-2, -1) @ v_input

    # 【算子 17: Softmax 反向】
    # - 操作: 复杂，涉及 softmax 的雅可比矩阵
    # - 公式: $\frac{\partial \text{Softmax}_i}{\partial x_j} = \text{Softmax}_i \cdot (\delta_{ij} - \text{Softmax}_j)$
    #   简化实现: $\frac{\partial L}{\partial x} = \text{Softmax}(x) \odot (\frac{\partial L}{\partial y} - \sum_i \frac{\partial L}{\partial y_i} \cdot \text{Softmax}_i)$
    # - 【CUDA 算子: 优化的 softmax 反向 kernel】
    # - 时间: 约 0.5ms
    grad_attn_scores = softmax_backward(grad_attn_weights, attn_scores)

    # 【算子 18: Scale 反向】
    # - 操作: 除以 sqrt(head_dim)
    # - 时间: ~0.1ms
    grad_attn_scores = grad_attn_scores / math.sqrt(qk_head_dim)

    # 【算子 19: QK^T 反向 (复杂)】
    # - 需要分别计算 d_q 和 d_k
    # - d_q = grad_attn_scores @ k
    # - d_k = grad_attn_scores.T @ q
    # - 【CUDA 算子: cublasGemmBatched × 2】
    # - 时间: 约 1ms × 2 = 2ms
    grad_q = grad_attn_scores @ k
    grad_k = grad_attn_scores.transpose(-2, -1) @ q

    # 【算子 20: Query/Key 权重梯度 - GEMM】
    # - 时间: 约 2ms × 2 = 4ms
    layer.q_weight.grad += grad_q.transpose(-2, -1) @ q_input
    layer.k_weight.grad += grad_k.transpose(-2, -1) @ k_input

    # 【算子 21: Query/Key 输入梯度 - GEMM】
    # - 时间: 约 2ms × 2 = 4ms
    grad_q_input = grad_q @ layer.q_weight.weight.T
    grad_k_input = grad_k @ layer.k_weight.weight.T

    # 【算子 22: Value 输入梯度 - GEMM】
    # - 时间: 约 2ms
    grad_v_input = grad_v @ layer.v_weight.weight.T

    # 【算子 23: QKV 梯度合并】
    # - 操作: 合并 d_q_input, d_k_input, d_v_input
    # - 时间: ~0.1ms
    grad_qkv = torch.cat([grad_q_input, grad_k_input, grad_v_input], dim=-1)

    # 【算子 24: QKV Projection 权重梯度 - GEMM】
    # - 时间: 约 2ms
    layer.qkv_proj.weight.grad += grad_qkv.transpose(-2, -1) @ hidden_states

    # 【算子 25: QKV Projection 输入梯度 - GEMM】
    # - 时间: 约 2ms
    grad_hidden = grad_qkv @ layer.qkv_proj.weight.T

    # ====== 残差连接梯度 ======
    # 【算子 26: 梯度累加】
    # - 操作: grad_residual = grad_hidden + grad_mlp
    # - 时间: ~0.1ms
    grad_residual = grad_hidden + grad_mlp

    # ====== LayerNorm 反向 ======
    # 【算子 27: LayerNorm 反向】
    # - 操作: 计算 d_x, d_gamma, d_beta
    # - 公式复杂，涉及均值、方差的反向传播
    # - 【CUDA 算子: 融合 LayerNorm 反向 kernel】
    # - 时间: 约 0.5ms
    grad_hidden, grad_gamma, grad_beta = layer_norm_backward(
        grad_residual, hidden_states, weight, bias
    )
    layer.layer_norm.weight.grad += grad_gamma
    layer.layer_norm.bias.grad += grad_beta

# === Embedding 反向 ===
# 【算子 28: Embedding 梯度】
# - 操作: 根据 token_id 累积梯度
# - 实现: index_add_ (scatter add)
# - 【CUDA 算子: 优化的 embedding 反向 kernel】
# - 时间: 约 0.5ms
embedding.weight.grad = torch.zeros_like(embedding.weight)
for i, token_id in enumerate(tokens.flatten()):
    embedding.weight.grad[token_id] += grad_embedding.flatten()[i]

# ====== 反向传播算子总结 ======
# 单层 Transformer (70B 模型, TP=4):
# - GEMM: 约25个 (所有线性层的正向和反向)
# - LayerNorm 反向: 2个
# - Softmax 反向: 1个
# - All-Reduce: 2个
# - 逐元素操作: 约20个
#
# 80层总计:
# - GEMM: 约2000个
# - LayerNorm 反向: 160个
# - Softmax 反向: 80个
# - All-Reduce: 160个
#
# 预估时间:
# - GEMM: 2000 × 2ms = 4000ms
# - Attention 反向: 80 × 3ms = 240ms
# - LayerNorm 反向: 160 × 0.5ms = 80ms
# - 通信: 160 × 0.5ms = 80ms
# - 其他: 约100ms
# 总计: ~4500ms (4.5秒/步，反向)

# ====== 单步总时间估算 ======
# - 前向: ~1370ms
# - 反向: ~4500ms
# - 梯度同步: ~50ms
# - 优化器更新: ~100ms
# 总计: ~6秒/步 (70B 模型, 8×8 GPU, TP=4, PP=2)
```

---

## 步骤 3: 梯度同步 (Gradient Synchronization)

### 3.1 DP All-Reduce

```python
# megatron/core/distributed/distributed_data_parallel.py
def finalize_model_grads(model, ...):
    """
    梯度同步（DP All-Reduce）

    ====== 算子分析 ======
    目的: 将所有数据并行 rank 的梯度聚合

    涉及算子:
    1. All-Reduce: 标准梯度同步
    2. Reduce-Scatter: DistributedOptimizer 优化
    3. All-Gather: 收集分片梯度
    """

    # === DP All-Reduce ===
    if use_distributed_optimizer:
        # DistributedOptimizer: 使用 Reduce-Scatter
        # ====== 算子: Reduce-Scatter ======
        # - 操作: 聚合 + 分散
        # - 输入: 每个 rank 有完整梯度
        # - 输出: 每个 rank 有部分梯度 (分片)
        # - 通信量: model_size (比 All-Reduce 少一半)
        # - 【NCCL 算子: ncclReduceScatter】
        # - 时间: 约 5ms (70B 模型, 8 GPU)
        # - 算法: Ring Reduce-Scatter
        for model_chunk in model:
            model_chunk.allreduce_gradients()
    else:
        # DDP: 使用标准 All-Reduce
        # ====== 算子: All-Reduce ======
        # - 操作: 所有 rank 聚合完整梯度
        # - 输入: 每个 rank 有完整梯度
        # - 输出: 每个 rank 有完整梯度 (相同)
        # - 通信量: 2 × model_size
        # - 【NCCL 算子: ncclAllReduce】
        # - 时间: 约 10ms (比 Reduce-Scatter 慢)
        # - 算法: Ring All-Reduce
        for model_chunk in model:
            model_chunk.allreduce_params_grads()

    # === 序列并行: LayerNorm 梯度 All-Reduce ===
    if config.sequence_parallel:
        # SP 模式下，LayerNorm 输入是分片的
        # 需要额外的 All-Reduce
        for model_chunk in model:
            for layer in model_chunk.decoder.layers:
                # ====== 算子: All-Reduce ======
                layer.layer_norm_weight.allreduce_params_grads()
                layer.layer_norm_bias.allreduce_params_grads()
```

### 3.2 梯度同步算子详解

```python
# ====== All-Reduce 详解 ======
"""
算子: NCCL All-Reduce (Ring 算法)

步骤:
1. Reduce-Scatter 阶段:
   rank 0:  [A0, B0, C0, D0]  →  [A0+A1+A2+A3, ..., ...]
   rank 1:  [A1, B1, C1, D1]  →  [..., B0+B1+B2+B3, ...]
   rank 2:  [A2, B2, C2, D2]  →  [..., ..., C0+C1+C2+C3, ...]
   rank 3:  [A3, B3, C3, D3]  →  [..., ..., ..., D0+D1+D2+D3]

   每个步骤: send next chunk, recv prev chunk
   通信: (N-1) × (size / N)

2. All-Gather 阶段:
   rank 0:  [A_sum, ..., ...]  →  [A_sum, B_sum, C_sum, D_sum]
   rank 1:  [..., B_sum, ...]  →  [A_sum, B_sum, C_sum, D_sum]
   ...

   每个步骤: send next chunk, recv prev chunk
   通信: (N-1) × (size / N)

总通信量: 2 × size × (N-1) / N ≈ 2 × size
总延迟: 2 × (N-1) × (latency + size/N/bandwidth)

示例: 70B 模型 (280GB FP32), 8 节点, InfiniBand (400Gbps)
- 通信量: 2 × 280GB = 560GB
- 时间: 560GB / 400Gbps ≈ 11.2 秒
- 实际: 约 50ms (因为梯度分片 + 重叠计算)
"""

# ====== Reduce-Scatter 详解 ======
"""
算子: NCCL Reduce-Scatter (Ring 算法)

步骤:
只有 Reduce-Scatter 阶段，没有 All-Gather

rank 0:  [A0, B0, C0, D0]  →  [A0+A1+A2+A3]
rank 1:  [A1, B1, C1, D1]  →  [B0+B1+B2+B3]
rank 2:  [A2, B2, C2, D2]  →  [C0+C1+C2+C3]
rank 3:  [A3, B3, C3, D3]  →  [D0+D1+D2+D3]

总通信量: size × (N-1) / N ≈ size
总延迟: (N-1) × (latency + size/N/bandwidth)

示例: 70B 模型, 8 节点
- 通信量: 280GB (比 All-Reduce 少一半)
- 时间: 约 25ms (比 All-Reduce 快)

优势:
1. 通信量减半
2. 每个 rank 只存储部分梯度，节省内存
3. 优化器更新时不需要再 All-Gather
"""
```

---

## 步骤 4: 优化器更新

### 4.1 Adam 优化器（算子级别）

```python
# megatron/core/optimizer/optimizer.py
def step(self):
    """
    Adam 优化器更新步骤

    ====== 算子分析 ======
    涉及算子:
    1. norm: L2 范数计算
    2. sqrt: 平方根
    3. div: 除法
    4. mul: 乘法
    5. add: 加法
    6. clip_grad_norm: 梯度裁剪

    所有算子都是逐元素操作，高度并行
    """
    update_successful = True
    grad_norm = 0.0
    num_zeros_in_grad = 0

    for param_group in self.param_groups:
        for param in param_group['params']:
            # === 获取梯度 ===
            if param.grad is not None:
                grad = param.grad.data
            elif hasattr(param, 'main_grad'):
                grad = param.main_grad.data
            else:
                continue

            # === 梯度 Clipping ===
            if self.config.clip_grad > 0.0:
                # ====== 算子: 梯度裁剪 ======
                # - 操作: if norm > clip: grad = grad / norm * clip
                # - 【CUDA 算子: 融合 norm + clip kernel】
                # - 时间: 约 0.5ms (遍历所有参数)
                grad = clip_grad_norm_(grad, self.config.clip_grad)

            # === 计算梯度范数 (用于 logging) ===
            # ====== 算子: L2 Norm ======
            # - 公式: sqrt(sum(x^2))
            # - 【CUDA 算子: cblasNrm2】
            # - 时间: 约 0.5ms
            # - 实现: 两轮 pass (sum of squares + sqrt)
            grad_norm += grad.data.norm() ** 2

            # === 统计零梯度 ===
            # ====== 算子: 比较 + 计数 ======
            # - 操作: count(x == 0)
            # - 【CUDA 算子: 融合计数 kernel】
            # - 时间: 约 0.1ms
            num_zeros_in_grad += (grad.data == 0).sum()

            # === Adam 更新 ======
            if self.config.optimizer == 'adam':
                # ====== 一阶矩 (动量) 更新 ======
                # - 公式: m = beta1 * m + (1 - beta1) * grad
                # - 【CUDA 算子: 融合 axpy kernel】
                # - 时间: 约 0.2ms
                exp_avg = param_group['exp_avg']
                # m = m * beta1
                exp_avg.mul_(self.config.beta1)
                # m = m + grad * (1 - beta1)
                exp_avg.add_(grad, alpha=1 - self.config.beta1)

                # ====== 二阶矩 (自适应学习率) 更新 ======
                # - 公式: v = beta2 * v + (1 - beta2) * grad^2
                # - 【CUDA 算子: 融合 axpy + square kernel】
                # - 时间: 约 0.2ms
                exp_avg_sq = param_group['exp_avg_sq']
                exp_avg_sq.mul_(self.config.beta2)
                # v = v + grad * grad * (1 - beta2)
                exp_avg_sq.addcmul_(grad, grad, value=1 - self.config.beta2)

                # ====== 偏差校正 ======
                # - 公式: m_hat = m / (1 - beta1^t)
                # - 【算子: CPU 标量运算】
                # - 时间: ~0 (标量)
                bias_correction1 = 1 - self.config.beta1 ** step
                bias_correction2 = 1 - self.config.beta2 ** step

                # ====== 参数更新 ======
                # - 公式: param = param - lr * m_hat / (sqrt(v_hat) + eps)
                # - 【CUDA 算子: 融合 sqrt + div + axpy kernel】
                # - 时间: 约 0.3ms
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(self.config.eps)
                step_size = param_group['lr'] / bias_correction1

                # param = param - lr * m / denom
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

    # === 计算总梯度范数 ===
    # ====== 算子: sqrt ======
    # - 【算子: CPU 标量运算】
    grad_norm = math.sqrt(grad_norm)

    return update_successful, grad_norm, num_zeros_in_grad
```

### 4.2 优化器更新算子详解

```python
# ====== Adam 算子详解 ======
"""
Adam 优化器的数学公式:

对于 70B 模型，单个参数的更新:

1. 一阶矩更新 (m - Momentum):
   m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
   - 操作: mul + add
   - FLOPs: 2n (n = 参数数量)
   - 时间: ~0.2ms (并行度高)

2. 二阶矩更新 (v - Adaptive Learning Rate):
   v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
   - 操作: mul + square + add
   - FLOPs: 3n
   - 时间: ~0.2ms

3. 偏差校正 (Bias Correction):
   m̂_t = m_t / (1 - β₁^t)
   v̂_t = v_t / (1 - β₂^t)
   - 操作: div (标量)
   - FLOPs: 2n
   - 时间: ~0.1ms

4. 参数更新 (Parameter Update):
   θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
   - 操作: sqrt + add + div + mul
   - FLOPs: 4n
   - 时间: ~0.3ms

总计:
- FLOPs: 11n ≈ 11 × 280B = 3.08T FLOPs
- 时间: 约 100ms (内存带宽限制，而非计算)
- 瓶颈: 参数读写 (需要加载完整参数)

超参数 (典型值):
- β₁ = 0.9 (一阶矩衰减率)
- β₂ = 0.999 (二阶矩衰减率)
- ε = 1e-8 (数值稳定项)
- α = 1e-4 (学习率)
"""
```

---

## 步骤 5: 学习率更新

```python
# megatron/core/optimizer_param_scheduler.py
def step(self, increment):
    """
    学习率调度器更新

    ====== 算子分析 ======
    涉及算子: 基础算术运算 (CPU)
    - 时间: ~0.01ms (标量运算，可忽略)
    """
    # === 更新计数器 ======
    # 【算子: 加法】
    self.num_steps_taken += increment

    # === 计算学习率 ======
    if self.lr_decay_style == 'cosine':
        if self.num_steps_taken < self.lr_warmup_steps:
            # ====== 预热阶段: 线性增长 ======
            # 公式: lr(t) = lr_{min} + (lr_{max} - lr_{min}) · t / T_{warmup}
            # 【算子: mul + add】
            lr = self.init_lr + (self.max_lr - self.init_lr) * (
                self.num_steps_taken / self.lr_warmup_steps
            )
        else:
            # ====== 余弦衰减 ======
            # 公式: lr(t) = lr_{min} + 0.5 · (lr_{max} - lr_{min}) · (1 + cos(π · (t - T_{warmup}) / (T_{max} - T_{warmup})))
            # 【算子: cos + mul + add】
            progress = (
                self.num_steps_taken - self.lr_warmup_steps
            ) / (self.lr_decay_steps - self.lr_warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
    elif self.lr_decay_style == 'inverse-square-root':
        # ====== 倒平方根衰减 ======
        # 公式: lr(t) = lr_{max} / √(max(t, T_{warmup}))
        # 【算子: sqrt + div】
        lr = self.max_lr / math.sqrt(max(self.num_steps_taken, self.warmup_steps))

    # === 更新优化器学习率 ======
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
```

---

## 关键优化技术详解（算子级别）

### 1. 梯度累积融合

```python
# ====== 标准 PyTorch 方式 ======
"""
grad_weight = 0
for microbatch in microbatches:
    # 【算子 1: GEMM】计算梯度
    grad = microbatch_input.T @ microbatch_grad
    # FLOPs: M × N × K
    # 时间: 约 2ms

    # 【算子 2: 加法】累积
    grad_weight += grad
    # FLOPs: N × K
    # 时间: 约 0.1ms

总计 (4 个微批次):
- GEMM: 4 × 2ms = 8ms
- 加法: 4 × 0.1ms = 0.4ms
- 总计: 8.4ms
"""

# ====== 融合方式 ======
"""
# 【融合算子: wgrad_gemm_accum_fp32】
fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
    microbatch_input,   # [M, K]
    microbatch_grad,   # [M, N]
    grad_buffer,        # [N, K], 直接累积
)

# 内部实现:
for microbatch in microbatches:
    # 【融合 GEMM + 累加】
    # grad_buffer = grad_buffer + input.T @ grad
    # 一次 kernel 调用完成 GEMM 和累积
    cublasGemmEx(
        transa='T',
        transb='N',
        m=K,
        n=N,
        M=M,
        alpha=1.0,
        A=microbatch_input,
        B=microbatch_grad,
        beta=1.0,  # 关键: 累加而非覆盖
        C=grad_buffer,
    )

总计 (4 个微批次):
- 融合 GEMM: 4 × 2ms = 8ms
- 无额外加法
- 总计: 8ms

优势:
1. 减少内存访问 (grad_buffer 保持在寄存器/cache)
2. 减少 kernel 启动开销 (1 个 vs 2 个)
3. 提高 cache 利用率
"""

# ====== 性能分析 ======
"""
内存访问分析 (单个微批次):
- 标准: 读取 input, grad, 读取 grad_weight, 写回 grad_weight
  - 内存访问: 2 × M×K + 2 × M×N + 2 × N×K
  - 对于 [2048, 8192] × [8192, 8192]:
    = 32MB + 32MB + 128MB = 192MB

- 融合: 读取 input, grad, 写回 grad_buffer
  - 内存访问: 2 × M×K + 2 × M×N
  - = 64MB (减少 67%)

Cache 利用率:
- 标准: grad_weight 每次都要从 L2/内存加载
- 融合: grad_buffer 保持在 L1/L2 cache
"""
```

### 2. 异步梯度归约

```python
# ====== 同步 All-Reduce ======
"""
def allreduce_sync(grad_input, tp_group):
    # 【算子: All-Reduce (阻塞)】
    # - CPU 等待 GPU 完成
    # - 时间: 0.5ms (通信) + 等待时间
    torch.distributed.all_reduce(grad_input, group=tp_group)

    # 【算子: GEMM】计算权重梯度
    grad_weight = grad_output.t().matmul(total_input)
    # 时间: 2ms

    # 总时间: 0.5ms (等待) + 0.5ms (通信) + 2ms (计算) = 3ms
"""

# ====== 异步 All-Reduce ======
"""
def allreduce_async(grad_input, tp_group):
    # 【算子 1: matmul - 计算输入梯度】
    grad_input = grad_output.matmul(weight)
    # 时间: 1ms

    # 【算子 2: All-Reduce (异步)】
    # - CPU 不等待，立即返回
    # - GPU 后台执行
    # - 时间: ~0 (CPU 侧)
    handle = torch.distributed.all_reduce(
        grad_input,
        group=tp_group,
        async_op=True  # 关键: 异步
    )

    # 【算子 3: GEMM - 计算权重梯度】
    # 与 All-Reduce 并行执行！
    grad_weight = grad_output.t().matmul(total_input)
    # 时间: 2ms

    # 【算子 4: wait】等待 All-Reduce 完成
    handle.wait()
    # 时间: max(0, 2ms - 0.5ms) = 1.5ms (已经完成大部分)

    # 总时间: 1ms + max(0.5ms, 2ms) = 3ms (理论上相同)
    #
    # 实际优势:
    # - 如果网络快 (0.2ms): 1ms + max(0.2ms, 2ms) = 3ms
    # - 如果计算快 (1ms): 1ms + max(0.5ms, 1ms) = 2ms (节省 33%)

    # 最佳情况:
    # - All-Reduce 与计算完全重叠
    # - 时间: max(1ms + 0.5ms, 2ms) = 2ms
"""

# ====== 时间线对比 ======
"""
同步:
|----- 计算 grad_input -----|--- All-Reduce ---|----- 计算 grad_weight -----|
                              ^^^^^ 串行 ^^^^^
总时间: 1ms + 0.5ms + 2ms = 3.5ms

异步:
|----- 计算 grad_input -----|--- All-Reduce (后台) ---|
                              |----- 计算 grad_weight -----|--- wait ---|
总时间: max(1ms + 0.5ms, 2ms) + 0.1ms = 2.6ms (节省 26%)
"""
```

### 3. Flash Attention

```python
# ====== 标准 Attention ======
"""
def standard_attention(q, k, v, attention_mask):
    # ====== 算子 1: QK^T - GEMM ======
    # - 输入: q [B, H, S, D], k [B, H, S, D]
    # - 输出: attn_scores [B, H, S, S]
    # - 公式: S_{ij} = Q_i · K_j^T / √d_k
    # - 【CUDA 算子: cublasGemmBatched】
    # - 时间: 约 1ms
    # - 内存: B × H × S × S (2048×2048 = 4M 元素)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    # 对于 [2, 32, 2048, 128]:
    # 内存: 2 × 32 × 2048 × 2048 × 2 bytes = 512 MB

    # ====== 算子 2: Scale ======
    attn_scores = attn_scores / math.sqrt(q.size(-1))

    # ====== 算子 3: Add mask ======
    attn_scores = attn_scores + attention_mask

    # ====== 算子 4: Softmax ======
    # - 公式: A_{ij} = exp(S_{ij}) / Σ_k exp(S_{ik})
    # - 【CUDA 算子: softmax kernel】
    # - 时间: 约 0.5ms
    attn_weights = F.softmax(attn_scores, dim=-1)

    # ====== 算子 5: Dropout ======
    attn_weights = F.dropout(attn_weights, p=dropout, training=training)

    # ====== 算子 6: Attention × V - GEMM ======
    # - 输入: attn_weights [B, H, S, S], v [B, H, S, D]
    # - 输出: output [B, H, S, D]
    # - 公式: O_i = Σ_j A_{ij} · V_j
    # - 【CUDA 算子: cublasGemmBatched】
    # - 时间: 约 1ms
    output = torch.matmul(attn_weights, v)

    # 总计:
    # - 时间: 1ms + 0.5ms + 0.1ms + 0.5ms + 1ms = 3.1ms
    # - 内存峰值: 512 MB (attn_scores + attn_weights)
"""

# ====== Flash Attention ======
"""
def flash_attention(q, k, v, attention_mask):
    # ====== 融合算子: FlashAttention ======
    # - 【CUDA 算子: 融合 attention kernel】
    # - 优化: 分块计算，避免存储完整的 S×S 矩阵
    # - 内存: O(S × block_size) vs O(S²)
    # - 时间: 约 1.5ms (更快)

    # 内部实现 (分块):
    for block in split_sequence_into_blocks(seq_len, block_size=128):
        # 1. 计算 Q @ K^T (仅当前 block)
        block_scores = q @ k[block].T  # [B, H, block, D]

        # 2. Softmax (仅当前 block)
        block_weights = softmax(block_scores)

        # 3. @ V (仅当前 block)
        block_output = block_weights @ v[block]

        # 4. 累积到输出
        output[block] = block_output

    # 总计:
    # - 时间: 1.5ms (约 2x 加速)
    # - 内存: 32 MB (减少 94%)

    # 优势:
    # 1. 内存访问模式更好 (cache friendly)
    # 2. 融合 kernel，减少 kernel 启动
    # 3. 数值稳定性更好 (在线 softmax)
"""

# ====== 性能对比 ======
"""
标准 Attention vs Flash Attention (2048 序列):

指标             | 标准        | Flash      | 提升
----------------|------------|------------|------
时间            | 3.1ms      | 1.5ms      | 2.1x
内存 (S×S)      | 512 MB     | 32 MB      | 16x
HBM 带宽        | 165 GB/s   | 85 GB/s    | 2x
Cache 命中率     | 60%        | 95%        | 1.6x
"""
```

### 4. Activation Checkpointing

```python
# ====== 无 Checkpointing ======
"""
def forward_no_checkpointing(hidden_states, layers):
    """
    前向传播: 保存所有激活值
    """
    activations = []

    for layer in layers:
        # ====== 算子: LayerNorm ======
        hidden_states = layer.layer_norm(hidden_states)
        # 【内存操作: 保存激活值】
        activations.append(hidden_states.detach())  # 保存

        # ====== 算子: Attention ======
        hidden_states = layer.attention(hidden_states)
        activations.append(hidden_states.detach())  # 保存

        # ====== 算子: MLP ======
        hidden_states = layer.mlp(hidden_states)
        activations.append(hidden_states.detach())  # 保存

    # 内存占用:
    # - 每层: 2 × seq_len × batch × hidden_size
    # - 80 层: 80 × 2 × 2048 × 2 × 8192 × 2 bytes = 53 GB

    return hidden_states, activations

def backward_no_checkpointing(activations, grad_output):
    """
    反向传播: 直接使用保存的激活值
    """
    for layer, activation in zip(reversed(layers), reversed(activations)):
        # 【算子: 反向传播】使用保存的 activation
        grad_activation = layer.backward(grad_output, activation=activation)
        # 时间: ~2ms (无需重算)
"""

# ====== 有 Checkpointing ======
"""
def forward_with_checkpointing(hidden_states, layers):
    """
    前向传播: 只保存部分激活值
    """
    checkpointed_layers = []

    for i, layer in enumerate(layers):
        # ====== 算子: LayerNorm ======
        hidden_states = layer.layer_norm(hidden_states)

        # ====== 算子: Attention ======
        hidden_states = layer.attention(hidden_states)

        # ====== 算子: MLP ======
        hidden_states = layer.mlp(hidden_states)

        # 【内存操作: 每 N 层 checkpoint 一次】
        if i % checkpoint_interval == 0:
            checkpointed_layers.append((i, hidden_states.detach()))

    # 内存占用 (每 10 层 checkpoint 一次):
    # - 只保存: 8 个 checkpoint
    # - 内存: 8 × 2 × 2048 × 2 × 8192 × 2 bytes = 5.3 GB (减少 90%)

    return hidden_states, checkpointed_layers

def backward_with_checkpointing(checkpointed_layers, grad_output):
    """
    反向传播: 重算未保存的激活值
    """
    # 从最后一个 checkpoint 开始
    for i, (checkpoint_idx, checkpoint_activation) in enumerate(reversed(checkpointed_layers)):
        # 【算子: 重算前向】
        # 重新计算从 checkpoint_idx 到下一个 checkpoint 的所有层
        hidden_states = checkpoint_activation
        for layer in layers[checkpoint_idx:]:
            # ====== 重算 LayerNorm ======
            hidden_states = layer.layer_norm(hidden_states)
            # ====== 重算 Attention ======
            hidden_states = layer.attention(hidden_states)
            # ====== 重算 MLP ======
            hidden_states = layer.mlp(hidden_states)

        # 【算子: 反向传播】
        grad_activation = layer.backward(grad_output, activation=hidden_states)
        # 时间: ~2ms (反向) + ~3ms (重算) = 5ms

# ====== 权衡分析 ======
"""
无 Checkpointing:
- 前向: 1370ms
- 反向: 4500ms
- 总计: 5870ms
- 内存: 53 GB

有 Checkpointing (每 10 层):
- 前向: 1370ms (相同)
- 反向: 4500ms + 9 × 10层 × 3ms (重算) = 7200ms
- 总计: 8570ms
- 内存: 5.3 GB

权衡:
- 时间增加: 46%
- 内存减少: 90%
- 适用场景: 内存受限时使用
"""
```

---

## 性能分析（算子级别）

### 单步时间分解

```python
"""
单步训练时间 = T_forward + T_backward + T_optimizer + T_comm

典型时间分布 (70B 模型, 8×8 GPU, TP=4, PP=2):

阶段                  | 时间    | 占比   | 主要算子
---------------------|---------|--------|------------------------
前向传播             | 1370ms  | 23%    | GEMM (70%), Softmax (15%)
反向传播             | 4500ms  | 75%    | GEMM (80%), 其他 (20%)
梯度同步 (DP)        | 50ms    | 1%     | All-Reduce (100%)
优化器更新           | 100ms   | 2%     | 逐元素 (100%)
学习率调度           | <1ms    | 0%     | CPU 运算
---------------------|---------|--------|------------------------
总计                 | 6020ms  | 100%   |

算子时间细分 (前向):
- GEMM: 960ms (70%)
- Softmax + Attention: 200ms (15%)
- LayerNorm: 80ms (6%)
- 通信 (TP): 80ms (6%)
- 其他: 50ms (3%)

算子时间细分 (反向):
- GEMM: 4000ms (89%)
- Softmax 反向: 240ms (5%)
- LayerNorm 反向: 80ms (2%)
- 通信 (TP): 80ms (2%)
- 其他: 100ms (2%)
"""
```

### 算子 FLOPs 分析

```python
"""
70B 模型单步 FLOPs 分析 (TP=4):

前向传播:
- QKV Projection: 80 × 2048 × 2 × 8192 × 6144 × 2 = 33T FLOPs
- Attention (QK^T + V@attn): 80 × 2 × 32 × 2048 × 2048 × 128 × 2 = 17T FLOPs
- Output Projection: 80 × 2048 × 2 × 8192 × 2048 × 2 = 11T FLOPs
- MLP (fc1 + fc2 + proj): 80 × 2048 × 2 × 8192 × 8192 × 2 × 3 = 66T FLOPs
- LayerNorm: 160 × 2048 × 2 × 8192 × 5 = 3T FLOPs
- Embedding: 2048 × 2 × 8192 = 0.03T FLOPs
- Output Layer: 2048 × 2 × 8192 × 24000 × 2 = 1.6T FLOPs
前向总计: 131T FLOPs

反向传播 (约为前向 2x):
- GEMM 梯度: 131T × 2 = 262T FLOPs
- Softmax 反向: 17T × 1.5 = 26T FLOPs
- LayerNorm 反向: 3T × 2 = 6T FLOPs
反向总计: 294T FLOPs

单步总计: 425T FLOPs

实际性能:
- 理论 FLOPS (H100): 67 TFLOPS (FP16)
- 实际时间: 6 秒
- 实际 FLOPS: 425T / 6s = 71 TFLOPS
- MFU (Model FLOP Utilization): 71 / 67 = 106% (因为 Tensor Core 加速)

优化后的性能:
- Flash Attention: 减少 10% 时间 → 5.4s → 79 TFLOPS
- 异步通信: 减少 5% 时间 → 5.1s → 83 TFLOPS
- 梯度融合: 减少 5% 时间 → 4.8s → 89 TFLOPS
- FP8: 减少 50% 时间 → 2.4s → 177 TFLOPS (但 H100 FP8 理论 2000 TFLOPS)
"""
```

---

## 总结

### 单步训练算子清单（含数学公式）

```
【前向传播】
1. Embedding lookup: F.embedding
   公式: h_i = E[token_i]

2. LayerNorm: F.layer_norm
   公式: LN(x) = γ · (x - μ) / √(σ² + ε) + β
   其中: μ = (1/n)Σx_i, σ² = (1/n)Σ(x_i - μ)²

3. QKV Projection: cublasGemmEx (3个并行)
   公式: [Q, K, V] = X · [W_Q, W_K, W_V]^T

4. RoPE: 逐元素操作
   公式: RoPE(x, m) = [x_{2i}cos(mθ_i) - x_{2i+1}sin(mθ_i), x_{2i}sin(mθ_i) + x_{2i+1}cos(mθ_i)]

5. QK^T: cublasGemmBatched
   公式: S_{ij} = Q_i · K_j^T / √d_k

6. Scale: 逐元素 mul
   公式: S' = S / √d_k

7. Add mask: 逐元素 add
   公式: S''_{ij} = S'_{ij} + M_{ij}, M_{ij} = {0 if i≥j, -∞ if i<j}

8. Softmax: 优化 softmax kernel
   公式: A_{ij} = exp(S''_{ij}) / Σ_k exp(S''_{ik})

9. Dropout: 随机 mask
   公式: y = x · Bernoulli(p) / (1-p)

10. Attention × V: cublasGemmBatched
    公式: O_i = Σ_j A_{ij} · V_j

11. All-Reduce (TP): ncclAllReduce
    公式: y_i^{(k)} = Σ_{r=1}^{TP} x_i^{(r)}

12. Output Projection: cublasGemmEx
    公式: Y = X · W_O^T

13. MLP fc1 (gate): cublasGemmEx
    公式: gate = X · W_gate^T

14. MLP fc2 (up): cublasGemmEx
    公式: up = X · W_up^T

15. Activation: SwiGLU (逐元素)
    公式: SwiGLU = SiLU(gate) ⊙ up = (gate / (1 + e^{-gate})) ⊙ up

16. MLP proj: cublasGemmEx
    公式: Y = act · W_down^T

17. All-Reduce (TP): ncclAllReduce
    同步骤 11

18. Final LayerNorm: F.layer_norm
    同步骤 2

19. Output Layer: cublasGemmEx
    公式: logits = H · W_vocab^T

20. All-Gather (可选): ncclAllGather
    公式: y^{(k)} = [x^{(1)}, x^{(2)}, ..., x^{(TP)}]

21. CrossEntropy: 融合 log_softmax + nll_loss
    公式: CE = -log(Softmax(logit_{target})) = log(Σ_j e^{logit_j}) - logit_{target}

【反向传播】
1. CrossEntropy 反向: 融合 kernel
   公式: ∂CE/∂y_i = Softmax(y)_i - 1[i = target]

2. Output Layer 反向: cublasGemmEx × 2
   公式: ∂CE/∂W = (∂CE/∂y) · x^T
         ∂CE/∂x = (∂CE/∂y) · W

3. MLP 反向: cublasGemmEx × 6
   包含 gate, up, down 三个投影的正向和反向梯度

4. Attention 反向: cublasGemmEx × 6 + softmax 反向
   公式: ∂Softmax_i/∂x_j = Softmax_i · (δ_{ij} - Softmax_j)

5. LayerNorm 反向: 融合 kernel
   公式涉及均值、方差的雅可比矩阵

6. All-Reduce (TP): ncclAllReduce
   同前向步骤 11

7. Embedding 反向: index_add
   公式: ∂CE/∂E[token] += ∂CE/∂h_i

【梯度同步 - DP】
1. All-Reduce / Reduce-Scatter: ncclAllReduce / ncclReduceScatter
   公式 (All-Reduce): y_i^{(k)} = Σ_{r=1}^{DP} grad_i^{(r)}
   公式 (Reduce-Scatter): y_i^{(k)} = Σ_{r=1}^{DP} grad_{k·DP+i}^{(r)}

【优化器更新 - Adam】
1. 梯度裁剪: 融合 norm + clip kernel
   公式: g' = g · clip / ||g||_2 if ||g||_2 > clip

2. Adam 更新: 融合 axpy + sqrt + div kernel
   公式: m_t = β₁·m_{t-1} + (1-β₁)·g_t
         v_t = β₂·v_{t-1} + (1-β₂)·g_t²
         m̂_t = m_t / (1-β₁^t)
         v̂_t = v_t / (1-β₂^t)
         θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)

【学习率调度 - Cosine】
1. 预热阶段: lr(t) = lr_{min} + (lr_{max} - lr_{min}) · t / T_{warmup}
2. 衰减阶段: lr(t) = lr_{min} + 0.5 · (lr_{max} - lr_{min}) · (1 + cos(π·(t-T_{warmup})/(T_{max}-T_{warmup})))
```

### 性能优化建议

```
1. 计算:
   - 使用 Flash Attention (2x 加速)
   - 使用 FP8 (H100 上 3-5x 加速)
   - 融合 kernel (减少 10-20% 时间)

2. 通信:
   - 使用 Reduce-Scatter (减少 50% 通信)
   - 异步通信 (减少 5-10% 时间)
   - 梯度压缩 (极端情况)

3. 内存:
   - Activation Checkpointing (减少 90% 内存)
   - 序列并行 (减少 TP× 内存)
   - 梯度累积融合 (减少内存访问)

4. 吞吐量:
   - 增加 micro_batch_size (提高 GPU 利用率)
   - 减少同步点 (流水线并行)
   - CUDA Graph (推理时有效)
```
