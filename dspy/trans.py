import numpy as np
import math


def masked_multi_head_self_attention(input_str):
    """
    实现Masked Multi-Head Self-Attention

    这个函数实现了Transformer中的多头自注意力机制，包含以下7个步骤：
    1. 生成Q, K, V矩阵
    2. 拆分Q, K, V为多个头
    3. 计算注意力分数
    4. 应用掩码
    5. 应用Softmax得到注意力权重
    6. 计算注意力输出
    7. 拼接多头输出并通过线性层投影

    Args:
        input_str: 输入字符串，格式为 "num_heads;X;W_Q;W_K;W_V;W_O"
                  - num_heads: 多头数量
                  - X: 输入序列 [batch_size, seq_len, d_model]
                  - W_Q, W_K, W_V: Query, Key, Value的权重矩阵
                  - W_O: 输出投影矩阵

    Returns:
        List: 最终输出结果，保留两位小数
    """
    # 解析输入
    parts = input_str.strip().rstrip(';').split(';')

    num_heads = int(parts[0])
    X = eval(parts[1])  # 输入序列
    Q_weights = eval(parts[2])  # W_Q权重矩阵
    K_weights = eval(parts[3])  # W_K权重矩阵
    V_weights = eval(parts[4])  # W_V权重矩阵
    W_O = eval(parts[5])  # 输出投影矩阵

    # 转换为numpy数组
    X = np.array(X, dtype=np.float64)
    Q_weights = np.array(Q_weights, dtype=np.float64)
    K_weights = np.array(K_weights, dtype=np.float64)
    V_weights = np.array(V_weights, dtype=np.float64)
    W_O = np.array(W_O, dtype=np.float64)

    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads

    # 第1步：生成Q, K, V矩阵
    Q = np.matmul(X, Q_weights)  # [batch_size, seq_len, d_model]
    K = np.matmul(X, K_weights)  # [batch_size, seq_len, d_model]
    V = np.matmul(X, V_weights)  # [batch_size, seq_len, d_model]

    # 第2步：拆分Q, K, V为多个头
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, d_k]
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, d_k]
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, d_k]

    # 第3步：计算注意力分数
    attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(
        d_k)  # [batch_size, num_heads, seq_len, seq_len]

    # 第4步：应用掩码（这里创建下三角掩码，防止未来信息泄露）
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)  # 上三角掩码
    mask = np.broadcast_to(mask, (batch_size, num_heads, seq_len, seq_len))

    masked_scores = attention_scores.copy()
    masked_scores[mask] = -np.inf

    # 第5步：应用Softmax得到注意力权重
    softmax_scores = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    softmax_scores = softmax_scores / np.sum(softmax_scores, axis=-1, keepdims=True)

    # 处理NaN值（当整行都是-inf时）
    softmax_scores = np.nan_to_num(softmax_scores, nan=0.0)

    # 第6步：计算注意力输出
    attention = np.matmul(softmax_scores, V)  # [batch_size, num_heads, seq_len, d_k]

    # 第7步：拼接多头输出并通过线性层投影
    attention = attention.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)  # [batch_size, seq_len, d_model]
    output = np.matmul(attention, W_O)  # [batch_size, seq_len, d_model]

    # 转换为List并保留两位小数
    output_list = np.round(output, 2).tolist()

    return output_list


# 测试样例
if __name__ == "__main__":
    # 样例1
    input1 = "2;[[[1.92, 1.48], [0.67, -1.23], [0.35, -0.68]],[[-1.11, 0.09], [-0.3, -0.39], [-0.59, -0.06]]];[[1.0, 2.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];"
    result1 = masked_multi_head_self_attention(input1)
    print("样例1结果:", result1)

    # 样例2
    input2 = "2;[[[1.92, 1.48], [0.67, -1.23], [0.35, -0.68]],[[-1.11, 0.09], [-0.3, -0.39], [-0.59, -0.06]]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];[[1.0, 1.0], [2.0, 2.0]];"
    result2 = masked_multi_head_self_attention(input2)
    print("样例2结果:", result2)