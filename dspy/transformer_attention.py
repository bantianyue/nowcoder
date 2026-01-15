import numpy as np

def masked_multi_head_attention(num_heads, X, W_Q, W_K, W_V, W_O, mask=None):
    """
    实现Masked Multi-Head Self-Attention
    
    参数:
    num_heads: 注意力头的数量
    X: 输入序列，形状为[batch_size, seq_len, d_model]
    W_Q, W_K, W_V: 权重矩阵，用于生成Q, K, V
    W_O: 输出投影矩阵
    mask: 掩码矩阵，形状为[batch_size, seq_len, seq_len]，True表示需要掩码的位置
    
    返回:
    output: 多头注意力的输出，形状为[batch_size, seq_len, d_model]
    """
    batch_size, seq_len, d_model = np.array(X).shape
    
    # 步骤1: 生成Q, K, V矩阵
    Q = np.matmul(X, W_Q)  # [batch_size, seq_len, d_model]
    K = np.matmul(X, W_K)  # [batch_size, seq_len, d_model]
    V = np.matmul(X, W_V)  # [batch_size, seq_len, d_model]
    
    # 步骤2: 拆分Q, K, V为多个头
    d_k = d_model // num_heads
    
    # 重塑为[batch_size, num_heads, seq_len, d_k]
    Q_reshaped = np.reshape(Q, (batch_size, seq_len, num_heads, d_k))
    Q_reshaped = np.transpose(Q_reshaped, (0, 2, 1, 3))  # [batch_size, num_heads, seq_len, d_k]
    
    K_reshaped = np.reshape(K, (batch_size, seq_len, num_heads, d_k))
    K_reshaped = np.transpose(K_reshaped, (0, 2, 1, 3))  # [batch_size, num_heads, seq_len, d_k]
    
    V_reshaped = np.reshape(V, (batch_size, seq_len, num_heads, d_k))
    V_reshaped = np.transpose(V_reshaped, (0, 2, 1, 3))  # [batch_size, num_heads, seq_len, d_k]
    
    # 步骤3: 计算注意力分数
    # K_reshaped的转置: [batch_size, num_heads, d_k, seq_len]
    K_transposed = np.transpose(K_reshaped, (0, 1, 3, 2))
    
    # 计算注意力分数: [batch_size, num_heads, seq_len, seq_len]
    attention_scores = np.matmul(Q_reshaped, K_transposed) / np.sqrt(d_k)
    
    # 步骤4: 应用掩码
    if mask is not None:
        # 扩展mask维度以适应多头
        expanded_mask = np.expand_dims(mask, axis=1)  # [batch_size, 1, seq_len, seq_len]
        # 将掩码位置设为负无穷
        attention_scores = np.where(expanded_mask, -1e9, attention_scores)
    
    # 步骤5: 应用Softmax得到注意力权重
    # 为了数值稳定性，先减去最大值
    max_scores = np.max(attention_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(attention_scores - max_scores)
    softmax_scores = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 步骤6: 计算注意力输出
    attention_output = np.matmul(softmax_scores, V_reshaped)  # [batch_size, num_heads, seq_len, d_k]
    
    # 步骤7: 拼接多头输出并通过线性层投影
    # 转置回[batch_size, seq_len, num_heads, d_k]
    attention_output = np.transpose(attention_output, (0, 2, 1, 3))
    
    # 重塑为[batch_size, seq_len, d_model]
    concat_attention = np.reshape(attention_output, (batch_size, seq_len, d_model))
    
    # 通过线性层投影
    output = np.matmul(concat_attention, W_O)
    
    # 保留两位小数
    output = np.round(output, 2)
    
    return output.tolist()

def parse_input(input_str):
    """解析输入字符串"""
    parts = input_str.strip().split(';')
    
    num_heads = int(parts[0])
    X = eval(parts[1])
    W_Q = eval(parts[2])
    W_K = eval(parts[3])
    W_V = eval(parts[4])
    W_O = eval(parts[5])
    
    return num_heads, X, W_Q, W_K, W_V, W_O

def main():
    # 读取输入
    input_str = input().strip()
    
    # 解析输入
    num_heads, X, W_Q, W_K, W_V, W_O = parse_input(input_str)
    
    # 计算多头注意力
    output = masked_multi_head_attention(num_heads, X, W_Q, W_K, W_V, W_O)
    
    # 输出结果
    print(output)

if __name__ == "__main__":
    main()