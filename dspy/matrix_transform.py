import numpy as np

def transform_matrix(A, T, S):
    """
    使用 T^(-1)AS 转换矩阵 A
    
    参数:
    A: 待转换的矩阵
    T: 转换矩阵T
    S: 转换矩阵S
    
    返回:
    转换后的矩阵，如果无解则返回-1
    """
    try:
        # 将输入转换为numpy数组
        A = np.array(A, dtype=float)
        T = np.array(T, dtype=float)
        S = np.array(S, dtype=float)
        
        # 检查矩阵T和S是否可逆
        T_det = np.linalg.det(T)
        S_det = np.linalg.det(S)
        
        if abs(T_det) < 1e-10 or abs(S_det) < 1e-10:
            return -1
        
        # 计算T的逆矩阵
        T_inv = np.linalg.inv(T)
        
        # 执行矩阵转换 T^(-1)AS
        result = T_inv @ A @ S
        
        # 将结果四舍五入到3位小数
        result = np.round(result, decimals=3)
        
        # 转换为嵌套列表并确保没有负零
        result_list = result.tolist()
        for i in range(len(result_list)):
            for j in range(len(result_list[i])):
                # 处理负零的情况
                if abs(result_list[i][j]) < 1e-10:
                    result_list[i][j] = 0.0
                # 确保显示整数为浮点数
                result_list[i][j] = float(result_list[i][j])
        
        return result_list
        
    except Exception as e:
        return -1

def main():
    # 读取输入
    A = eval(input().strip())
    T = eval(input().strip())
    S = eval(input().strip())
    
    # 执行矩阵转换
    result = transform_matrix(A, T, S)
    
    # 输出结果
    print(result)

if __name__ == "__main__":
    main()