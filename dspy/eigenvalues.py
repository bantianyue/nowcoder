import numpy as np

def calculate_eigenvalues(matrix):
    """
    计算2×2矩阵的特征值
    
    参数:
    matrix: 2×2的二维数组（矩阵）
    
    返回:
    list: 包含两个特征值的列表，按从大到小排序
    """
    # 将输入转换为numpy数组
    matrix = np.array(matrix, dtype=float)
    
    # 使用numpy的linalg.eigvals计算特征值
    eigenvalues = np.linalg.eigvals(matrix)
    
    # 将特征值按从大到小排序
    eigenvalues = sorted(eigenvalues, key=lambda x: (x.real, x.imag), reverse=True)
    
    # 如果特征值是实数，则去掉虚部的0
    eigenvalues = [float(x.real) if abs(x.imag) < 1e-10 else x for x in eigenvalues]
    
    return eigenvalues

def main():
    # 读取输入
    matrix = eval(input().strip())
    
    # 计算特征值
    result = calculate_eigenvalues(matrix)
    
    # 格式化输出
    # 如果结果是复数，保持复数形式；如果是实数，转换为一位小数
    formatted_result = []
    for val in result:
        if isinstance(val, complex):
            formatted_result.append(complex(round(val.real, 1), round(val.imag, 1)))
        else:
            formatted_result.append(round(float(val), 1))
    
    print(formatted_result)

if __name__ == "__main__":
    main()