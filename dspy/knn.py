import sys
import math
from collections import Counter

def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def knn_classify(k, test_sample, training_samples, training_labels):
    """KNN分类算法实现"""
    # 计算测试样本与所有训练样本的距离
    distances = []
    for i, sample in enumerate(training_samples):
        dist = calculate_distance(test_sample, sample)
        distances.append((dist, i))
    
    # 按距离排序
    distances.sort()
    
    # 获取前k个最近邻居的标签
    k_nearest_labels = [training_labels[idx] for _, idx in distances[:k]]
    
    # 统计标签出现次数
    label_counts = Counter(k_nearest_labels)
    most_common = label_counts.most_common()
    
    # 如果有并列第一，选择距离最近的那个标签
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # 找出并列第一的标签
        max_count = most_common[0][1]
        tied_labels = [label for label, count in most_common if count == max_count]
        
        # 在k个最近邻居中，找出属于tied_labels的最近邻居
        for dist, idx in distances:
            if training_labels[idx] in tied_labels:
                return training_labels[idx], label_counts[training_labels[idx]]
    
    # 没有并列，直接返回出现次数最多的标签
    most_common_label = most_common[0][0]
    return most_common_label, label_counts[most_common_label]

def main():
    # 从标准输入读取数据
    k, n, d, c = map(int, input().strip().split())
    
    # 读取待分类样本
    test_sample = list(map(float, input().strip().split()))
    
    # 读取训练样本
    training_samples = []
    training_labels = []
    for _ in range(n):
        data = list(map(float, input().strip().split()))
        training_samples.append(data[:d])
        training_labels.append(data[d])
    
    # 使用KNN算法进行分类
    label, count = knn_classify(k, test_sample, training_samples, training_labels)
    
    # 输出结果
    print(f"{int(label)} {count}")

if __name__ == "__main__":
    main()