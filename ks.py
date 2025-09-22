def solve_tokenization():
    # 读取输入
    text = input().strip()
    n = int(input())
    
    # 读取词汇表和置信度分数
    vocab = {}
    for _ in range(n):
        word, score = input().split()
        vocab[word] = int(score)
    
    m = int(input())
    
    # 读取转移分数表
    transitions = {}
    for _ in range(m):
        from_word, to_word, bonus = input().split()
        transitions[(from_word, to_word)] = int(bonus)
    
    # 动态规划求解
    # dp[i][j] 表示从位置i到位置j的最大分数
    # 但这样会超时，我们需要优化
    
    # 使用记忆化递归
    memo = {}
    
    def dfs(start):
        if start in memo:
            return memo[start]
        
        if start == len(text):
            return 0
        
        max_score = -float('inf')
        
        # 尝试所有可能的词元
        for end in range(start + 1, len(text) + 1):
            word = text[start:end]
            if word in vocab:
                # 当前词元的分数
                current_score = vocab[word]
                
                # 递归计算剩余部分的分数
                remaining_score = dfs(end)
                
                # 计算转移分数
                transition_bonus = 0
                if end < len(text):
                    # 找到下一个词元来计算转移分数
                    for next_end in range(end + 1, len(text) + 1):
                        next_word = text[end:next_end]
                        if next_word in vocab:
                            transition_key = (word, next_word)
                            if transition_key in transitions:
                                transition_bonus = transitions[transition_key]
                                break
                
                total_score = current_score + transition_bonus + remaining_score
                max_score = max(max_score, total_score)
        
        memo[start] = max_score
        return max_score
    
    result = dfs(0)
    
    # 如果无法分割，返回0
    if result == -float('inf'):
        return 0
    
    return result

if __name__ == "__main__":
    result = solve_tokenization()
    print(result)