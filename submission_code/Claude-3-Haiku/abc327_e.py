import math

def max_rating(performances):
    n = len(performances)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            if j == 1:
                dp[i][j] = performances[i - 1]
            else:
                dp[i][j] = max(dp[i - 1][j - 1] * 0.9 + performances[i - 1], dp[i - 1][j])
    
    max_score = float('-inf')
    for j in range(1, n + 1):
        rating = (sum(dp[n][j] * 0.9 ** (j - i) for i in range(j)) / sum(0.9 ** (j - i) for i in range(j))) - 1200 / math.sqrt(j)
        max_score = max(max_score, rating)
    
    return max_score

# Read input
n = int(input())
performances = [int(x) for x in input().split()]

# Compute the maximum rating
max_score = max_rating(performances)
print(f"{max_score:.15f}")