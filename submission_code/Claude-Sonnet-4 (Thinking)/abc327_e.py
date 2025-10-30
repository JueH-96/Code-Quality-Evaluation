import math

N = int(input())
P = list(map(int, input().split()))

max_rating = float('-inf')

for k in range(1, N + 1):
    # DP to find the maximum weighted sum for choosing exactly k contests
    dp = [[-float('inf')] * (k + 1) for _ in range(N + 1)]
    
    # Base case: choosing 0 contests gives sum 0
    for i in range(N + 1):
        dp[i][0] = 0
    
    for i in range(1, N + 1):
        for j in range(min(i, k) + 1):
            # Option 1: Don't choose contest i
            dp[i][j] = dp[i-1][j]
            
            # Option 2: Choose contest i (if j > 0)
            if j > 0:
                weight = (0.9) ** (k - j)
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + weight * P[i-1])
    
    # Calculate rating for choosing exactly k contests
    weighted_sum = dp[N][k]
    sum_of_weights = (1 - (0.9) ** k) / (1 - 0.9)
    rating = weighted_sum / sum_of_weights - 1200 / math.sqrt(k)
    max_rating = max(max_rating, rating)

print(max_rating)