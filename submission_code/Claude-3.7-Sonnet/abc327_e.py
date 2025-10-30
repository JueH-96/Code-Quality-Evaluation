def max_rating(N, P):
    max_r = float('-inf')
    
    for k in range(1, N+1):
        # Compute the maximum weighted sum for k chosen contests
        dp = [[-float('inf')] * (k+1) for _ in range(N+1)]
        dp[0][0] = 0
        
        for i in range(1, N+1):
            dp[i][0] = 0
            for j in range(1, min(i+1, k+1)):
                dp[i][j] = dp[i-1][j]  # Don't choose the i-th contest
                weight = (0.9) ** (k-j)
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + weight * P[i-1])  # Choose the i-th contest
        
        max_weighted_sum = dp[N][k]
        
        # Compute the rating
        denom = (1 - (0.9)**k) / (1 - 0.9)
        r = max_weighted_sum / denom - 1200 / (k ** 0.5)
        
        max_r = max(max_r, r)
    
    return max_r

# Read input
N = int(input().strip())
P = list(map(int, input().strip().split()))

# Compute the maximum rating
result = max_rating(N, P)

# Print the result
print(result)