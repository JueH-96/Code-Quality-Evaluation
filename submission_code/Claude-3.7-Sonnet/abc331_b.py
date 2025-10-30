def min_cost(N, S, M, L):
    # dp[i] represents the minimum cost to buy at least i eggs
    dp = [float('inf')] * (N + 1)
    dp[0] = 0
    
    for i in range(1, N + 1):
        dp[i] = min(
            S + dp[max(0, i - 6)],
            M + dp[max(0, i - 8)],
            L + dp[max(0, i - 12)]
        )
    
    return dp[N]

# Read input values
N, S, M, L = map(int, input().split())

# Calculate and print the answer
print(min_cost(N, S, M, L))