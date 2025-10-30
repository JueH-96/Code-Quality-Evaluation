# YOUR CODE HERE
def min_cost(N, S, M, L):
    # Initialize a large number for comparison
    INF = float('inf')
    
    # dp[i] will store the minimum cost to buy at least i eggs
    dp = [INF] * (N + 1)
    dp[0] = 0  # No cost for 0 eggs
    
    for i in range(1, N + 1):
        if i >= 6:
            dp[i] = min(dp[i], dp[i - 6] + S)
        if i >= 8:
            dp[i] = min(dp[i], dp[i - 8] + M)
        if i >= 12:
            dp[i] = min(dp[i], dp[i - 12] + L)
    
    return dp[N]

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
S = int(data[1])
M = int(data[2])
L = int(data[3])

# Output the result
print(min_cost(N, S, M, L))