# YOUR CODE HERE
N, M = map(int, input().split())

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

W = [0] + list(map(int, input().split()))
A = [0] + list(map(int, input().split()))

# Memoization for f(x)
memo = {}

def f(x):
    if x in memo:
        return memo[x]
    
    # Get all neighbors of x
    neighbors = adj[x]
    
    # Use DP to find the best subset
    # dp[w] = maximum sum of f(y) for subset with total weight exactly w
    # We want subsets with total weight < W[x]
    dp = [0] * W[x]  # dp[0] to dp[W[x]-1]
    
    for y in neighbors:
        f_y = f(y)
        w_y = W[y]
        if w_y < W[x]:  # Only consider if we can include this neighbor
            # Iterate in reverse to avoid using the same item twice
            for w in range(W[x] - 1, w_y - 1, -1):
                dp[w] = max(dp[w], dp[w - w_y] + f_y)
    
    # Find the maximum value among all valid weights
    max_value = max(dp)
    
    memo[x] = 1 + max_value
    return 1 + max_value

# Calculate total operations
total = 0
for i in range(1, N + 1):
    if A[i] > 0:
        total += A[i] * f(i)

print(total)