# YOUR CODE HERE
n, m = map(int, input().split())

graph = [[] for _ in range(n + 1)]

for _ in range(m):
    u, v, b, c = map(int, input().split())
    graph[u].append((v, b, c))

def can_achieve_ratio(ratio):
    # dp[i] = maximum value of sum(beauty - ratio * cost) for paths from 1 to i
    dp = [-float('inf')] * (n + 1)
    dp[1] = 0
    
    for i in range(1, n + 1):
        if dp[i] == -float('inf'):
            continue
        for j, beauty, cost in graph[i]:
            weight = beauty - ratio * cost
            dp[j] = max(dp[j], dp[i] + weight)
    
    return dp[n] >= 0

# Binary search on the ratio
left, right = 0.0, 10000.0

for _ in range(100):
    mid = (left + right) / 2
    if can_achieve_ratio(mid):
        left = mid
    else:
        right = mid

print(f"{left:.16f}")