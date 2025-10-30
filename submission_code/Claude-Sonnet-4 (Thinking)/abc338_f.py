from collections import defaultdict

N, M = map(int, input().split())

# Read edges
graph = defaultdict(list)
for _ in range(M):
    u, v, w = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1
    graph[u].append((v, w))

# DP with bitmask
INF = float('inf')
dp = [[INF] * N for _ in range(1 << N)]

# Initialize: start from each vertex
for i in range(N):
    dp[1 << i][i] = 0

# Fill DP table
for mask in range(1 << N):
    for u in range(N):
        if dp[mask][u] == INF:
            continue
        
        for v, w in graph[u]:
            new_mask = mask | (1 << v)
            dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + w)

# Find answer
full_mask = (1 << N) - 1
ans = min(dp[full_mask])

if ans == INF:
    print("No")
else:
    print(ans)