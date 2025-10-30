N = int(input())

# Build adjacency matrix
adj = [[0] * N for _ in range(N)]

for i in range(N-1):
    weights = list(map(int, input().split()))
    for j in range(len(weights)):
        adj[i][i+1+j] = weights[j]
        adj[i+1+j][i] = weights[j]  # undirected

# DP with bitmask
dp = [0] * (1 << N)

for mask in range(1 << N):
    for i in range(N):
        if mask & (1 << i):
            continue
        for j in range(i+1, N):
            if mask & (1 << j):
                continue
            # Try to add edge (i,j)
            new_mask = mask | (1 << i) | (1 << j)
            dp[new_mask] = max(dp[new_mask], dp[mask] + adj[i][j])

print(max(dp))