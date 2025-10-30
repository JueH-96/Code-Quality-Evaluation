# YOUR CODE HERE
n = int(input())
weight = [[0] * n for _ in range(n)]

# Read the weights
for i in range(n - 1):
    line = list(map(int, input().split()))
    for j in range(len(line)):
        weight[i][i + 1 + j] = line[j]
        weight[i + 1 + j][i] = line[j]

# DP with bitmask
dp = [0] * (1 << n)

for mask in range(1 << n):
    # Try to add a new edge
    for i in range(n):
        if mask & (1 << i):
            continue
        for j in range(i + 1, n):
            if mask & (1 << j):
                continue
            new_mask = mask | (1 << i) | (1 << j)
            dp[new_mask] = max(dp[new_mask], dp[mask] + weight[i][j])

print(max(dp))