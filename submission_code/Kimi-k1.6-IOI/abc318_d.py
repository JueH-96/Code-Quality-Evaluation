n = int(input())
d = [[0] * n for _ in range(n)]
for i in range(n-1):
    row = list(map(int, input().split()))
    for j in range(i+1, n):
        val = row[j - (i+1)]
        d[i][j] = val
        d[j][i] = val

size = 1 << n
dp = [-float('inf')] * size
dp[0] = 0

for mask in range(size):
    if dp[mask] == -float('inf'):
        continue
    available = []
    for v in range(n):
        if not (mask & (1 << v)):
            available.append(v)
    for i in range(len(available)):
        for j in range(i+1, len(available)):
            u = available[i]
            v = available[j]
            new_mask = mask | (1 << u) | (1 << v)
            if dp[new_mask] < dp[mask] + d[u][v]:
                dp[new_mask] = dp[mask] + d[u][v]

print(max(dp))