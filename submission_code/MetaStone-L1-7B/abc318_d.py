n = int(input())
D = [[0] * n for _ in range(n)]
for i in range(n - 1):
    line = list(map(int, input().split()))
    for j in range(i + 1, n):
        D[i][j] = line[j - (i + 1)]

max_mask = 1 << n
dp = [-1] * max_mask
dp[0] = 0

for mask in range(max_mask):
    if dp[mask] == -1:
        continue
    for u in range(n):
        if not (mask & (1 << u)):
            for v in range(u + 1, n):
                if not (mask & (1 << v)):
                    new_mask = mask | (1 << u) | (1 << v)
                    if dp[new_mask] < dp[mask] + D[u][v]:
                        dp[new_mask] = dp[mask] + D[u][v]

print(max(dp))