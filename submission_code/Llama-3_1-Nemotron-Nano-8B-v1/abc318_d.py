n = int(input())
d = [[0] * n for _ in range(n)]
current_line = 0
for i in range(n - 1):
    row = list(map(int, input().split()))
    for j in range(len(row)):
        d[i][i + j + 1] = row[j]
        d[i + j + 1][i] = row[j]

edges = []
for u in range(n):
    for v in range(u + 1, n):
        edges.append((u, v))

from collections import defaultdict
dp = defaultdict(int)
dp[0] = 0

mask_count = 1 << n
for mask in range(mask_count):
    if mask not in dp:
        continue
    available = []
    for i in range(n):
        if not (mask & (1 << i)):
            available.append(i)
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            u = available[i]
            v = available[j]
            new_mask = mask | (1 << u) | (1 << v)
            new_sum = dp[mask] + d[u][v]
            if new_sum > dp.get(new_mask, 0):
                dp[new_mask] = new_sum

print(max(dp.values()))