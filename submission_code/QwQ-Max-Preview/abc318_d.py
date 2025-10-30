n = int(input())
d = [[0] * n for _ in range(n)]
current_i = 0
for i in range(n-1):
    line = list(map(int, input().split()))
    current_j = i + 1
    for val in line:
        d[i][current_j] = val
        current_j += 1

max_mask = 1 << n
dp = [0] * max_mask

for mask in range(max_mask):
    for i in range(n):
        if not (mask & (1 << i)):
            for j in range(i + 1, n):
                if not (mask & (1 << j)):
                    new_mask = mask | (1 << i) | (1 << j)
                    if dp[new_mask] < dp[mask] + d[i][j]:
                        dp[new_mask] = dp[mask] + d[i][j]

print(max(dp))