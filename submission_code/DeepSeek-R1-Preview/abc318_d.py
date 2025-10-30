n = int(input())
D = [[0] * (n + 1) for _ in range(n + 1)]

for i in range(1, n):
    arr = list(map(int, input().split()))
    idx = 0
    for j in range(i + 1, n + 1):
        D[i][j] = arr[idx]
        D[j][i] = arr[idx]
        idx += 1

max_mask = 1 << n
dp = [0] * max_mask

for mask in range(max_mask):
    # Find the first unused vertex
    for u in range(1, n + 1):
        if not (mask & (1 << (u - 1))):
            break
    else:
        continue  # All vertices are used

    # Try to pair u with all possible v > u
    for v in range(u + 1, n + 1):
        if not (mask & (1 << (v - 1))):
            new_mask = mask | (1 << (u - 1)) | (1 << (v - 1))
            dp[new_mask] = max(dp[new_mask], dp[mask] + D[u][v])

print(max(dp))