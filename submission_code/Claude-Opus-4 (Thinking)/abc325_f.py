# YOUR CODE HERE
N = int(input())
D = list(map(int, input().split()))
L1, C1, K1 = map(int, input().split())
L2, C2, K2 = map(int, input().split())

# For each section, find all valid (x, y) pairs
valid_pairs = []
for d in D:
    pairs = []
    # Try all possible numbers of type-1 sensors
    for x in range(min(K1 + 1, (d + L1 - 1) // L1 + 1)):
        remaining = max(0, d - x * L1)
        y = (remaining + L2 - 1) // L2 if remaining > 0 else 0
        if y <= K2:
            pairs.append((x, y))
    
    if not pairs:
        print(-1)
        exit()
    valid_pairs.append(pairs)

# DP: dp[i][j][k] = min cost to monitor first i sections using exactly j type-1 and k type-2 sensors
INF = float('inf')
dp = [[[INF] * (K2 + 1) for _ in range(K1 + 1)] for _ in range(N + 1)]
dp[0][0][0] = 0

for i in range(1, N + 1):
    for j in range(K1 + 1):
        for k in range(K2 + 1):
            if dp[i - 1][j][k] == INF:
                continue
            for x, y in valid_pairs[i - 1]:
                if j + x <= K1 and k + y <= K2:
                    cost = x * C1 + y * C2
                    dp[i][j + x][k + y] = min(dp[i][j + x][k + y], dp[i - 1][j][k] + cost)

result = INF
for j in range(K1 + 1):
    for k in range(K2 + 1):
        result = min(result, dp[N][j][k])

if result == INF:
    print(-1)
else:
    print(result)