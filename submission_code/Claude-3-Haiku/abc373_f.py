import sys

# Read input
N, W = map(int, input().split())
items = []
for _ in range(N):
    w, v = map(int, input().split())
    items.append((w, v))

# Define the happiness function
def happiness(k):
    return k * v - k ** 2

# Solve the problem
def solve():
    dp = [[0] * (W + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        w, v = items[i - 1]
        for j in range(W + 1):
            for k in range(10 ** 10 + 1):
                if j >= k * w:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - k * w] + happiness(k))
    return dp[N][W]

print(solve())