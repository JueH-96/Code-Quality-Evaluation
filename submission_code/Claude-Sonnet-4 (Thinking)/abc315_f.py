import math

N = int(input())
points = []
for _ in range(N):
    x, y = map(int, input().split())
    points.append((x, y))

def distance(i, j):
    x1, y1 = points[i]
    x2, y2 = points[j]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def penalty(k):
    if k == 0:
        return 0
    else:
        return 2**(k-1)

# dp[i] = minimum cost to reach checkpoint i (0-indexed)
dp = [float('inf')] * N
dp[0] = 0

for i in range(1, N):
    for j in range(i):
        skipped = i - j - 1
        cost = dp[j] + distance(j, i) + penalty(skipped)
        dp[i] = min(dp[i], cost)

print(dp[N-1])