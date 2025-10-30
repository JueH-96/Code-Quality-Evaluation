import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def penalty(skipped):
    if skipped == 0:
        return 0
    return 2 ** (skipped - 1)

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

# dp[i] = minimum cost to reach checkpoint i (0-indexed)
dp = [float('inf')] * n
dp[0] = 0  # Starting at checkpoint 1 (index 0)

for i in range(1, n):
    for j in range(i):
        # Going from checkpoint j to checkpoint i
        # We skip (i - j - 1) checkpoints
        skipped = i - j - 1
        dist = distance(points[j][0], points[j][1], points[i][0], points[i][1])
        cost = dp[j] + dist + penalty(skipped)
        dp[i] = min(dp[i], cost)

print(dp[n-1])