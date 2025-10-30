import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def penalty(c):
    if c == 0:
        return 0
    else:
        return 2**(c - 1)

n = int(input())
points = []
for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))

# dp[i][c] = minimum distance to reach checkpoint i with c skips
# We limit the number of skips because the penalty grows exponentially
INF = float('inf')
max_skips = min(n - 2, 60)  # Limit to 60 skips
dp = [[INF] * (max_skips + 1) for _ in range(n)]

# Base case: start at checkpoint 0 (1-indexed becomes 0-indexed)
dp[0][0] = 0

# Fill the DP table
for i in range(1, n):
    for j in range(i):
        # Going from checkpoint j to i
        skips = i - j - 1
        if skips > max_skips:
            continue  # Skip if too many checkpoints are skipped
        
        dist = distance(points[j], points[i])
        
        for c in range(max_skips + 1):
            if dp[j][c] != INF and c + skips <= max_skips:
                dp[i][c + skips] = min(dp[i][c + skips], dp[j][c] + dist)

# Find the answer
ans = INF
for c in range(max_skips + 1):
    if dp[n-1][c] != INF:
        total_cost = dp[n-1][c] + penalty(c)
        ans = min(ans, total_cost)

print(f"{ans:.17f}")