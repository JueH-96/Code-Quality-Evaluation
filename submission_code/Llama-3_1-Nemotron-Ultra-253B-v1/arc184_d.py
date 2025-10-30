MOD = 998244353

n = int(input())
points = [tuple(map(int, input().split())) for _ in range(n)]
points.sort()
y = [y for x, y in points]

from collections import defaultdict

dp = defaultdict(int)
dp[float('inf')] = 1  # Start with no elements selected

for yi in y:
    new_dp = defaultdict(int)
    for y_prev, cnt in dp.items():
        # Option 1: Include the current ball
        if yi <= y_prev:
            new_dp[yi] = (new_dp[yi] + cnt) % MOD
        # Option 2: Exclude the current ball
        if y_prev < yi:
            new_dp[y_prev] = (new_dp[y_prev] + cnt) % MOD
    dp = new_dp

# The empty set is not considered a valid solution
print((sum(dp.values()) - 1) % MOD)