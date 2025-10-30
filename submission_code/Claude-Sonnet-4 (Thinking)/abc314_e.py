n, m = map(int, input().split())
wheels = []
for _ in range(n):
    line = list(map(int, input().split()))
    c = line[0]
    p = line[1]
    s = line[2:]
    wheels.append((c, p, s))

# dp[x] = expected cost to reach at least m points starting from x points
dp = [0.0] * m

# Fill dp from back to front
for x in range(m - 1, -1, -1):
    min_cost = float('inf')
    for c, p, s in wheels:
        expected_cost = c
        for outcome in s:
            next_points = x + outcome
            if next_points < m:
                expected_cost += dp[next_points] / p
        min_cost = min(min_cost, expected_cost)
    dp[x] = min_cost

print(dp[0])