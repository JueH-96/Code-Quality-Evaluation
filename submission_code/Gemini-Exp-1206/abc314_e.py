def solve():
    n, m = map(int, input().split())
    wheels = []
    for _ in range(n):
        line = list(map(int, input().split()))
        wheels.append((line[0], line[1], line[2:]))

    dp = [float('inf')] * (m + 1)
    dp[m] = 0

    for points in range(m - 1, -1, -1):
        best_expected_cost = float('inf')
        for cost, p, s_values in wheels:
            expected_cost = cost
            total_prob = 0
            for s in s_values:
                next_points = min(m, points + s)
                expected_cost += dp[next_points] / p
                total_prob += 1/p
            
            
            best_expected_cost = min(best_expected_cost, expected_cost)
        dp[points] = best_expected_cost

    print(dp[0])

solve()