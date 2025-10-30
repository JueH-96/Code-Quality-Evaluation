def solve():
    n, m = map(int, input().split())
    wheels = []
    for _ in range(n):
        line = list(map(int, input().split()))
        c = line[0]
        p = line[1]
        s = line[2:]
        wheels.append((c, p, s))

    dp = [float('inf')] * (m + 1)
    dp[0] = 0

    for target_points in range(1, m + 1):
        for c, p, s in wheels:
            expected_cost = c
            expected_next_dp = 0
            for score in s:
                next_points = min(m, target_points - score)
                expected_next_dp += dp[max(0, next_points)]
            
            expected_cost += expected_next_dp / p
            dp[target_points] = min(dp[target_points], expected_cost)
    
    print(f"{dp[m]:.15f}")

solve()