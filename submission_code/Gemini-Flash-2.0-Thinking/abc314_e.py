def solve():
    n, m = map(int, input().split())
    costs = []
    probs = []
    outcomes = []
    for _ in range(n):
        line = list(map(int, input().split()))
        c = line[0]
        p = line[1]
        s = line[2:]
        costs.append(c)
        probs.append(p)
        outcomes.append(s)

    dp = [0.0] * m
    epsilon = 1e-7
    while True:
        new_dp = list(dp)
        max_diff = 0
        for j in range(m):
            min_expected_cost = float('inf')
            for i in range(n):
                expected_cost_wheel_i = costs[i]
                for outcome in outcomes[i]:
                    next_state = min(m, j + outcome)
                    if next_state < m:
                        expected_cost_wheel_i += (1 / probs[i]) * new_dp[next_state]
                min_expected_cost = min(min_expected_cost, expected_cost_wheel_i)
            new_dp[j] = min_expected_cost
            max_diff = max(max_diff, abs(new_dp[j] - dp[j]))

        dp = new_dp
        if max_diff < epsilon:
            break

    print(f"{dp[0]:.10f}")

solve()