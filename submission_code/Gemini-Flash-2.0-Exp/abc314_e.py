def solve():
    n, m = map(int, input().split())
    wheels = []
    for _ in range(n):
        line = list(map(int, input().split()))
        c = line[0]
        p = line[1]
        s = line[2:]
        wheels.append((c, p, s))

    dp = {}

    def calculate_expected_cost(current_points):
        if current_points >= m:
            return 0.0
        
        if current_points in dp:
            return dp[current_points]

        min_expected_cost = float('inf')
        for c, p, s in wheels:
            expected_cost_for_wheel = 0.0
            for outcome in s:
                expected_cost_for_wheel += calculate_expected_cost(current_points + outcome)
            expected_cost_for_wheel /= p
            expected_cost_for_wheel += c
            min_expected_cost = min(min_expected_cost, expected_cost_for_wheel)

        dp[current_points] = min_expected_cost
        return min_expected_cost

    result = calculate_expected_cost(0)
    print(result)

solve()