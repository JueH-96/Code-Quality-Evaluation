def solve():
    n, k, p = map(int, input().split())
    plans = []
    for _ in range(n):
        plans.append(list(map(int, input().split())))

    dp = {}
    dp[(0,) * k] = 0

    for i in range(n):
        new_dp = dp.copy()
        cost_i = plans[i][0]
        a_i = plans[i][1:]

        for current_state, current_cost in dp.items():
            next_state = tuple(min(p, current_state[j] + a_i[j]) for j in range(k))
            new_dp[next_state] = min(new_dp.get(next_state, float('inf')), current_cost + cost_i)
        dp = new_dp

    min_cost = float('inf')
    for state, cost in dp.items():
        if all(val >= p for val in state):
            min_cost = min(min_cost, cost)

    if min_cost == float('inf'):
        print(-1)
    else:
        print(min_cost)

solve()