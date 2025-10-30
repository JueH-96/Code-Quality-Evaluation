def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    P = int(data[2])

    # Read in the cost and increment arrays for each plan
    idx = 3
    plans = []
    for _ in range(N):
        cost = int(data[idx])
        idx += 1
        increments = list(map(int, data[idx:idx+K]))
        idx += K
        plans.append((cost, increments))

    # We'll use a dictionary-based DP to store minimum cost for a given tuple of parameter levels.
    # Key: a tuple (p1, p2, ..., pK), where p_j is the current level of the j-th parameter.
    # Value: the minimum cost to achieve these parameter levels.
    dp = {(0,) * K: 0}
    INF = 10**18

    for cost, incs in plans:
        new_dp = dict(dp)  # We'll update a copy so each plan can only be used once
        for state, cur_cost in dp.items():
            # Compute new state
            new_state = []
            for i in range(K):
                new_state.append(min(P, state[i] + incs[i]))
            new_state = tuple(new_state)
            new_cost = cur_cost + cost

            if new_cost < new_dp.get(new_state, INF):
                new_dp[new_state] = new_cost

        dp = new_dp

    # Our goal state has all parameters at least P
    goal_state = (P,) * K
    answer = dp.get(goal_state, INF)
    print(answer if answer < INF else -1)

# Do not forget to call main()
main()