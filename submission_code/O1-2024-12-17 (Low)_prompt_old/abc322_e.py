def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K, P = map(int, input_data[:3])

    # Read the plan data
    plans = []
    idx = 3
    for _ in range(N):
        cost = int(input_data[idx]); idx += 1
        inc = list(map(int, input_data[idx:idx+K]))
        idx += K
        plans.append((cost, inc))

    # We'll use dynamic programming where dp is a dictionary:
    # Key (x1, x2, ..., xK) = minimum cost to achieve these parameter levels
    # Because P <= 5, we clamp each parameter level not to exceed P.
    from collections import defaultdict

    # Initialize dp with only state (0,0,...,0) at cost 0
    start_state = tuple([0]*K)
    dp = {start_state: 0}

    for cost, increments in plans:
        new_dp = dict(dp)  # Make a copy so we can update
        for state, current_cost in dp.items():
            # Calculate new state after using this plan
            new_state = list(state)
            for j in range(K):
                new_state[j] = min(P, new_state[j] + increments[j])
            new_state = tuple(new_state)

            new_cost = current_cost + cost
            if new_state not in new_dp or new_cost < new_dp[new_state]:
                new_dp[new_state] = new_cost
        dp = new_dp

    # The goal state is (P, P, ..., P)
    goal_state = tuple([P]*K)

    # Check if we can achieve it
    if goal_state in dp:
        print(dp[goal_state])
    else:
        print(-1)

# Call solve() after defining it
solve()