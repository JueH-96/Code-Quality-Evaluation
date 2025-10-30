def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    
    # Read the development plans
    plans = []
    for _ in range(N):
        cost = int(next(it))
        increases = [int(next(it)) for _ in range(K)]
        plans.append((cost, increases))
    
    # dp[state] holds the minimum total cost to achieve 'state'
    # Here, state is represented as a tuple of K numbers (each from 0 to P, where above P is capped to P).
    # Initially, all parameters are 0.
    dp = {tuple([0] * K): 0}
    
    for cost, inc in plans:
        # For each plan, update dp.
        # We create a temporary dictionary to record new states after using this plan.
        new_dp = dp.copy()
        for state, curr_cost in dp.items():
            # Compute new state by adding the plan's values, saturating each parameter at P.
            new_state = []
            for j in range(K):
                new_val = state[j] + inc[j]
                if new_val > P:
                    new_val = P
                new_state.append(new_val)
            new_state = tuple(new_state)
            new_cost = curr_cost + cost
            if new_state not in new_dp or new_cost < new_dp[new_state]:
                new_dp[new_state] = new_cost
        dp = new_dp

    # The target state is when all parameters are at least P.
    target = tuple([P] * K)
    print(dp[target] if target in dp else -1)

if __name__ == '__main__':
    main()