def main():
    import sys
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    N = int(next(it))
    K = int(next(it))
    P = int(next(it))
    plans = []
    for _ in range(N):
        cost = int(next(it))
        gains = [int(next(it)) for _ in range(K)]
        plans.append((cost, gains))
    
    # We use dynamic programming with a dictionary.
    # Each state is a tuple of K values where each value is at most P.
    # dp[state] = minimum cost to achieve that state
    dp = { (0,) * K: 0 }
    
    # Process each plan (each can be used at most once)
    for cost, gains in plans:
        # Copy current dp to update states using this plan
        new_dp = dp.copy()
        for state, curr_cost in dp.items():
            # Calculate new state by adding the gains, capped at P.
            new_state = tuple(min(P, state[i] + gains[i]) for i in range(K))
            new_cost = curr_cost + cost
            if new_state not in new_dp or new_dp[new_state] > new_cost:
                new_dp[new_state] = new_cost
        dp = new_dp
    
    target = (P,) * K
    # Output the minimum cost if the target state can be reached, otherwise -1.
    if target in dp:
        sys.stdout.write(str(dp[target]))
    else:
        sys.stdout.write("-1")
    
if __name__ == '__main__':
    main()