def main():
    import sys
    input = sys.stdin.readline
    
    N, K, P = map(int, input().split())
    plans = []
    for _ in range(N):
        data = list(map(int, input().split()))
        c = data[0]
        a = data[1:]
        plans.append((c, a))

    # dp[i] will map from a K-tuple (p1, p2, ..., pK) where each p_j is in [0, P]
    # to the minimum cost needed to achieve those parameter values using only
    # the first i development plans.
    dp = [dict() for _ in range(N + 1)]
    
    # Initialize: with 0 plans, the only achievable state is (0,0,...,0) at cost 0.
    init_state = tuple([0] * K)
    dp[0][init_state] = 0
    
    def update(dictionary, key, val):
        if key not in dictionary or dictionary[key] > val:
            dictionary[key] = val
    
    # Build dp[i] from dp[i-1]
    for i in range(1, N + 1):
        cost_i, inc = plans[i - 1]
        dp_next = {}
        for state, cost_so_far in dp[i - 1].items():
            # Option 1: Skip the i-th plan
            update(dp_next, state, cost_so_far)
            
            # Option 2: Use the i-th plan
            new_state = list(state)
            for j in range(K):
                new_state[j] = min(P, new_state[j] + inc[j])
            new_state = tuple(new_state)
            update(dp_next, new_state, cost_so_far + cost_i)
        dp[i] = dp_next
    
    # The goal is to have all parameters at least P, so the final "key" is (P, P, ..., P).
    final_state = tuple([P] * K)
    ans = dp[N].get(final_state, None)
    
    # Print result
    if ans is None:
        print(-1)
    else:
        print(ans)

# Do not forget to call main()!
if __name__ == "__main__":
    main()