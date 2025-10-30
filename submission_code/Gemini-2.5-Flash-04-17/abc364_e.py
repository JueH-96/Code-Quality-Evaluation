import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = []
    for _ in range(N):
        A, B = map(int, sys.stdin.readline().split())
        dishes.append((A, B))

    INF = float('inf')

    # Decide which dimension (sweetness or saltiness) to use for the DP state.
    # We use the one with the smaller limit (X or Y) to minimize state space.
    # This optimization assumes the DP structure is symmetric with respect to sweetness and saltiness.
    # Let's use sweetness as the dimension if X <= Y, and saltiness if Y < X.

    use_sweetness_as_state = X <= Y

    if use_sweetness_as_state:
        # dp[k][s] = minimum total saltiness 't' achievable using a *valid sequence*
        # of exactly k dishes chosen from the input dishes, such that the total sweetness sum is 's'.
        # A sequence d_1, ..., d_k is valid if for all p from 1 to k,
        # sum(A_d_i for i=1 to p) <= X and sum(B_d_i for i=1 to p) <= Y.
        # The value dp[k][s] stores the minimum such 't'. If no such sequence exists, it's infinity.

        MAX_DIM_VAL = X
        dp = [[INF for _ in range(MAX_DIM_VAL + 1)] for _ in range(N + 1)]

        # Base case: A valid sequence of 0 dishes has sweetness 0 and saltiness 0.
        dp[0][0] = 0

        # Iterate through each dish (a, b)
        for a, b in dishes:
            # To avoid using the same dish multiple times within the same sequence length k,
            # we can iterate through k and the state dimension in reverse order.
            # When adding dish (a, b) to a sequence of length k ending in (s_prev, t_prev),
            # the new sequence of length k+1 ends in (s_prev + a, t_prev + b).
            # This new sequence is valid only if s_prev + a <= X and t_prev + b <= Y.
            # We update dp[k+1][new_s] = min(dp[k+1][new_s], new_t) if valid.

            for k in range(N - 1, -1, -1):
                # The new sweetness will be s_prev + a. We only track new sweetness <= X.
                # So, the current sweetness `s` in dp[k][s] (which becomes s_prev)
                # needs to be such that s + a <= X. This means s <= X - a.
                # Iterating `s` from MAX_DIM_VAL down to 0 covers all possibilities.
                # We only make a transition if dp[k][s] is reachable.
                # new_s = s + a

                for s in range(MAX_DIM_VAL, -1, -1):
                    if dp[k][s] < INF:
                        new_s = s + a
                        new_t = dp[k][s] + b

                        # Check if adding this dish keeps the new sequence valid
                        # Both the new sweetness and the new total saltiness must be within limits.
                        if new_s <= X and new_t <= Y:
                            # Update the minimum saltiness for a valid sequence of length k+1 ending in new_s.
                            dp[k + 1][new_s] = min(dp[k + 1][new_s], new_t)

        # After processing all dishes, find the maximum k for which dp[k][s] is reachable (finite)
        # for at least one sweetness sum s (0 <= s <= MAX_DIM_VAL).
        max_k = 0
        for k in range(N, -1, -1):
            for s in range(MAX_DIM_VAL + 1):
                if dp[k][s] < INF:
                    max_k = k
                    # Since we are iterating k downwards, the first k for which we find a reachable state is the maximum.
                    # We found the largest k, no need to check smaller k.
                    break
            if max_k > 0:
                break

    else: # use_saltiness_as_state (Y < X)
        # dp[k][t] = minimum total sweetness 's' achievable using a *valid sequence*
        # of exactly k dishes chosen from the input dishes, such that the total saltiness sum is 't'.
        # A sequence d_1, ..., d_k is valid if for all p from 1 to k,
        # sum(A_d_i for i=1 to p) <= X and sum(B_d_i for i=1 to p) <= Y.
        # The value dp[k][t] stores the minimum such 's'. If no such sequence exists, it's infinity.

        MAX_DIM_VAL = Y
        dp = [[INF for _ in range(MAX_DIM_VAL + 1)] for _ in range(N + 1)]

        # Base case: A valid sequence of 0 dishes has sweetness 0 and saltiness 0.
        dp[0][0] = 0

        # Iterate through each dish (a, b)
        for a, b in dishes:
            # Swap roles of sweetness (a) and saltiness (b)
            for k in range(N - 1, -1, -1):
                # The new saltiness will be t_prev + b. We only track new saltiness <= Y.
                # So, the current saltiness `t` in dp[k][t] (which becomes t_prev)
                # needs to be such that t + b <= Y. This means t <= Y - b.
                # Iterating `t` from MAX_DIM_VAL down to 0 covers all possibilities.
                # We only make a transition if dp[k][t] is reachable.
                # new_t = t + b

                for t in range(MAX_DIM_VAL, -1, -1):
                    if dp[k][t] < INF:
                        new_t = t + b
                        new_s = dp[k][t] + a # The stored value is min sweetness

                        # Check if adding this dish keeps the new sequence valid
                        # Both the new saltiness and the new total sweetness must be within limits.
                        if new_t <= Y and new_s <= X:
                            # Update the minimum sweetness for a valid sequence of length k+1 ending in new_t.
                            dp[k + 1][new_t] = min(dp[k + 1][new_t], new_s)

        # After processing all dishes, find the maximum k for which dp[k][t] is reachable (finite)
        # for at least one saltiness sum t (0 <= t <= MAX_DIM_VAL).
        max_k = 0
        for k in range(N, -1, -1):
            for t in range(MAX_DIM_VAL + 1):
                if dp[k][t] < INF:
                    max_k = k
                    # Since we are iterating k downwards, the first k for which we find a reachable state is the maximum.
                    # We found the largest k, no need to check smaller k.
                    break
            if max_k > 0:
                break

    # The maximum number of dishes Snuke will eat is the maximum k found.
    # The base case k=0 is always possible (eating 0 dishes). max_k is initialized to 0.

    print(max_k)

solve()