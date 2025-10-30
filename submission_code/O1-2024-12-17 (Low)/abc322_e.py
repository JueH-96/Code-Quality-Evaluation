def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    P = int(input_data[2])

    # C[i] = cost of i-th plan
    # A[i] = list of increments for i-th plan (length = K)
    C = []
    A = []
    idx = 3
    for _ in range(N):
        c_i = int(input_data[idx]); idx += 1
        C.append(c_i)
        a_i = [int(x) for x in input_data[idx:idx+K]]
        idx += K
        A.append(a_i)

    # We use a dynamic programming approach where
    # dp[state] represents the minimum cost to achieve state,
    # where state is a K-tuple (v1, v2, ..., vK) with 0 <= vj <= P.
    # For each plan, we can either skip it or use it once.
    # We start from dp[(0,0,...,0)] = 0.

    from collections import defaultdict
    INF = 10**18
    # dp will map a K-tuple to a minimum cost
    dp = {(0,)*K: 0}

    for i in range(N):
        cost_i = C[i]
        inc = A[i]
        new_dp = defaultdict(lambda: INF)

        for state, curr_cost in dp.items():
            # Skip plan i: keep the same state
            if curr_cost < new_dp[state]:
                new_dp[state] = curr_cost

            # Use plan i: move to the new state
            new_state = []
            for j in range(K):
                nxt_val = state[j] + inc[j]
                if nxt_val > P:
                    nxt_val = P
                new_state.append(nxt_val)
            new_state = tuple(new_state)
            use_cost = curr_cost + cost_i
            if use_cost < new_dp[new_state]:
                new_dp[new_state] = use_cost

        dp = new_dp

    # The goal state is (P, P, ..., P)
    goal_state = (P,)*K
    answer = dp.get(goal_state, INF)
    if answer == INF:
        print(-1)
    else:
        print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()