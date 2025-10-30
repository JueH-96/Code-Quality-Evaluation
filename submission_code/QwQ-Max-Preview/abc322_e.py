def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    P = int(input[idx]); idx +=1

    plans = []
    for _ in range(N):
        C_i = int(input[idx]); idx +=1
        A_i = list(map(int, input[idx:idx+K]))
        idx += K
        plans.append( (C_i, A_i) )

    # Initialize DP with initial state (0, 0, ..., 0)
    initial_state = tuple([0]*K)
    dp = {initial_state: 0}

    for (C_i, A_i) in plans:
        temp_dp = {}
        for state in list(dp.keys()):
            current_cost = dp[state]
            new_params = list(state)
            for j in range(K):
                new_params[j] = min(new_params[j] + A_i[j], P)
            new_state = tuple(new_params)
            new_cost = current_cost + C_i
            if new_state not in temp_dp or new_cost < temp_dp[new_state]:
                temp_dp[new_state] = new_cost
        # Merge temp_dp into dp
        for state in temp_dp:
            if state not in dp or temp_dp[state] < dp[state]:
                dp[state] = temp_dp[state]

    min_cost = float('inf')
    for state in dp:
        if all(s >= P for s in state):
            if dp[state] < min_cost:
                min_cost = dp[state]

    print(-1 if min_cost == float('inf') else min_cost)

if __name__ == "__main__":
    main()