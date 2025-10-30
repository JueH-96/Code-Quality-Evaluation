def main():
    import sys
    input = sys.stdin.readline
    N, K, P = map(int, input().split())
    plans = []
    for _ in range(N):
        data = list(map(int, input().split()))
        cost = data[0]
        gains = data[1:]
        plans.append((cost, gains))
    
    # We use a dictionary for DP: key is a tuple representing parameters progress (each in [0, P])
    # and value is the minimum cost to achieve that state.
    init_state = tuple([0] * K)
    dp = {init_state: 0}
    
    # For each development plan, update dp states.
    for cost, gain in plans:
        # We iterate over a copy to avoid conflicts.
        curr_dp = list(dp.items())
        for state, curr_cost in curr_dp:
            new_state = []
            for s, a in zip(state, gain):
                ns = s + a
                if ns > P:
                    ns = P
                new_state.append(ns)
            new_state = tuple(new_state)
            new_cost = curr_cost + cost
            if new_state not in dp or dp[new_state] > new_cost:
                dp[new_state] = new_cost

    # The target state is one where every parameter is at least P.
    target = tuple([P] * K)
    if target in dp:
        print(dp[target])
    else:
        print(-1)

if __name__ == '__main__':
    main()