def main():
    import sys
    import itertools

    # Read N, K, P
    N, K, P = map(int, sys.stdin.readline().split())
    # Read N plans
    plans = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        C_i = parts[0]
        A_j = parts[1:]
        plans.append((C_i, A_j))
    
    # Precompute base multipliers
    bases = [(P + 1) ** i for i in range(K-1, -1, -1)]
    
    # Define state_to_index
    def state_to_index(state):
        return sum(s * b for s, b in zip(state, bases))
    
    # Calculate size of DP array
    size = (P + 1) ** K
    # Initialize DP array
    INF = 10**18
    dp_prev = [INF] * size
    start_state = (0,) * K
    start_index = state_to_index(start_state)
    dp_prev[start_index] = 0
    
    # Iterate through each plan
    for plan in plans:
        C_i, A_j = plan
        dp_next = dp_prev[:]
        for state in itertools.product(range(P + 1), repeat=K):
            current_index = state_to_index(state)
            if dp_prev[current_index] == INF:
                continue
            # Calculate new_state
            new_state = tuple(min(s + A_j[k], P) for k, s in enumerate(state))
            new_index = state_to_index(new_state)
            if dp_next[new_index] > dp_prev[current_index] + C_i:
                dp_next[new_index] = dp_prev[current_index] + C_i
        dp_prev = dp_next
    
    # Find the goal state
    goal_state = (P,) * K
    goal_index = state_to_index(goal_state)
    if dp_prev[goal_index] == INF:
        print(-1)
    else:
        print(dp_prev[goal_index])

if __name__ == '__main__':
    main()