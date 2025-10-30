def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Read the wheels information.
    wheels = []
    for _ in range(N):
        C = int(next(it))
        P = int(next(it))
        S = [int(next(it)) for __ in range(P)]
        wheels.append((C, P, S))
    
    # dp[x] will hold the minimal expected additional cost to reach at least M points given current points x.
    # For states x >= M we have dp[x] = 0, thus we keep an array sized M+1 and set dp[M]=0 (in our recurrence, we treat
    # any state x+s >= M as success, cost 0).
    dp = [0.0] * (M + 1)
    
    # Compute dp from state M-1 backwards to state 0.
    for x in range(M - 1, -1, -1):
        best_cost = float('inf')
        # Evaluate each wheel choice.
        for (C, P, Slist) in wheels:
            expected_cost = C  # cost for playing this wheel once.
            total = 0.0
            for score in Slist:
                new_points = x + score
                if new_points >= M:
                    total += 0.0  # If threshold reached, no further cost.
                else:
                    total += dp[new_points]
            # The expected cost for this wheel is:
            expected_cost += total / P
            if expected_cost < best_cost:
                best_cost = expected_cost
        dp[x] = best_cost

    # Print the minimal expected cost starting from 0 points.
    print(dp[0])

if __name__ == '__main__':
    main()